# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import gc
import logging
import pickle
from pathlib import Path

from omegaconf import DictConfig

from vipe.slam.system import SLAMOutput, SLAMSystem
from vipe.streams.base import (
    AssignAttributesProcessor,
    FrameAttribute,
    MultiviewVideoList,
    ProcessedVideoStream,
    StreamProcessor,
    VideoStream,
)
from vipe.utils import io
from vipe.utils.cameras import CameraType
from vipe.utils.device import get_device
from vipe.utils.logging import progress_stage
from vipe.utils.visualization import save_projection_video

from . import AnnotationPipelineOutput, Pipeline
from .processors import (
    AdaptiveDepthProcessor,
    GeoCalibIntrinsicsProcessor,
    MultiviewDepthProcessor,
    TrackAnythingProcessor,
    add_invisible_masks,
)

logger = logging.getLogger(__name__)


class DefaultAnnotationPipeline(Pipeline):
    def __init__(self, init: DictConfig, slam: DictConfig, post: DictConfig, output: DictConfig) -> None:
        super().__init__()
        self.init_cfg = init
        self.slam_cfg = slam
        if get_device().type == "cpu":
            self.slam_cfg.ba.fused = False
        self.post_cfg = post
        self.out_cfg = output
        self.out_path = Path(self.out_cfg.path)
        if self.out_cfg.save_artifacts or self.out_cfg.save_viz or self.out_cfg.save_slam_map:
            self.out_path.mkdir(exist_ok=True, parents=True)
        self.camera_type = CameraType(self.init_cfg.camera_type)

    def _add_init_processors(self, video_stream: VideoStream) -> ProcessedVideoStream:
        init_processors: list[StreamProcessor] = []

        # The assertions make sure that the attributes are not estimated previously.
        # Otherwise it will be overwritten by the processors.
        assert FrameAttribute.INTRINSICS not in video_stream.attributes()
        assert FrameAttribute.CAMERA_TYPE not in video_stream.attributes()
        assert FrameAttribute.METRIC_DEPTH not in video_stream.attributes()
        assert FrameAttribute.INSTANCE not in video_stream.attributes()

        init_processors.append(
            GeoCalibIntrinsicsProcessor(video_stream, camera_type=self.camera_type, model_cache=self.model_cache)
        )
        if self.init_cfg.instance is not None:
            init_processors.append(
                TrackAnythingProcessor(
                    self.init_cfg.instance.phrases,
                    add_sky=self.init_cfg.instance.add_sky,
                    sam_run_gap=int(video_stream.fps() * self.init_cfg.instance.kf_gap_sec),
                    model_cache=self.model_cache,
                )
            )
        return ProcessedVideoStream(video_stream, init_processors)

    def _add_post_processors(
        self, view_idx: int, video_stream: VideoStream, slam_output: SLAMOutput
    ) -> ProcessedVideoStream:
        post_processors: list[StreamProcessor] = [
            AssignAttributesProcessor(
                {
                    FrameAttribute.POSE: slam_output.get_view_trajectory(view_idx),  # type: ignore
                    FrameAttribute.INTRINSICS: [slam_output.intrinsics[view_idx]] * len(video_stream),
                }
            )
        ]
        if (depth_align_model := self.post_cfg.depth_align_model) is not None:
            if depth_align_model.startswith("mvd_"):
                post_processors.append(MultiviewDepthProcessor(slam_output, model=depth_align_model))
            else:
                post_processors.append(
                    AdaptiveDepthProcessor(
                        slam_output,
                        view_idx,
                        depth_align_model,
                        cpu_knn_chunk_size=self.post_cfg.cpu_knn_chunk_size,
                    )
                )
        return ProcessedVideoStream(video_stream, post_processors)

    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        if isinstance(video_data, MultiviewVideoList):
            video_streams = [video_data[view_idx] for view_idx in range(len(video_data))]
            artifact_paths = [io.ArtifactPath(self.out_path, video_stream.name()) for video_stream in video_streams]
            slam_rig = video_data.rig()

        else:
            assert isinstance(video_data, VideoStream)
            video_streams = [video_data]
            artifact_paths = [io.ArtifactPath(self.out_path, video_data.name())]
            slam_rig = None

        annotate_output = AnnotationPipelineOutput()

        if all([self.should_filter(video_stream.name()) for video_stream in video_streams]):
            logger.info(f"{video_data.name()} has been proccessed already, skip it!!")
            return annotate_output

        async_prefetch = self.init_cfg.async_prefetch
        prefetch_queue_size = self.init_cfg.prefetch_queue_size
        slam_streams: list[VideoStream] = [
            self._add_init_processors(video_stream).cache(
                "process",
                online=True,
                async_prefetch=async_prefetch,
                prefetch_queue_size=prefetch_queue_size,
            )
            for video_stream in video_streams
        ]

        slam_pipeline = SLAMSystem(device=get_device(), config=self.slam_cfg, model_cache=self.model_cache)
        slam_output = slam_pipeline.run(slam_streams, rig=slam_rig, camera_type=self.camera_type)
        annotate_output.ba_residual = float(slam_output.ba_residual)

        if self.post_cfg.release_completed_models:
            slam_pipeline.release_transient_state()
            for prefix in ("geocalib/", "track_anything/", "slam/", "depth/"):
                self.model_cache.clear_prefix(prefix)
            del slam_pipeline
            gc.collect()

        if self.return_payload:
            annotate_output.payload = slam_output
            return annotate_output

        output_streams: list[VideoStream]
        if self.post_cfg.depth_align_model is not None:
            with progress_stage("Estimate and align metric depth"):
                output_streams = [
                    self._add_post_processors(view_idx, slam_stream, slam_output).cache(
                        "Aligning metric depth", online=True
                    )
                    for view_idx, slam_stream in enumerate(slam_streams)
                ]
                # Materialize post-processing so this stage owns its frame progress and timing.
                for output_stream in output_streams:
                    _ = output_stream[len(output_stream) - 1]
        else:
            output_streams = [
                self._add_post_processors(view_idx, slam_stream, slam_output).cache("Post-processing", online=True)
                for view_idx, slam_stream in enumerate(slam_streams)
            ]

        # Dumping artifacts for all views in the streams
        writes_output = self.out_cfg.save_artifacts or self.out_cfg.save_viz or self.out_cfg.save_slam_map
        needs_final_masks = self.out_cfg.save_artifacts or self.out_cfg.save_viz or self.return_output_streams
        if self.out_cfg.invisible_mask.enabled and needs_final_masks:
            output_streams = [
                add_invisible_masks(output_stream, self.out_cfg.invisible_mask.threshold)
                for output_stream in output_streams
            ]

        if writes_output:
            with progress_stage("Write output files"):
                for output_stream, artifact_path in zip(output_streams, artifact_paths):
                    artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
                    if self.out_cfg.save_artifacts:
                        annotate_output.written_paths = [
                            *annotate_output.written_paths,
                            *io.save_artifacts(artifact_path, output_stream),
                        ]
                        with artifact_path.meta_info_path.open("wb") as f:
                            pickle.dump({"ba_residual": slam_output.ba_residual}, f)
                        annotate_output.written_paths = [
                            *annotate_output.written_paths,
                            artifact_path.meta_info_path,
                        ]

                    if self.out_cfg.save_viz:
                        save_projection_video(
                            artifact_path.meta_vis_path,
                            output_stream,
                            slam_output,
                            self.out_cfg.viz_downsample,
                            self.out_cfg.viz_attributes,
                        )
                        annotate_output.written_paths = [
                            *annotate_output.written_paths,
                            artifact_path.meta_vis_path,
                        ]

                    if self.out_cfg.save_slam_map and slam_output.slam_map is not None:
                        slam_output.slam_map.save(artifact_path.slam_map_path)
                        annotate_output.written_paths = [
                            *annotate_output.written_paths,
                            artifact_path.slam_map_path,
                        ]

        if self.return_output_streams:
            annotate_output.output_streams = output_streams

        return annotate_output
