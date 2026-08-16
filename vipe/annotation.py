# SPDX-License-Identifier: Apache-2.0

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import torch

from vipe.streams.base import FrameAttribute, VideoFrame, VideoStream
from vipe.utils import io
from vipe.utils.cameras import CameraType


class _FrameListStream(VideoStream):
    def __init__(self, data: "VipeAnnotationData") -> None:
        self.data = data

    def frame_size(self) -> tuple[int, int]:
        return self.data.frames[0].size()

    def name(self) -> str:
        return self.data.sequence_name

    def fps(self) -> float:
        return self.data.fps

    def __len__(self) -> int:
        return len(self.data.frames)

    def __iter__(self):
        return iter(self.data.frames)

    def attributes(self) -> set[FrameAttribute]:
        return self.data.frames[0].attributes()


@dataclass(frozen=True, slots=True)
class VipeAnnotationData:
    """Synchronized, CPU-resident output from one ViPE sequence."""

    sequence_name: str
    fps: float
    frames: tuple[VideoFrame, ...]
    ba_residual: float | None = None

    def __post_init__(self) -> None:
        if not self.sequence_name or "/" in self.sequence_name or "\\" in self.sequence_name:
            raise ValueError("sequence_name must be a non-empty file name")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive and finite")
        if not self.frames:
            raise ValueError("annotation data must contain at least one frame")
        if self.ba_residual is not None and not math.isfinite(self.ba_residual):
            raise ValueError("ba_residual must be finite when present")
        expected_size = self.frames[0].size()
        labels: dict[int, str] = {}
        for index, frame in enumerate(self.frames):
            if frame.raw_frame_idx != index:
                raise ValueError("frame indices must be contiguous and start at zero")
            if frame.size() != expected_size:
                raise ValueError("all frame resolutions must match")
            values = (frame.rgb, frame.metric_depth, frame.instance, frame.intrinsics)
            if any(value is None for value in values[1:]) or frame.pose is None or frame.camera_type is None:
                raise ValueError("every frame must contain RGB, depth, instance IDs, pose, intrinsics, and camera type")
            if any(value is not None and value.device.type != "cpu" for value in values) or frame.pose.device.type != "cpu":
                raise ValueError("annotation tensors and poses must be CPU resident")
            if frame.rgb.shape != (*expected_size, 3) or frame.metric_depth.shape != expected_size:
                raise ValueError("RGB or depth shape does not match the frame resolution")
            if not torch.isfinite(frame.rgb).all() or frame.rgb.min() < 0 or frame.rgb.max() > 1:
                raise ValueError("RGB values must be finite and in [0, 1]")
            if frame.instance.shape != expected_size or frame.instance.dtype != torch.uint8:
                raise ValueError("instance IDs must have dtype uint8 and shape [H, W]")
            if frame.intrinsics.shape != (4,) or not torch.isfinite(frame.intrinsics).all():
                raise ValueError("intrinsics must be finite [fx, fy, cx, cy] values")
            if frame.intrinsics[0] <= 0 or frame.intrinsics[1] <= 0:
                raise ValueError("focal lengths must be positive")
            if frame.camera_type is not CameraType.PINHOLE:
                raise ValueError("only pinhole camera annotations are supported")
            pose_matrix = frame.pose.matrix()
            identity = torch.eye(3, dtype=pose_matrix.dtype)
            if (
                pose_matrix.shape != (4, 4)
                or not torch.isfinite(pose_matrix).all()
                or not torch.allclose(pose_matrix[3], torch.tensor([0, 0, 0, 1], dtype=pose_matrix.dtype), atol=1e-5)
                or not torch.allclose(pose_matrix[:3, :3].T @ pose_matrix[:3, :3], identity, atol=1e-4)
                or not torch.allclose(torch.linalg.det(pose_matrix[:3, :3]), torch.ones((), dtype=pose_matrix.dtype), atol=1e-4)
            ):
                raise ValueError("camera-to-world poses must be finite rigid 4x4 transforms")
            labels.update(frame.instance_phrases or {})
        if any(label_id < 0 or label_id > 255 for label_id in labels):
            raise ValueError("mask label IDs must be in 0..255")

    @property
    def mask_labels(self) -> dict[int, str]:
        labels: dict[int, str] = {}
        for frame in self.frames:
            labels.update(frame.instance_phrases or {})
        return dict(sorted(labels.items()))

    @property
    def rgb(self) -> torch.Tensor:
        return torch.stack([frame.rgb for frame in self.frames]).to(torch.float32)

    @property
    def metric_depth(self) -> torch.Tensor:
        return torch.stack([frame.metric_depth for frame in self.frames]).to(torch.float32)  # type: ignore[arg-type]

    @property
    def instance_ids(self) -> torch.Tensor:
        return torch.stack([frame.instance for frame in self.frames]).to(torch.uint8)  # type: ignore[arg-type]

    @property
    def intrinsics(self) -> torch.Tensor:
        return torch.stack([frame.intrinsics for frame in self.frames]).to(torch.float32)  # type: ignore[arg-type]

    @property
    def poses(self) -> torch.Tensor:
        return torch.stack([frame.pose.matrix() for frame in self.frames]).to(torch.float32)  # type: ignore[union-attr]

    @property
    def camera_models(self) -> tuple[str, ...]:
        return tuple(frame.camera_type.value for frame in self.frames)  # type: ignore[union-attr]


def annotation_from_stream(stream: VideoStream, ba_residual: float | None = None) -> VipeAnnotationData:
    frames_list = []
    for index, source in enumerate(stream):
        frame = source.cpu()
        frame.raw_frame_idx = index
        if frame.instance is None:
            frame.instance = torch.zeros(frame.size(), dtype=torch.uint8)
            frame.instance_phrases = {}
        frames_list.append(frame)
    frames = tuple(frames_list)
    return VipeAnnotationData(stream.name(), float(stream.fps()), frames, ba_residual)


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"global {module}.{name} is not allowed in ViPE metadata")


def _load_info(path: Path) -> float | None:
    with path.open("rb") as handle:
        info = _RestrictedUnpickler(handle).load()
    if not isinstance(info, dict):
        raise ValueError("ViPE info metadata must be a dictionary")
    value = info.get("ba_residual")
    if value is None:
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("ViPE BA residual must be finite")
    return float(value)


def save_annotation_data(data: VipeAnnotationData, base_path: str | Path) -> tuple[Path, ...]:
    base_path = Path(base_path)
    artifacts = io.ArtifactPath(base_path, data.sequence_name)
    written = io.save_artifacts(artifacts, _FrameListStream(data))
    artifacts.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
    with artifacts.meta_info_path.open("wb") as handle:
        pickle.dump({"ba_residual": data.ba_residual, "fps": data.fps}, handle)
    return tuple([*written, artifacts.meta_info_path])


def load_annotation_data(base_path: str | Path, sequence_name: str | None = None) -> VipeAnnotationData:
    base_path = Path(base_path)
    candidates = sorted((base_path / "rgb").glob("*.mp4"))
    if sequence_name is not None:
        candidates = [path for path in candidates if path.stem == sequence_name]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one ViPE RGB sequence, found {len(candidates)}")
    artifacts = io.ArtifactPath(base_path, candidates[0].stem)
    required = (
        artifacts.rgb_path,
        artifacts.mask_path,
        artifacts.depth_path,
        artifacts.pose_path,
        artifacts.intrinsics_path,
        artifacts.camera_type_path,
        artifacts.meta_info_path,
    )
    missing = [str(path.relative_to(base_path)) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"missing ViPE artifacts: {', '.join(missing)}")

    rgb = dict(io.read_rgb_artifacts(artifacts.rgb_path))
    depth = dict(io.read_depth_artifacts(artifacts.depth_path))
    instances = dict(io.read_instance_artifacts(artifacts.mask_path))
    pose_inds, poses = io.read_pose_artifacts(artifacts.pose_path)
    intr_inds, intrinsics, cameras = io.read_intrinsics_artifacts(
        artifacts.intrinsics_path, artifacts.camera_type_path
    )
    pose_map = {int(index): poses[position].cpu() for position, index in enumerate(pose_inds.tolist())}
    intr_map = {int(index): intrinsics[position].cpu().to(torch.float32) for position, index in enumerate(intr_inds.tolist())}
    camera_map = {int(index): cameras[position] for position, index in enumerate(intr_inds.tolist())}
    expected = list(range(len(rgb)))
    streams = (sorted(rgb), sorted(depth), sorted(instances), sorted(pose_map), sorted(intr_map), sorted(camera_map))
    if any(indices != expected for indices in streams):
        raise ValueError("ViPE artifacts are not synchronized or have non-contiguous frame indices")
    labels = io.read_instance_phrases(artifacts.mask_phrase_path) if artifacts.mask_phrase_path.is_file() else {}
    frames = tuple(
        VideoFrame(
            raw_frame_idx=index,
            rgb=rgb[index].cpu().to(torch.float32),
            pose=pose_map[index],
            camera_type=camera_map[index],
            intrinsics=intr_map[index],
            instance=instances[index].cpu().to(torch.uint8),
            instance_phrases=labels,
            metric_depth=depth[index].cpu().to(torch.float32),
        )
        for index in expected
    )
    capture = cv2.VideoCapture(str(artifacts.rgb_path))
    try:
        video_fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    info_fps = None
    with artifacts.meta_info_path.open("rb") as handle:
        info = _RestrictedUnpickler(handle).load()
        if isinstance(info, dict):
            info_fps = info.get("fps")
    fps = float(info_fps) if isinstance(info_fps, (int, float)) and float(info_fps) > 0 else video_fps
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("RGB artifact has invalid FPS")
    return VipeAnnotationData(artifacts.artifact_name, fps, frames, _load_info(artifacts.meta_info_path))
