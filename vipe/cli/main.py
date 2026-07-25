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

import time
from pathlib import Path

import click

from vipe import make_pipeline
from vipe.config import parse_typed_config
from vipe.streams.base import ProcessedVideoStream
from vipe.streams.frame_dir_stream import FrameDirStream
from vipe.streams.raw_mp4_stream import RawMp4Stream
from vipe.utils.device import configure_device
from vipe.utils.logging import configure_logging, console_message, progress_stage
from vipe.utils.viser import run_viser


@click.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path), required=False)
@click.option(
    "--image-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing image frames",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory (default: current directory)",
    default=Path.cwd() / "vipe_results",
)
@click.option("--pipeline", "-p", default="default", help="Pipeline configuration to use (default: 'default')")
@click.option("--visualize", "-v", is_flag=True, help="Enable visualization of intermediate results")
@click.option("--quiet", "-q", is_flag=True, help="Only show warnings and errors")
@click.option("--verbose", is_flag=True, help="Show detailed model and optimization diagnostics")
def infer(
    video: Path | None,
    image_dir: Path | None,
    output: Path,
    pipeline: str,
    visualize: bool,
    quiet: bool,
    verbose: bool,
):
    """Run inference on a video file or directory of images."""

    if quiet and verbose:
        raise click.UsageError("--quiet and --verbose cannot be used together")

    # Validate that exactly one input source is provided
    if not video and not image_dir:
        click.echo("Error: Must provide either a video file or --image-dir", err=True)
        raise click.Abort()

    if video and image_dir:
        click.echo("Error: Cannot provide both video file and --image-dir", err=True)
        raise click.Abort()

    overrides = [f"pipeline={pipeline}", f"pipeline.output.path={output}", "pipeline.output.save_artifacts=true"]
    if visualize:
        overrides.append("pipeline.output.save_viz=true")
        overrides.append("pipeline.slam.visualize=true")
    else:
        overrides.append("pipeline.output.save_viz=false")

    # Set up stream configuration based on input type
    if image_dir:
        overrides.extend(["streams=frame_dir_stream", f"streams.base_path={image_dir}"])
        input_desc = f"image directory {image_dir}"
    else:
        input_desc = f"video {video}"

    args = parse_typed_config("default", hydra_args=overrides)
    has_depth_stage = args.pipeline.post.depth_align_model is not None
    writes_output = any(
        bool(getattr(args.pipeline.output, option, False))
        for option in ("save_artifacts", "save_viz", "save_slam_map")
    )
    stage_total = 3 + int(has_depth_stage) + int(writes_output)
    logger = configure_logging(quiet=quiet, verbose=verbose, stage_total=stage_total)
    configure_device(args.device)

    started = time.monotonic()
    logger.info(f"Processing {input_desc}")
    vipe_pipeline = make_pipeline(args.pipeline)

    with progress_stage("Read and validate input frames"):
        if image_dir:
            video_stream = ProcessedVideoStream(FrameDirStream(image_dir), []).cache(desc="Reading image frames")
        else:
            assert video is not None
            # Cache the stream to obtain a reliable frame count even for malformed videos.
            video_stream = ProcessedVideoStream(RawMp4Stream(video), []).cache(desc="Reading video")

    result = vipe_pipeline.run(video_stream)
    elapsed = time.monotonic() - started
    console_message(f"Finished processing {len(video_stream)} frames in {elapsed:.1f}s")
    if result.written_paths:
        console_message("Outputs:")
        for path in result.written_paths:
            console_message(f"  {path.resolve()}")


@click.command()
@click.argument("data_path", type=click.Path(exists=True, path_type=Path), default=Path.cwd() / "vipe_results")
@click.option("--port", "-p", default=20540, type=int, help="Port for the visualization server (default: 20540)")
def visualize(data_path: Path, port: int):
    run_viser(data_path, port)


@click.group()
@click.version_option(package_name="nvidia-vipe")
def main():
    """NVIDIA Video Pose Engine (ViPE) CLI"""
    pass


# Add subcommands
main.add_command(infer)
main.add_command(visualize)


if __name__ == "__main__":
    main()
