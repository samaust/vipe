# SPDX-License-Identifier: Apache-2.0

import math
from collections.abc import Iterator

import torch

from vipe.streams.base import VideoFrame, VideoStream
from vipe.utils.device import get_device


class TensorVideoStream(VideoStream):
    """A read-only RGB video stream backed by a ComfyUI-style tensor."""

    def __init__(self, frames: torch.Tensor, fps: float, name: str = "tensor") -> None:
        if not isinstance(frames, torch.Tensor):
            raise TypeError("frames must be a torch.Tensor")
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("frames must have shape [N, H, W, 3]")
        if frames.shape[0] == 0 or frames.shape[1] == 0 or frames.shape[2] == 0:
            raise ValueError("frames must contain at least one non-empty frame")
        if not torch.is_floating_point(frames):
            raise TypeError("frames must use a floating-point dtype")
        if not torch.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
            raise ValueError("frames must contain finite RGB values in [0, 1]")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")
        if not name or "/" in name or "\\" in name:
            raise ValueError("name must be a non-empty sequence name")

        self._frames = frames.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        self._fps = float(fps)
        self._name = name

    def frame_size(self) -> tuple[int, int]:
        return int(self._frames.shape[1]), int(self._frames.shape[2])

    def name(self) -> str:
        return self._name

    def fps(self) -> float:
        return self._fps

    def __len__(self) -> int:
        return int(self._frames.shape[0])

    def __iter__(self) -> Iterator[VideoFrame]:
        for index, rgb in enumerate(self._frames):
            yield VideoFrame(raw_frame_idx=index, rgb=rgb.to(get_device()))
