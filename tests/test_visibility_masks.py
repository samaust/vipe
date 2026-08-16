from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pytest
import torch

from vipe.pipeline.processors import InvisibleMaskProcessor, add_invisible_masks
from vipe.streams.base import VideoFrame, VideoStream
from vipe.utils import device as device_module
from vipe.utils import io
from vipe.utils.cameras import CameraType
from vipe.utils.visibility import invisible_pixels_from_depth


@pytest.fixture(autouse=True)
def _cpu_runtime_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device_module, "_runtime_device", torch.device("cpu"))


class _ListStream(VideoStream):
    def __init__(self, frames: list[VideoFrame], name: str = "sequence") -> None:
        self.frames = frames
        self._name = name

    def frame_size(self) -> tuple[int, int]:
        return self.frames[0].size()

    def name(self) -> str:
        return self._name

    def fps(self) -> float:
        return 30.0

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[VideoFrame]:
        return iter(self.frames)


def _frame(
    *,
    instance: torch.Tensor | None = None,
    depth: torch.Tensor | None = None,
    phrases: dict[int, str] | None = None,
) -> VideoFrame:
    height, width = instance.shape if instance is not None else (2, 2)
    return VideoFrame(
        raw_frame_idx=0,
        rgb=torch.zeros(height, width, 3),
        camera_type=CameraType.PINHOLE,
        intrinsics=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        instance=instance,
        instance_phrases=phrases,
        mask=torch.tensor([[True, False], [False, True]]),
        metric_depth=depth,
    )


def test_front_facing_plane_remains_visible() -> None:
    invisible = invisible_pixels_from_depth(
        torch.ones(3, 4),
        torch.tensor([2.0, 2.0, 1.5, 1.0]),
        threshold=0.1,
    )

    assert not invisible.any()


def test_qualifying_face_marks_its_three_pixel_vertices() -> None:
    invisible = invisible_pixels_from_depth(
        torch.tensor([[0.1, 0.1], [1.0, 1.0]]),
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        threshold=0.1,
    )

    torch.testing.assert_close(
        invisible,
        torch.tensor([[True, False], [True, True]]),
    )


@pytest.mark.parametrize("invalid_depth", [0.0, float("nan"), float("inf")])
def test_invalid_or_degenerate_faces_do_not_change_pixels(invalid_depth: float) -> None:
    depth = torch.ones(2, 2)
    depth[0, 0] = invalid_depth

    invisible = invisible_pixels_from_depth(depth, torch.tensor([1.0, 1.0, 0.0, 0.0]), threshold=0.1)

    assert not invisible.any()


def test_processor_replaces_background_and_foreground_but_not_other_pixels() -> None:
    original_instance = torch.tensor([[0, 2], [3, 0]], dtype=torch.uint8)
    frame = _frame(
        instance=original_instance,
        depth=torch.tensor([[0.1, 0.1], [1.0, 1.0]]),
        phrases={0: "background", 2: "animal", 3: "pet"},
    )
    original_binary_mask = frame.mask.clone()

    result = InvisibleMaskProcessor(invisible_id=4, threshold=0.1)(0, frame)

    torch.testing.assert_close(result.instance, torch.tensor([[4, 2], [4, 4]], dtype=torch.uint8))
    torch.testing.assert_close(result.mask, original_binary_mask)
    assert result.instance_phrases == {0: "background", 2: "animal", 3: "pet", 4: "invisible"}


def test_sequence_allocates_one_greater_than_pixel_and_phrase_ids() -> None:
    frames = [
        _frame(
            instance=torch.tensor([[0, 2], [3, 0]], dtype=torch.uint8),
            depth=torch.ones(2, 2),
            phrases={0: "background", 2: "animal", 3: "pet", 4: "reserved"},
        ),
        _frame(
            instance=torch.tensor([[0, 2], [3, 0]], dtype=torch.uint8),
            depth=torch.ones(2, 2),
            phrases={0: "background", 2: "animal", 3: "pet"},
        ),
    ]

    finalized = add_invisible_masks(_ListStream(frames), threshold=0.1)

    assert [frame.instance_phrases[5] for frame in finalized] == ["invisible", "invisible"]


def test_sequence_errors_when_uint8_ids_are_exhausted() -> None:
    frame = _frame(
        instance=torch.zeros(2, 2, dtype=torch.uint8),
        depth=torch.ones(2, 2),
        phrases={255: "last"},
    )

    with pytest.raises(ValueError, match="uint8 ID 255"):
        add_invisible_masks(_ListStream([frame]), threshold=0.1)


def test_sequence_without_instance_masks_is_unchanged() -> None:
    stream = _ListStream([_frame()])

    finalized = add_invisible_masks(stream, threshold=0.1)

    assert finalized[0].instance is None


def test_finalized_archive_and_labels_use_the_same_invisible_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame(
        instance=torch.tensor([[0, 2], [3, 0]], dtype=torch.uint8),
        depth=torch.tensor([[0.1, 0.1], [1.0, 1.0]]),
        phrases={0: "background", 2: "animal", 3: "pet"},
    )
    finalized = add_invisible_masks(_ListStream([frame]), threshold=0.1)
    artifact = io.ArtifactPath(tmp_path, "sequence")
    monkeypatch.setattr(io, "save_pose_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(io, "save_intrinsics_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(io, "save_rgb_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(io, "save_depth_artifacts", lambda *args, **kwargs: None)

    io.save_artifacts(artifact, finalized)

    with zipfile.ZipFile(artifact.mask_path) as archive:
        encoded = np.frombuffer(archive.read("00000.png"), dtype=np.uint8)
        mask = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(mask, np.array([[4, 2], [4, 4]], dtype=np.uint8))
    assert artifact.mask_phrase_path.read_text(encoding="utf-8").splitlines() == [
        "0: background",
        "2: animal",
        "3: pet",
        "4: invisible",
    ]
