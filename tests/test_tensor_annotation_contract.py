from pathlib import Path
import tempfile

import pytest
import torch

from vipe.streams import TensorVideoStream
from vipe.utils.device import configure_device


def test_tensor_video_stream_validates_shape_range_and_uses_cpu_storage() -> None:
    configure_device("cpu")
    source = torch.rand(2, 3, 4, 3)
    stream = TensorVideoStream(source, 29.97, "sequence")
    assert len(stream) == 2
    assert stream.frame_size() == (3, 4)
    assert stream.fps() == pytest.approx(29.97)
    frames = list(stream)
    assert all(frame.rgb.device.type == "cpu" for frame in frames)
    source.zero_()
    assert frames[0].rgb.max() > 0


@pytest.mark.parametrize(
    "frames",
    [torch.rand(2, 3, 4), torch.ones(1, 2, 2, 3) * 2, torch.zeros(0, 2, 2, 3)],
)
def test_tensor_video_stream_rejects_invalid_input(frames: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        TensorVideoStream(frames, 30)


def test_tensor_stream_creation_does_not_write_files() -> None:
    configure_device("cpu")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        TensorVideoStream(torch.zeros(1, 2, 2, 3), 30)
        assert list(root.iterdir()) == []
