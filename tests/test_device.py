import pytest
import torch

from vipe.streams.base import VideoFrame
from vipe.utils import device as device_module
from vipe.utils.device import configure_device, get_device


@pytest.fixture(autouse=True)
def _isolate_runtime_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device_module, "_runtime_device", torch.device("cpu"))


def test_configure_cpu_device() -> None:
    assert configure_device("cpu") == torch.device("cpu")
    assert get_device() == torch.device("cpu")


def test_configure_cuda_resolves_current_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)

    assert configure_device("cuda") == torch.device("cuda:2")
    assert get_device() == torch.device("cuda:2")


def test_rejects_unsupported_device() -> None:
    with pytest.raises(ValueError, match="Unsupported ViPE device"):
        configure_device("meta")


def test_video_frame_to_moves_all_tensor_attributes() -> None:
    frame = VideoFrame(
        raw_frame_idx=3,
        rgb=torch.zeros(2, 3, 3),
        mask=torch.ones(2, 3, dtype=torch.bool),
        instance=torch.ones(2, 3, dtype=torch.uint8),
        metric_depth=torch.ones(2, 3),
        intrinsics=torch.ones(4),
    )

    moved = frame.to("cpu")

    assert moved.rgb.device.type == "cpu"
    assert moved.mask is not None and moved.mask.device.type == "cpu"
    assert moved.instance is not None and moved.instance.device.type == "cpu"
    assert moved.metric_depth is not None and moved.metric_depth.device.type == "cpu"
    assert moved.intrinsics is not None and moved.intrinsics.device.type == "cpu"
    assert moved.raw_frame_idx == frame.raw_frame_idx
