from __future__ import annotations

import torch
import pytest

from vipe.ext.lietorch import SE3
from vipe.pipeline.processors import AdaptiveDepthProcessor
from vipe.priors.depth.base import DepthEstimationResult
from vipe.priors.depth.priorda.depth_completion import DepthCompletion
from vipe.slam.interface import SLAMOutput
from vipe.streams.base import VideoFrame
from vipe.utils.cameras import CameraType
from vipe.utils import device as device_module
from vipe.utils.model_cache import ModelCache


@pytest.fixture(autouse=True)
def _cpu_runtime_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device_module, "_runtime_device", torch.device("cpu"))


class _FakeDepthModel:
    def estimate(self, src) -> DepthEstimationResult:
        assert src.rgb is not None
        return DepthEstimationResult(metric_depth=torch.ones(src.rgb.shape[:2]))


class _FakeMap:
    def project_map(self, *args, target_size=None, **kwargs) -> torch.Tensor:
        if target_size is None:
            target_size = args[2]
        return torch.ones(target_size)


def _slam_output() -> SLAMOutput:
    return SLAMOutput(
        trajectory=SE3.Identity(1),
        intrinsics=torch.tensor([[10.0, 10.0, 5.0, 5.0]]),
        rig=SE3.Identity(1),
        slam_map=_FakeMap(),  # type: ignore[arg-type]
    )


def _frame() -> VideoFrame:
    return VideoFrame(
        raw_frame_idx=0,
        rgb=torch.zeros(10, 10, 3),
        intrinsics=torch.tensor([10.0, 10.0, 5.0, 5.0]),
        camera_type=CameraType.PINHOLE,
    )


def test_model_cache_evicts_by_prefix() -> None:
    cache = ModelCache()
    cache.get("track_anything/sam", object)
    cache.get("track_anything/aot", object)
    retained = cache.get("other/model", object)

    assert cache.clear_prefix("track_anything/") == 2
    assert cache.keys() == ("other/model",)
    assert cache.get("other/model", object) is retained


def test_adaptive_depth_models_are_lazy_and_metric_branch_is_exclusive(monkeypatch) -> None:
    processor = AdaptiveDepthProcessor(_slam_output(), model="adaptive_unidepth-l")
    assert processor.video_depth_model is None
    assert processor.depth_model is None
    assert processor.prompt_model is None

    metric_model = _FakeDepthModel()
    monkeypatch.setattr(processor, "_compute_min_uv_score", lambda frame, slam_map: 0.1)
    monkeypatch.setattr(processor, "_make_metric_depth_model", lambda: metric_model)
    monkeypatch.setattr(
        processor,
        "_make_prompt_depth_model",
        lambda: (_ for _ in ()).throw(AssertionError("PriorDA must not be constructed")),
    )

    output = list(processor.update_iterator(iter([_frame()]), 0))
    assert len(output) == 1
    assert output[0].metric_depth is not None
    assert processor.depth_model is None
    assert processor.prompt_model is None


def test_adaptive_depth_prompt_branch_is_exclusive(monkeypatch) -> None:
    processor = AdaptiveDepthProcessor(_slam_output(), model="adaptive_unidepth-l")
    prompt_model = _FakeDepthModel()
    monkeypatch.setattr(processor, "_compute_min_uv_score", lambda frame, slam_map: 0.8)
    monkeypatch.setattr(processor, "_make_prompt_depth_model", lambda: prompt_model)
    monkeypatch.setattr(
        processor,
        "_make_metric_depth_model",
        lambda: (_ for _ in ()).throw(AssertionError("UniDepth must not be constructed")),
    )

    output = list(processor.update_iterator(iter([_frame()]), 0))
    assert len(output) == 1
    assert output[0].metric_depth is not None
    assert processor.depth_model is None
    assert processor.prompt_model is None


def test_cpu_knn_is_processed_in_query_chunks(monkeypatch) -> None:
    import vipe_ext

    completion = DepthCompletion.__new__(DepthCompletion)
    torch.nn.Module.__init__(completion)
    completion.cpu_knn_chunk_size = 3
    calls: list[int] = []

    def nearest(query: torch.Tensor, tree: torch.Tensor, k: int):
        calls.append(query.shape[0])
        distances = (query[:, None] - tree[None]).square().sum(-1)
        values, indices = torch.topk(distances, k, dim=-1, largest=False)
        return values, indices.to(torch.int32)

    monkeypatch.setattr(vipe_ext.utils_ext, "nearest_neighbours", nearest)
    sparse_mask = torch.zeros(1, 3, 4, dtype=torch.bool)
    sparse_mask[0, 0, 0] = True
    sparse_mask[0, 2, 3] = True
    complete_mask = torch.ones_like(sparse_mask)
    sparse = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)

    distances, sparse_targets, predicted_targets = completion.knn_aligns(
        sparse,
        sparse + 1,
        sparse_mask,
        complete_mask,
        K=1,
    )

    assert calls == [3, 3, 3, 3]
    assert distances.shape == (12, 1)
    assert sparse_targets.shape == (12, 1)
    assert predicted_targets.shape == (12, 1)
