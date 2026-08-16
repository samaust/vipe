from pathlib import Path
import zipfile

import numpy as np
import OpenImageIO as oiio
import torch

from vipe.streams.base import VideoFrame, VideoStream
from vipe.utils.io import ArtifactPath, read_depth_artifacts, save_depth_artifacts


class _DepthStream(VideoStream):
    def __init__(self, depths: torch.Tensor) -> None:
        self._frames = [
            VideoFrame(
                raw_frame_idx=index,
                rgb=torch.zeros((*depth.shape, 3), dtype=torch.float32),
                metric_depth=depth,
            )
            for index, depth in enumerate(depths)
        ]

    def __iter__(self):
        return iter(self._frames)

    def __len__(self) -> int:
        return len(self._frames)

    def frame_size(self) -> tuple[int, int]:
        return self._frames[0].size()

    def fps(self) -> float:
        return 30.0

    def name(self) -> str:
        return "depth-test"


def test_depth_artifacts_round_trip_as_z_channel_exr(tmp_path: Path) -> None:
    depths = torch.tensor(
        [
            [[1.25, 2.5, 3.75], [4.0, 5.25, 6.5]],
            [[7.75, 8.0, 9.5], [10.25, 11.5, 12.75]],
        ],
        dtype=torch.float32,
    )
    artifacts = ArtifactPath(tmp_path, "sequence")
    save_depth_artifacts(artifacts, _DepthStream(depths))

    with zipfile.ZipFile(artifacts.depth_path) as archive:
        assert archive.namelist() == ["00000.exr", "00001.exr"]
        extracted = tmp_path / "frame.exr"
        extracted.write_bytes(archive.read("00000.exr"))
    image_input = oiio.ImageInput.open(str(extracted))
    assert image_input is not None
    try:
        assert image_input.spec().channelnames == ("Z",)
    finally:
        image_input.close()

    loaded = list(read_depth_artifacts(artifacts.depth_path))
    assert [index for index, _ in loaded] == [0, 1]
    actual = torch.stack([depth for _, depth in loaded])
    np.testing.assert_allclose(actual.numpy(), depths.numpy(), rtol=0, atol=1e-3)
