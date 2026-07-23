from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest
import torch

from vipe.streams.base import CachedVideoStream, ProcessedVideoStream, SourceMedia, VideoFrame, VideoStream
from vipe.streams.raw_mp4_stream import RawMp4Stream
from vipe.utils.io import ArtifactPath, save_rgb_artifacts


class _MemoryStream(VideoStream):
    def __init__(self, source_media: SourceMedia | None = None, fps: float = 4.0, frame_count: int = 4) -> None:
        self._source_media = source_media
        self._fps = fps
        self.frames = [
            VideoFrame(raw_frame_idx=index, rgb=torch.from_numpy(np.full((16, 16, 3), index / 4, np.float32)))
            for index in range(frame_count)
        ]

    def frame_size(self) -> tuple[int, int]:
        return (16, 16)

    def fps(self) -> float:
        return self._fps

    def name(self) -> str:
        return "memory"

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def source_media(self) -> SourceMedia | None:
        return self._source_media


def _run_ffmpeg(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
        check=True,
    )


def _make_source_video(path: Path, *, audio: bool = True) -> None:
    inputs = ["-f", "lavfi", "-i", "color=c=blue:s=16x16:r=4:d=2"]
    if audio:
        inputs.extend(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=2", "-shortest"])
    _run_ffmpeg(*inputs, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path))


def _probe_stderr(path: Path) -> str:
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stderr


def test_raw_mp4_source_media_survives_stream_wrappers(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _make_source_video(source)

    raw = RawMp4Stream(source, seek_range=range(2, 8, 2))
    expected = SourceMedia(source, start_time_seconds=0.5)
    assert raw.source_media() == expected
    assert ProcessedVideoStream(raw, []).source_media() == expected
    assert CachedVideoStream(raw).source_media() == expected


@pytest.mark.parametrize("with_audio", [True, False])
def test_save_rgb_artifacts_preserves_audio_when_present(tmp_path: Path, with_audio: bool) -> None:
    source = tmp_path / "source.mp4"
    _make_source_video(source, audio=with_audio)
    stream = _MemoryStream(SourceMedia(source))
    output = ArtifactPath(tmp_path / "results", "sample")

    save_rgb_artifacts(output, stream)

    probe = _probe_stderr(output.rgb_path)
    assert "Video:" in probe
    assert ("Audio:" in probe) is with_audio


def test_save_rgb_artifacts_uses_segment_timing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.mp4"
    source.touch()
    stream = _MemoryStream(SourceMedia(source, start_time_seconds=1.25), fps=5.0, frame_count=3)
    output = ArtifactPath(tmp_path / "results", "sample")
    command: list[str] = []

    def fake_run(args, **kwargs):
        command.extend(args)
        Path(args[-1]).touch()
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    save_rgb_artifacts(output, stream)

    assert command[command.index("-ss") + 1] == "1.25"
    assert command[command.index("-t") + 1] == "0.6"


def test_mux_failure_preserves_existing_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.mp4"
    source.touch()
    output = ArtifactPath(tmp_path / "results", "sample")
    output.rgb_path.parent.mkdir(parents=True)
    output.rgb_path.write_bytes(b"existing")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", "mux failed"),
    )

    with pytest.raises(RuntimeError, match=r"source\.mp4.*sample\.mp4.*mux failed"):
        save_rgb_artifacts(output, _MemoryStream(SourceMedia(source)))

    assert output.rgb_path.read_bytes() == b"existing"


def test_save_rgb_artifacts_without_source_media_is_video_only(tmp_path: Path) -> None:
    output = ArtifactPath(tmp_path / "results", "sample")
    save_rgb_artifacts(output, _MemoryStream())

    probe = _probe_stderr(output.rgb_path)
    assert "Video:" in probe
    assert "Audio:" not in probe
