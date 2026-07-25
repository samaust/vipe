import importlib
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from vipe.pipeline import AnnotationPipelineOutput

cli_module = importlib.import_module("vipe.cli.main")


class _FakeStream:
    def __len__(self) -> int:
        return 12


class _FakeProcessedStream:
    def __init__(self, stream, processors) -> None:
        pass

    def cache(self, desc: str):
        return _FakeStream()


def _patch_inference(monkeypatch, tmp_path: Path, *, written_paths=(), depth_model="depth-model", writes=True):
    config = SimpleNamespace(
        device="cpu",
        pipeline=SimpleNamespace(
            post=SimpleNamespace(depth_align_model=depth_model),
            output=SimpleNamespace(save_artifacts=writes, save_viz=False, save_slam_map=False),
        ),
    )
    pipeline = SimpleNamespace(run=lambda stream: AnnotationPipelineOutput(written_paths=written_paths))
    monkeypatch.setattr(cli_module, "parse_typed_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(cli_module, "configure_device", lambda device: None)
    monkeypatch.setattr(cli_module, "make_pipeline", lambda config: pipeline)
    monkeypatch.setattr(cli_module, "RawMp4Stream", lambda path: object())
    monkeypatch.setattr(cli_module, "ProcessedVideoStream", _FakeProcessedStream)
    input_path = tmp_path / "input.mp4"
    input_path.touch()
    return input_path


def test_infer_rejects_quiet_and_verbose(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.touch()

    result = CliRunner().invoke(cli_module.infer, [str(input_path), "--quiet", "--verbose"])

    assert result.exit_code == 2
    assert "--quiet and --verbose cannot be used together" in result.output


def test_infer_reports_stages_and_outputs(monkeypatch, tmp_path: Path) -> None:
    output_path = tmp_path / "result.npz"
    input_path = _patch_inference(monkeypatch, tmp_path, written_paths=[output_path])

    result = CliRunner().invoke(cli_module.infer, [str(input_path)])

    assert result.exit_code == 0
    assert "[1/5] Read and validate input frames" in result.output
    assert "Finished processing 12 frames in" in result.output
    assert str(output_path.resolve()) in result.output


def test_infer_quiet_suppresses_success_output(monkeypatch, tmp_path: Path) -> None:
    input_path = _patch_inference(monkeypatch, tmp_path)

    result = CliRunner().invoke(cli_module.infer, [str(input_path), "--quiet"])

    assert result.exit_code == 0
    assert result.output == ""


def test_infer_stage_count_follows_enabled_pipeline_work(monkeypatch, tmp_path: Path) -> None:
    input_path = _patch_inference(monkeypatch, tmp_path, depth_model=None, writes=False)

    result = CliRunner().invoke(cli_module.infer, [str(input_path)])

    assert result.exit_code == 0
    assert "[1/3] Read and validate input frames" in result.output
