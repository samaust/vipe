import logging

import pytest

from vipe.utils import logging as logging_module
from vipe.utils.logging import configure_logging, pbar, progress_stage


@pytest.fixture(autouse=True)
def _reset_console() -> None:
    configure_logging(quiet=True, interactive=False)
    yield
    logging.getLogger("vipe").handlers.clear()


def test_configure_logging_is_idempotent() -> None:
    logger = configure_logging(interactive=False)
    first_handler = logger.handlers[0]

    configure_logging(verbose=True, interactive=False)

    assert len(logger.handlers) == 1
    assert logger.handlers[0] is not first_handler
    assert logger.level == logging.DEBUG


def test_non_tty_stage_uses_plain_lines(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(interactive=False, stage_total=1)

    with progress_stage("Read input"):
        pass

    stderr = capsys.readouterr().err
    assert "[1/1] Read input..." in stderr
    assert "[1/1] Read input completed in" in stderr


def test_failed_stage_reports_failure_and_preserves_exception(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(interactive=False, stage_total=1)

    with pytest.raises(RuntimeError, match="broken"):
        with progress_stage("Process"):
            raise RuntimeError("broken")

    assert "[1/1] Process failed after" in capsys.readouterr().err


def test_cancelled_stage_reports_cancellation(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(interactive=False, stage_total=1)

    with pytest.raises(KeyboardInterrupt):
        with progress_stage("Process"):
            raise KeyboardInterrupt

    assert "[1/1] Process cancelled after" in capsys.readouterr().err


def test_quiet_mode_suppresses_progress(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(quiet=True, interactive=False, stage_total=1)

    with progress_stage("Read input"):
        pass

    assert capsys.readouterr().err == ""


def test_detailed_progress_requires_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(logging_module.tqdm, "tqdm", lambda iterable, **kwargs: calls.append(kwargs) or iterable)

    configure_logging(interactive=True)
    list(pbar(range(1), level="detail"))
    assert calls == []

    configure_logging(verbose=True, interactive=True)
    list(pbar(range(1), level="detail"))
    assert len(calls) == 1
