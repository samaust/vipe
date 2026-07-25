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

import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, TextIO, TypeVar

import tqdm

ProgressLevel = Literal["normal", "detail"]
T = TypeVar("T")

disable_progress_bar: bool = False


@dataclass
class _ConsoleState:
    quiet: bool = False
    verbose: bool = False
    interactive: bool = False
    stage_total: int = 0
    stage_index: int = 0
    stream: TextIO = sys.stderr


_state = _ConsoleState()


class TqdmLoggingHandler(logging.Handler):
    """Write log records without corrupting an active tqdm progress bar."""

    def __init__(self, stream: TextIO) -> None:
        super().__init__()
        self.stream = stream

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.tqdm.write(self.format(record), file=self.stream)
        except Exception:
            self.handleError(record)


def configure_logging(
    *,
    quiet: bool = False,
    verbose: bool = False,
    interactive: bool | None = None,
    stage_total: int = 0,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure ViPE's user-facing logging and progress behavior."""
    global disable_progress_bar

    output_stream = sys.stderr if stream is None else stream
    _state.quiet = quiet
    _state.verbose = verbose
    _state.interactive = output_stream.isatty() if interactive is None else interactive
    _state.stage_total = stage_total
    _state.stage_index = 0
    _state.stream = output_stream
    disable_progress_bar = quiet or not _state.interactive

    logger = logging.getLogger("vipe")
    logger.handlers.clear()
    handler = TqdmLoggingHandler(output_stream)
    if verbose:
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.setLevel(logging.DEBUG)
    else:
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.setLevel(logging.WARNING if quiet else logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_stage_total(total: int) -> None:
    _state.stage_total = total
    _state.stage_index = 0


def _stage_label(name: str, index: int) -> str:
    if _state.stage_total:
        return f"[{index}/{_state.stage_total}] {name}"
    return name


def console_message(message: str, *, error: bool = False) -> None:
    if _state.quiet and not error:
        return
    tqdm.tqdm.write(message, file=_state.stream)


@contextmanager
def progress_stage(name: str) -> Iterator[None]:
    """Report a coarse pipeline stage while preserving the original exception."""
    _state.stage_index += 1
    label = _stage_label(name, _state.stage_index)
    started = time.monotonic()
    console_message(f"{label}...")
    try:
        yield
    except KeyboardInterrupt:
        console_message(f"{label} cancelled after {time.monotonic() - started:.1f}s", error=True)
        raise
    except BaseException:
        console_message(f"{label} failed after {time.monotonic() - started:.1f}s", error=True)
        raise
    else:
        console_message(f"{label} completed in {time.monotonic() - started:.1f}s")


def pbar(iterable: Iterable[T], *, level: ProgressLevel = "normal", **kwargs) -> Iterable[T]:
    """Render frame progress in a TTY; detailed bars require verbose mode."""
    if disable_progress_bar or _state.quiet or (level == "detail" and not _state.verbose):
        return iterable
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("unit", "frame")
    return tqdm.tqdm(iterable, file=_state.stream, **kwargs)
