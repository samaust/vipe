# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from vipe.config.base_schema import BaseConfigSchema, Field
from vipe.config.pipeline import PipelineConfig
from vipe.config.streams import StreamsConfig


class ViPEConfig(BaseConfigSchema):
    """Top-level ViPE runtime configuration."""

    device: Literal["cpu", "cuda"] = Field(description="Torch device used for model inference and SLAM.")
    streams: StreamsConfig = Field(
        description="Input stream list that supplies videos or frame directories to process."
    )
    pipeline: PipelineConfig = Field(description="Annotation pipeline and all pipeline-specific runtime options.")
