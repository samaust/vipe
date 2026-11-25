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

"""Cache-first Hugging Face Hub loading.

``hf_hub_download`` and ``from_pretrained`` contact the Hub on every call to
check whether the cached revision is still current, even when every file is
already on disk. ViPE builds its models once per pipeline, so in long-lived
hosts that rebuild pipelines (for example ComfyUI) this produced a burst of
metadata requests on every run. These helpers resolve from the local cache
first and only reach the network when the cache is missing the requested files.
"""

from typing import Any, Callable, TypeVar

from huggingface_hub import hf_hub_download

T = TypeVar("T")


def load_local_first(loader: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Call ``loader`` with ``local_files_only=True``, retrying online on a cache miss.

    ``loader`` is any Hub-aware loader accepting ``local_files_only`` such as
    ``hf_hub_download``, ``AutoTokenizer.from_pretrained``, ``AutoConfig.from_pretrained``
    or ``PyTorchModelHubMixin.from_pretrained``. Cache misses surface as ``OSError``
    (``LocalEntryNotFoundError`` subclasses it, and transformers wraps misses in it).
    """
    try:
        return loader(*args, local_files_only=True, **kwargs)
    except OSError:
        return loader(*args, **kwargs)


def hf_hub_file(repo_id: str, filename: str, **kwargs: Any) -> str:
    """Return the local path of a Hub file, downloading it only when not cached."""
    return load_local_first(hf_hub_download, repo_id=repo_id, filename=filename, **kwargs)
