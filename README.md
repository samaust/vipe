# ViPE: Video Pose Engine for Geometric 3D Perception

<p align="center">
  <img src="assets/teaser.gif" alt="teaser"/>
</p>

**TL;DR: ViPE is a useful open-source spatial AI tool for annotating camera poses and dense depth maps from raw videos!**

ViPE estimates camera intrinsics, camera motion, and dense near-metric depth maps from unconstrained raw videos, including pinhole, wide-angle, and 360-degree panorama footage.

<p align="center">
  <a href="https://research.nvidia.com/labs/toronto-ai/vipe"><img src="https://img.shields.io/badge/Project%20Page-76B900?logo=nvidia&logoColor=white" alt="Project Page"/></a>
  <a href="https://arxiv.org/abs/2508.10934"><img src="https://img.shields.io/badge/arXiv-2508.10934-B31B1B?logo=arxiv&logoColor=white" alt="arXiv"/></a>
  <a href="https://pypi.org/project/nvidia-vipe/"><img src="https://img.shields.io/pypi/v/nvidia-vipe?logo=pypi&logoColor=white&label=PyPI" alt="PyPI"/></a>
  <a href="https://nv-tlabs.github.io/vipe/"><img src="https://img.shields.io/badge/Documentation-blue?logo=readthedocs&logoColor=white" alt="Documentation"/></a>
  <a href="https://nv-tlabs.github.io/vipe/dataset/"><img src="https://img.shields.io/badge/Datasets-4B8BBE?logo=databricks&logoColor=white" alt="Datasets"/></a>
</p>

## News

- **2026/06**: 🚀🚀🚀 Released ViPE 1.2.0: **2.7x speed-up with no loss of accuracy**, enabled by CUDA fused kernels, model and pipeline caching, prefetching, and other optimizations.
- **2026/05**: Merged Panorama estimation pipeline & bump release version to 1.0.0.
- **2026/01**: Integration with [Depth-Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) for depth estimation (use `dav3` pipeline).
- **2025/10**: Add support to run on wide-angle videos.
- **2025/09**: Add support to run [Lyra](https://github.com/nv-tlabs/lyra) pipeline.
- **2025/08**: Initial release of ViPE.

## In-memory integration API

`TensorVideoStream` accepts float RGB tensors shaped `[frames, height, width, 3]` in `[0,1]` and keeps its backing storage on CPU. Set `pipeline.return_output_streams = True` to receive processed streams without enabling artifact output, then convert one with `annotation_from_stream` to a validated, CPU-resident `VipeAnnotationData` value.

`save_annotation_data` and `load_annotation_data` round-trip one synchronized pinhole sequence using ViPE's standard `rgb`, `mask`, `depth`, `pose`, `intrinsics`, and `vipe` artifact directories. Loading rejects missing, non-contiguous, mismatched, or non-pinhole data.

## Hugging Face model cache

ViPE loads Hugging Face Hub files from the local cache (`$HF_HOME/hub`, by default `~/.cache/huggingface/hub`) and only contacts the Hub when a file is missing. Cached files are not checked for updates, so a model updated upstream keeps using the cached revision until you refresh it. This applies to these repositories:

| Repository | Files | Used by |
| --- | --- | --- |
| `bert-base-uncased` | tokenizer and `config.json` | GroundingDINO (instance masks) |
| `lpiccinelli/unidepth-v2-vit{s,b,l}14` | `config.json`, `model.safetensors` | UniDepth keyframe/metric depth |
| `Rain729/Prior-Depth-Anything` | `depth_anything_v2_vitb.pth`, `prior_depth_anything_vitb.pth` | depth alignment |
| `depth-anything/DA3-GIANT`, `depth-anything/DA3METRIC-LARGE` | `model.safetensors` | `dav3` pipeline |

To fetch the latest revision, re-download the files ViPE uses with the `hf` CLI. This updates the cache in place, and the next run uses the new files. The commands name the files explicitly because these repositories also contain large files ViPE never loads (for example a 1.4 GB `pytorch_model.bin` next to UniDepth's safetensors). `--force-download` also fetches files that are not cached yet, so skip the groups for pipelines you do not use:

```bash
# GroundingDINO text encoder (instance masks, all pipelines)
hf download bert-base-uncased config.json tokenizer.json tokenizer_config.json vocab.txt --force-download

# UniDepth (default, no_vda and static_vda pipelines use vitl14)
hf download lpiccinelli/unidepth-v2-vitl14 config.json model.safetensors --force-download
hf download lpiccinelli/unidepth-v2-vitb14 config.json model.safetensors --force-download
hf download lpiccinelli/unidepth-v2-vits14 config.json model.safetensors --force-download

# Prior-Depth-Anything (depth alignment)
hf download Rain729/Prior-Depth-Anything depth_anything_v2_vitb.pth prior_depth_anything_vitb.pth --force-download

# Depth-Anything 3 (dav3 pipeline)
hf download depth-anything/DA3METRIC-LARGE model.safetensors --force-download
hf download depth-anything/DA3-GIANT model.safetensors --force-download
```

Alternatively, remove the cached repositories, and ViPE downloads the ones it needs again on the next run. Repositories that are not cached are skipped; use `--dry-run` to preview:

```bash
hf cache rm \
  model/bert-base-uncased \
  model/lpiccinelli/unidepth-v2-vitl14 \
  model/lpiccinelli/unidepth-v2-vitb14 \
  model/lpiccinelli/unidepth-v2-vits14 \
  model/Rain729/Prior-Depth-Anything \
  model/depth-anything/DA3METRIC-LARGE \
  model/depth-anything/DA3-GIANT
```

### Other model weights

SAM, DeAOT, DROID-SLAM, GeoCalib, GroundingDINO and Video-Depth-Anything weights are downloaded from fixed release URLs into the PyTorch hub directory (`$TORCH_HOME/hub`, by default `~/.cache/torch/hub`) and are never re-checked. To refresh them, delete the files; ViPE downloads them again on the next run:

```bash
TORCH_HUB="${TORCH_HOME:-$HOME/.cache/torch}/hub"
rm -f \
  "$TORCH_HUB/geocalib/pinhole.tar" \
  "$TORCH_HUB/droid_slam/droid.pth" \
  "$TORCH_HUB/sam/sam_vit_b_01ec64.pth" \
  "$TORCH_HUB/aot/R50_DeAOTL_PRE_YTB_DAV.pth" \
  "$TORCH_HUB/checkpoints/groundingdino_swint_ogc.pth" \
  "$TORCH_HUB/checkpoints/video_depth_anything_vits.pth" \
  "$TORCH_HUB/checkpoints/video_depth_anything_vitl.pth"
```

Long-running hosts such as ComfyUI keep loaded models in memory for at most one pipeline run, so a refresh takes effect on the next run without a restart.

## License

This project will download and install additional third-party **models and softwares**. Note that these models or softwares are not distributed by NVIDIA. Review the license terms of these models and projects before use. This source code, **except for the Unik3D part (which is under the BY-NC-SA 4.0 license)** , is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).
