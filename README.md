# PhysReflect: Geometry and Perception Guided Diffusion for Physically-Plausible Mirror Reflections

> Official repository for **PhysReflect: Geometry and Perception Guided Diffusion for Physically-Plausible Mirror Reflections**.

**The paper has been accepted by SIGGRAPH Asia 2026.**

**[Paper — ACM Digital Library](https://dl.acm.org/doi/10.1145/3829340.3842250)**

## Release Status

**Last updated: 2026-11-22.** Inference code, dependency setup, and three pretrained PhysReflect checkpoints are available.

**[Download pretrained checkpoints (Google Drive)](https://drive.google.com/drive/folders/1ombIge3e_bAvOn24rcQLhxfCzZ44wFpG?usp=sharing)** — single-object, multi-object, and MSD models.

- [Datasets: SynMirrorV2 and MSD](#datasets)
- [Environment setup](#getting-started)
- [Checkpoint downloads and verification](#pretrained-checkpoints)
- [Inference instructions](#inference)
- [Input format](#input-format)
- [Release plan](#release-plan)

## Overview

Generating convincing mirror reflections requires more than producing visually plausible content: the reflected scene must also agree with the geometry and perceptual structure of the physical world. PhysReflect studies geometry- and perception-guided diffusion for synthesizing physically plausible mirror reflections.

The project is built on the experimental setting introduced by [MirrorVerse: Pushing Diffusion Models to Realistically Reflect the World](https://github.com/val-iisc/MirrorVerse) (CVPR 2025). We follow MirrorVerse for the baseline implementation and dataset setup to support consistent training and evaluation.

## Method

PhysReflect is designed around two complementary sources of guidance:

- **Geometry guidance** encourages the generated reflection to remain consistent with the spatial structure of the scene.
- **Perceptual guidance** promotes visually coherent reflections while preserving important semantic and appearance cues.
- **Diffusion-based generation** integrates these constraints within a flexible image-generation framework.

The current release provides inference code and pretrained checkpoints. Additional method details and resources will be added as they become available.

## Datasets

### SynMirrorV2

We adopt **SynMirrorV2**, introduced in MirrorVerse, as the primary synthetic training and evaluation resource. SynMirrorV2 contains **207K samples** and provides full scene geometry, including:

- RGB images;
- depth maps;
- surface-normal maps; and
- segmentation masks.

The dataset includes varied object poses, occlusions, camera viewpoints, and multi-object configurations. For download instructions, terms of use, and the latest dataset updates, please refer to the official [MirrorVerse repository](https://github.com/val-iisc/MirrorVerse) and [SynMirrorV2 dataset page](https://huggingface.co/datasets/ankitIIsc/SynMirrorV2).

> PhysReflect does not redistribute SynMirrorV2. Please obtain the dataset from its official source and comply with the original license and usage terms.

### MSD (Mirror Segmentation Dataset)

**MSD**, introduced in *Where Is My Mirror?* (ICCV 2019), contains **4,018 real-world RGB images** with manually annotated **pixel-level mirror masks**, covering both indoor and outdoor scenes. We use MSD to evaluate mirror-reflection generation on real-world photographs. The paper and dataset access instructions are available on the authors' [official MSD project page](https://mhaiyang.github.io/ICCV2019_MirrorNet/index.html).

For `--model msd`, prepare paired RGB images, mirror masks, and depth maps as described in [Input Format](#input-format). Depth maps must be prepared separately.

## Baselines

Our experiments follow the baseline setting provided by MirrorVerse, including its released checkpoints:

| Baseline | Description |
| --- | --- |
| **MirrorFusion-v2** | Trained on the single- and multi-object samples of SynMirrorV2. |
| **MirrorFusion-v2-MSD** | Fine-tuned on the real-world Mirror Segmentation Dataset (MSD). |

Baseline code, checkpoints, and usage instructions are maintained by the [MirrorVerse authors](https://github.com/val-iisc/MirrorVerse). PhysReflect input preparation and inference usage are documented below; evaluation scripts remain on the release plan.

## Release Plan

- ✅ **2026-11-22** — [Inference code](inference-code/)
- ✅ **2026-11-22** — [Pretrained PhysReflect checkpoints: single-object, multi-object, and MSD](#pretrained-checkpoints)
- ✅ **2026-11-22** — [Inference environment and dependency instructions](#getting-started)
- [ ] Training and fine-tuning pipeline
- [ ] Dataset preparation tools
- [ ] Evaluation scripts and benchmark configuration

## Getting Started

Use Python 3.11 and a CUDA-capable GPU. The inference environment was checked with Python 3.11.9, PyTorch 2.3.0, torchvision 0.18.0, and CUDA 12.1 wheels.

```bash
git clone https://github.com/LeaFoRin/PhysReflect.git
cd PhysReflect
conda env create -f environment.yml
conda activate physreflect
python -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

`environment.yml` creates the Python environment; the two pip commands install the CUDA build and inference dependencies. Alternatively, use an existing Python 3.11 environment and run the pip commands. See the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v230) for other supported builds.

The entry point uses the bundled modified Diffusers source in `inference-code/src/diffusers`. No separate Diffusers installation is required. The pinned dependencies are listed in [inference-code/requirements.txt](inference-code/requirements.txt).

## Pretrained Checkpoints

Download from the [PhysReflect checkpoint folder on Google Drive](https://drive.google.com/drive/folders/1ombIge3e_bAvOn24rcQLhxfCzZ44wFpG?usp=sharing), open its [model subfolder](https://drive.google.com/drive/folders/1zFNpFAirkYPW3RBI3TBVlZacHEQdGtXP), or use the individual links below:

| Model | Download | Extracted directory | Archive size |
|---|---|---|---|
| Single-object (`--model single`) | [PhysReflect-single-object-fp32.zip](https://drive.google.com/file/d/19K-WKom6fmEHWElie4HFvBdJvSbVUo8r/view) | `models/single-object` | 5.91 GB |
| Multi-object (`--model multi`) | [PhysReflect-multi-object-fp32.zip](https://drive.google.com/file/d/1IxghAI1k1D0VTwcfBuXdK5olEHWeh1nb/view) | `models/multi-object` | 5.91 GB |
| MSD (`--model msd`) | [PhysReflect-msd-fp32.zip](https://drive.google.com/file/d/1rFw8WGrwR0jbFGEw5tPL_L2SPHRyNQMs/view) | `models/msd` | 5.91 GB |

Each archive contains a paired BrushNet and UNet, their architecture configurations, a README, and file checksums. Extract the archive with its directory structure intact to restore both weights and `config.json` files; model configurations are supplied by the downloads. The stored weights are FP32; the launcher loads them in FP16 by default. Model archives are hosted on Drive, with their expected SHA-256 digests in [CHECKPOINTS_SHA256SUMS](CHECKPOINTS_SHA256SUMS).

If all three archives are downloaded into this repository directory, verify them and extract them with:

```bash
sha256sum -c CHECKPOINTS_SHA256SUMS
python -m zipfile -e PhysReflect-single-object-fp32.zip models
python -m zipfile -e PhysReflect-multi-object-fp32.zip models
python -m zipfile -e PhysReflect-msd-fp32.zip models
```

You may download and extract only the model you need. After extraction, verify that model's files by running `sha256sum -c SHA256SUMS` from its model directory.

Always load BrushNet and UNet from the same model directory. A complete Stable Diffusion 1.5 base model is required separately for the tokenizer, text encoder, VAE, and scheduler. Use safetensors component weights, including `text_encoder/model.safetensors`; the pinned dependencies do not support legacy `.bin` text-encoder weights.

## Inference

Set `BASE_MODEL`, `DATA_ROOT`, and `OUTPUT_DIR` to your local locations, then run from the repository root:

```bash
python inference-code/infer.py \
  --model single \
  --base-model "$BASE_MODEL" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR"
```

Select `--model multi` or `--model msd` for the other checkpoints. Use `--models-root` if the three model folders are stored elsewhere. All modes read `test.csv` under the data root by default; `--csv` selects another CSV filename.

The defaults are FP16, UniPC, 15 denoising steps, guidance scale 7.5, seed 1234, and four candidates per input. All three models require concatenated depth conditioning; normal and latent-depth conditioning are unsupported. Optional `--layout_*` settings apply only when `--two_stage_layout` is enabled.

- Add `--dry-run` to inspect the command without loading ML dependencies or models.
- Add `--num_samples 1 --num_images_per_validation 2` for a small inference check.
- Additional options such as `--num_inference_steps`, `--scheduler`, `--CFG`, and `--seed` are passed to the bundled entry point. Use `python inference-code/examples/brushnet/test_brushnet.py --help` to inspect them.

The output is a two-row candidate grid per input. The candidate count must be an even integer of at least two. Existing output files are skipped, so use a fresh output directory when changing inference settings.

## Input Format

The input CSV contains `uid`, `path`, and `auto_caption`. The identifier is used in the output filename.

For `single` and `multi`, each `path` points to an HDF5 file relative to the data root. Required fields are `colors`, `category_id_segmaps`, `depth`, and `normals`. Category 1 marks the mirror. The reader prepares the masked RGB image and depth conditioning.

For `msd`, each `path` names a `.jpg` image in `images/`. The corresponding `.png` mask belongs in `masks/`, and the corresponding `.npz` archive belongs in `depth/`. Use the same relative filename with the appropriate extension. Masks use white for the mirror region. Each depth archive must contain an array named `depth`; prepare depth maps separately.

RGB arrays have shape `(H, W, 3)` and 8-bit values; masks and depth maps have shape `(H, W)` and must match the image dimensions. Use a nonempty mirror mask and finite depth values. Depth preprocessing preserves spatial alignment, including for rectangular inputs. The pipeline handles resizing, and output dimensions are rounded down to multiples of eight.

Each checkpoint has passed a one-input GPU inference check with the bundled code. This verifies loading and generation; a complete benchmark rerun is not included in this release.

## Results

Quantitative comparisons, qualitative results, and ablation studies will be added after the paper and associated artifacts are publicly available.

## Citation

If you use PhysReflect in your research, please cite the [ACM conference paper](https://dl.acm.org/doi/10.1145/3829340.3842250) using the following BibTeX entry. Author names, venue, and pages follow the DOI publication metadata.

```bibtex
@inproceedings{ge2026physreflect,
  author    = {Ge, Shuheng and Ren, Hongwei and Zhang, Li and Wu, Xiangqian},
  title     = {{PhysReflect}: Geometry and Perception Guided Diffusion for Physically-Plausible Mirror Reflections},
  booktitle = {Proceedings of the SIGGRAPH Asia 2026 Conference Papers},
  series    = {SA Conference Papers '26},
  year      = {2026},
  pages     = {1--11},
  publisher = {ACM},
  doi       = {10.1145/3829340.3842250},
  url       = {https://doi.org/10.1145/3829340.3842250}
}
```

Please also cite MirrorVerse when using its baseline or SynMirrorV2 dataset.

## Acknowledgements

We thank the authors of [MirrorVerse](https://github.com/val-iisc/MirrorVerse) for releasing the SynMirrorV2 dataset, baseline implementation, and pretrained checkpoints. PhysReflect uses these resources as its baseline and data foundation.

## License

The PhysReflect repository retains its [MIT License](LICENSE). Bundled Diffusers and BrushNet source retains its [Apache-2.0 license](inference-code/LICENSE), and the upstream [Reflecting-Reality MIT license](inference-code/licenses/Reflecting-Reality-LICENSE) is included. Third-party datasets, model weights, checkpoints, and code remain subject to their respective licenses and terms of use.

## Contact

Questions and release-related requests may be submitted through this repository's issue tracker.
