# PhysReflect: Geometry and Perception Guided Diffusion for Physically-Plausible Mirror Reflections

> Official repository for **PhysReflect: Geometry and Perception Guided Diffusion for Physically-Plausible Mirror Reflections**.
>
##Latest status: 
**The paper has been accepted by SIGGRAPH Asia 2026.**  Code & Checkpoint is Coming Soon.
>

## Release Status

🚧 **This repository is currently under preparation.** The source code, pretrained models, inference scripts, training pipeline, and evaluation instructions are planned for release **after December 2026**.

Please watch or star this repository to receive future release updates.

## Overview

Generating convincing mirror reflections requires more than producing visually plausible content: the reflected scene must also agree with the geometry and perceptual structure of the physical world. PhysReflect studies geometry- and perception-guided diffusion for synthesizing physically plausible mirror reflections.

The project is built on the experimental setting introduced by [MirrorVerse: Pushing Diffusion Models to Realistically Reflect the World](https://github.com/val-iisc/MirrorVerse) (CVPR 2025). We follow MirrorVerse for the baseline implementation and dataset setup to support consistent training and evaluation.

## Method

PhysReflect is designed around two complementary sources of guidance:

- **Geometry guidance** encourages the generated reflection to remain consistent with the spatial structure of the scene.
- **Perceptual guidance** promotes visually coherent reflections while preserving important semantic and appearance cues.
- **Diffusion-based generation** integrates these constraints within a flexible image-generation framework.

Further architectural details, objectives, and implementation notes will be added when the paper and code are released.

## Dataset

We adopt **SynMirrorV2**, introduced in MirrorVerse, as the primary synthetic training and evaluation resource. SynMirrorV2 contains **207K samples** and provides full scene geometry, including:

- RGB images;
- depth maps;
- surface-normal maps; and
- segmentation masks.

The dataset includes varied object poses, occlusions, camera viewpoints, and multi-object configurations. For download instructions, terms of use, and the latest dataset updates, please refer to the official [MirrorVerse repository](https://github.com/val-iisc/MirrorVerse) and [SynMirrorV2 dataset page](https://huggingface.co/datasets/ankitIIsc/SynMirrorV2).

> PhysReflect does not redistribute SynMirrorV2. Please obtain the dataset from its official source and comply with the original license and usage terms.

## Baselines

Our experiments follow the baseline setting provided by MirrorVerse, including its released checkpoints:

| Baseline | Description |
| --- | --- |
| **MirrorFusion-v2** | Trained on the single- and multi-object samples of SynMirrorV2. |
| **MirrorFusion-v2-MSD** | Fine-tuned on the real-world Mirror Segmentation Dataset (MSD). |

Baseline code, checkpoints, and usage instructions are maintained by the [MirrorVerse authors](https://github.com/val-iisc/MirrorVerse). Any project-specific preprocessing or evaluation changes will be documented in this repository upon release.

## Planned Release

The following resources are planned for release after December 2026:

- [ ] Inference code
- [ ] Pretrained PhysReflect checkpoints
- [ ] Training and fine-tuning pipeline
- [ ] Dataset preparation instructions
- [ ] Evaluation scripts and benchmark configuration
- [ ] Configuration files and reproducibility notes

## Getting Started

Installation and inference instructions will be provided together with the public code release. Until then, the MirrorVerse repository can be used to access the baseline implementation, SynMirrorV2 dataset, and baseline checkpoints.

## Results

Quantitative comparisons, qualitative results, and ablation studies will be added after the paper and associated artifacts are publicly available.

## Citation

If you find this project useful, please cite the PhysReflect paper. The complete BibTeX entry will be added once the publication metadata is available.

Please also cite MirrorVerse when using its baseline or SynMirrorV2 dataset:

```bibtex
@inproceedings{Ge2026PhysReflectGA,
  title={PhysReflect: Geometry and Perception Guided Diffusion for Physically-Plausible Mirror Reflections},
  author={Shu-Heng Ge and Hongwei Ren and Li Zhang and Xiang-Qian Wu},
  year={2026},
  url={https://api.semanticscholar.org/CorpusID:292220083}
}
```

## Acknowledgements

We thank the authors of [MirrorVerse](https://github.com/val-iisc/MirrorVerse) for releasing the SynMirrorV2 dataset, baseline implementation, and pretrained checkpoints. PhysReflect uses these resources as its baseline and data foundation.

## License

The PhysReflect code is released under the [MIT License](LICENSE). Third-party datasets, model weights, checkpoints, and code remain subject to their respective licenses and terms of use.

## Contact

Questions and release-related requests may be submitted through this repository's issue tracker. Please note that we cannot provide unreleased code or model weights before the public release.
