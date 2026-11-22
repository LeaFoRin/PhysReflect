# PhysReflect inference

See the [package README](../README.md) for installation, model selection, and input formats.

`infer.py` selects a model and calls the bundled inference entry point. `src/diffusers` contains the modified model and pipeline implementation. Data preprocessing is limited to image, mirror-mask, and depth preparation.

Use `python infer.py --help` for launcher options. Use `python examples/brushnet/test_brushnet.py --help` for additional inference options. Model, base-model, data, and output paths are supplied at runtime.
