#!/usr/bin/env python3
"""Launch the bundled PhysReflect inference entry point.

Only standard-library modules are imported here, including in --dry-run mode.
Additional arguments are forwarded to examples/brushnet/test_brushnet.py.
"""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


PACKAGE_ROOT = Path(__file__).resolve().parent
PRESETS = {
    "single": ("single-object", "test.csv", False),
    "multi": ("multi-object", "test.csv", False),
    "msd": ("msd", "test.csv", True),
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run PhysReflect on HDF5 or MSD image data.",
        epilog=(
            "Additional arguments are forwarded to the inference entry point. "
            "For example: --num_samples 1 --num_inference_steps 50 --scheduler ddim. "
            "Use --csv to select an input CSV instead of test.csv."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--model", choices=tuple(PRESETS), required=True)
    parser.add_argument(
        "--models-root", type=Path,
        default=PACKAGE_ROOT.parent / "models",
        help="Directory containing single-object/, multi-object/, and msd/.",
    )
    parser.add_argument(
        "--base-model", required=True,
        help="SD 1.5 base-model directory or Hugging Face model ID.",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output directory; defaults to inference-code/outputs/<model>.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the command without importing ML packages or loading models.")
    args, extra = parser.parse_known_args(argv)
    if extra[:1] == ["--"]:
        extra = extra[1:]

    checkpoint_name, csv_name, image_mode = PRESETS[args.model]
    checkpoint = args.models_root.expanduser().resolve() / checkpoint_name
    output_dir = (args.output_dir or PACKAGE_ROOT / "outputs" / args.model).expanduser().resolve()
    command = [
        sys.executable, str(PACKAGE_ROOT / "examples" / "brushnet" / "test_brushnet.py"),
        "--brushnet_path", str(checkpoint),
        "--base_model_path", args.base_model,
        "--data_root", str(args.data_root.expanduser().resolve()),
        "--output_dir", str(output_dir),
        "--csv", csv_name,
        "--caption_column", "auto_caption",
        "--weight_dtype", "fp16",
        "--depth_conditioning_mode", "concat",
        "--depth_source", "gt",
        "--scheduler", "unipc",
        "--num_inference_steps", "15",
        "--CFG", "7.5",
        "--seed", "1234",
        "--num_images_per_validation", "4",
    ]
    if image_mode:
        command.append("--image_mode")
    command.extend(extra)
    if args.dry_run:
        print(shlex.join(command))
        return 0
    missing = [
        checkpoint / component / filename
        for component in ("brushnet", "unet")
        for filename in ("config.json", "diffusion_pytorch_model.safetensors")
        if not (checkpoint / component / filename).is_file()
    ]
    if missing:
        parser.error("Checkpoint files are missing:\n" + "\n".join(str(path) for path in missing))
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
