import sys
from pathlib import Path

# Always load the BrushNet-enabled Diffusers shipped with this release.
# Keep the caller's working directory unchanged for relative data/output paths.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PACKAGE_ROOT / "src"))

import os
# Configure HF_ENDPOINT externally when an alternative Hugging Face endpoint is needed.
import numpy as np
import logging
from tqdm.auto import tqdm
import h5py
from dataset.dataset import HDF5Dataset
import pandas as pd
from accelerate import PartialState
from accelerate.logging import get_logger
from transformers import pipeline
from diffusers import UniPCMultistepScheduler, DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.models.brushnet import BrushNetModel
from diffusers.pipelines.brushnet.pipeline_brushnet import StableDiffusionBrushNetPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
import torch
import argparse
from PIL import Image, ImageFilter, ImageOps


logger = get_logger(__name__)


def read_from_marigold(geometric_data_path, uid, f_name):
    marigold_path = os.path.join(geometric_data_path, 'marigold')
    #Find the uid in this path
    file_name = os.path.join(marigold_path, 'depth_npy', f"{uid}_{f_name}_pred.npy")
    if not os.path.exists(file_name):
        return None
    else:
        return np.load(file_name)

def read_from_depth_pro(geometric_data_path, rel_path):
    depth_pro_path = os.path.join(geometric_data_path, "depth_pro", rel_path.replace(".hdf5", ".npz"))
    if not os.path.exists(depth_pro_path):
        print(f'File does not exist: {depth_pro_path}')
        return None
    else:
        return np.load(depth_pro_path)["depth"]


def read_from_geowizard(geometric_data_path, uid, f_name, mode):
    geowizard_path = os.path.join(geometric_data_path, 'geowizard')
    if mode == 'depth':
        file_name = os.path.join(geowizard_path, 'depth_npy', f"{uid}_{f_name}_pred.npy")
        if not os.path.exists(file_name):
            return None
        else:
            return np.load(file_name)
    elif mode == 'normal':
        file_name = os.path.join(geowizard_path, 'normal_npy', f"{uid}_{f_name}_pred.npy")
        if not os.path.exists(file_name):
            return None
        else:
            return np.load(file_name)
    else:
        logger.error(f"Wrong mode for reading from geowizard: {mode}")
        return None


def image_grid(imgs, args):
    """Create a grid of images from a list of images. Note that the number of images must be equal to rows * cols."""
    rows = 2
    assert len(imgs) == args.num_images_per_validation, f"Number of images must be equal to {args.num_images_per_validation}"
    cols = args.num_images_per_validation // rows
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def summerize_caption(caption, summarizer):
    return summarizer(caption, max_length=50, min_length=10, do_sample=False)[0]["summary_text"]


def get_blended_image(gt_image: Image.Image, gen_image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    create final output with mask region from gen image and rest from gt image
    """
    gt_image = gt_image.convert("RGBA")
    gen_image = gen_image.convert("RGBA")
    mask = mask.convert("RGBA")
    blended_image = Image.blend(gt_image, gen_image, alpha=0.5)
    blended_image.paste(gen_image, (0, 0), mask)
    return blended_image


def make_layout_condition(
    gt_image: Image.Image,
    layout_image: Image.Image,
    original_condition: Image.Image,
    mask: Image.Image,
    mode: str,
) -> Image.Image:
    """
    Build the intermediate layout image used by the second-stage generation.

    The mask area is the mirror target region. Outside the mask we keep the original
    scene context, matching the conditioning style used by BrushNet.
    """
    layout_image = layout_image.convert("RGB")
    gt_image = gt_image.convert("RGB").resize(layout_image.size, Image.Resampling.BICUBIC)
    original_condition = original_condition.convert("RGB").resize(layout_image.size, Image.Resampling.BICUBIC)
    mask = mask.convert("L").resize(layout_image.size, Image.Resampling.NEAREST)

    if mode == "rgb":
        return layout_image
    if mode == "composite":
        return Image.composite(layout_image, gt_image, mask)
    if mode == "masked":
        return Image.composite(layout_image, original_condition, mask)
    if mode == "edge":
        edge = ImageOps.grayscale(layout_image).filter(ImageFilter.FIND_EDGES)
        edge_rgb = ImageOps.colorize(edge, black="black", white="white")
        return Image.composite(edge_rgb, original_condition, mask)

    raise ValueError(f"Unsupported layout mode: {mode}")


def ensure_condition_tensor_0_1(x):
    """Ensure conditioning tensor/array is in [0, 1] to satisfy diffusers image processor expectations."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.to(dtype=torch.float32)
        if torch.min(x) < 0:
            x = (x + 1.0) / 2.0
        return torch.clamp(x, 0.0, 1.0)
    if isinstance(x, np.ndarray):
        x = x.astype(np.float32)
        if np.min(x) < 0:
            x = (x + 1.0) / 2.0
        return np.clip(x, 0.0, 1.0)
    return x


def main(args):
    test_df = pd.read_csv(os.path.join(args.data_root, args.csv))
    if args.infer_list:
        with open(args.infer_list, "r") as f:
            infer_list = f.readlines()
        infer_list = [x.strip() for x in infer_list]
        test_df = test_df[test_df["path"].isin(infer_list)]
        print(f"Processing {len(test_df)} files from the list.")
    if not args.infer_list and args.num_samples:
        test_df = test_df.sample(args.num_samples, random_state=args.seed)

    def get_data(index):
        row = test_df.iloc[index]
        caption = str(row[args.caption_column])
        hdf5_path = os.path.join(args.data_root, str(row["path"]))
        uid = row["uid"]
        f_name = os.path.split(hdf5_path)[1].split('.')[0] # View identifier from the input filename.
        with h5py.File(hdf5_path, "r") as hdf5_data:
            data = HDF5Dataset.extract_data_from_hdf5(hdf5_data)
        return data, caption, uid, f_name, str(row["path"])

    def get_img_data(index):
        # Read paired RGB images, mirror masks, and prepared depth maps.
        row = test_df.iloc[index]
        caption = str(row[args.caption_column])
        img_path = os.path.join(args.data_root, "images", str(row["path"]))
        mask_path = os.path.join(args.data_root, "masks", str(row["path"]).replace(".jpg", ".png"))
        depth_path = os.path.join(args.data_root, "depth", str(row["path"]).replace(".jpg", ".npz"))
        uid = row["uid"]
        return (img_path, mask_path, depth_path), caption, uid, 0 # Default view identifier for image inputs.
        
    weight_dtype = torch.float32
    if args.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif args.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16

    # summarizer for long captions
    summarizer = None
    if args.summarizer:
        summarizer = pipeline("summarization", model=args.summarizer)

    def run_inference(brushnet_path: str, output_dir: str):
        print("Loading PhysReflect model.")
        os.makedirs(output_dir, exist_ok=True)
        subfolder = ""
        if os.path.isdir(os.path.join(brushnet_path, "brushnet")):
            subfolder = "brushnet"

        brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=weight_dtype, subfolder=subfolder)
        unet = None
        # Load the accompanying UNet weights when present.
        if os.path.isdir(os.path.join(brushnet_path, "unet")):
            unet = UNet2DConditionModel.from_pretrained(
                brushnet_path, subfolder="unet", torch_dtype=weight_dtype
            )
        pipe = StableDiffusionBrushNetPipeline.from_pretrained(
            args.base_model_path,
            brushnet=brushnet,
            unet=unet,
            torch_dtype=weight_dtype,
            low_cpu_mem_usage=False,
            safety_checker=None,
            depth_conditioning_mode=args.depth_conditioning_mode,
            normals_conditioning_mode=args.normals_conditioning_mode,
        )
        pipe.set_progress_bar_config(disable=True)
        # speed up diffusion process with faster scheduler and memory optimization
        if args.scheduler == "unipc":
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        elif args.scheduler == "dpm++":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True
            )
        else:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        # memory optimization.
        # pipe.enable_model_cpu_offload()

        # for distributed inference
        distributed_state = PartialState()
        pipe.to(distributed_state.device)

        generator = torch.Generator("cuda").manual_seed(args.seed)
        with distributed_state.split_between_processes(list(range(test_df.shape[0]))) as idxs:
            idxs = list(idxs)
            progress_bar = tqdm(total=len(idxs), disable=not distributed_state.is_local_main_process)
            for i in idxs:
                if args.image_mode:
                    # Reading from images (MSD dataset)
                    # override mirror prompt
                    args.mirror_prompt = ""
                    img_data, caption, uid, f_name = get_img_data(i)
                else:
                    #HDF5 mode
                    data, caption, uid, f_name, rel_path = get_data(i)

                save_name = f"{uid}.png" if uid == f_name else f"{uid}_{f_name}.png"
                save_path = os.path.join(output_dir, save_name)

                if os.path.exists(save_path):
                    logger.warning(f"Skipping {save_path} as it already exists.")
                    progress_bar.update(1)
                    continue

                depth_image = None
                normal_image = None

                validation_prompt = args.mirror_prompt + caption

                if summarizer:
                    validation_prompt = summerize_caption(validation_prompt, summarizer)

                if args.image_mode:
                    gt_image = Image.open(img_data[0]).convert("RGB")
                    validation_mask = Image.open(img_data[1]).convert("L")  # Convert mask to grayscale

                    # Create a black image of the same size as the validation image
                    black_image = Image.new("RGB", gt_image.size, "black")

                    # Apply the mask to the black image
                    validation_image = Image.composite(black_image, gt_image, validation_mask)

                    if args.depth_conditioning_mode is not None:
                        depth_image = np.load(img_data[2])["depth"]
                        depth_image = HDF5Dataset.apply_transforms_depth(
                            depth_image, np.array(validation_mask)
                        )
                else:
                    gt_image = Image.fromarray(data["image"], mode="RGB")
                    if args.hint_map_dir is not None:
                        validation_image = Image.open(os.path.join(args.data_root, args.hint_map_dir, rel_path.replace(".hdf5", ".png")))
                    else:
                        validation_image = Image.fromarray(data["masked_image"], mode="RGB")
                    validation_mask = Image.fromarray(data["mask"]).convert("RGB")
                    if args.depth_conditioning_mode is not None:
                        if args.depth_source == 'gt':
                            depth_image = HDF5Dataset.apply_transforms_depth(data["depth"], data["mask"])
                        elif args.depth_source == 'marigold':
                            depth_data = read_from_marigold(args.geometric_input_data_dir, uid, f_name)
                            if depth_data is None:
                                logger.error(f"Marigold depth doesn't exist for {uid}_{f_name}")
                                progress_bar.update(1)
                                continue
                            depth_image = HDF5Dataset.apply_transforms_depth(depth_data, data["mask"])
                        elif args.depth_source == "depth_pro":
                            depth_data = read_from_depth_pro(args.geometric_input_data_dir, rel_path)
                            if depth_data is None:
                                # logger.error(f"depth_pro depth doesn't exist for {rel_path}")
                                progress_bar.update(1)
                                continue
                            depth_image = HDF5Dataset.apply_transforms_depth(depth_data, data["mask"])

                    if args.normals_conditioning_mode is not None:
                        if args.normal_source == 'gt':
                            normal_image = Image.fromarray(data["normals"], mode="RGB")
                        elif args.normal_source == 'geowizard':
                            normal_data = read_from_geowizard(args.geometric_input_data_dir, uid, f_name, mode='normal')
                            if normal_data is None:
                                logger.error(f"Marigold depth doesnot exist for {uid}_{f_name}")
                                progress_bar.update(1)
                                continue
                            normal_image = Image.fromarray(normal_data, mode="RGB")

                depth_image = ensure_condition_tensor_0_1(depth_image)
                normal_image = ensure_condition_tensor_0_1(normal_image)

                images = []
                layout_images = []
                layout_dir = os.path.join(output_dir, args.layout_dir)
                if args.two_stage_layout:
                    os.makedirs(layout_dir, exist_ok=True)

                for image_idx in range(args.num_images_per_validation):
                    stage1_prompt = validation_prompt
                    if args.two_stage_layout and args.layout_prompt is not None:
                        stage1_prompt = args.layout_prompt + caption

                    stage1_image = pipe(
                        stage1_prompt,
                        validation_image,
                        validation_mask,
                        depth=depth_image,
                        normals=normal_image,
                        num_inference_steps=(
                            args.layout_num_inference_steps
                            if args.two_stage_layout and args.layout_num_inference_steps is not None
                            else args.num_inference_steps
                        ),
                        guidance_scale=(
                            args.layout_CFG
                            if args.two_stage_layout and args.layout_CFG is not None
                            else args.CFG
                        ),
                        generator=generator,
                        brushnet_conditioning_scale=(
                            args.layout_brushnet_conditioning_scale
                            if args.two_stage_layout else args.brushnet_conditioning_scale
                        ),
                    ).images[0]

                    if args.two_stage_layout:
                        layout_image = make_layout_condition(
                            gt_image=gt_image,
                            layout_image=stage1_image,
                            original_condition=validation_image,
                            mask=validation_mask,
                            mode=args.layout_mode,
                        )
                        layout_images.append(layout_image)
                        layout_save_path = os.path.join(
                            layout_dir,
                            f"{os.path.splitext(save_name)[0]}_layout_{image_idx}.png",
                        )
                        layout_image.save(layout_save_path)
                        image = pipe(
                            validation_prompt,
                            layout_image,
                            validation_mask,
                            depth=depth_image,
                            normals=normal_image,
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=args.CFG,
                            generator=generator,
                            brushnet_conditioning_scale=args.brushnet_conditioning_scale,
                        ).images[0]
                    else:
                        image = stage1_image

                    # choose whether to use blended operations
                    if args.blended:
                        image = get_blended_image(gt_image, image, validation_mask)

                    images.append(image)

                grid = image_grid(images, args)
                grid.save(save_path)
                if args.two_stage_layout:
                    layout_grid = image_grid(layout_images, args)
                    layout_grid.save(os.path.join(layout_dir, save_name))
                progress_bar.update(1)

            progress_bar.close()

    output_dir = os.path.join(args.brushnet_path, args.output_dir)
    run_inference(args.brushnet_path, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brushnet_path", type=str, required=True, help="Model directory containing brushnet/ and unet/.")
    parser.add_argument("--weight_dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--base_model_path", type=str, required=True, help="SD 1.5 base-model directory or Hugging Face model ID.")
    parser.add_argument("--brushnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument(
        "--scheduler", type=str, default="ddim",
        choices=["ddim", "unipc", "dpm++"],
        help="Inference scheduler: DDIM, UniPC, or DPM++.",
    )
    parser.add_argument("--CFG", type=float, default=7.5)
    parser.add_argument(
        "--mirror_prompt", type=str, default="A perfect plane mirror reflection of ", help="Mirror Prompt"
    )
    parser.add_argument(
        "--summarizer",
        type=str,
        default=None,
        help="Optional summarization model name or path for long captions.",
    )
    parser.add_argument(
        "--num_images_per_validation",
        type=int,
        default=4,
        help="Number of images to be generated per validation image",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to generate. Default: all. Set no. for quick testing")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Dataset directory containing the input CSV and its referenced images or HDF5 files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference",
        help="The output directory name where the model predictions will be written. Default: `inference` in `args.brushnet_path`.",
    )
    parser.add_argument("--csv", type=str, default="test.csv")
    parser.add_argument("--caption_column", type=str, default="auto_caption", choices=["caption", "auto_caption"])
    parser.add_argument("--blended", action="store_true", help="Whether to use blended operation")
    parser.add_argument(
        "--two_stage_layout",
        action="store_true",
        help="Generate and save an intermediate layout first, then use it as the second-stage BrushNet condition.",
    )
    parser.add_argument(
        "--layout_dir",
        type=str,
        default="_layouts",
        help="Directory name, under output_dir, where intermediate layouts are saved.",
    )
    parser.add_argument(
        "--layout_mode",
        type=str,
        default="composite",
        choices=["composite", "rgb", "masked", "edge"],
        help=(
            "How to convert the first-stage geometry image into the layout condition: "
            "`composite` keeps GT outside mask and layout inside mask; "
            "`masked` keeps the original masked condition outside mask; "
            "`rgb` uses the raw first-stage image; `edge` uses first-stage edges inside mask."
        ),
    )
    parser.add_argument(
        "--layout_prompt",
        type=str,
        default=None,
        help="Optional prompt prefix for the layout stage. If omitted, the normal validation prompt is reused.",
    )
    parser.add_argument(
        "--layout_num_inference_steps",
        type=int,
        default=None,
        help="Inference steps for the layout stage. Defaults to --num_inference_steps.",
    )
    parser.add_argument(
        "--layout_CFG",
        type=float,
        default=None,
        help="CFG for the layout stage. Defaults to --CFG.",
    )
    parser.add_argument(
        "--layout_brushnet_conditioning_scale",
        type=float,
        default=1.0,
        help="BrushNet conditioning scale for the layout stage.",
    )
    parser.add_argument("--image_mode", action="store_true", help="If true then changes dataloader to read images and mask directly from the path")
    parser.add_argument(
        "--depth_conditioning_mode",
        type=str,
        default="concat",
        choices=["concat"],
        help="The packaged models require concatenated depth conditioning.",
    )
    parser.add_argument(
        "--normals_conditioning_mode",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--geometric_input_data_dir",
        type=str,
        default=None,
        help="Directory containing optional depth predictions.",
    )
    parser.add_argument(
        "--depth_source",
        type=str,
        default='gt',
        choices=['gt', 'marigold', 'depth_pro'],
        help="HDF5 depth source. External predictions require --geometric_input_data_dir. Image mode always reads depth/*.npz.",
    )
    parser.add_argument(
        "--normal_source",
        type=str,
        default='gt',
        choices=['gt', 'geowizard'],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--hint_map_dir",
        type=str,
        default=None,
        help=(
            "If using, set to the directory relative to data_root containing the hint maps. \
            File paths should match the CSV paths with a .png extension."
        ),
    )
    parser.add_argument(
        "--infer_list",
        type=str,
        default=None,
        help=(
            "A list of hdf5 file paths to process. If provided, the script will ignore other paths in `csv` file."
        ),
    )
    args = parser.parse_args()
    if args.num_images_per_validation < 2 or args.num_images_per_validation % 2 != 0:
        parser.error("--num_images_per_validation must be an even integer of at least 2.")
    if args.normals_conditioning_mode is not None:
        parser.error("The packaged models use depth conditioning only; normal conditioning is unsupported.")
    if not args.image_mode and args.depth_source != 'gt' and not args.geometric_input_data_dir:
        parser.error("--geometric_input_data_dir is required for external depth predictions.")
    main(args)
