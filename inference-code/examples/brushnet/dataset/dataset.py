"""Image, mirror-mask, and depth preprocessing used by PhysReflect inference."""

import numpy as np
import torch


class HDF5Dataset:
    """Static preprocessing helpers for HDF5 inference samples."""

    @staticmethod
    def get_masked_image(image, mask, invert=True):
        masked_image = image.copy()
        if invert:
            masked_image[mask == 255] = 0
        else:
            masked_image[mask == 0] = 0
        return masked_image

    @staticmethod
    def apply_transforms_depth(
        depth_map: np.ndarray,
        mask: np.ndarray = None,
        normalization_method="max_scene_depth",
        max_scene_depth=5.0,
        norm_range=[-1, 1],
        delta=0.5,
    ):
        """Normalize depth without changing its alignment with the input image."""
        depth_map = np.copy(depth_map)  # Make a copy to ensure positive strides
        if depth_map.ndim != 2 or depth_map.size == 0:
            raise ValueError("Depth must be a nonempty two-dimensional array.")
        if not np.isfinite(depth_map).all():
            raise ValueError("Depth must contain only finite values.")

        if mask is not None:
            mask = np.asarray(mask)
            # Accept image masks with channels by using their first channel.
            if mask.ndim == 3 and mask.shape[2] > 0:
                mask = mask[:, :, 0]
            if mask.ndim != 2 or mask.shape != depth_map.shape:
                raise ValueError("The mirror mask must match the depth array's height and width.")
            if not np.any(mask > 0):
                raise ValueError("The mirror mask must contain a nonempty mirror region.")

        if normalization_method == "percentile":
            # Calculate the 2% and 98% percentiles
            d_2 = np.percentile(depth_map, 2)
            d_98 = np.percentile(depth_map, 98)
            if d_98 <= d_2:
                raise ValueError("Percentile normalization requires a nonzero depth range.")

            # Clip the depth_map map to the range [d_2, d_98]
            clipped_depth_map = np.clip(depth_map, d_2, d_98)

            # Normalize to the specified range
            if norm_range == [0, 1]:
                normalized_depth = (clipped_depth_map - d_2) / (d_98 - d_2)
            elif norm_range == [-1, 1]:
                normalized_depth = 2.0 * (clipped_depth_map - d_2) / (d_98 - d_2) - 1.0
            else:
                raise ValueError("Unsupported normalization range. Use [0, 1] or [-1, 1].")
        elif normalization_method == "max_scene_depth":
            if mask is not None:
                # Convert mask to boolean where non-zero values are True
                bool_mask = mask > 0
                # Calculate the maximum depth value over the mask
                max_depth_over_mask = np.max(depth_map[bool_mask])
                max_scene_depth = max_depth_over_mask + delta
            if not np.isfinite(max_scene_depth) or max_scene_depth <= 0:
                raise ValueError("The depth normalization scale must be finite and positive.")

            # Clip the depth map to the range [0, max_scene_depth]
            clipped_depth = np.clip(depth_map, 0, max_scene_depth)

            # Normalize to the specified range
            if norm_range == [0, 1]:
                normalized_depth = clipped_depth / max_scene_depth
            elif norm_range == [-1, 1]:
                normalized_depth = 2.0 * (clipped_depth / max_scene_depth) - 1.0
            else:
                raise ValueError("Unsupported normalization range. Use [0, 1] or [-1, 1].")
        else:
            raise ValueError("Unsupported normalization method. Use 'percentile' or 'max_scene_depth'.")

        # Keep the full field of view; the pipeline resizes all image conditions.
        return torch.tensor(normalized_depth, dtype=torch.float32).unsqueeze(0)

    @staticmethod
    def extract_data_from_hdf5(hdf5_data):
        data = {
            "image": np.array(hdf5_data["colors"], dtype=np.uint8),
            "mask": (np.array(hdf5_data["category_id_segmaps"], dtype=np.uint8) == 1).astype(np.uint8) * 255,
            "depth": np.array(hdf5_data["depth"]),
            "normals": np.array(hdf5_data["normals"]),
        }
        data["masked_image"] = HDF5Dataset.get_masked_image(data["image"], data["mask"])
        return data
