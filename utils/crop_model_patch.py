from contextlib import contextmanager
from typing import Any, Iterator, TypeAlias

import torch

from .logging import debug_log
from .usdu_utils import resize_region

CropRegion: TypeAlias = tuple[int, int, int, int]
CropRegions: TypeAlias = list[CropRegion] | CropRegion


@contextmanager
def crop_model_cond(
    model: Any,
    crop_regions: CropRegions,
    init_size: tuple[int, int],
    canvas_size: tuple[int, int],
    tile_size: tuple[int, int],
    latent_crop: bool = False,
) -> Iterator[Any]:
    """Clone model and crop compatible model patches for tile-local sampling."""
    try:
        patched_model = model.clone()
    except Exception:
        # Fallback to original model when clone/patch access is unavailable.
        yield model
        return

    patches = (
        patched_model
        .model_options
        .get("transformer_options", {})
        .get("patches", {})
    )
    applied_croppers = {}
    for _module, module_patches in patches.items():
        for patch in module_patches:
            if id(patch) in applied_croppers:
                continue
            if type(patch).__name__ not in ("DiffSynthCnetPatch", "ZImageControlPatch"):
                continue
            try:
                cropper = ModelPatchCropper(patch).crop(crop_regions, canvas_size, latent_crop)
                applied_croppers[id(patch)] = cropper
            except Exception as exc:
                debug_log(f"crop_model_cond: patch crop skipped for {type(patch).__name__}: {exc}")
    try:
        yield patched_model
    finally:
        for cropper in applied_croppers.values():
            del cropper


class ModelPatchCropper:
    """Stateful crop helper that restores model patch tensors on cleanup."""

    def __init__(self, patch):
        self.patch = patch
        self.original_state = {
            "image": patch.image.clone() if isinstance(patch.image, torch.Tensor) else patch.image,
            "encoded_image": patch.encoded_image.clone() if isinstance(patch.encoded_image, torch.Tensor) else patch.encoded_image,
            "encoded_image_size": patch.encoded_image_size,
        }
        self.patch_class = type(patch).__name__
        required_attrs = (
            "image",
            "model_patch",
            "vae",
            "strength",
            "encoded_image",
            "encoded_image_size",
        )
        missing = [attr for attr in required_attrs if not hasattr(patch, attr)]
        if missing:
            raise AttributeError(
                f"{self.patch_class} missing required attrs: {', '.join(missing)}"
            )

    def __del__(self):
        self.patch.image = self.original_state["image"]
        self.patch.encoded_image = self.original_state["encoded_image"]
        self.patch.encoded_image_size = self.original_state["encoded_image_size"]

    def crop(
        self,
        crop_regions: CropRegions,
        canvas_size: tuple[int, int],
        latent_crop: bool = True,
    ) -> "ModelPatchCropper":
        patch = self.patch

        if not isinstance(crop_regions, list):
            crop_regions = [crop_regions]

        image_size = (patch.image.shape[2], patch.image.shape[1])  # (W,H)

        cropped_images = []
        for crop_region in crop_regions:
            resized_crop = resize_region(crop_region, canvas_size, image_size)
            x1, y1, x2, y2 = resized_crop
            cropped_image = patch.image[:, y1:y2, x1:x2, :]
            cropped_images.append(cropped_image)

        concatenated_image = torch.cat(cropped_images, dim=0)
        patch.image = concatenated_image
        patch.encoded_image_size = (
            concatenated_image.shape[1],
            concatenated_image.shape[2],
        )

        if latent_crop:
            downscale_ratio = patch.vae.spacial_compression_encode()
            cropped_latents = []
            for crop_region in crop_regions:
                resized_crop = resize_region(crop_region, canvas_size, image_size)
                x1, y1, x2, y2 = tuple(x // downscale_ratio for x in resized_crop)
                cropped_latent = patch.encoded_image[:, :, y1:y2, x1:x2]
                cropped_latents.append(cropped_latent)
            patch.encoded_image = torch.cat(cropped_latents, dim=0)
        else:
            patch.__init__(
                patch.model_patch,
                patch.vae,
                concatenated_image,
                patch.strength,
                inpaint_image=patch.inpaint_image,
                mask=patch.mask,
            )
        return self
