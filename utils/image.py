"""
Image and tensor conversion utilities for ComfyUI-Distributed.
"""
import base64
import io
from typing import Any

import torch
import numpy as np
from PIL import Image

def tensor_to_pil(img_tensor, batch_index=0):
    """Takes a batch of images in tensor form [B, H, W, C] and returns an RGB PIL Image."""
    return Image.fromarray((255 * img_tensor[batch_index].cpu().numpy()).astype(np.uint8))

def pil_to_tensor(image):
    """Takes a PIL image and returns a tensor of shape [1, H, W, C]."""
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    if len(image.shape) == 3:  # If grayscale, add channel dimension
        image = image.unsqueeze(-1)
    return image

def ensure_contiguous(tensor):
    """Ensure tensor is contiguous in memory."""
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor


def encode_tensor_png_data_url(image_batch, batch_index=0):
    """Encode one batched tensor image as a PNG data URL."""
    image = tensor_to_pil(image_batch, batch_index)
    byte_io = io.BytesIO()
    image.save(byte_io, format="PNG", compress_level=0)
    encoded = base64.b64encode(byte_io.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def blend_processed_batch_item(
    result_images: list[Any],
    processed_batch: Any,
    batch_index: int,
    blend_fn: Any,
    x1: int,
    y1: int,
    ew: int,
    eh: int,
    tile_mask: Any,
    padding: int,
) -> None:
    """Blend one processed batch item back into result_images in-place."""
    tile_pil = tensor_to_pil(processed_batch, batch_index)
    if tile_pil.size != (ew, eh):
        tile_pil = tile_pil.resize((ew, eh), Image.LANCZOS)
    result_images[batch_index] = blend_fn(
        result_images[batch_index],
        tile_pil,
        x1,
        y1,
        (ew, eh),
        tile_mask,
        padding,
    )
