"""
Image and tensor conversion utilities for ComfyUI-Distributed.
"""
import torch
import numpy as np
from PIL import Image

def tensor_to_pil(img_tensor, batch_index=0):
    """Takes a batch of images in tensor form [B, H, W, C] and returns a PIL Image.

    Values are clipped to 0..1 before the (truncating) 8-bit conversion, like
    utils.transfer_codec.tensor_batch_to_uint8: resamplers such as bicubic
    overshoot slightly, and an unclipped 1.01 would wrap around to 2 (black).
    Half-precision tensors are converted to float32 first, and a single
    channel becomes a greyscale ("L") image.
    """
    array = img_tensor[batch_index].detach().cpu().float().numpy()
    array = np.clip(255.0 * array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array)

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

