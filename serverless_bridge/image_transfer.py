"""Image-return policy shared by the master and standalone endpoint bridge.

This module is deliberately dependency-free so graph compilation does not import
Torch, Pillow, PyAV or ComfyUI. Existing Collector settings stay unchanged.
"""

FORMATS = ("exact", "legacy_png", "png", "ffv1", "jpeg", "webp", "h264", "h265", "av1")
DEFAULT_FORMAT = "exact"
DEFAULT_QUALITY = 90
FORMAT_TOOLTIP = (
    "Image returns only; master-local images and other media stay unchanged. "
    "exact: original floating-point tensor, no quantization. "
    "legacy_png / png / ffv1: lossless 8-bit pixels, not exact float tensors. "
    "jpeg / webp: lossy stills; h264 / h265 / av1: lossy frame-batch video. "
    "Alpha is preserved separately when the codec cannot carry it. "
    "Requires matching transfer support on the worker/endpoint."
)
QUALITY_TOOLTIP = (
    "Quality 1–100 for lossy image returns; ignored by exact and lossless modes. "
    "JPEG/WebP use it directly; video formats use the existing Collector CRF mapping."
)


def validate_options(transfer_format=DEFAULT_FORMAT, transfer_quality=DEFAULT_QUALITY):
    """Reject malformed controls rather than repairing or silently substituting."""
    if type(transfer_format) is not str or transfer_format not in FORMATS:
        raise ValueError("Unsupported image transfer_format")
    if type(transfer_quality) is not int or not 1 <= transfer_quality <= 100:
        raise ValueError("Image transfer_quality must be an integer from 1 to 100")
    return transfer_format, transfer_quality
