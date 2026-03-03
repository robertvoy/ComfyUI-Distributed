from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UpscaleCoreArgs:
    """Shared denoise/sampling arguments for USDU tile processing."""

    model: Any
    positive: Any
    negative: Any
    vae: Any
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    denoise: float
    tiled_decode: bool
