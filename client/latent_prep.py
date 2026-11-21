"""
Latent Prep — client/latent_prep.py
====================================
Encodes N images into latent space using the shared FP16 VAE encoder,
adds forward Gaussian noise corresponding to t_start (SDEdit),
and provides serialization helpers for base64 transport.
"""

import base64
import io
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DTYPE, RESOLUTION


# ── Image preprocessing ──────────────────────────────────────────────

def images_to_tensor(images: List[Image.Image], resolution: int = RESOLUTION) -> torch.Tensor:
    """
    Convert a list of N PIL Images to a batched tensor [N, 3, H, W]
    normalized to [-1, 1] for VAE encoding.
    """
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),           # [0, 1]
        transforms.Normalize([0.5], [0.5]),  # [-1, 1]
    ])
    tensors = [transform(img.convert("RGB")) for img in images]
    return torch.stack(tensors)  # [N, 3, H, W]


# ── VAE Encoding ─────────────────────────────────────────────────────

@torch.no_grad()
def encode_images(
    images: List[Image.Image],
    vae,
    device: str = "cuda",
    dtype: torch.dtype = DTYPE,
) -> torch.Tensor:
    """
    Encode N PIL images into latent space using the shared VAE encoder.
    
    Args:
        images: List of N PIL Images (already 512x512 RGB).
        vae: The shared VAE model (will be moved to device temporarily).
        device: Target device for encoding.
        dtype: Tensor precision (default: float16).
    
    Returns:
        Latents tensor of shape [N, 4, 64, 64] in FP16.
    """
    pixel_values = images_to_tensor(images).to(device=device, dtype=dtype)
    
    # Encode through VAE
    latent_dist = vae.encode(pixel_values).latent_dist
    latents = latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    
    return latents  # [N, 4, 64, 64]


# ── Forward Noise Injection (SDEdit) ─────────────────────────────────

@torch.no_grad()
def add_noise(
    latents: torch.Tensor,
    scheduler,
    t_start: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Add forward Gaussian noise at timestep t_start for SDEdit.
    
    Args:
        latents: Clean encoded latents [N, 4, 64, 64].
        scheduler: DPMSolverMultistepScheduler instance.
        t_start: The starting timestep index from the schedule.
        generator: Optional torch Generator for reproducibility.
    
    Returns:
        Noisy latents [N, 4, 64, 64] at timestep t_start.
    """
    noise = torch.randn(
        latents.shape,
        generator=generator,
        device=latents.device,
        dtype=latents.dtype,
    )
    
    if isinstance(t_start, torch.Tensor):
        timestep = t_start.to(device=latents.device)
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
    else:
        timestep = torch.tensor([t_start], device=latents.device)
    
    noisy_latents = scheduler.add_noise(latents, noise, timestep)
    
    return noisy_latents


# ── Base64 Serialization ─────────────────────────────────────────────

def latents_to_b64(latents: torch.Tensor) -> str:
    """
    Serialize FP16 latent tensor [N, 4, 64, 64] to base64 string.
    """
    arr = latents.cpu().to(torch.float16).numpy()
    buffer = io.BytesIO()
    np.save(buffer, arr)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def b64_to_latents(b64_str: str, device: str = "cpu") -> torch.Tensor:
    """
    Deserialize base64 string back to FP16 latent tensor.
    """
    raw = base64.b64decode(b64_str)
    buffer = io.BytesIO(raw)
    arr = np.load(buffer)
    return torch.from_numpy(arr).to(dtype=torch.float16, device=device)
