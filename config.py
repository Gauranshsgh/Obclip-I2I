"""
ObCLIP-I2I Configuration
========================
Centralized model IDs, default hyperparameters, and paths.
"""

import os
import torch

# ── Model Identifiers ────────────────────────────────────────────────
CLOUD_MODEL_ID  = "SG161222/Realistic_Vision_V5.1_noVAE"
CLOUD_UNET_ID   = CLOUD_MODEL_ID                       # Alias for backward compatibility
CLIENT_MODEL_ID = "segmind/small-sd"
CLIENT_UNET_ID  = CLIENT_MODEL_ID                      # Alias for backward compatibility
VAE_MODEL_ID    = "stabilityai/sd-vae-ft-mse"
SHARED_MODEL_ID = CLOUD_MODEL_ID                       # Tokenizer, Text Encoder

# ── Prompts ──────────────────────────────────────────────────────────
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, deformed, bad anatomy, plastic skin, waxy, "
    "cartoon, 3d render, oversaturated, artifacts"
)
PROMPT_WRAPPER = (
    "RAW photo, realistic portrait of a person {prompt}, "
    "natural skin texture, sharp focus, 85mm DSLR, natural lighting"
)

def wrap_prompt(prompt: str) -> str:
    """Wrap a user prompt with photorealistic prefix/suffix."""
    return PROMPT_WRAPPER.format(prompt=prompt)

# ── Hyperparameters ──────────────────────────────────────────────────
NUM_INFERENCE_STEPS = 30          # Total scheduler steps
DEFAULT_STRENGTH    = 0.55        # SDEdit noise level
DEFAULT_K           = 10          # Cloud switch point (first k denoising steps)
N                   = 4           # Batch size: 1 real + (N-1) decoys
CLOUD_GUIDANCE_SCALE  = 8.5       # Cloud CFG scale
CLIENT_GUIDANCE_SCALE = 6.0       # Client CFG scale
RESOLUTION          = 512         # Image resolution (px)

# Backward-compatible aliases (used by test_smoke, eval scripts, etc.)
TOTAL_STEPS    = NUM_INFERENCE_STEPS
STRENGTH       = DEFAULT_STRENGTH
K              = DEFAULT_K
GUIDANCE_SCALE = CLOUD_GUIDANCE_SCALE

# ── FreeU parameters ─────────────────────────────────────────────────
FREEU_S1 = 0.9
FREEU_S2 = 0.2
FREEU_B1 = 1.2
FREEU_B2 = 1.4

# ── Precision & Device ───────────────────────────────────────────────
DTYPE           = torch.float16
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT    = os.path.dirname(os.path.abspath(__file__))
DECOY_DIR       = os.path.join(PROJECT_ROOT, "decoys")

# ── API ──────────────────────────────────────────────────────────────
CLOUD_API_URL   = "http://127.0.0.1:8000"
CLOUD_ENDPOINT  = f"{CLOUD_API_URL}/cloud_i2i_denoise"
