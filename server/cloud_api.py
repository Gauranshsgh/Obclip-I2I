"""
Cloud API — server/cloud_api.py
================================
FastAPI server simulating the untrusted cloud inference server.
Accepts a batch of N noisy latents + prompt, runs the first k denoising
steps on the large cloud UNet, and returns intermediate latents.

The server has NO access to real_index — privacy bound = 1/N.
"""

import time
import sys
import os
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CLOUD_MODEL_ID, CLOUD_UNET_ID, SHARED_MODEL_ID,
    TOTAL_STEPS, STRENGTH, K, CLOUD_GUIDANCE_SCALE,
    DTYPE, DEFAULT_NEGATIVE_PROMPT,
    FREEU_S1, FREEU_S2, FREEU_B1, FREEU_B2,
)
from client.latent_prep import b64_to_latents, latents_to_b64


# ── Global model references (loaded once at startup) ─────────────────
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load shared models at startup."""
    from diffusers import UNet2DConditionModel, AutoencoderKL, DPMSolverMultistepScheduler
    from transformers import CLIPTextModel, CLIPTokenizer

    print("[Cloud] Loading shared text encoder and tokenizer...")
    models["tokenizer"] = CLIPTokenizer.from_pretrained(
        SHARED_MODEL_ID, subfolder="tokenizer"
    )
    models["text_encoder"] = CLIPTextModel.from_pretrained(
        SHARED_MODEL_ID, subfolder="text_encoder", torch_dtype=DTYPE
    )

    print(f"[Cloud] Loading cloud UNet from {CLOUD_UNET_ID}...")
    models["unet_cloud"] = UNet2DConditionModel.from_pretrained(
        CLOUD_UNET_ID, subfolder="unet", torch_dtype=DTYPE
    )

    # Enable FreeU
    models["unet_cloud"].enable_freeu(s1=FREEU_S1, s2=FREEU_S2, b1=FREEU_B1, b2=FREEU_B2)
    print(f"[Cloud] ✓ FreeU enabled (s1={FREEU_S1}, s2={FREEU_S2}, b1={FREEU_B1}, b2={FREEU_B2})")

    print("[Cloud] Loading scheduler (sde-dpmsolver++ with Karras sigmas)...")
    models["scheduler"] = DPMSolverMultistepScheduler.from_pretrained(
        CLOUD_MODEL_ID, subfolder="scheduler",
        algorithm_type="sde-dpmsolver++", use_karras_sigmas=True
    )

    # Keep everything on CPU initially
    models["text_encoder"].to("cpu")
    models["unet_cloud"].to("cpu")

    print("[Cloud] ✓ All models loaded (on CPU). Ready to serve requests.")
    yield
    # Cleanup
    models.clear()
    torch.cuda.empty_cache()


app = FastAPI(title="ObCLIP-I2I Cloud Server", lifespan=lifespan)


class DenoiseRequest(BaseModel):
    noisy_latents_b64: str
    prompt: str
    negative_prompt: Optional[str] = DEFAULT_NEGATIVE_PROMPT
    total_steps: int = TOTAL_STEPS
    strength: float = STRENGTH
    k: int = K
    cloud_guidance_scale: float = CLOUD_GUIDANCE_SCALE
    use_accel: bool = True
    r: int = 2
    s: int = 3


class DenoiseResponse(BaseModel):
    intermediate_latents_b64: str
    cloud_latency_s: float
    payload_size_kb: float
    use_accel: bool = False


@app.post("/cloud_i2i_denoise", response_model=DenoiseResponse)
async def cloud_i2i_denoise(req: DenoiseRequest):
    """
    Cloud partial denoising endpoint.
    
    Receives N noisy latents (no real_index!), runs k denoising steps
    with the cloud UNet, returns intermediate latents.
    """
    t_start = time.time()

    # ── Deserialize input ──────────────────────────────────────────
    noisy_latents = b64_to_latents(req.noisy_latents_b64, device="cpu")
    batch_size = noisy_latents.shape[0]

    print(f"\n{'='*60}")
    print(f"[Cloud] Received batch shape: {list(noisy_latents.shape)}")
    print(f"[Cloud] Prompt: '{req.prompt}'")
    print(f"[Cloud] ⚠ NO real_index received — privacy bound = 1/{batch_size}")
    print(f"[Cloud] Will run k={req.k} of {req.total_steps} total steps "
          f"(strength={req.strength})")
    print(f"{'='*60}")

    # ── Prepare scheduler ──────────────────────────────────────────
    scheduler = models["scheduler"]
    scheduler.set_timesteps(req.total_steps)
    
    n_active_steps = int(req.total_steps * req.strength)
    active_timesteps = scheduler.timesteps[-n_active_steps:]
    cloud_timesteps = active_timesteps[:req.k]
    
    print(f"[Cloud] Active denoising steps: {n_active_steps}")
    print(f"[Cloud] Cloud timesteps: {cloud_timesteps.tolist()}")

    # ── Encode prompt (move text_encoder to GPU) ───────────────────
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]

    text_encoder.to("cuda")
    
    text_input = tokenizer(
        req.prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to("cuda"))[0]

    neg_prompt = req.negative_prompt if req.negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT
    uncond_input = tokenizer(
        neg_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to("cuda"))[0]

    text_encoder.to("cpu")
    torch.cuda.empty_cache()

    text_embeddings = text_embeddings.expand(batch_size, -1, -1)
    uncond_embeddings = uncond_embeddings.expand(batch_size, -1, -1)
    combined_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    combined_embeddings = combined_embeddings.to(device="cuda", dtype=DTYPE)

    # ── Move cloud UNet to GPU and denoise ─────────────────────────
    unet_cloud = models["unet_cloud"]
    unet_cloud.to("cuda")

    attn_ctx = None
    block_ctrl = None
    if req.use_accel:
        from server.accel_processors import AttnAccelContext, set_accel_processors, restore_vanilla_processors
        from server.block_skip import BlockSkipController

        print(f"[Cloud] ⚡ Enabling I2I Acceleration: temporal attn2 cache (r={req.r}), interleaved block skip (s={req.s})")
        attn_ctx = AttnAccelContext(r=req.r, refresh_rate=2)
        attn_ctx.enabled = True
        attn_ctx.N = batch_size
        set_accel_processors(unet_cloud, attn_ctx)

        block_ctrl = BlockSkipController(s=req.s)
        block_ctrl.enabled = True
    else:
        print("[Cloud] Running Vanilla Cloud Denoising (use_accel=False)")

    latents = noisy_latents.to(device="cuda", dtype=DTYPE)

    with torch.no_grad():
        for i, t in enumerate(cloud_timesteps):
            if req.use_accel and attn_ctx is not None and block_ctrl is not None:
                attn_ctx.set_step(i)
                block_ctrl.set_step(i)

            latent_model_input = torch.cat([latents, latents])
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            if req.use_accel and block_ctrl is not None:
                noise_pred = block_ctrl.forward(
                    unet_cloud,
                    latent_model_input,
                    t,
                    encoder_hidden_states=combined_embeddings,
                )
            else:
                noise_pred = unet_cloud(
                    latent_model_input,
                    t,
                    encoder_hidden_states=combined_embeddings,
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + req.cloud_guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            latents = scheduler.step(noise_pred, t, latents).prev_sample
            assert not torch.isnan(latents).any(), f"Cloud step {i+1} produced NaN latents!"
            assert not torch.isinf(latents).any(), f"Cloud step {i+1} produced Inf latents!"

            skip_flag = ""
            if req.use_accel and block_ctrl is not None and i >= req.s:
                skip_flag = " (skipped)" if (i - req.s) % 2 == 1 else " (full+cache)"
            print(f"[Cloud] Step {i+1}/{req.k} (t={t.item()}){skip_flag} done")

    # ── Cleanup and Offload cloud UNet ─────────────────────────────
    if req.use_accel:
        block_ctrl.reset()
        restore_vanilla_processors(unet_cloud)

    intermediate_latents = latents.cpu()
    unet_cloud.to("cpu")
    del latents, combined_embeddings, noise_pred, latent_model_input
    torch.cuda.empty_cache()

    # ── Serialize response ─────────────────────────────────────────
    result_b64 = latents_to_b64(intermediate_latents)
    
    cloud_latency = time.time() - t_start
    payload_size_kb = len(result_b64) * 3 / 4 / 1024

    print(f"[Cloud] ✓ Cloud denoising complete in {cloud_latency:.2f}s (accel={req.use_accel})")
    print(f"[Cloud] Response payload: {payload_size_kb:.1f} KB")

    return DenoiseResponse(
        intermediate_latents_b64=result_b64,
        cloud_latency_s=round(cloud_latency, 3),
        payload_size_kb=round(payload_size_kb, 1),
        use_accel=req.use_accel,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
