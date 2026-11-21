"""
Client Runner — client/client_runner.py
========================================
Orchestrates the full ObCLIP-I2I client pipeline:
1. Prepare shuffled batch (real + decoys)
2. VAE encode → add noise → serialize
3. Send to cloud API for partial denoising
4. Extract real latent, run remaining steps locally on client UNet
5. VAE decode → optional postprocessing → final edited image
"""

import time
import sys
import os
from typing import Optional, Dict, Any

import torch
import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CLOUD_MODEL_ID, CLIENT_MODEL_ID, VAE_MODEL_ID,
    CLOUD_UNET_ID, CLIENT_UNET_ID, SHARED_MODEL_ID,
    TOTAL_STEPS, STRENGTH, K, N,
    CLOUD_GUIDANCE_SCALE, CLIENT_GUIDANCE_SCALE, GUIDANCE_SCALE,
    RESOLUTION, DTYPE, DECOY_DIR, CLOUD_ENDPOINT,
    DEFAULT_NEGATIVE_PROMPT,
    FREEU_S1, FREEU_S2, FREEU_B1, FREEU_B2,
    wrap_prompt,
)
from client.decoy_manager import prepare_batch
from client.latent_prep import (
    encode_images, add_noise,
    latents_to_b64, b64_to_latents,
)


def load_shared_models():
    """
    Load shared VAE, tokenizer, text_encoder (once), and both UNets.
    All models start on CPU; they are moved to CUDA only when needed.
    """
    from diffusers import (
        AutoencoderKL,
        UNet2DConditionModel,
        DPMSolverMultistepScheduler,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    print(f"[Client] Loading shared VAE from {VAE_MODEL_ID}...")
    vae = AutoencoderKL.from_pretrained(
        VAE_MODEL_ID, torch_dtype=DTYPE
    ).to("cpu")

    print(f"[Client] Loading shared tokenizer & text encoder from {CLOUD_MODEL_ID}...")
    tokenizer = CLIPTokenizer.from_pretrained(
        CLOUD_MODEL_ID, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        CLOUD_MODEL_ID, subfolder="text_encoder", torch_dtype=DTYPE
    ).to("cpu")

    print(f"[Client] Loading client UNet from {CLIENT_MODEL_ID}...")
    unet_client = UNet2DConditionModel.from_pretrained(
        CLIENT_MODEL_ID, subfolder="unet", torch_dtype=DTYPE
    ).to("cpu")

    # Enable FreeU on client UNet
    unet_client.enable_freeu(s1=FREEU_S1, s2=FREEU_S2, b1=FREEU_B1, b2=FREEU_B2)
    print(f"[Client] ✓ FreeU enabled on client UNet")

    print("[Client] Loading scheduler (sde-dpmsolver++ with Karras sigmas)...")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        CLOUD_MODEL_ID, subfolder="scheduler",
        algorithm_type="sde-dpmsolver++", use_karras_sigmas=True
    )

    return {
        "vae": vae,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "unet_client": unet_client,
        "scheduler": scheduler,
    }


@torch.no_grad()
def run_pipeline(
    real_image: Image.Image,
    prompt: str,
    models_dict: dict,
    negative_prompt: Optional[str] = DEFAULT_NEGATIVE_PROMPT,
    n: int = N,
    strength: float = STRENGTH,
    k: int = K,
    total_steps: int = TOTAL_STEPS,
    cloud_guidance_scale: float = CLOUD_GUIDANCE_SCALE,
    client_guidance_scale: float = CLIENT_GUIDANCE_SCALE,
    cloud_url: str = CLOUD_ENDPOINT,
    decoy_dir: str = DECOY_DIR,
    use_accel: bool = True,
    r: int = 2,
    s: int = 3,
    match_lighting: bool = True,
    use_prompt_wrapper: bool = True,
    use_lab_match: bool = True,
    use_gfpgan: bool = False,
    # Legacy compat
    guidance_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run the full ObCLIP-I2I pipeline.
    """
    # Legacy guidance_scale compat
    if guidance_scale is not None:
        cloud_guidance_scale = guidance_scale
        client_guidance_scale = guidance_scale

    torch.cuda.reset_peak_memory_stats()
    result = {"use_accel": use_accel}

    vae = models_dict["vae"]
    tokenizer = models_dict["tokenizer"]
    text_encoder = models_dict["text_encoder"]
    unet_client = models_dict["unet_client"]
    scheduler = models_dict["scheduler"]

    # Apply prompt wrapper if enabled
    effective_prompt = wrap_prompt(prompt) if use_prompt_wrapper else prompt

    # ── Step 1: Prepare shuffled batch ─────────────────────────────
    print(f"\n[Client] Preparing batch: N={n}, strength={strength}, k={k}")
    shuffled_images, real_index = prepare_batch(
        real_image, N=n, decoy_dir=decoy_dir, resolution=RESOLUTION, match_lighting=match_lighting
    )
    result["shuffled_images"] = shuffled_images
    result["real_index"] = real_index
    print(f"[Client] ✓ Batch prepared. Real image at secret index={real_index}")

    # ── Step 2: VAE Encode + Noise Injection ───────────────────────
    print("[Client] Encoding images with VAE...")
    vae.to("cuda")
    latents = encode_images(shuffled_images, vae, device="cuda", dtype=DTYPE)
    print(f"[Client] Encoded latents shape: {list(latents.shape)}")

    scheduler.set_timesteps(total_steps)
    n_active_steps = int(total_steps * strength)
    active_timesteps = scheduler.timesteps[-n_active_steps:]
    
    print(f"[Client] Active denoising steps: {n_active_steps}")
    noisy_latents = add_noise(latents, scheduler, active_timesteps[0].item(), generator=None)

    vae.to("cpu")
    del latents
    torch.cuda.empty_cache()

    # ── Step 3: Send to Cloud ──────────────────────────────────────
    print(f"[Client] Sending to cloud (use_accel={use_accel}, r={r}, s={s})...")
    noisy_b64 = latents_to_b64(noisy_latents)
    upload_size_kb = len(noisy_b64) * 3 / 4 / 1024
    result["upload_payload_kb"] = round(upload_size_kb, 1)

    del noisy_latents
    torch.cuda.empty_cache()

    payload = {
        "noisy_latents_b64": noisy_b64,
        "prompt": effective_prompt,
        "negative_prompt": negative_prompt if negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT,
        "total_steps": total_steps,
        "strength": strength,
        "k": k,
        "cloud_guidance_scale": cloud_guidance_scale,
        "use_accel": use_accel,
        "r": r,
        "s": s,
    }

    response = requests.post(cloud_url, json=payload, timeout=300)
    response.raise_for_status()
    cloud_result = response.json()

    result["cloud_latency_s"] = cloud_result["cloud_latency_s"]
    result["download_payload_kb"] = cloud_result["payload_size_kb"]
    result["cloud_steps"] = k
    print(f"[Client] ✓ Cloud returned in {cloud_result['cloud_latency_s']:.2f}s")

    # ── Step 4: Extract real latent ────────────────────────────────
    intermediate_latents = b64_to_latents(cloud_result["intermediate_latents_b64"])
    real_latent = intermediate_latents[real_index:real_index + 1]
    del intermediate_latents

    # ── Step 5: Client-side denoising ──────────────────────────────
    client_start = time.time()
    
    client_timesteps = active_timesteps[k:]
    n_client_steps = len(client_timesteps)
    result["client_steps"] = n_client_steps

    if n_client_steps > 0:
        text_encoder.to("cuda")
        text_input = tokenizer(
            effective_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = text_encoder(text_input.input_ids.to("cuda"))[0]
        
        neg_prompt = negative_prompt if negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT
        uncond_input = tokenizer(
            neg_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to("cuda"))[0]
        
        combined_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        combined_embeddings = combined_embeddings.to(dtype=DTYPE)
        
        text_encoder.to("cpu")
        torch.cuda.empty_cache()

        unet_client.to("cuda")
        client_latents = real_latent.to(device="cuda", dtype=DTYPE)

        scheduler.set_timesteps(total_steps)

        for i, t in enumerate(client_timesteps):
            latent_model_input = torch.cat([client_latents, client_latents])
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = unet_client(
                latent_model_input,
                t,
                encoder_hidden_states=combined_embeddings,
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + client_guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            client_latents = scheduler.step(noise_pred, t, client_latents).prev_sample

            if (i + 1) % 5 == 0 or i == n_client_steps - 1:
                print(f"[Client] Step {i+1}/{n_client_steps} (t={t.item()}) done")

        unet_client.to("cpu")
        del combined_embeddings, noise_pred, latent_model_input
        torch.cuda.empty_cache()
    else:
        client_latents = real_latent.to(device="cuda", dtype=DTYPE)

    client_latency = time.time() - client_start
    result["client_latency_s"] = round(client_latency, 3)

    # ── Step 6: VAE Decode ─────────────────────────────────────────
    vae.to("cuda")
    decoded = vae.decode(client_latents / vae.config.scaling_factor).sample
    vae.to("cpu")
    del client_latents
    torch.cuda.empty_cache()

    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
    final_image = Image.fromarray((decoded[0] * 255).astype("uint8"))

    # ── Step 7: Optional Post-processing ───────────────────────────
    if use_lab_match or use_gfpgan:
        from client.postprocess import lab_color_match, gfpgan_enhance
        if use_lab_match:
            print("[Client] Applying Lab color/lighting match...")
            final_image = lab_color_match(final_image, real_image, strength=0.35)
        if use_gfpgan:
            print("[Client] Applying GFPGAN face restoration...")
            final_image = gfpgan_enhance(final_image, weight=0.65)

    result["final_image"] = final_image
    result["peak_vram_mb"] = round(
        torch.cuda.max_memory_allocated() / 1024 / 1024, 1
    )
    
    print(f"[Client] ✓ Final image generated ({final_image.size})")
    print(f"[Client] Peak VRAM: {result['peak_vram_mb']} MB")

    return result


def main():
    """CLI entry point for testing."""
    test_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test_input.jpg"
    )
    if os.path.exists(test_path):
        real_image = Image.open(test_path).convert("RGB")
    else:
        import numpy as np
        arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        real_image = Image.fromarray(arr)

    prompt = "wearing glasses and a formal suit"
    models_dict = load_shared_models()

    result = run_pipeline(
        real_image=real_image,
        prompt=prompt,
        models_dict=models_dict,
    )

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output_client.png"
    )
    result["final_image"].save(output_path)
    print(f"\n[Client] Output saved to {output_path}")
    print(f"[Client] Cloud latency: {result['cloud_latency_s']}s")
    print(f"[Client] Client latency: {result['client_latency_s']}s")
    print(f"[Client] Peak VRAM: {result['peak_vram_mb']} MB")


if __name__ == "__main__":
    main()
