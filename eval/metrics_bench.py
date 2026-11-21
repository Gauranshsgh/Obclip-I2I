"""
Paper Benchmarks — eval/metrics_bench.py
=========================================
Runs comprehensive Phase 3 evaluations across 10 leave-one-out images:
  1. Four Baselines:
     - Cloud-Only (k=15, N=1)
     - Client-Only (k=0, N=1)
     - Vanilla ObCLIP-I2I (k=5, N=4, use_accel=False)
     - Accelerated ObCLIP-I2I (k=5, N=4, use_accel=True, r=2, s=4)
  2. Sweeps:
     - Cloud switch point k in [3, 5, 7] (N=4, strength=0.6)
     - Decoy candidate pool N in [2, 4, 6] (k=5, strength=0.6)

Metrics recorded:
  - CLIP Score (open_clip ViT-B-32 prompt-image alignment)
  - LPIPS Distance (lpips AlexNet perceptual fidelity)
  - Cloud Latency (s) & Client Latency (s)
  - Communication Payload (KB)
  - Peak VRAM (MB)

Outputs:
  - eval/results_table.csv
  - eval/tradeoff_plots.png
"""

import os
import sys
import time
import subprocess
import requests
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

import open_clip
import lpips

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TOTAL_STEPS, STRENGTH, GUIDANCE_SCALE, RESOLUTION, DECOY_DIR,
    CLOUD_API_URL, CLOUD_ENDPOINT, DEFAULT_NEGATIVE_PROMPT,
    VAE_MODEL_ID, CLOUD_MODEL_ID, CLIENT_MODEL_ID, DTYPE
)
from client.decoy_manager import load_all_decoys
from client.client_runner import load_shared_models, run_pipeline
from client.latent_prep import encode_images, add_noise


def ensure_server_running(project_root: str):
    """Ensure cloud API server is alive on port 8000; launch if needed."""
    try:
        resp = requests.get(f"{CLOUD_API_URL}/docs", timeout=2)
        if resp.status_code == 200:
            print("[Metrics] ✓ Cloud server is already active.")
            return None
    except (requests.ConnectionError, requests.Timeout):
        pass

    print("[Metrics] Spawning cloud API server on port 8000...")
    venv_python = os.path.join(project_root, "venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        venv_python = sys.executable

    log_path = os.path.join(project_root, "eval", "cloud_server.log")
    log_file = open(log_path, "w", encoding="utf-8")
    server_proc = subprocess.Popen(
        [venv_python, "-m", "uvicorn", "server.cloud_api:app",
         "--host", "127.0.0.1", "--port", "8000"],
        cwd=project_root,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    start = time.time()
    max_wait = 180
    while time.time() - start < max_wait:
        if server_proc.poll() is not None:
            print("[Metrics] ✗ Server process exited unexpectedly:")
            print(server_proc.communicate()[0])
            sys.exit(1)
        try:
            resp = requests.get(f"{CLOUD_API_URL}/docs", timeout=2)
            if resp.status_code == 200:
                print(f"[Metrics] ✓ Cloud server is ready ({int(time.time() - start)}s).")
                return server_proc
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(3)

    print("[Metrics] ✗ Cloud server failed to start within timeout.")
    server_proc.terminate()
    sys.exit(1)


class MetricEvaluator:
    """Computes CLIP Score and LPIPS distance."""
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print("[Metrics] Initializing LPIPS (AlexNet)...")
        self.lpips_fn = lpips.LPIPS(net="alex").to(self.device).eval()
        
        print("[Metrics] Initializing OpenCLIP (ViT-B-32)...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.clip_model = self.clip_model.to(self.device).eval()
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

    @torch.no_grad()
    def compute_clip_score(self, image: Image.Image, prompt: str) -> float:
        img_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = self.clip_tokenizer([prompt]).to(self.device)
        
        img_feats = self.clip_model.encode_image(img_tensor)
        text_feats = self.clip_model.encode_text(text_tokens)
        
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)
        similarity = (img_feats @ text_feats.T).item()
        return float(similarity)

    @torch.no_grad()
    def compute_lpips(self, img_ref: Image.Image, img_gen: Image.Image) -> float:
        t = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        t_ref = t(img_ref.convert("RGB")).unsqueeze(0).to(self.device)
        t_gen = t(img_gen.convert("RGB")).unsqueeze(0).to(self.device)
        dist = self.lpips_fn(t_ref, t_gen).item()
        return float(dist)


@torch.no_grad()
def run_client_only(
    real_image: Image.Image,
    prompt: str,
    models_dict: dict,
    strength: float = STRENGTH,
    total_steps: int = TOTAL_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
) -> dict:
    """Executes Client-Only baseline (all active steps on compact client UNet)."""
    torch.cuda.reset_peak_memory_stats()
    vae = models_dict["vae"]
    tokenizer = models_dict["tokenizer"]
    text_encoder = models_dict["text_encoder"]
    unet_client = models_dict["unet_client"]
    scheduler = models_dict["scheduler"]

    # 1. Encode single image
    vae.to("cuda")
    latents = encode_images([real_image], vae, device="cuda", dtype=DTYPE)
    scheduler.set_timesteps(total_steps)
    n_active = int(total_steps * strength)
    active_timesteps = scheduler.timesteps[-n_active:]
    noisy_latents = add_noise(latents, scheduler, active_timesteps[0].item())
    vae.to("cpu")
    torch.cuda.empty_cache()

    # 2. Text Embeddings
    text_encoder.to("cuda")
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to("cuda"))[0]
    uncond_input = tokenizer(negative_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to("cuda"))[0]
    combined_emb = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=DTYPE)
    text_encoder.to("cpu")
    torch.cuda.empty_cache()

    # 3. Denoise all steps on Client UNet
    t0 = time.time()
    unet_client.to("cuda")
    cur_latents = noisy_latents.to(device="cuda", dtype=DTYPE)
    for t in active_timesteps:
        lat_in = torch.cat([cur_latents, cur_latents])
        lat_in = scheduler.scale_model_input(lat_in, t)
        noise_pred = unet_client(lat_in, t, encoder_hidden_states=combined_emb).sample
        u, c = noise_pred.chunk(2)
        noise_cfg = u + guidance_scale * (c - u)
        cur_latents = scheduler.step(noise_cfg, t, cur_latents).prev_sample

    unet_client.to("cpu")
    torch.cuda.empty_cache()
    client_latency = time.time() - t0

    # 4. Decode
    vae.to("cuda")
    final_latents = cur_latents / vae.config.scaling_factor
    dec_out = vae.decode(final_latents).sample
    vae.to("cpu")
    torch.cuda.empty_cache()

    img_np = ((dec_out[0].permute(1, 2, 0).clamp(-1, 1).cpu().float().numpy() + 1.0) * 127.5).astype(np.uint8)
    final_img = Image.fromarray(img_np)

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return {
        "final_image": final_img,
        "cloud_latency_s": 0.0,
        "client_latency_s": round(client_latency, 3),
        "upload_payload_kb": 0.0,
        "download_payload_kb": 0.0,
        "peak_vram_mb": round(peak_vram_mb, 1),
    }


def plot_paper_tradeoffs(df: pd.DataFrame, output_path: str):
    """Generates 4-panel publication-grade benchmark plots."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=200)
    plt.subplots_adjust(hspace=0.35, wspace=0.28)

    # Panel A: Quality vs Latency Pareto Curve (Baselines)
    ax = axes[0, 0]
    base_df = df[df["config_type"] == "baseline"]
    grouped_base = base_df.groupby("baseline_name")[["clip_score", "total_latency_s", "peak_vram_mb"]].mean()
    
    colors = {"Cloud-Only": "#3498DB", "Client-Only": "#9B59B6", "Vanilla ObCLIP-I2I": "#E67E22", "Accelerated ObCLIP-I2I": "#2ECC71"}
    markers = {"Cloud-Only": "s", "Client-Only": "^", "Vanilla ObCLIP-I2I": "o", "Accelerated ObCLIP-I2I": "*"}

    for name, row in grouped_base.iterrows():
        c = colors.get(name, "black")
        m = markers.get(name, "o")
        ax.scatter(row["total_latency_s"], row["clip_score"], color=c, marker=m, s=160, zorder=4, label=name)
        ax.annotate(name, (row["total_latency_s"], row["clip_score"]),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")

    ax.set_title("A. Generation Quality vs Total Latency Pareto", fontsize=11, fontweight="bold")
    ax.set_xlabel("Total Latency (Cloud + Client, s)", fontsize=10)
    ax.set_ylabel("CLIP Score (Prompt Alignment)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel B: Privacy Candidate Pool (N) vs Communication Payload
    ax = axes[0, 1]
    n_df = df[df["config_type"] == "n_sweep"]
    grouped_n = n_df.groupby("N")[["total_payload_kb"]].mean()
    
    ax.plot(grouped_n.index, grouped_n["total_payload_kb"], marker="D", color="#E74C3C", linewidth=2.5, markersize=8)
    for n_val, p_val in zip(grouped_n.index, grouped_n["total_payload_kb"]):
        ax.annotate(f"{p_val:.1f} KB\n(1/{n_val} = {100/n_val:.0f}%)", (n_val, p_val),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")

    ax.set_title("B. Privacy Guarantee (1/N) vs Payload Overhead", fontsize=11, fontweight="bold")
    ax.set_xlabel("Candidate Batch Size (N)", fontsize=10)
    ax.set_ylabel("Total Network Payload (KB)", fontsize=10)
    ax.set_xticks(grouped_n.index)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel C: Cloud Speedup vs Cloud Steps (k)
    ax = axes[1, 0]
    k_df = df[df["config_type"] == "k_sweep"]
    grouped_k = k_df.groupby(["k", "use_accel"])[["cloud_latency_s"]].mean().unstack()
    
    if False in grouped_k.columns and True in grouped_k.columns:
        speedups = grouped_k["cloud_latency_s"][False] / grouped_k["cloud_latency_s"][True]
        ax.bar(speedups.index.astype(str), speedups.values, color="#1ABC9C", width=0.45, edgecolor="black")
        for i, (k_val, sp) in enumerate(zip(speedups.index.astype(str), speedups.values)):
            ax.annotate(f"{sp:.2f}x", (k_val, sp), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=10, fontweight="bold")
    else:
        # Fallback if unstacked is formatted differently
        grouped_k_acc = k_df[k_df["use_accel"] == True].groupby("k")["cloud_latency_s"].mean()
        ax.bar(grouped_k_acc.index.astype(str), grouped_k_acc.values, color="#1ABC9C", width=0.45, edgecolor="black")

    ax.set_title("C. Cloud Acceleration Speedup vs Switch Point (k)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Cloud Switch Point (k steps)", fontsize=10)
    ax.set_ylabel("Speedup Factor (x)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")

    # Panel D: LPIPS Perceptual Distance vs Cloud Switch Point (k)
    ax = axes[1, 1]
    grouped_lpips = k_df[k_df["use_accel"] == True].groupby("k")[["lpips_distance"]].mean()
    
    ax.plot(grouped_lpips.index, grouped_lpips["lpips_distance"], marker="o", color="#8E44AD", linewidth=2.5, markersize=8)
    for k_val, dist in zip(grouped_lpips.index, grouped_lpips["lpips_distance"]):
        ax.annotate(f"{dist:.3f}", (k_val, dist), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")

    ax.set_title("D. LPIPS Perceptual Distance vs Switch Point (k)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Cloud Switch Point (k steps)", fontsize=10)
    ax.set_ylabel("LPIPS Distance (Lower = Closer to Source)", fontsize=10)
    ax.set_xticks(grouped_lpips.index)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Metrics] ✓ Saved publication tradeoff plots to: {output_path}")


def main():
    print("=" * 70)
    print("ObCLIP-I2I Phase 3 — Paper Benchmarks (Multi-Baseline & Sweeps)")
    print("=" * 70)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    server_proc = ensure_server_running(project_root)

    try:
        # 1. Load decoders & shared client models
        print("\n[Metrics] Loading models...")
        evaluator = MetricEvaluator()
        models_dict = load_shared_models()

        # 2. Select 10 test images
        all_decoys = load_all_decoys(DECOY_DIR)
        assert len(all_decoys) >= 20, "Need at least 20 images in decoys/."
        np.random.seed(42)
        test_indices = np.random.choice(len(all_decoys), size=10, replace=False)
        test_paths = [all_decoys[i] for i in test_indices]
        print(f"[Metrics] Selected {len(test_paths)} test images (leave-one-out from {len(all_decoys)} pool).")

        prompt = "wearing glasses and a formal suit"
        records = []

        # ── Benchmark 1: The 4 Baselines (10 images) ───────────────────
        print("\n" + "=" * 60)
        print("STAGE 1: Evaluating 4 Baselines across 10 images...")
        print("=" * 60)

        for img_idx, p in enumerate(test_paths):
            real_img = Image.open(p).convert("RGB")
            print(f"\n[Test Image {img_idx+1}/10] {os.path.basename(p)}")

            # 1. Cloud-Only (k=15, N=1, no acceleration needed)
            res_cloud = run_pipeline(
                real_image=real_img, prompt=prompt, models_dict=models_dict,
                n=1, strength=0.6, k=15, use_accel=False, match_lighting=False
            )
            c_score = evaluator.compute_clip_score(res_cloud["final_image"], prompt)
            lp_dist = evaluator.compute_lpips(real_img, res_cloud["final_image"])
            records.append({
                "image_idx": img_idx, "config_type": "baseline", "baseline_name": "Cloud-Only",
                "k": 15, "N": 1, "use_accel": False,
                "clip_score": c_score, "lpips_distance": lp_dist,
                "cloud_latency_s": res_cloud["cloud_latency_s"], "client_latency_s": res_cloud["client_latency_s"],
                "total_latency_s": round(res_cloud["cloud_latency_s"] + res_cloud["client_latency_s"], 3),
                "upload_payload_kb": res_cloud["upload_payload_kb"], "download_payload_kb": res_cloud["download_payload_kb"],
                "total_payload_kb": round(res_cloud["upload_payload_kb"] + res_cloud["download_payload_kb"], 1),
                "peak_vram_mb": res_cloud["peak_vram_mb"],
            })

            # 2. Client-Only (k=0, N=1)
            res_client = run_client_only(
                real_image=real_img, prompt=prompt, models_dict=models_dict, strength=0.6
            )
            c_score = evaluator.compute_clip_score(res_client["final_image"], prompt)
            lp_dist = evaluator.compute_lpips(real_img, res_client["final_image"])
            records.append({
                "image_idx": img_idx, "config_type": "baseline", "baseline_name": "Client-Only",
                "k": 0, "N": 1, "use_accel": False,
                "clip_score": c_score, "lpips_distance": lp_dist,
                "cloud_latency_s": res_client["cloud_latency_s"], "client_latency_s": res_client["client_latency_s"],
                "total_latency_s": round(res_client["cloud_latency_s"] + res_client["client_latency_s"], 3),
                "upload_payload_kb": res_client["upload_payload_kb"], "download_payload_kb": res_client["download_payload_kb"],
                "total_payload_kb": round(res_client["upload_payload_kb"] + res_client["download_payload_kb"], 1),
                "peak_vram_mb": res_client["peak_vram_mb"],
            })

            # 3. Vanilla ObCLIP-I2I (k=5, N=4, use_accel=False)
            res_vanilla = run_pipeline(
                real_image=real_img, prompt=prompt, models_dict=models_dict,
                n=4, strength=0.6, k=5, use_accel=False, match_lighting=True
            )
            c_score = evaluator.compute_clip_score(res_vanilla["final_image"], prompt)
            lp_dist = evaluator.compute_lpips(real_img, res_vanilla["final_image"])
            records.append({
                "image_idx": img_idx, "config_type": "baseline", "baseline_name": "Vanilla ObCLIP-I2I",
                "k": 5, "N": 4, "use_accel": False,
                "clip_score": c_score, "lpips_distance": lp_dist,
                "cloud_latency_s": res_vanilla["cloud_latency_s"], "client_latency_s": res_vanilla["client_latency_s"],
                "total_latency_s": round(res_vanilla["cloud_latency_s"] + res_vanilla["client_latency_s"], 3),
                "upload_payload_kb": res_vanilla["upload_payload_kb"], "download_payload_kb": res_vanilla["download_payload_kb"],
                "total_payload_kb": round(res_vanilla["upload_payload_kb"] + res_vanilla["download_payload_kb"], 1),
                "peak_vram_mb": res_vanilla["peak_vram_mb"],
            })

            # 4. Accelerated ObCLIP-I2I (k=5, N=4, use_accel=True, r=2, s=4)
            res_accel = run_pipeline(
                real_image=real_img, prompt=prompt, models_dict=models_dict,
                n=4, strength=0.6, k=5, use_accel=True, r=2, s=4, match_lighting=True
            )
            c_score = evaluator.compute_clip_score(res_accel["final_image"], prompt)
            lp_dist = evaluator.compute_lpips(real_img, res_accel["final_image"])
            records.append({
                "image_idx": img_idx, "config_type": "baseline", "baseline_name": "Accelerated ObCLIP-I2I",
                "k": 5, "N": 4, "use_accel": True,
                "clip_score": c_score, "lpips_distance": lp_dist,
                "cloud_latency_s": res_accel["cloud_latency_s"], "client_latency_s": res_accel["client_latency_s"],
                "total_latency_s": round(res_accel["cloud_latency_s"] + res_accel["client_latency_s"], 3),
                "upload_payload_kb": res_accel["upload_payload_kb"], "download_payload_kb": res_accel["download_payload_kb"],
                "total_payload_kb": round(res_accel["upload_payload_kb"] + res_accel["download_payload_kb"], 1),
                "peak_vram_mb": res_accel["peak_vram_mb"],
            })

        # ── Benchmark 2: Sweeps across a representative subset ────────
        print("\n" + "=" * 60)
        print("STAGE 2: Evaluating k and N sweeps...")
        print("=" * 60)

        sweep_img = Image.open(test_paths[0]).convert("RGB")

        # k sweep: k in [3, 5, 7], compare vanilla and accelerated
        for k_val in [3, 5, 7]:
            s_val = min(4, k_val - 1)
            # Vanilla
            res_v = run_pipeline(
                real_image=sweep_img, prompt=prompt, models_dict=models_dict,
                n=4, strength=0.6, k=k_val, use_accel=False, match_lighting=True
            )
            c_v = evaluator.compute_clip_score(res_v["final_image"], prompt)
            lp_v = evaluator.compute_lpips(sweep_img, res_v["final_image"])
            records.append({
                "image_idx": 0, "config_type": "k_sweep", "baseline_name": f"k={k_val}_vanilla",
                "k": k_val, "N": 4, "use_accel": False,
                "clip_score": c_v, "lpips_distance": lp_v,
                "cloud_latency_s": res_v["cloud_latency_s"], "client_latency_s": res_v["client_latency_s"],
                "total_latency_s": round(res_v["cloud_latency_s"] + res_v["client_latency_s"], 3),
                "upload_payload_kb": res_v["upload_payload_kb"], "download_payload_kb": res_v["download_payload_kb"],
                "total_payload_kb": round(res_v["upload_payload_kb"] + res_v["download_payload_kb"], 1),
                "peak_vram_mb": res_v["peak_vram_mb"],
            })
            # Accelerated
            res_a = run_pipeline(
                real_image=sweep_img, prompt=prompt, models_dict=models_dict,
                n=4, strength=0.6, k=k_val, use_accel=True, r=2, s=s_val, match_lighting=True
            )
            c_a = evaluator.compute_clip_score(res_a["final_image"], prompt)
            lp_a = evaluator.compute_lpips(sweep_img, res_a["final_image"])
            records.append({
                "image_idx": 0, "config_type": "k_sweep", "baseline_name": f"k={k_val}_accel",
                "k": k_val, "N": 4, "use_accel": True,
                "clip_score": c_a, "lpips_distance": lp_a,
                "cloud_latency_s": res_a["cloud_latency_s"], "client_latency_s": res_a["client_latency_s"],
                "total_latency_s": round(res_a["cloud_latency_s"] + res_a["client_latency_s"], 3),
                "upload_payload_kb": res_a["upload_payload_kb"], "download_payload_kb": res_a["download_payload_kb"],
                "total_payload_kb": round(res_a["upload_payload_kb"] + res_a["download_payload_kb"], 1),
                "peak_vram_mb": res_a["peak_vram_mb"],
            })

        # N sweep: N in [2, 4, 6] (at k=5, accelerated)
        for n_val in [2, 4, 6]:
            res_n = run_pipeline(
                real_image=sweep_img, prompt=prompt, models_dict=models_dict,
                n=n_val, strength=0.6, k=5, use_accel=True, r=2, s=4, match_lighting=True
            )
            c_n = evaluator.compute_clip_score(res_n["final_image"], prompt)
            lp_n = evaluator.compute_lpips(sweep_img, res_n["final_image"])
            records.append({
                "image_idx": 0, "config_type": "n_sweep", "baseline_name": f"N={n_val}_accel",
                "k": 5, "N": n_val, "use_accel": True,
                "clip_score": c_n, "lpips_distance": lp_n,
                "cloud_latency_s": res_n["cloud_latency_s"], "client_latency_s": res_n["client_latency_s"],
                "total_latency_s": round(res_n["cloud_latency_s"] + res_n["client_latency_s"], 3),
                "upload_payload_kb": res_n["upload_payload_kb"], "download_payload_kb": res_n["download_payload_kb"],
                "total_payload_kb": round(res_n["upload_payload_kb"] + res_n["download_payload_kb"], 1),
                "peak_vram_mb": res_n["peak_vram_mb"],
            })

        # ── Save CSV & Generate Plots ─────────────────────────────────
        df = pd.DataFrame(records)
        csv_path = os.path.join(project_root, "eval", "results_table.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[Metrics] ✓ Saved full benchmark results to: {csv_path}")

        plot_path = os.path.join(project_root, "eval", "tradeoff_plots.png")
        plot_paper_tradeoffs(df, plot_path)

        # ── Print Summary Table ───────────────────────────────────────
        base_df = df[df["config_type"] == "baseline"]
        agg = base_df.groupby("baseline_name").agg({
            "clip_score": ["mean", "std"],
            "lpips_distance": ["mean", "std"],
            "cloud_latency_s": "mean",
            "client_latency_s": "mean",
            "total_latency_s": "mean",
            "total_payload_kb": "mean",
            "peak_vram_mb": "max",
        }).round(3)

        print("\n" + "=" * 80)
        print("PHASE 3 PAPER BENCHMARK SUMMARY (4 BASELINES, 10 TEST IMAGES)")
        print("=" * 80)
        print(agg)
        print("=" * 80)

    finally:
        if server_proc is not None:
            print("[Metrics] Stopping cloud server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()
            print("[Metrics] ✓ Cloud server stopped.")


if __name__ == "__main__":
    main()
