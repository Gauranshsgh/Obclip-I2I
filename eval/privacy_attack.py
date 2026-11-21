"""
Privacy Attack Simulation — eval/privacy_attack.py
===================================================
Evaluates the semi-honest cloud distinguishing attacker on N=4 candidate batches.
The attacker attempts to identify the user's real latent among N-1 decoys
using spatial, histogram, and 2D FFT spectral feature representations.

Compares:
  - Random Decoy Selection (Uniform random sampling)
  - CLIP-Matched Decoy Selection (ObCLIP-I2I with openai/clip-vit-base-patch32)
Target: CLIP-matched decoys collapse attacker accuracy to ~25% (1/N chance).
"""

import os
import sys
import time
from typing import List, Tuple, Dict
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    VAE_MODEL_ID, CLOUD_MODEL_ID, DTYPE, DEVICE, RESOLUTION,
    TOTAL_STEPS, STRENGTH, N, DECOY_DIR
)
from client.decoy_manager import prepare_batch, load_all_decoys
from client.latent_prep import encode_images, add_noise


def extract_latent_features(latents_np: np.ndarray) -> np.ndarray:
    """
    Extract 52 spatial, histogram, and 2D FFT spectral features from a single latent tensor.
    
    Args:
        latents_np: numpy array of shape [4, 64, 64] (channels, H, W).
        
    Returns:
        1D feature vector of shape [52].
    """
    feats = []
    
    # 1. Spatial channel-wise moments (4 means + 4 stds = 8 features)
    ch_means = np.mean(latents_np, axis=(1, 2))
    ch_stds = np.std(latents_np, axis=(1, 2))
    feats.extend(ch_means)
    feats.extend(ch_stds)
    
    # 2. Channel histograms: 8 bins per channel (4 * 8 = 32 features)
    for c in range(4):
        hist, _ = np.histogram(latents_np[c], bins=8, range=(-4.0, 4.0), density=True)
        feats.extend(hist)
        
    # 3. 2D FFT Spectral energy: Low-freq, High-freq, Total energy (4 * 3 = 12 features)
    for c in range(4):
        fft_c = np.fft.fftshift(np.fft.fft2(latents_np[c]))
        mag = np.abs(fft_c)
        total_energy = np.sum(mag ** 2)
        
        # Central 16x16 box is low frequency
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        low_freq_energy = np.sum(mag[cy-8:cy+8, cx-8:cx+8] ** 2)
        high_freq_energy = total_energy - low_freq_energy
        
        feats.append(float(low_freq_energy / (total_energy + 1e-8)))
        feats.append(float(high_freq_energy / (total_energy + 1e-8)))
        feats.append(float(np.log1p(total_energy)))
        
    return np.array(feats, dtype=np.float32)


def generate_attack_dataset(
    vae,
    scheduler,
    decoy_paths: List[str],
    num_batches: int = 40,
    n_candidates: int = 4,
    strength: float = STRENGTH,
    match_lighting: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Generate dataset of candidate latent features and real_index ground truth labels.
    
    Returns:
        X: [num_batches * n_candidates, 52] feature matrix
        y: [num_batches * n_candidates] binary labels (1=real, 0=decoy)
        real_indices: [num_batches] ground truth indices of the real candidate
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Calculate active timesteps for noise injection
    scheduler.set_timesteps(TOTAL_STEPS)
    n_active_steps = int(TOTAL_STEPS * strength)
    t_start = scheduler.timesteps[-n_active_steps].item()
    
    X_list = []
    y_list = []
    real_indices = []
    
    # Select num_batches distinct images to serve as user real images
    selected_indices = np.random.choice(len(decoy_paths), size=num_batches, replace=False)
    
    vae.to("cuda")
    with torch.no_grad():
        for b, idx in enumerate(selected_indices):
            real_path = decoy_paths[idx]
            real_img = Image.open(real_path).convert("RGB")
            
            # Decoy pool excludes the real image (leave-one-out)
            available_decoys = [p for i, p in enumerate(decoy_paths) if i != idx]
            
            shuffled_images, real_idx = prepare_batch(
                real_img,
                N=n_candidates,
                decoy_dir=DECOY_DIR,
                resolution=RESOLUTION,
                match_lighting=match_lighting,
            )
            real_indices.append(real_idx)
            
            latents = encode_images(shuffled_images, vae, device="cuda", dtype=DTYPE)
            noisy_latents = add_noise(latents, scheduler, t_start)
            noisy_np = noisy_latents.cpu().float().numpy()  # [N, 4, 64, 64]
            
            for j in range(n_candidates):
                feats = extract_latent_features(noisy_np[j])
                X_list.append(feats)
                y_list.append(1 if j == real_idx else 0)
                
            del latents, noisy_latents
            if (b + 1) % 10 == 0:
                print(f"[Attack Gen] Processed {b+1}/{num_batches} batches (matched={match_lighting})...")
                
    vae.to("cpu")
    torch.cuda.empty_cache()
    
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int32)
    return X, y, real_indices


def evaluate_attacker(
    X: np.ndarray,
    y: np.ndarray,
    real_indices: List[int],
    n_candidates: int = 4,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Run 5-fold cross-validation of distinguishing attacker on batches.
    An attacker succeeds on a batch if argmax_{j in batch} P(real | candidate_j) == real_index.
    """
    num_batches = len(real_indices)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    rf_batch_correct = 0
    lr_batch_correct = 0
    total_evaluated_batches = 0
    
    batch_indices = np.arange(num_batches)
    
    for train_batch_idx, test_batch_idx in kf.split(batch_indices):
        # Map batch indices to row indices in X
        train_rows = np.concatenate([np.arange(b * n_candidates, (b + 1) * n_candidates) for b in train_batch_idx])
        test_rows = np.concatenate([np.arange(b * n_candidates, (b + 1) * n_candidates) for b in test_batch_idx])
        
        X_train, y_train = X[train_rows], y[train_rows]
        X_test, y_test = X[test_rows], y[test_rows]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. Random Forest Attacker
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed, class_weight="balanced")
        rf.fit(X_train, y_train)
        rf_probs = rf.predict_proba(X_test)[:, 1] if len(rf.classes_) > 1 else np.zeros(len(X_test))
        
        # 2. Logistic Regression Attacker
        lr = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
        lr.fit(X_train_scaled, y_train)
        lr_probs = lr.predict_proba(X_test_scaled)[:, 1] if len(lr.classes_) > 1 else np.zeros(len(X_test_scaled))
        
        # Batch-level top-1 selection
        for i_local, b in enumerate(test_batch_idx):
            start = i_local * n_candidates
            end = start + n_candidates
            true_real = real_indices[b]
            
            # Predict candidate with highest P(real)
            rf_pred = np.argmax(rf_probs[start:end])
            lr_pred = np.argmax(lr_probs[start:end])
            
            if rf_pred == true_real:
                rf_batch_correct += 1
            if lr_pred == true_real:
                lr_batch_correct += 1
            total_evaluated_batches += 1
            
    rf_acc = (rf_batch_correct / total_evaluated_batches) * 100
    lr_acc = (lr_batch_correct / total_evaluated_batches) * 100
    return {"rf_accuracy": rf_acc, "lr_accuracy": lr_acc}


def plot_attack_results(
    results_random: Dict[str, float],
    results_matched: Dict[str, float],
    n_candidates: int,
    output_path: str,
):
    """Plot bar chart comparing attack accuracy against random guessing privacy bound."""
    categories = ["Random Forest Attacker", "Logistic Regression Attacker"]
    x = np.arange(len(categories))
    width = 0.32
    
    acc_random = [results_random["rf_accuracy"], results_random["lr_accuracy"]]
    acc_matched = [results_matched["rf_accuracy"], results_matched["lr_accuracy"]]
    
    theoretical_bound = (1.0 / n_candidates) * 100
    
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    
    rects1 = ax.bar(
        x - width/2, acc_random, width,
        label="Random Decoys (No Matching)",
        color="#EF4444", alpha=0.9, edgecolor="#B91C1C", linewidth=1.2
    )
    rects2 = ax.bar(
        x + width/2, acc_matched, width,
        label="CLIP-Matched Decoys (ObCLIP-I2I)",
        color="#10B981", alpha=0.9, edgecolor="#047857", linewidth=1.2
    )
    
    ax.axhline(
        theoretical_bound,
        color="#1E293B",
        linestyle="--",
        linewidth=1.8,
        label=f"Theoretical Privacy Bound (1/N = {theoretical_bound:.0f}%)"
    )
    
    ax.set_ylabel("Attacker Distinguishing Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Semi-Honest Server Distinguishing Attack on Noisy Latents (N=4, strength=0.6)",
        fontsize=13,
        fontweight="bold",
        pad=14
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 118)
    ax.legend(frameon=True, fontsize=10, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 4 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Attack] ✓ Saved attack comparison figure to: {output_path}")


def main():
    print("=" * 70)
    print("ObCLIP-I2I Phase 3 — Semi-Honest Privacy Attack Simulation")
    print("=" * 70)
    
    from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
    
    # 1. Load VAE and Scheduler
    print("[Attack] Loading shared VAE from:", VAE_MODEL_ID)
    vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID, torch_dtype=DTYPE).to("cpu")
    
    print("[Attack] Loading scheduler from:", CLOUD_MODEL_ID)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        CLOUD_MODEL_ID, subfolder="scheduler", algorithm_type="dpmsolver++", use_karras_sigmas=True
    )
    
    # 2. Gather decoy pool
    decoy_paths = load_all_decoys(DECOY_DIR)
    print(f"[Attack] Available images in pool: {len(decoy_paths)}")
    assert len(decoy_paths) >= 40, "Need at least 40 images for attack simulation."
    
    num_batches = 40
    n_candidates = 4
    
    # ── Condition A: Random Decoys ─────────────────────────────────────
    print("\n" + "-" * 60)
    print(f"Condition A: Extracting features with RANDOM Decoys ({num_batches} batches of N={n_candidates})...")
    print("-" * 60)
    X_rnd, y_rnd, real_idx_rnd = generate_attack_dataset(
        vae, scheduler, decoy_paths, num_batches=num_batches, n_candidates=n_candidates, match_lighting=False, seed=42
    )
    
    print("[Attack] Evaluating Attacker on Random Decoys (5-Fold CV)...")
    res_random = evaluate_attacker(X_rnd, y_rnd, real_idx_rnd, n_candidates=n_candidates)
    print(f"  • Random Forest Accuracy:      {res_random['rf_accuracy']:.1f}%")
    print(f"  • Logistic Regression Accuracy: {res_random['lr_accuracy']:.1f}%")
    
    # ── Condition B: CLIP-Matched Decoys ──────────────────────────────
    print("\n" + "-" * 60)
    print(f"Condition B: Extracting features with CLIP-MATCHED Decoys ({num_batches} batches of N={n_candidates})...")
    print("-" * 60)
    X_mat, y_mat, real_idx_mat = generate_attack_dataset(
        vae, scheduler, decoy_paths, num_batches=num_batches, n_candidates=n_candidates, match_lighting=True, seed=42
    )
    
    print("[Attack] Evaluating Attacker on CLIP-Matched Decoys (5-Fold CV)...")
    res_matched = evaluate_attacker(X_mat, y_mat, real_idx_mat, n_candidates=n_candidates)
    print(f"  • Random Forest Accuracy:      {res_matched['rf_accuracy']:.1f}%")
    print(f"  • Logistic Regression Accuracy: {res_matched['lr_accuracy']:.1f}%")
    
    # ── Summary & Plotting ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PRIVACY ATTACK SUMMARY (Theoretical Bound = 25.0%)")
    print("=" * 70)
    print(f"  Random Decoys:       RF = {res_random['rf_accuracy']:.1f}%, LR = {res_random['lr_accuracy']:.1f}%")
    print(f"  CLIP-Matched Decoys: RF = {res_matched['rf_accuracy']:.1f}%, LR = {res_matched['lr_accuracy']:.1f}%")
    print("=" * 70)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_png = os.path.join(project_root, "eval", "attack_accuracy.png")
    plot_attack_results(res_random, res_matched, n_candidates, output_png)
    print("[Attack] ✓ Privacy Attack evaluation finished successfully.")


if __name__ == "__main__":
    main()
