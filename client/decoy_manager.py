"""
Decoy Manager — client/decoy_manager.py
========================================
Manages the decoy pool for ObCLIP-I2I.
Features:
  - CLIP Image-Embedding Retrieval (openai/clip-vit-base-patch32)
  - Disk caching of decoy embeddings in decoys/clip_cache.npz (FP16, normalized)
  - Duplicate/near-duplicate exclusion (similarity > 0.98)
  - Low-VRAM footprint: CLIP model is loaded to CUDA in FP16 only during encoding
    and immediately offloaded to CPU with torch.cuda.empty_cache().
  - Fast sub-second retrieval on cached runs.
"""

import os
import random
import time
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DECOY_DIR, RESOLUTION, DTYPE

# Global in-memory cache for fast repeated queries
_IN_MEMORY_CACHE: Dict[str, Any] = {}
_CLIP_MODEL = None
_CLIP_PROCESSOR = None

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_clip_components():
    """Lazily load CLIP model and processor on CPU to preserve VRAM."""
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is None or _CLIP_PROCESSOR is None:
        from transformers import CLIPModel, CLIPProcessor
        print("[DecoyManager] Loading openai/clip-vit-base-patch32...")
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _CLIP_MODEL = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", torch_dtype=DTYPE
        ).to("cpu")
        _CLIP_MODEL.eval()
    return _CLIP_MODEL, _CLIP_PROCESSOR


def load_all_decoys(decoy_dir: str = DECOY_DIR) -> List[str]:
    """Return all valid image paths in decoy_dir sorted alphabetically."""
    candidates = []
    if os.path.isdir(decoy_dir):
        for fname in sorted(os.listdir(decoy_dir)):
            if fname.lower() == "clip_cache.npz":
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTS:
                candidates.append(os.path.join(decoy_dir, fname))
    return candidates


def center_crop_and_resize(img: Image.Image, size: int = RESOLUTION) -> Image.Image:
    """Center-crop to square, then resize to size×size RGB."""
    img = img.convert("RGB")
    w, h = img.size
    crop_size = min(w, h)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    img = img.crop((left, top, left + crop_size, top + crop_size))
    img = img.resize((size, size), Image.LANCZOS)
    return img


def _generate_placeholder(index: int, size: int = RESOLUTION) -> Image.Image:
    """Generate a synthetic placeholder decoy image (colored rectangle with label)."""
    colors = [
        (180, 120, 100), (100, 180, 120), (120, 100, 180),
        (180, 180, 100), (180, 100, 180), (100, 180, 180),
        (200, 150, 100), (100, 200, 150), (150, 100, 200),
    ]
    color = colors[index % len(colors)]
    img = Image.new("RGB", (size, size), color)
    draw = ImageDraw.Draw(img)
    label = f"Decoy #{index + 1}"
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except (IOError, OSError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size - tw) // 2, (size - th) // 2), label, fill=(255, 255, 255), font=font)
    return img


def compute_hsv_features(img: Image.Image) -> np.ndarray:
    """Legacy HSV histogram & luminance feature extractor."""
    hsv = img.convert("HSV")
    arr = np.array(hsv, dtype=np.float32)
    h = arr[:, :, 0]
    s = arr[:, :, 1]
    v = arr[:, :, 2]

    h_hist, _ = np.histogram(h, bins=16, range=(0, 256), density=True)
    s_hist, _ = np.histogram(s, bins=8, range=(0, 256), density=True)
    v_hist, _ = np.histogram(v, bins=8, range=(0, 256), density=True)

    mean_v = float(np.mean(v) / 255.0)
    mean_s = float(np.mean(s) / 255.0)
    return np.concatenate([h_hist, s_hist, v_hist, [mean_v, mean_s]])


# ── CLIP Embedding Cache ─────────────────────────────────────────────

def get_or_build_clip_cache(
    decoy_dir: str = DECOY_DIR,
    batch_size: int = 64,
) -> Tuple[List[str], np.ndarray]:
    """
    Load or build disk cache for all decoy CLIP image embeddings.
    
    Returns:
        filenames: List of image basenames in decoy_dir.
        embeddings: L2-normalized float16 array of shape [num_images, 512].
    """
    global _IN_MEMORY_CACHE

    cache_file = os.path.join(decoy_dir, "clip_cache.npz")
    current_paths = load_all_decoys(decoy_dir)
    current_filenames = [os.path.basename(p) for p in current_paths]

    # 1. Check in-memory cache first
    if _IN_MEMORY_CACHE.get("decoy_dir") == decoy_dir:
        if _IN_MEMORY_CACHE.get("filenames") == current_filenames:
            return _IN_MEMORY_CACHE["filenames"], _IN_MEMORY_CACHE["embeddings"]

    # 2. Check disk cache
    if os.path.exists(cache_file):
        try:
            data = np.load(cache_file, allow_pickle=True)
            cached_filenames = list(data["filenames"])
            cached_embeddings = data["embeddings"]
            if cached_filenames == current_filenames and len(cached_embeddings) == len(current_filenames):
                print(f"[DecoyManager] ✓ Loaded CLIP cache from {cache_file} ({len(cached_filenames)} images)")
                _IN_MEMORY_CACHE = {
                    "decoy_dir": decoy_dir,
                    "filenames": cached_filenames,
                    "embeddings": cached_embeddings,
                }
                return cached_filenames, cached_embeddings
            else:
                print("[DecoyManager] Decoy directory contents changed; rebuilding CLIP cache...")
        except Exception as e:
            print(f"[DecoyManager] Warning: failed to load existing cache: {e}. Rebuilding...")

    # 3. Build cache if missing or invalid
    if not current_paths:
        print("[DecoyManager] No images found in decoy directory.")
        return [], np.empty((0, 512), dtype=np.float16)

    print(f"[DecoyManager] Precomputing CLIP embeddings for {len(current_paths)} decoys in batches of {batch_size}...")
    model, processor = get_clip_components()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    all_embeddings = []
    t_start = time.time()

    with torch.no_grad():
        for i in range(0, len(current_paths), batch_size):
            batch_paths = current_paths[i : i + batch_size]
            batch_imgs = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    batch_imgs.append(img)
                except Exception as e:
                    print(f"[DecoyManager] Error opening {p}: {e}")
                    batch_imgs.append(Image.new("RGB", (224, 224), (128, 128, 128)))

            inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
            feats = model.get_image_features(**inputs)
            if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            elif hasattr(feats, "image_embeds") and feats.image_embeds is not None:
                feats = feats.image_embeds
            # L2 normalize
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeddings.append(feats.cpu().to(dtype=DTYPE).numpy())

            if (i + batch_size) % (batch_size * 4) == 0 or (i + len(batch_paths)) == len(current_paths):
                print(f"[DecoyManager] Processed {min(i + batch_size, len(current_paths))}/{len(current_paths)} decoys...")

    # Offload model to CPU and clear CUDA cache immediately
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    embeddings_np = np.concatenate(all_embeddings, axis=0).astype(np.float16)
    np.savez_compressed(cache_file, filenames=np.array(current_filenames), embeddings=embeddings_np)
    
    elapsed = time.time() - t_start
    print(f"[DecoyManager] ✓ Saved CLIP cache to {cache_file} in {elapsed:.2f}s ({len(current_filenames)} images)")

    _IN_MEMORY_CACHE = {
        "decoy_dir": decoy_dir,
        "filenames": current_filenames,
        "embeddings": embeddings_np,
    }
    return current_filenames, embeddings_np


# ── CLIP-Based Decoy Retrieval ───────────────────────────────────────

@torch.no_grad()
def retrieve_matched_decoys_with_scores(
    real_image: Image.Image,
    decoy_dir: str = DECOY_DIR,
    n_decoys: int = 3,
    resolution: int = RESOLUTION,
    duplicate_threshold: float = 0.98,
) -> Tuple[List[Image.Image], List[str], List[float]]:
    """
    Retrieve top n_decoys using CLIP image embedding cosine similarity.
    Excludes exact or near-duplicates with cosine similarity > duplicate_threshold.
    
    Returns:
        selected_images: List of PIL Images (center-cropped and resized to resolution).
        matched_filenames: List of filenames for the selected decoys.
        similarity_scores: List of cosine similarity scores (float).
    """
    cached_filenames, cached_embeddings = get_or_build_clip_cache(decoy_dir)

    if len(cached_filenames) == 0:
        placeholders = [_generate_placeholder(i, resolution) for i in range(n_decoys)]
        return placeholders, [f"placeholder_{i}.png" for i in range(n_decoys)], [0.0] * n_decoys

    # 1. Encode user's real image using CLIP
    model, processor = get_clip_components()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = processor(images=real_image.convert("RGB"), return_tensors="pt").to(device)
    real_feats = model.get_image_features(**inputs)
    if hasattr(real_feats, "pooler_output") and real_feats.pooler_output is not None:
        real_feats = real_feats.pooler_output
    elif hasattr(real_feats, "image_embeds") and real_feats.image_embeds is not None:
        real_feats = real_feats.image_embeds
    real_feats = real_feats / real_feats.norm(dim=-1, keepdim=True)
    real_vec = real_feats.cpu().float().numpy()[0]  # [512]

    # Offload model to CPU and clear CUDA cache immediately
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Compute cosine similarities (dot product since both are L2 normalized)
    cached_matrix = cached_embeddings.astype(np.float32)  # [M, 512]
    sims = np.dot(cached_matrix, real_vec)  # [M]

    # 3. Filter out near/exact duplicates (e.g. if real image is in the pool)
    valid_mask = sims <= duplicate_threshold
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        # Fallback if all images somehow exceeded threshold
        valid_indices = np.arange(len(cached_filenames))

    # 4. Sort by highest similarity
    sorted_order = np.argsort(sims[valid_indices])[::-1]
    top_indices = valid_indices[sorted_order[:n_decoys]]

    selected_images = []
    matched_filenames = []
    similarity_scores = []

    for idx in top_indices:
        fname = cached_filenames[idx]
        fpath = os.path.join(decoy_dir, fname)
        score = float(sims[idx])
        matched_filenames.append(fname)
        similarity_scores.append(score)
        try:
            img = Image.open(fpath)
            selected_images.append(center_crop_and_resize(img, resolution))
        except Exception as e:
            print(f"[DecoyManager] Error opening matched decoy {fpath}: {e}")

    # Fill with placeholders if pool had fewer than n_decoys
    while len(selected_images) < n_decoys:
        idx = len(selected_images)
        selected_images.append(_generate_placeholder(idx, resolution))
        matched_filenames.append(f"placeholder_{idx}.png")
        similarity_scores.append(0.0)

    return selected_images, matched_filenames, similarity_scores


def retrieve_matched_decoys(
    real_image: Image.Image,
    decoy_dir: str = DECOY_DIR,
    n_decoys: int = 3,
    resolution: int = RESOLUTION,
) -> List[Image.Image]:
    """
    Standard retrieval interface returning only the list of selected PIL images.
    Preserves exact backward compatibility.
    """
    images, _, _ = retrieve_matched_decoys_with_scores(
        real_image=real_image,
        decoy_dir=decoy_dir,
        n_decoys=n_decoys,
        resolution=resolution,
    )
    return images


def load_decoys(decoy_dir: str = DECOY_DIR, n_decoys: int = 3, random_sample: bool = True) -> List[Image.Image]:
    """Random/sequential fallback decoy loader."""
    paths = load_all_decoys(decoy_dir)
    if random_sample and len(paths) >= n_decoys:
        selected_paths = random.sample(paths, n_decoys)
    else:
        selected_paths = paths[:n_decoys] if len(paths) >= n_decoys else paths

    decoys = []
    for p in selected_paths:
        try:
            img = Image.open(p)
            decoys.append(center_crop_and_resize(img))
        except Exception:
            pass

    while len(decoys) < n_decoys:
        decoys.append(_generate_placeholder(len(decoys)))

    return decoys


def prepare_batch(
    real_image: Image.Image,
    N: int = 4,
    decoy_dir: str = DECOY_DIR,
    resolution: int = RESOLUTION,
    match_lighting: bool = True,
) -> Tuple[List[Image.Image], int]:
    """
    Prepare a shuffled batch of N images (1 real + N-1 decoys).
    If match_lighting=True, uses CLIP image-embedding retrieval.
    """
    real_processed = center_crop_and_resize(real_image, resolution)

    if match_lighting:
        decoys = retrieve_matched_decoys(
            real_processed, decoy_dir=decoy_dir, n_decoys=N - 1, resolution=resolution
        )
    else:
        decoys = load_decoys(decoy_dir, n_decoys=N - 1)

    batch = [real_processed] + decoys

    indices = list(range(N))
    random.shuffle(indices)

    shuffled_batch = [batch[i] for i in indices]
    real_index = indices.index(0)

    return shuffled_batch, real_index
