"""
Test CLIP Retrieval & Caching — test_clip_retrieval.py
======================================================
1. Runs retrieve_matched_decoys_with_scores on test_input.jpg, building decoys/clip_cache.npz.
2. Measures latency of second call to verify sub-second (<0.5s) retrieval.
3. Prints top 3 matched decoy filenames and cosine similarity scores.
"""

import os
import sys
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DECOY_DIR, RESOLUTION
from client.decoy_manager import (
    retrieve_matched_decoys_with_scores,
    get_or_build_clip_cache,
    load_all_decoys
)


def main():
    print("=" * 70)
    print("ObCLIP-I2I — CLIP Image Retrieval & Cache Verification")
    print("=" * 70)

    project_root = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(project_root, "test_input.jpg")

    if not os.path.exists(test_path):
        print(f"[Test] Generating dummy test image at {test_path}...")
        img_arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        Image.fromarray(img_arr).save(test_path)

    test_image = Image.open(test_path).convert("RGB")
    print(f"[Test] Loaded test image: {test_path} ({test_image.size})")

    cache_path = os.path.join(DECOY_DIR, "clip_cache.npz")
    if os.path.exists(cache_path):
        print(f"[Test] Existing cache found at {cache_path}, removing to test clean build...")
        os.remove(cache_path)

    # ── Call 1: Build Cache and Retrieve ──────────────────────────────
    print("\n[Test] Call 1: Building disk cache and retrieving top-3 matched decoys...")
    t0 = time.time()
    imgs1, names1, scores1 = retrieve_matched_decoys_with_scores(
        test_image, decoy_dir=DECOY_DIR, n_decoys=3, resolution=RESOLUTION
    )
    t_call1 = time.time() - t0
    print(f"[Test] Call 1 completed in {t_call1:.2f}s")

    assert os.path.exists(cache_path), f"Expected cache at {cache_path} but not found!"
    cache_data = np.load(cache_path, allow_pickle=True)
    n_cached = len(cache_data["filenames"])
    emb_shape = cache_data["embeddings"].shape
    print(f"[Test] ✓ Cache verified: {n_cached} filenames, embeddings shape: {emb_shape}")

    # ── Call 2: Benchmark Fast Retrieval (<0.5s) ──────────────────────
    print("\n[Test] Call 2: Testing fast retrieval from cache (<0.5s target)...")
    t1 = time.time()
    imgs2, names2, scores2 = retrieve_matched_decoys_with_scores(
        test_image, decoy_dir=DECOY_DIR, n_decoys=3, resolution=RESOLUTION
    )
    t_call2 = time.time() - t1
    print(f"[Test] Call 2 completed in {t_call2:.3f}s")

    # ── Print Results ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TOP 3 CLIP-MATCHED DECOYS")
    print("=" * 70)
    for rank, (fname, score) in enumerate(zip(names2, scores2), start=1):
        print(f"  {rank}. Filename: {fname:<25} Cosine Similarity: {score:.4f}")
    print("=" * 70)

    # ── Assertions ───────────────────────────────────────────────────
    assert len(imgs2) == 3, f"Expected 3 images, got {len(imgs2)}"
    assert len(names2) == 3, f"Expected 3 names, got {len(names2)}"
    assert all(s <= 0.98 for s in scores2), f"Expected no duplicate above 0.98, got {scores2}"
    assert t_call2 < 0.5, f"Call 2 took {t_call2:.3f}s (exceeded 0.5s target)!"

    print(f"\n[Test] ✓ ALL CHECKS PASSED: Second retrieval latency: {t_call2:.3f}s (< 0.5s)")


if __name__ == "__main__":
    main()
