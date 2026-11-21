"""
Post-Processing — client/postprocess.py
=======================================
Optional post-processing for the final decoded image:
  1. Lab Color/Lighting Match: transfer color statistics from reference using OpenCV.
  2. GFPGAN Face Restoration: enhance facial details.
"""

import numpy as np
from PIL import Image
import cv2
import torch


def lab_color_match(
    edited: Image.Image,
    reference: Image.Image,
    strength: float = 0.35,
) -> Image.Image:
    """
    Match the color/lighting of `edited` to `reference` in Lab color space via OpenCV.
    Blends channel means and stds with the given strength.
    """
    # Resize reference to match edited size if needed
    ref_resized = reference.resize(edited.size).convert("RGB")
    
    edited_np = np.array(edited.convert("RGB"), dtype=np.float32)
    ref_np = np.array(ref_resized, dtype=np.float32)

    # Convert RGB to Lab (float32, 0..255 range in cv2)
    edited_lab = cv2.cvtColor(edited_np / 255.0, cv2.COLOR_RGB2LAB)
    ref_lab = cv2.cvtColor(ref_np / 255.0, cv2.COLOR_RGB2LAB)

    # Match channel-wise mean and std
    for c in range(3):
        e_mean, e_std = float(edited_lab[..., c].mean()), float(edited_lab[..., c].std()) + 1e-6
        r_mean, r_std = float(ref_lab[..., c].mean()), float(ref_lab[..., c].std()) + 1e-6

        target_mean = e_mean + strength * (r_mean - e_mean)
        target_std = e_std + strength * (r_std - e_std)

        edited_lab[..., c] = (edited_lab[..., c] - e_mean) * (target_std / e_std) + target_mean

    # Convert back to RGB
    matched_rgb = cv2.cvtColor(edited_lab, cv2.COLOR_LAB2RGB)
    matched_rgb = np.clip(matched_rgb * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(matched_rgb)


def gfpgan_enhance(
    img: Image.Image,
    weight: float = 0.65,
) -> Image.Image:
    """
    Enhance facial details using GFPGANv1.4.
    Downloads weights on first use. Offloads to CPU after processing.
    """
    try:
        from gfpgan import GFPGANer
    except ImportError:
        print("[Postprocess] GFPGAN not installed — skipping face restoration.")
        return img

    import os

    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "weights", "GFPGANv1.4.pth"
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    try:
        restorer = GFPGANer(
            model_path=model_path,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )

        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        _, _, restored_img = restorer.enhance(
            img_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=weight,
        )

        del restorer
        torch.cuda.empty_cache()

        if restored_img is None:
            return img

        return Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"[Postprocess] GFPGAN enhancement failed: {e}. Returning original.")
        return img
