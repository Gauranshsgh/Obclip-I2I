"""
Streamlit App — app.py
=======================
ObCLIP-I2I Demo UI with two panels:
  Left:  Untrusted Cloud View (batch sent, no real_index, acceleration metrics)
  Right: Private Client View (real image, lighting-matched decoys, final edit)
"""

import streamlit as st
import sys
import os
import time
import math

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TOTAL_STEPS, STRENGTH, K, N,
    CLOUD_GUIDANCE_SCALE, CLIENT_GUIDANCE_SCALE,
    RESOLUTION, DECOY_DIR, CLOUD_ENDPOINT, DEFAULT_NEGATIVE_PROMPT,
)


def main():
    st.set_page_config(
        page_title="ObCLIP-I2I Demo",
        page_icon="🔒",
        layout="wide",
    )

    st.title("🔒 ObCLIP-I2I: Privacy-Preserving Image-to-Image Generation")
    st.caption(
        "Oblivious Cloud-Device Hybrid I2I (extending ObCLIP, NeurIPS 2025). "
        "User reference images are obfuscated inside an N-candidate batch with I2I-aware "
        "acceleration and block skipping on the untrusted cloud."
    )

    # ── Sidebar Controls ───────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Controls")

        uploaded = st.file_uploader(
            "Upload your photo",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )

        prompt = st.text_input(
            "Edit prompt",
            value="portrait wearing a formal black suit in an office",
        )

        use_prompt_wrapper = st.checkbox(
            "🎨 Photorealistic Prompt Wrapper",
            value=True,
            help="Wraps your prompt with RAW photo, natural skin, 85mm DSLR styling.",
        )

        negative_prompt = st.text_area(
            "Negative prompt (optional)",
            value=DEFAULT_NEGATIVE_PROMPT,
            height=70,
        )

        st.divider()
        st.subheader("Hybrid Parameters")

        n_val = st.slider("N (batch size: 1 real + N-1 decoys)", 2, 6, N)

        strength_val = st.slider(
            "Strength (SDEdit noise level)",
            0.3, 0.8, STRENGTH, 0.05,
        )

        k_val = st.slider("k (cloud switch point)", 1, 15, K)

        st.divider()
        st.subheader("⚡ Acceleration & Quality")

        use_accel = st.checkbox("Enable I2I Acceleration", value=True)

        r_val = st.slider(
            "r (Temporal Attn Cache threshold)",
            1, min(k_val, 5), min(2, k_val),
            help="Steps > r cache & reuse cross-attention outputs (refresh every 2 steps).",
        )

        s_val = st.slider(
            "s (Interleaved Block Skip threshold)",
            1, min(k_val, 8), min(5, k_val),
            help="Steps >= s alternate between full UNet and cached down/mid-block skip.",
        )

        st.divider()
        st.subheader("CFG Scales")

        cloud_cfg = st.slider("Cloud CFG Scale", 1.0, 15.0, CLOUD_GUIDANCE_SCALE, 0.5)
        client_cfg = st.slider("Client CFG Scale", 1.0, 15.0, CLIENT_GUIDANCE_SCALE, 0.5)

        st.divider()
        st.subheader("Post-Processing")

        match_lighting = st.checkbox("CLIP-Matched Decoys", value=True)
        use_lab_match = st.checkbox("Lab Color/Lighting Match", value=True)
        use_gfpgan = st.checkbox("GFPGAN Face Restoration", value=False)

        st.divider()
        cloud_url = st.text_input("Cloud API URL", value=CLOUD_ENDPOINT)

        run_button = st.button("🚀 Run ObCLIP-I2I", type="primary", use_container_width=True)

    # ── Main Area ──────────────────────────────────────────────────
    if uploaded is None:
        st.info("👈 Upload a photo in the sidebar to get started.")

        st.subheader("System Architecture")
        st.markdown("""
        ```
        ┌───────────────────────────────────┐         ┌──────────────────────────────────────────────┐
        │        CLIENT (Private)           │         │            CLOUD (Untrusted)                 │
        │                                   │         │                                              │
        │  Real Image + CLIP-Matched Decoys │         │  Receives N noisy latents                    │
        │  → VAE Encode                     │         │  (oblivious to which latent is real, 1/N)    │
        │  → Add Noise (SDEdit t_start)     │  HTTP   │                                              │
        │  → Shuffle & Send ────────────────┼────────►│  → Cloud UNet: first k steps                 │
        │                                   │         │     • Steps <= r: Per-candidate cross-attn   │
        │                                   │         │     • Steps >= s: Interleaved block skip     │
        │                                   │         │     • Steps > r:  Temporal attn2 cache       │
        │  ← Extract real latent at j* ─────┼─────────┤  → Return N intermediate latents             │
        │  → Client UNet: finish steps      │         └──────────────────────────────────────────────┘
        │  → VAE Decode + Post-processing   │
        │  → Final Edited Image             │
        └───────────────────────────────────┘

        Privacy Guarantee: Cloud sees N identical-looking noisy latents with CLIP-matched decoys.
        Guessing probability = 1/N (e.g. 25% for N=4).
        ```
        """)
        return

    real_image = Image.open(uploaded).convert("RGB")

    if not run_button:
        st.image(real_image, caption="Uploaded Image (preview)", width=300)
        st.info("Adjust parameters in the sidebar and click **Run ObCLIP-I2I**.")
        return

    # ── Run Pipeline ───────────────────────────────────────────────
    with st.spinner("Loading models..."):
        from client.client_runner import load_shared_models, run_pipeline

        if "models_dict" not in st.session_state:
            st.session_state.models_dict = load_shared_models()

    with st.spinner(f"Running ObCLIP-I2I pipeline (N={n_val}, k={k_val}, accel={use_accel})..."):
        try:
            result = run_pipeline(
                real_image=real_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                models_dict=st.session_state.models_dict,
                n=n_val,
                strength=strength_val,
                k=k_val,
                cloud_guidance_scale=cloud_cfg,
                client_guidance_scale=client_cfg,
                cloud_url=cloud_url,
                decoy_dir=DECOY_DIR,
                use_accel=use_accel,
                r=r_val,
                s=s_val,
                match_lighting=match_lighting,
                use_prompt_wrapper=use_prompt_wrapper,
                use_lab_match=use_lab_match,
                use_gfpgan=use_gfpgan,
            )
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.exception(e)
            return

    # ── Display Results ────────────────────────────────────────────
    col_cloud, col_client = st.columns(2)

    with col_cloud:
        st.subheader("☁️ Untrusted Cloud View")
        st.caption("What the cloud server sees — NO knowledge of which image is real.")

        c_priv, c_acc = st.columns(2)
        c_priv.metric("🔒 Privacy Bound", f"1/{n_val} = {100/n_val:.1f}%")
        acc_status = f"⚡ ON (r={r_val}, s={s_val})" if use_accel else "OFF (Vanilla)"
        c_acc.metric("Acceleration", acc_status)

        n_cols = min(n_val, 3)
        n_rows = math.ceil(n_val / n_cols)

        st.markdown("**Shuffled batch sent to cloud** (CLIP-matched decoys):")
        for row in range(n_rows):
            cols = st.columns(n_cols)
            for col_idx in range(n_cols):
                img_idx = row * n_cols + col_idx
                if img_idx < n_val:
                    with cols[col_idx]:
                        st.image(
                            result["shuffled_images"][img_idx],
                            caption=f"Candidate #{img_idx + 1}",
                            use_container_width=True,
                        )

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Cloud Steps", f"{result['cloud_steps']}")
        c2.metric("Cloud Latency", f"{result['cloud_latency_s']:.2f}s")

        c3, c4 = st.columns(2)
        c3.metric("Upload Payload", f"{result['upload_payload_kb']:.1f} KB")
        c4.metric("Download Payload", f"{result['download_payload_kb']:.1f} KB")

    with col_client:
        st.subheader("🔐 Private Client View")
        st.caption("Only the client knows secret index j* and receives the final edit.")

        st.image(real_image, caption="Original Photo (private)", width=250)
        st.metric("🔑 Secret Real Index", f"j* = {result['real_index']} (of {n_val})")

        st.divider()
        st.markdown("**Final Edited Image:**")
        st.image(result["final_image"], caption=f'Edit: "{prompt}"', use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric("Client Steps", f"{result['client_steps']}")
        c2.metric("Client Latency", f"{result['client_latency_s']:.2f}s")

        st.metric("Peak VRAM", f"{result['peak_vram_mb']:.0f} MB")

    # ── Summary ────────────────────────────────────────────────────
    st.divider()
    acc_text = "with I2I-aware acceleration" if use_accel else "in vanilla baseline mode"
    st.success(
        f"✅ Execution complete {acc_text}! Cloud executed {result['cloud_steps']} steps on {n_val} images "
        f"in {result['cloud_latency_s']:.2f}s without learning identity. "
        f"Client completed remaining {result['client_steps']} steps locally in {result['client_latency_s']:.2f}s. "
        f"Peak GPU VRAM: {result['peak_vram_mb']:.0f} MB."
    )


if __name__ == "__main__":
    main()
