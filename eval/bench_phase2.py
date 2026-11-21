"""
Phase 2 Benchmark — eval/bench_phase2.py
========================================
Compares Vanilla ObCLIP-I2I vs Accelerated ObCLIP-I2I.
Profiles cloud latency, speedup, client latency, peak VRAM, NaN checks.
Saves visual comparison to compare_phase2.png.
"""

import os
import sys
import time
import subprocess
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TOTAL_STEPS, STRENGTH, K, N,
    CLOUD_GUIDANCE_SCALE, CLIENT_GUIDANCE_SCALE,
    RESOLUTION, DECOY_DIR, CLOUD_API_URL, CLOUD_ENDPOINT,
)
from client.client_runner import load_shared_models, run_pipeline


def ensure_server_running(project_root: str):
    """Ensure cloud API server is alive on port 8000; launch if needed."""
    try:
        resp = requests.get(f"{CLOUD_API_URL}/docs", timeout=2)
        if resp.status_code == 200:
            print("[Bench] ✓ Cloud server is already running.")
            return None
    except (requests.ConnectionError, requests.Timeout):
        pass

    print("[Bench] Starting cloud API server subprocess...")
    venv_python = os.path.join(project_root, "venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        venv_python = sys.executable

    log_path = os.path.join(project_root, "eval", "server_bench.log")
    log_file = open(log_path, "w")

    server_proc = subprocess.Popen(
        [venv_python, "-m", "uvicorn", "server.cloud_api:app",
         "--host", "127.0.0.1", "--port", "8000"],
        cwd=project_root,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    start = time.time()
    max_wait = 300
    while time.time() - start < max_wait:
        if server_proc.poll() is not None:
            log_file.close()
            print("[Bench] ✗ Server process exited unexpectedly. Check eval/server_bench.log")
            sys.exit(1)
        try:
            resp = requests.get(f"{CLOUD_API_URL}/docs", timeout=2)
            if resp.status_code == 200:
                print(f"[Bench] ✓ Cloud server is ready ({int(time.time() - start)}s).")
                return server_proc
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(3)

    print("[Bench] ✗ Cloud server failed to start within timeout.")
    server_proc.terminate()
    log_file.close()
    sys.exit(1)


def create_comparison_grid(
    original_img: Image.Image,
    vanilla_img: Image.Image,
    accel_img: Image.Image,
    t_vanilla: float,
    t_accel: float,
    speedup: float,
    output_path: str,
    bench_params: dict,
):
    """Create annotated side-by-side comparison image."""
    w, h = RESOLUTION, RESOLUTION
    header_h = 70
    footer_h = 50
    canvas_w = w * 3 + 40
    canvas_h = h + header_h + footer_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (24, 26, 32))
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("arial.ttf", 26)
        font_sub = ImageFont.truetype("arial.ttf", 18)
        font_box = ImageFont.truetype("arial.ttf", 20)
    except (IOError, OSError):
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_box = ImageFont.load_default()

    title = "ObCLIP-I2I: Vanilla vs Accelerated Cloud Denoising"
    draw.text((20, 15), title, fill=(255, 255, 255), font=font_title)

    p = bench_params
    sub = (f"N={p['n']}, k={p['k']}, strength={p['strength']}, r={p['r']}, s={p['s']} | "
           f"Speedup: {speedup:.2f}x ({(1 - t_accel/t_vanilla)*100:.1f}% reduction)")
    draw.text((20, 45), sub, fill=(100, 220, 140), font=font_sub)

    canvas.paste(original_img.resize((w, h)), (10, header_h))
    canvas.paste(vanilla_img.resize((w, h)), (20 + w, header_h))
    canvas.paste(accel_img.resize((w, h)), (30 + w * 2, header_h))

    draw.text((10, header_h + h + 10), "Original Input (Private)", fill=(200, 200, 200), font=font_box)
    draw.text((20 + w, header_h + h + 10), f"Vanilla ({t_vanilla:.2f}s cloud)", fill=(255, 180, 100), font=font_box)
    draw.text((30 + w * 2, header_h + h + 10), f"Accelerated ({t_accel:.2f}s, {speedup:.2f}x)", fill=(100, 255, 150), font=font_box)

    canvas.save(output_path)
    print(f"[Bench] ✓ Saved visual comparison to: {output_path}")


def main():
    print("=" * 70)
    print("ObCLIP-I2I — Acceleration Benchmark (Final Polish)")
    print("=" * 70)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    test_path = os.path.join(project_root, "test_input.jpg")
    if os.path.exists(test_path):
        real_image = Image.open(test_path).convert("RGB")
    else:
        arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        real_image = Image.fromarray(arr)
        real_image.save(test_path)

    server_proc = ensure_server_running(project_root)

    # Benchmark parameters
    bp = {
        "n": 4,
        "strength": 0.55,
        "k": 10,
        "r": 2,
        "s": 5,
    }

    try:
        print("\n[Bench] Loading shared client models...")
        models_dict = load_shared_models()

        prompt = "wearing glasses and a formal suit"

        print(f"\n[Bench] Running Benchmark (N={bp['n']}, strength={bp['strength']}, k={bp['k']}):")

        # ── Run 1: Vanilla ─────────────────────────────────────────────
        print("\n" + "-" * 60)
        print(">>> Configuration 1: Vanilla ObCLIP-I2I (use_accel=False)")
        print("-" * 60)
        res_vanilla = run_pipeline(
            real_image=real_image,
            prompt=prompt,
            models_dict=models_dict,
            n=bp["n"],
            strength=bp["strength"],
            k=bp["k"],
            use_accel=False,
            match_lighting=True,
        )

        # ── Run 2: Accelerated ─────────────────────────────────────────
        print("\n" + "-" * 60)
        print(f">>> Configuration 2: Accelerated ObCLIP-I2I (r={bp['r']}, s={bp['s']})")
        print("-" * 60)
        res_accel = run_pipeline(
            real_image=real_image,
            prompt=prompt,
            models_dict=models_dict,
            n=bp["n"],
            strength=bp["strength"],
            k=bp["k"],
            use_accel=True,
            r=bp["r"],
            s=bp["s"],
            match_lighting=True,
        )

        # ── NaN checks ────────────────────────────────────────────────
        vanilla_arr = np.array(res_vanilla["final_image"])
        accel_arr = np.array(res_accel["final_image"])
        assert not np.isnan(vanilla_arr).any(), "Vanilla final image has NaN values!"
        assert not np.isnan(accel_arr).any(), "Accelerated final image has NaN values!"
        assert accel_arr.mean() > 10.0, f"Accelerated image is too dark (mean={accel_arr.mean():.2f})!"

        # ── Results ────────────────────────────────────────────────────
        t_v = res_vanilla["cloud_latency_s"]
        t_a = res_accel["cloud_latency_s"]
        speedup = t_v / max(t_a, 0.001)

        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)
        print(f"  • Vanilla Cloud Latency:     {t_v:.2f} s")
        print(f"  • Accelerated Cloud Latency: {t_a:.2f} s")
        print(f"  • Cloud Speedup Factor:      {speedup:.2f}x")
        print(f"  • Cloud Latency Reduction:   {(1.0 - t_a / t_v) * 100:.1f}%")
        print(f"  • Client Latency (Vanilla):  {res_vanilla['client_latency_s']:.2f} s")
        print(f"  • Client Latency (Accel):    {res_accel['client_latency_s']:.2f} s")
        print(f"  • Peak VRAM (Vanilla):       {res_vanilla['peak_vram_mb']:.1f} MB")
        print(f"  • Peak VRAM (Accel):         {res_accel['peak_vram_mb']:.1f} MB")
        assert res_accel["peak_vram_mb"] < 3072, f"Peak VRAM {res_accel['peak_vram_mb']} MB exceeds 3.0 GB!"
        print("  • ✓ 0 NaNs, VRAM < 3.0 GB")
        print("=" * 70)

        compare_path = os.path.join(project_root, "compare_phase2.png")
        create_comparison_grid(
            original_img=real_image,
            vanilla_img=res_vanilla["final_image"],
            accel_img=res_accel["final_image"],
            t_vanilla=t_v,
            t_accel=t_a,
            speedup=speedup,
            output_path=compare_path,
            bench_params=bp,
        )

        print("\n[Bench] ✓ Benchmark completed successfully!")

    finally:
        if server_proc is not None:
            print("[Bench] Stopping spawned cloud server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()
            print("[Bench] ✓ Cloud server stopped.")


if __name__ == "__main__":
    main()
