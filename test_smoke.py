"""
Smoke Test — test_smoke.py
===========================
Headless end-to-end test for ObCLIP-I2I Phase 1.
Starts the cloud API server, runs the client pipeline, and verifies
tensor shapes, output validity, and VRAM usage.
"""

import subprocess
import sys
import os
import time
import signal

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TOTAL_STEPS, STRENGTH, K, N, GUIDANCE_SCALE, RESOLUTION


def main():
    print("=" * 70)
    print("ObCLIP-I2I Phase 1 — Smoke Test")
    print("=" * 70)

    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # ── Load or generate test image ────────────────────────────────
    test_path = os.path.join(project_root, "test_input.jpg")
    if os.path.exists(test_path):
        real_image = Image.open(test_path).convert("RGB")
        print(f"[Test] Loaded test image: {test_path} ({real_image.size})")
    else:
        print("[Test] No test_input.jpg found, generating dummy image...")
        arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        real_image = Image.fromarray(arr)
        real_image.save(test_path)
        print(f"[Test] Saved dummy test image to {test_path}")

    # ── Start cloud API server in background ───────────────────────
    print("\n[Test] Starting cloud API server on port 8000...")
    
    venv_python = os.path.join(project_root, "venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        venv_python = sys.executable
    
    log_path = os.path.join(project_root, "server_smoke.log")
    log_file = open(log_path, "w")

    server_proc = subprocess.Popen(
        [venv_python, "-m", "uvicorn", "server.cloud_api:app",
         "--host", "127.0.0.1", "--port", "8000"],
        cwd=project_root,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for server to be ready
    import requests
    max_wait = 600  # seconds (models may need to download on first run)
    start = time.time()
    server_ready = False
    print("[Test] Waiting for cloud server to load models...")
    
    while time.time() - start < max_wait:
        if server_proc.poll() is not None:
            print("[Test] ✗ Cloud server process terminated unexpectedly!")
            out = server_proc.communicate()[0]
            print(out)
            sys.exit(1)
        try:
            resp = requests.get("http://127.0.0.1:8000/docs", timeout=2)
            if resp.status_code == 200:
                server_ready = True
                break
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(3)
        elapsed = int(time.time() - start)
        if elapsed % 15 == 0:
            print(f"[Test] Still waiting... ({elapsed}s elapsed)")

    if not server_ready:
        print("[Test] ✗ Cloud server failed to start within timeout!")
        server_proc.terminate()
        out = server_proc.communicate()[0]
        print(out)
        sys.exit(1)

    print(f"[Test] ✓ Cloud server ready ({int(time.time() - start)}s)")

    try:
        # ── Run the pipeline ───────────────────────────────────────
        import torch
        from client.client_runner import load_shared_models, run_pipeline

        print("\n[Test] Loading client models...")
        models_dict = load_shared_models()

        prompt = "wearing glasses and a formal suit"
        print(f"\n[Test] Running pipeline: prompt='{prompt}', N={N}, "
              f"strength={STRENGTH}, k={K}")

        result = run_pipeline(
            real_image=real_image,
            prompt=prompt,
            models_dict=models_dict,
            n=N,
            strength=STRENGTH,
            k=K,
        )

        # ── Assertions ─────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("VERIFICATION RESULTS")
        print("=" * 70)

        passed = 0
        total = 0

        # Check 1: shuffled_images count
        total += 1
        assert len(result["shuffled_images"]) == N, \
            f"Expected {N} shuffled images, got {len(result['shuffled_images'])}"
        print(f"  ✓ Shuffled images count: {len(result['shuffled_images'])} == N={N}")
        passed += 1

        # Check 2: real_index in valid range
        total += 1
        assert 0 <= result["real_index"] < N, \
            f"real_index {result['real_index']} out of range [0, {N})"
        print(f"  ✓ Real index: {result['real_index']} (valid, in [0, {N}))")
        passed += 1

        # Check 3: cloud steps
        total += 1
        assert result["cloud_steps"] == K, \
            f"Expected {K} cloud steps, got {result['cloud_steps']}"
        print(f"  ✓ Cloud steps: {result['cloud_steps']} == k={K}")
        passed += 1

        # Check 4: client steps
        total += 1
        n_active = int(TOTAL_STEPS * STRENGTH)
        expected_client_steps = n_active - K
        assert result["client_steps"] == expected_client_steps, \
            f"Expected {expected_client_steps} client steps, got {result['client_steps']}"
        print(f"  ✓ Client steps: {result['client_steps']} == {expected_client_steps}")
        passed += 1

        # Check 5: final image is valid
        total += 1
        final = result["final_image"]
        assert isinstance(final, Image.Image), "Final output is not a PIL Image"
        assert final.size == (RESOLUTION, RESOLUTION), \
            f"Expected {RESOLUTION}x{RESOLUTION}, got {final.size}"
        assert final.mode == "RGB", f"Expected RGB, got {final.mode}"
        print(f"  ✓ Final image: {final.size} {final.mode}")
        passed += 1

        # Check 6: VRAM usage
        total += 1
        peak_vram_gb = result["peak_vram_mb"] / 1024
        assert peak_vram_gb < 7.5, \
            f"Peak VRAM {peak_vram_gb:.2f} GB exceeds 7.5 GB limit"
        print(f"  ✓ Peak VRAM: {result['peak_vram_mb']:.0f} MB ({peak_vram_gb:.2f} GB < 7.5 GB)")
        passed += 1

        # Check 7: latencies are positive
        total += 1
        assert result["cloud_latency_s"] > 0, "Cloud latency should be positive"
        assert result["client_latency_s"] > 0, "Client latency should be positive"
        print(f"  ✓ Cloud latency: {result['cloud_latency_s']:.2f}s")
        print(f"  ✓ Client latency: {result['client_latency_s']:.2f}s")
        passed += 1

        # Check 8: payload sizes
        total += 1
        assert result["upload_payload_kb"] > 0, "Upload payload should be positive"
        assert result["download_payload_kb"] > 0, "Download payload should be positive"
        print(f"  ✓ Upload payload: {result['upload_payload_kb']:.1f} KB")
        print(f"  ✓ Download payload: {result['download_payload_kb']:.1f} KB")
        passed += 1

        # Save output
        output_path = os.path.join(project_root, "output_smoke.png")
        result["final_image"].save(output_path)
        print(f"\n  ✓ Output saved to: {output_path}")

        print(f"\n{'=' * 70}")
        print(f"SMOKE TEST: {passed}/{total} checks PASSED ✅")
        print(f"{'=' * 70}")

        # Print launch commands
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                     LAUNCH COMMANDS                                ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Terminal 1 — Cloud API Server:                                    ║
║  cd {project_root}
║  .\\venv\\Scripts\\Activate.ps1                                      ║
║  uvicorn server.cloud_api:app --host 0.0.0.0 --port 8000           ║
║                                                                    ║
║  Terminal 2 — Streamlit App:                                       ║
║  cd {project_root}
║  .\\venv\\Scripts\\Activate.ps1                                      ║
║  streamlit run app.py                                              ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    except Exception as e:
        print(f"\n[Test] ✗ SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup: stop the server
        print("[Test] Stopping cloud server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        try:
            log_file.close()
        except Exception:
            pass
        print("[Test] ✓ Server stopped.")


if __name__ == "__main__":
    main()
