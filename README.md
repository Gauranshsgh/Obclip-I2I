# ObCLIP-I2I: Privacy-Preserving Cloud-Device Collaborative Image-to-Image Generation

ObCLIP-I2I is a framework for privacy-preserving Image-to-Image (I2I) synthesis. It enables secure collaborative inference between an edge device and a cloud server:
- User images are obfuscated inside an N-candidate batch using semantic decoy matching.
- The cloud server runs the initial denoising steps on a photorealistic foundation model without learning which candidate is real (privacy bound: 1/N).
- The client extracts the target latent and finishes denoising locally with a lightweight model, followed by color matching and enhancement.

---

## Architecture Overview

```
+------------------------------------+           +-------------------------------------+
|          CLIENT (Private)          |           |           CLOUD (Untrusted)         |
|                                    |           |                                     |
| Real Image + Matched Decoys (N)    |   HTTP    | Receives N noisy latents (1/N bound)|
| -> VAE Encode + Noise Injection    +---------->+ -> Cloud UNet (Realistic Vision V5) |
|                                    |           |    - Temporal Attention Caching     |
|                                    |           |    - Interleaved Block Skipping     |
| <- Extract real latent at index j* <-----------+ <- Returns N intermediate latents   |
| -> Client UNet (small-sd finish)   |           +-------------------------------------+
| -> VAE Decode + Color Match        |
+------------------------------------+
```

---

## Features

- **Decoy Matching**: Uses CLIP image embeddings to select top matching decoy candidates from a pool, matching lighting and composition.
- **Cloud Acceleration**:
  - Memory-efficient attention (SDPA) to keep peak VRAM under 2 GB.
  - Temporal cross-attention caching.
  - Interleaved down/mid-block skipping.
- **Split Guidance**: Dual CFG scales for cloud and edge stages.
- **Edge Post-Processing**: Lab color transfer and optional facial detail enhancement.

---

## Project Structure

```
obclip-i2i-sim/
├── app.py                  # Streamlit Web UI
├── config.py               # Hyperparameters and model configurations
├── requirements.txt        # Python dependencies
├── test_smoke.py           # Automated smoke test suite
├── test_clip_retrieval.py  # Decoy retrieval test
├── client/
│   ├── decoy_manager.py    # CLIP decoy selection & cache
│   ├── latent_prep.py      # VAE encode & noise injection
│   ├── postprocess.py      # Lab color transfer & enhancement
│   └── client_runner.py    # Client pipeline execution
├── server/
│   ├── cloud_api.py        # FastAPI cloud server
│   ├── accel_processors.py # Attention processors
│   └── block_skip.py       # Block skipping controller
├── decoys/                 # Decoy image pool and CLIP embedding cache
├── docs/                   # System documentation
└── eval/                   # Benchmarks and evaluation scripts
```

---

## Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Gauranshsgh/Obclip-I2I.git
cd Obclip-I2I

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Smoke Test

```bash
python test_smoke.py
```

### 3. Start the Application

Terminal 1 (Cloud Server):
```bash
python -m uvicorn server.cloud_api:app --host 0.0.0.0 --port 8000
```

Terminal 2 (Client UI):
```bash
python -m streamlit run app.py
```

---

## Evaluation Scripts

- `python eval/bench_phase2.py`: Profile latency and acceleration speedup.
- `python eval/privacy_attack.py`: Run semi-honest distinguishing attack evaluation.
- `python eval/metrics_bench.py`: Run parameter sweeps and metric evaluation.

---

## License

MIT License
