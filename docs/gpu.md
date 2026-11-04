# GPU setup

## Recommended host

- RTX 3090 24 GB
- CUDA 12.5 driver support
- 64 GB disk
- Ubuntu 22.04 or the RAPIDS base image

## Vast.ai template

Create a minimal template instead of using the portal/Jupyter preset:

| Field | Value |
|---|---|
| Image | `rapidsai/base:24.10-cuda12.5-py3.11` |
| Launch mode | Interactive shell server, SSH |
| Direct SSH | checked |
| On-start script | empty |
| Docker options | empty |
| Ports | none |
| Environment variables | none |
| Disk | 64 GB |
| Extra filters | `cuda_max_good>=12.5 compute_cap>=800` |

The generated CLI command should look like:

```bash
vastai create instance <OFFER_ID> \
  --image rapidsai/base:24.10-cuda12.5-py3.11 \
  --disk 64 \
  --ssh --direct
```

Do not pass portal ports, `OPEN_BUTTON_*`, or `entrypoint.sh`. Those belong to the generic vast.ai portal template and can stop the RAPIDS image.

If the host supports only CUDA 12.4, use a matching RAPIDS image, for example `rapidsai/base:24.06-cuda12.4-py3.11`, or use the conda fallback below.

## Setup

From the repository root:

```bash
bash scripts/setup_gpu.sh
```

The script checks cuML, PyTorch, XGBoost, Optuna, and the tracking dependencies, then runs a tiny cuML fit, a tiny XGBoost CUDA fit, and a PyTorch CUDA check.

## Smoke test

```bash
fpd smoke --config configs/smoke.yaml
```

This uses a small CPU subset and should finish quickly. If it fails, fix the pipeline before renting a GPU.

## Run the benchmark

```bash
fpd download
fpd features --config configs/stage_a.yaml
fpd benchmark --config configs/stage_a.yaml
fpd benchmark --config configs/stage_b.yaml
fpd final --config configs/final.yaml
fpd cnn --config configs/cnn.yaml
fpd ablation --config configs/ablation.yaml
fpd plot --config configs/final.yaml
```

Run the stages inside `tmux` or `nohup`; the full sequence is several hours.

## Sync results

```bash
rsync -avz <host>:<repo>/results ./results
```

The results directory contains the evaluation rows, run summaries, final test table, CNN table, ablation table, metadata, and plots.

## Conda fallback

Use this if the RAPIDS image cannot be pulled on the host:

```bash
mamba create -n fpd -c rapidsai -c conda-forge -c nvidia \
  rapids=24.10 python=3.11 cuda-version=12.5
conda activate fpd
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install xgboost optuna scikit-image opencv-python-headless nvidia-ml-py psutil
pip install -e . --no-deps
```

For CUDA 12.4 hosts, use the matching RAPIDS release and CUDA version, for example `rapids=24.06` and `cuda-version=12.4`.

## Blackwell note

The RTX PRO 4000 Blackwell (compute capability 12.0) is not supported by RAPIDS 24.10 / CUDA 12.5. CuPy fails with an NVRTC invalid-architecture error and cuML cannot compile kernels.

For Blackwell, use RAPIDS 26.10 / CUDA 13:

```bash
mamba create -y -n fpd -c rapidsai -c conda-forge -c nvidia \
  "cuml=26.10.00" "cuda-version=13.0" python=3.11
/opt/conda/envs/fpd/bin/pip install "cupy-cuda13x[ctk]==14.2.0"
/opt/conda/envs/fpd/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
/opt/conda/envs/fpd/bin/pip install xgboost optuna scikit-image opencv-python-headless nvidia-ml-py psutil pandas matplotlib seaborn huggingface-hub pyyaml tqdm joblib pillow
```

The CuPy `[ctk]` extra supplies the CUDA headers. A duplicate `cupy` warning is harmless if the cuML smoke passes. The 3090 path with RAPIDS 24.10 is simpler and remains the recommended setup when available.

## Notes

- If XGBoost was installed from a CPU-only wheel, `device="cuda"` fails. The smoke check reports this; install the GPU build or use the conda `py-xgboost-gpu` package.
- cuML and PyTorch can conflict if pip upgrades NumPy or scikit-learn. Install the repo with `--no-deps` on the GPU host.
- The benchmark stores feature caches under `~/downloads/datasets/x_fake_profile_detection/features`. The disk requirement is roughly 5 GB for the three feature families plus the dataset.
