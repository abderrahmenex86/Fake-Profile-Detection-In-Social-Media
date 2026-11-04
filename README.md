# Fake Profile Detection Benchmark

A script-first research benchmark for classifying social-media profile screenshots into four account types: `BOT`, `CYBORG`, `REAL`, and `VERIFIED`.

The benchmark compares handcrafted image features, classical and GPU-accelerated classifiers, a small MLP, a small CNN, and eight hyperparameter optimizers under an equal evaluation budget. It uses the official dataset splits, train-only preprocessing, and a single held-out test evaluation per final model.

## Research questions

1. Do game-based optimizers (TOW, LCA, DGO) beat classical optimizers (Random Search, TPE, GA, PSO, SA) under the same budget?
2. Which feature representation is stronger: HOG or SIFT?
3. Does PCA help or hurt, and does the answer depend on the classifier?
4. Can classical features plus a GPU-trained MLP match a small end-to-end CNN?
5. How much of the accuracy comes from the profile template rather than account behavior?

## Dataset

The dataset is the public X fake-profile dataset from Hugging Face:

- 15,000 profile screenshots
- Four balanced classes: BOT, CYBORG, REAL, VERIFIED
- Official splits: 10,000 train / 1,000 validation / 4,000 test
- All images are 600×800 RGB screenshots

The dataset is downloaded with `fpd download`. It is not redistributed in this repository.

## Features

| Feature | Configuration | Dimensions |
|---|---|---:|
| HOG | 450×600, 16×16 cells, 2×2 blocks, 9 bins | 34,992 |
| SIFT | top 256 keypoints by response, 128-dim descriptors | 32,768 |
| Raw + PCA | 150×200×3 flattened, PCA to 1,000 | 1,000 |

HOG and SIFT are dimension-matched for a fair comparison. Raw pixels are used only after PCA. The CNN is the only model that consumes raw images end to end.

All scalers and PCA transforms are fit on the training split only.

## Models

- Logistic Regression
- Linear SVM
- RBF SVM (PCA features only)
- Random Forest
- XGBoost
- MLP with two hidden layers
- Small CNN with three convolutional blocks

The MLP consumes the same feature vectors as the classical models. The CNN consumes raw images.

## Optimizers

Game-based: TOW, LCA, DGO.

Classical: Random Search, TPE, GA, PSO, SA.

Stage A compares all eight optimizers with 60 evaluations and 3 seeds on PCA-1,000 versions of HOG, SIFT, and raw. Population methods use six candidates over ten evaluations per generation period; SA uses 60 steps; Random Search and TPE use 60 trials. Stage B compares models with TPE using 40 evaluations and 2 seeds; the final MLP retrains for 30 epochs.

Results are reported as mean ± standard deviation across seeds, with best-so-far curves and evaluations to best/95% of best. No p-values are used.

## Fair comparison

- Official 10k / 1k / 4k splits.
- Hyperparameter search on a 2,000-image stratified subset of the official train split and the official 1,000-image validation split.
- Final models retrain on the full 10,000 train + 1,000 validation = 11,000 images.
- The 4,000-image test split is used once per final model.
- Stage A uses PCA-1,000 versions of HOG, SIFT, and raw, with the same search space and 60-evaluation budget for every optimizer. No-PCA comparisons are in Stage B.
- Stage B uses TPE with 40 evaluations and 2 seeds; the final MLP retrains for 30 epochs.
- Random Forest is constrained to `max_features` in {`sqrt`, `log2`} to keep 35k-feature fits tractable.
- The same model seed is used across optimizers; only the optimizer seed varies.
- No early stopping; each evaluation uses a fixed training budget.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

fpd download
fpd smoke --config configs/smoke.yaml
```

The smoke run uses a small stratified subset, one feature, two models, and two optimizers. It validates the data, feature, optimizer, and tracking pipeline before a GPU run.

## GPU run

Use a RAPIDS image with cuML prebuilt, or follow the conda fallback in [docs/gpu.md](docs/gpu.md).

```bash
bash scripts/setup_gpu.sh

fpd features --config configs/stage_a.yaml
fpd benchmark --config configs/stage_a.yaml
fpd benchmark --config configs/stage_b.yaml
fpd final --config configs/final.yaml
fpd cnn --config configs/cnn.yaml
fpd ablation --config configs/ablation.yaml
fpd plot --config configs/final.yaml
```

The recommended GPU is a 24 GB RTX 3090. A 16 GB card works for the PCA and subset configurations.

## Results

The benchmark writes:

- `results/evaluations.csv` — every objective evaluation.
- `results/runs.csv` — every optimizer/seed run.
- `results/final.csv` — final test results for every feature/model.
- `results/cnn.csv` — CNN test results.
- `results/ablation.csv` — template-masking and brightness-baseline results.
- `results/metadata.json` — versions, GPU, CUDA, and config.
- `results/plots/` — convergence, heatmap, boxplot, PCA/HOG effects, cost, CNN, ablation, and confusion plots.

The paper-style report is [docs/benchmark.md](docs/benchmark.md).

## Repository layout

```text
src/fpd/
├── config.py
├── data.py
├── features.py
├── search_spaces.py
├── models.py
├── optimizers/
├── benchmark.py
├── final_eval.py
├── cnn.py
├── ablation.py
├── tracking.py
├── plotting.py
└── cli.py
configs/
├── smoke.yaml
├── stage_a.yaml
├── stage_b.yaml
├── final.yaml
├── cnn.yaml
└── ablation.yaml
docs/
├── benchmark.md
└── gpu.md
scripts/
└── setup_gpu.sh
tests/
```

## Limitations

- The dataset is synthetic. Profile layout, text, badges, and template cues are highly correlated with the label. The benchmark includes a template ablation to measure this.
- Three seeds and no formal statistical tests mean small differences should not be over-interpreted.
- LCA and DGO are simplified implementations. They use the same budget and search space but are not bit-for-bit reproductions of the published algorithms.
- SIFT is represented by a fixed top-256 descriptor matrix. A bag-of-visual-words representation would be a different comparison.
- GPU results depend on the RAPIDS, CUDA, and PyTorch versions recorded in `results/metadata.json`.

## Background

The project started as a graduation thesis on fake-profile detection. This repository is a clean re-evaluation: it uses the fair 34,992-dim HOG, official splits, train-only preprocessing, and an equal optimizer budget. The thesis’s full-resolution HOG result is discussed in [docs/benchmark.md](docs/benchmark.md) as background, not as the benchmark result.

## Citation

If you use the dataset, cite:

Dracewicz, W., & Sepczuk, M. (2024). Detecting Fake Accounts on Social Media Portals—The X Portal Case Study. *Electronics*, 13(13), 2542. https://doi.org/10.3390/electronics13132542

## License

MIT
