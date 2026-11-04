# Benchmark Report

## Abstract

This report evaluates image-based fake-profile detection on the X profile dataset. It compares HOG and SIFT features, PCA and no PCA, classical and GPU-accelerated classifiers, a small MLP, a small CNN, and eight hyperparameter optimizers under an equal evaluation budget. The protocol uses the official dataset splits, train-only preprocessing, and a single held-out test evaluation per final model.

## 1. Introduction

The benchmark answers five questions:

1. Do game-based optimizers (TOW, LCA, DGO) beat classical optimizers (Random Search, TPE, GA, PSO, SA) under the same budget?
2. Which feature representation is stronger: HOG or SIFT?
3. Does PCA help or hurt, and does the answer depend on the classifier?
4. Can classical features plus a GPU-trained MLP match a small end-to-end CNN?
5. How much of the accuracy comes from the profile template rather than account behavior?

## 2. Background

The project started as a graduation thesis on fake-profile detection. The thesis used full-resolution HOG (63,504 dimensions) and reported near-perfect test accuracy. A corrected reproduction on the official test split reached 99.65% macro F1 with full-resolution HOG and XGBoost. That result is useful context, but it is not the benchmark result: full-resolution HOG is roughly twice the size of SIFT and can read the profile template directly.

This benchmark uses a fair 34,992-dimension HOG so that HOG and SIFT are dimension-matched. The template ablation measures how much of the accuracy comes from the template.

## 3. Dataset and split

| Item | Value |
|---|---|
| Images | 15,000 |
| Classes | BOT, CYBORG, REAL, VERIFIED |
| Image size | 600×800 RGB |
| Official train | 10,000 |
| Official validation | 1,000 |
| Official test | 4,000 |
| Class balance | 2,500 / 250 / 1,000 per class |

Hyperparameter search uses a stratified 2,000-image subset of the official train split and the official 1,000-image validation split. Final models retrain on the full 11,000 train+validation images. The 4,000-image test split is used once per final model.

## 4. Features

| Feature | Configuration | Dimensions |
|---|---|---:|
| HOG | 450×600, 16×16 cells, 2×2 blocks, 9 orientations | 34,992 |
| SIFT | top 256 keypoints by response, 128-dim descriptors | 32,768 |
| Raw + PCA | 150×200×3 flattened, PCA to 1,000 | 1,000 |

Scalers and PCA transforms are fit on the training portion only. HOG and SIFT are dimension-matched; raw pixels are used only after PCA. The CNN is the only model that consumes raw images end to end.

SIFT is represented as a fixed top-256 descriptor matrix. This is simpler than a bag-of-visual-words representation and is documented as a limitation.

## 5. Models

| Model | Backend | Notes |
|---|---|---|
| Logistic Regression | cuML / scikit-learn | linear baseline |
| Linear SVM | cuML / scikit-learn | linear max-margin baseline |
| RBF SVM | cuML / scikit-learn | PCA features only |
| Random Forest | cuML / scikit-learn | tree ensemble |
| XGBoost | GPU / CPU | gradient boosting |
| MLP | PyTorch | two hidden layers, feature input |
| CNN | PyTorch | three convolutional blocks, raw image input |

The MLP uses the same feature vectors as the classical models. The CNN uses 150×200 RGB images and has roughly 100k parameters.

## 6. Optimizers and fairness

| Optimizer | Type | Budget |
|---|---|---:|
| Random Search | classical | 60 trials |
| TPE | classical / Bayesian | 60 trials |
| GA | classical | 6 candidates × 10 evaluations |
| PSO | classical | 6 particles × 10 iterations |
| SA | classical | 60 steps |
| TOW | game-based | 6 agents × 10 iterations |
| LCA | game-based | 6 teams × 10 weeks |
| DGO | game-based | 6 players × 10 rounds |

All optimizers use the same search space per model, the same 2,000/1,000 search split, the same model seed, and 60 objective evaluations. The optimizer seed varies. Each evaluation trains a model from scratch and scores it on the validation split.

Results are reported as mean ± standard deviation across three seeds. No p-values are used. Best-so-far curves, evaluations to best, and evaluations to 95% of best are reported as descriptive measures.

LCA and DGO are simplified implementations. They preserve the population and evaluation budget but are not bit-for-bit reproductions of the published algorithms.

## 7. Experimental protocol

1. Download the official dataset.
2. Extract HOG, SIFT, and raw+PCA features on the official train, validation, and test splits.
3. Build the stratified 2,000-image search split from the official train split.
4. Fit the scaler and PCA on the search train split.
5. Run every optimizer for 60 evaluations and three seeds.
6. Select the best configuration per feature/model by validation macro F1.
7. Retrain the selected configuration on the full 11,000 train+validation images.
8. Evaluate once on the official 4,000-image test split.
9. Train the CNN with the fixed recipe and three seeds.
10. Run the template ablation and brightness baseline.
11. Generate the tables and figures.

## 8. Results

The results are generated by the pipeline and written to:

- `results/evaluations.csv`
- `results/runs.csv`
- `results/final.csv`
- `results/cnn.csv`
- `results/ablation.csv`
- `results/metadata.json`
- `results/plots/`

### 8.1 Optimizer comparison

Table: mean best validation macro F1 per optimizer and feature/model.

Figure: `fig_convergence_<feature>.png`, `fig_optimizer_box.png`.

### 8.2 HOG vs SIFT

Table: final test macro F1 per feature and model.

Figure: `fig_hog_vs_sift.png`.

### 8.3 PCA vs no PCA

Table: paired PCA minus no-PCA deltas.

Figure: `fig_pca_effect.png`.

### 8.4 Classical, MLP, and CNN

Table: final test accuracy, macro F1, model size, training time, inference time, and GPU memory.

Figure: `fig_cnn_comparison.png`, `fig_final_heatmap.png`.

### 8.5 Cost and accuracy

Figure: `fig_cost_accuracy.png`.

### 8.6 Template ablation

Table: test macro F1 for full image, header crop, bottom mask, and brightness/edge baseline.

Figure: `fig_template_ablation.png`.

### 8.7 Best confusion matrix

Figure: `fig_best_confusion.png`.

## 9. Threats to validity

- The dataset is synthetic. Layout, text, badges, and template cues are highly correlated with the label. The template ablation measures this but does not eliminate it.
- Three seeds provide descriptive stability, not formal inference. Differences smaller than the seed standard deviation should not be interpreted as real.
- The 1,000-image official validation split is small. Optimizer ranking is therefore noisy.
- LCA and DGO are simplified implementations.
- SIFT is represented by a fixed top-256 descriptor matrix, not a bag-of-visual-words or Fisher-vector encoding.
- GPU results depend on the RAPIDS, CUDA, and PyTorch versions recorded in `results/metadata.json`.
- The final model is selected by validation macro F1 across optimizers and seeds. This selection can be optimistic on the test split.

## 10. Conclusion

The conclusion is written after the full run, using the generated tables. It should state:

- whether any game-based optimizer beat Random Search or TPE,
- whether HOG outperformed SIFT,
- whether PCA helped or hurt,
- whether the MLP or CNN matched the classical models,
- how much accuracy dropped under template masking.

## 11. Reproducibility

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

The exact command, config, versions, and GPU are recorded in `results/metadata.json`.
