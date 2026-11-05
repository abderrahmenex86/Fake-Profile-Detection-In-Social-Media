---
license: mit
library_name: xgboost
pipeline_tag: image-classification
tags:
  - fake-profile-detection
  - social-media
  - hog
  - xgboost
  - tabular
datasets:
  - drveronika/x_fake_profile_detection
metrics:
  - accuracy
  - f1
---

# HOG + XGBoost fake-profile classifier

Four-class image classifier for social-media profile screenshots: `BOT`, `CYBORG`, `REAL`, `VERIFIED`.

This is the best accuracy/cost model from the [fake-profile detection benchmark](https://github.com/abderrahmenex86/Fake-Profile-Detection-In-Social-Media). It classifies a screenshot using HOG features and a small gradient-boosted tree ensemble. No neural network, no OCR, no metadata.

Try it at **[fpd.abderrahmene.me](https://fpd.abderrahmene.me)**, which runs this artifact in the browser.

## Results

Official 4,000-image test split of the X fake-profile dataset. The exported artifact was retrained on CPU from the 11,000 train+validation images; the reference row is the same configuration trained on a GPU in the benchmark run.

| | Exported artifact | Reference benchmark |
|---|---:|---:|
| Test accuracy | 99.28% | 99.35% |
| Test macro F1 | 0.9928 | 0.9935 |
| Precision (macro) | 0.9929 | 0.9936 |
| Recall (macro) | 0.9928 | 0.9935 |
| Train time | 123.5s (CPU) | 8.8s (RTX PRO 4000) |
| Model size | 189 KB | 189 KB |

Per class, exported artifact:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| BOT | 0.9980 | 1.0000 | 0.9990 | 1000 |
| CYBORG | 1.0000 | 0.9920 | 0.9960 | 1000 |
| REAL | 0.9737 | 1.0000 | 0.9867 | 1000 |
| VERIFIED | 1.0000 | 0.9790 | 0.9894 | 1000 |

29 of 4,000 test images are misclassified: 2 `CYBORG` → `BOT`, 6 `CYBORG` → `REAL`, 21 `VERIFIED` → `REAL`. The remaining errors are between visually similar classes.

The small difference from the benchmark (0.9928 vs 0.9935) comes from retraining on CPU with a different XGBoost build (`hist` on CPU vs CUDA), not from a change in features or protocol.

## How it works

1. Read the image, convert to RGB, resize to **450×600** (`INTER_AREA`).
2. Convert to grayscale.
3. HOG with 9 orientations, 16×16 cells, 2×2 blocks, `L2-Hys` normalization → **34,992 features**.
4. `StandardScaler` fit on the 11,000 train+validation images (saved as `scaler.joblib`).
5. XGBoost classifier: 50 trees, depth 3, learning rate 0.174, subsample 0.614, column sample 0.920.
6. Softmax over 4 classes.

The tree ensemble is only 189 KB; almost all of the artifact size is the scaler (0.80 MB), because it stores per-feature mean and scale for 34,992 features.

## Usage

Install and predict a single screenshot:

```bash
pip install -r requirements.txt
python inference.py screenshot.png
```

```text
screenshot.png: REAL (0.989)  BOT=0.001  CYBORG=0.000  REAL=0.989  VERIFIED=0.010
```

`--json` prints machine-readable output. From Python:

```python
import json
from pathlib import Path

import joblib
from inference import extract, load_artifacts

config, labels, scaler, model = load_artifacts(Path("."))
features = extract("screenshot.png", config).reshape(1, -1)
probabilities = model.predict_proba(scaler.transform(features))[0]
```

Latency on CPU, batch 1: about **35 ms** for HOG extraction and **3 ms** for the classifier, roughly 38 ms per image end to end. Throughput scales with batch size.

## Files

| File | Contents |
|---|---|
| `model.ubj` | XGBoost booster, 189 KB |
| `scaler.joblib` | fitted `StandardScaler`, 0.80 MB |
| `labels.json` | class order, class names, dataset revision |
| `feature_config.json` | exact resize, HOG, and preprocessing settings |
| `metrics.json` | training config, test metrics, cost, reference benchmark |
| `inference.py` | dependency-light predictor |
| `requirements.txt` | runtime dependencies |

To rebuild the artifacts from the dataset:

```bash
python scripts/export_best_model.py --data-dir ~/downloads/datasets/x_fake_profile_detection
```

## Limitations

- **Synthetic dataset.** The X dataset is generated screenshots, and profile layout, badges, and template cues are correlated with the label. In the benchmark, masking one template region still leaves ~92% test macro F1, so the model leans on layout, not account behavior.
- **Screenshots only.** There is no metadata, text history, follower graph, or posting behavior. It cannot detect fakes from a profile URL alone.
- **Balanced classes.** The dataset has 25% per class. Real traffic is highly imbalanced, so precision on rare classes will be lower than these numbers.
- **Cross-domain use is untested.** Different platform layout, language, dark mode, cropping, or compression will shift the HOG distribution. Evaluate before relying on it.
- **Fine detail matters.** One BOT test image scores 0.999 as a 450×600 launch PNG but 0.898 CYBORG after being resized up to 600×800 and back down. HOG reads small layout details, so resampling or lossy compression can flip a confident prediction.
- **Not a moderation decision system.** It outputs a probability over four synthetic classes; it is not evidence that an account is fake.

## Citation

Dataset and task:

Dracewicz, W., & Sepczuk, M. (2024). Detecting Fake Accounts on Social Media Portals—The X Portal Case Study. *Electronics*, 13(13), 2542. https://doi.org/10.3390/electronics13132542

## License

MIT
