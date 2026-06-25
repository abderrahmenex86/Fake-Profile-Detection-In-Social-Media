# 🎭 Fake Profile Detection (Machine Learning)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![Metaheuristics](https://img.shields.io/badge/Optimization-Scikit--Opt-purple.svg)

A professional, production-ready Machine Learning pipeline for detecting fake profiles from screenshots. Features strict modularity, `joblib` artifact tracking, incremental PCA for memory safety, and metaheuristic hyperparameter optimization (GA, PSO).

## ✨ Features
* **Zero Deep Learning Frameworks**: Pure Scikit-Learn and XGBoost architectures.
* **Hardware-Aware Memory Management**: Uses `IncrementalPCA` and nested batched feature extraction to prevent RAM crashes on 10,000+ SIFT descriptors.
* **Metaheuristic Optima**: Employs Genetic Algorithms and Particle Swarm Optimization to dynamically explore `n_estimators`, `max_depth`, and `C` spaces.
* **Smart Inference**: Dynamically rebuilds scaling pipelines and models entirely from JSON artifact logs.

## 🚀 Workflow

**1. Download Dataset**
```bash
python tools.py --mode download
```

**2. Sanity Check**

```bash
python main.py --mode test --model rf --feature sift --pca
```

**3. Optimize Hyperparameters (Metaheuristics)**

```bash
python main.py --mode optimize --model xgb --feature hog --optimizer pso
```

**4. Train Model**

```bash
python main.py --mode train --model xgb --feature hog --pca
```

**5. Smart Inference**

```bash
python main.py --mode infer
```
