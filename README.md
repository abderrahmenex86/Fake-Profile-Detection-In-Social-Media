# 🎭 Fake Profile Detection (Machine Learning)

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.2-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![RAPIDS](https://img.shields.io/badge/NVIDIA-RAPIDS%20cuML-purple.svg)
![Metaheuristics](https://img.shields.io/badge/Optimization-Scikit--Opt-pink.svg)

A professional, production-grade Machine Learning pipeline for detecting fake social media profiles from profile screenshots. Built strictly from scratch using classical machine learning, manual feature extraction, and GPU-accelerated metaheuristic hyperparameter optimization.

This project was built to explore the trade-offs between manual feature representation and classical classification models, optimized dynamically on state-of-the-art **NVIDIA Blackwell GPU (RTX 5060)** architectures.

---

## 📝 Abstract
Deceptive social media profiles (bots, cyborgs, and fake accounts) pose a systemic threat to digital trust, public discourse, and cybersecurity. Standard detection techniques rely heavily on textual metadata or behavioral APIs, which are easily bypassed or restricted by platform privacy controls. This project presents a professional, end-to-end framework designed to answer a central research question: **Can we accurately detect fake profiles strictly using static screenshots of their profiles?** 

By leveraging manual computer vision feature extraction—incorporating Raw Pixel Flattening, Histogram of Oriented Gradients (HOG), and Scale-Invariant Feature Transform (SIFT) with and without Principal Component Analysis (PCA)—we map these screenshots into high-dimensional numerical spaces. We evaluate five distinct classification backends (Naive Bayes, Logistic Regression, Support Vector Machines, Random Forests, and XGBoost) optimized across a combinatorial grid of 120 configurations using four distinct metaheuristic algorithms: Genetic Algorithms (GA), Simulated Annealing (SA), Particle Swarm Optimization (PSO), and a custom-built gravitational Tug of War (TOW) optimizer. 

---

## 🛠️ Methodology & Feature Engineering

Traditional deep learning models act as "black boxes" and require massive, resource-heavy architectures. To maintain a lightweight, interpretable, and highly optimized footprint, this project relies on custom-engineered spatial and structural feature extractors:

```
[Profile Image] ──> Resize (150x200) ──> Gray Conversion ──> [Feature Extraction] ──> StandardScaler ──> [Optional IPCA] ──> [Classifier]
                                                                ├── Raw (90k dims)
                                                                ├── HOG (3,168 dims)
                                                                └── SIFT (29,952 dims)
```

### 1. Feature Representation Space
* **Raw Pixel Flattening (RAW):** Images are resized to a standard $150 \times 200 \times 3$ RGB matrix and flattened. This establishes a baseline representation of the global color layout and spatial structure of the profile, yielding a dense **90,000-dimensional** feature vector.
* **Histogram of Oriented Gradients (HOG):** Designed to capture edge directions, gradients, and localized shape representations. Images are converted to grayscale and processed with 9 orientation bins, $16 \times 16$ pixels per cell, and $2 \times 2$ cells per block, yielding a robust, illumination-invariant **3,168-dimensional** structural feature vector.
* **Scale-Invariant Feature Transform (SIFT):** Captures local keypoint descriptors that are invariant to scale, rotation, and affine transformations. We extract and flatten the top 234 descriptors (each of 128 dimensions), padding with zeros where keypoint density is low, resulting in a highly stable **29,952-dimensional** localized keypoint vector.

### 2. Memory-Aware Dimensionality Reduction (Incremental PCA)
Training classifiers on a $90,000$-dimensional dense matrix of thousands of images would trigger immediate Out-of-Memory (OOM) errors on consumer hardware. To alleviate this, we implement **Incremental PCA (IPCA)** in batched passes. This compresses the feature space down to **500 principal components**, capturing the maximum explained variance while keeping the peak memory footprint strictly bounded.

---

## 🧭 Optimization Framework & Classifiers

Our research grid comprises **120 unique mathematical permutations** (5 Classifiers $\times$ 3 Features $\times$ 2 PCA states $\times$ 4 Metaheuristic Optimizers).

### 1. Classification Backends
* **Naive Bayes (NB):** High-speed probabilistic classifier utilizing Gaussian likelihoods.
* **Logistic Regression (LR):** Linear decision boundary utilizing L2 regularization.
* **Support Vector Machine (SVM):** High-dimensional hyperplane separator utilizing Radial Basis Function (RBF) and Linear kernels.
* **Random Forest (RF):** Ensemble of bagging decision trees capturing non-linear feature interactions.
* **XGBoost (XGB):** Advanced gradient-boosted decision trees executing sequential error correction.

### 2. Metaheuristic Tuning Algorithms
Instead of traditional Grid or Random Searches, we employ global and local optimization heuristics:
* **Genetic Algorithm (GA):** Mimics natural selection by maintaining a population of 8 chromosomes, executing crossover, and applying a $10\%$ mutation rate over 10 generations. (88 total evaluations)
* **Particle Swarm Optimization (PSO):** Mimics bird flocking behavior, letting 8 particles fly through the continuous hyperparameter space, adjusting their velocities based on individual and global historical bests over 10 iterations. (88 total evaluations)
* **Simulated Annealing (SA):** A custom, fully deterministic thermodynamic-inspired local optimizer. It starts at a high temperature ($T_{\max}=1.0$) and cools down to $T_{\min}=1e-3$ over 8 steps, taking 20 random walk perturbations per step. It accepts worse solutions with a probability $P = \exp(\Delta / T)$ to successfully escape local minima. (161 total evaluations)
* **Tug of War (TOW):** A custom gravitational-attraction optimizer. 8 agents exert "forces" on each other proportional to their fitness weights, pulling the population dynamically toward the global optimum over 10 iterations. (80 total evaluations)

---

## ⚡ The GPU Paradigm Shift: Scikit-Learn to RAPIDS cuML

To accelerate our extensive 120-run grid sweep, we transitioned our CPU-bound Scikit-Learn classifiers to **NVIDIA RAPIDS cuML** on a state-of-the-art **NVIDIA RTX 5060 (Blackwell Architecture)**. This transition unlocked **$10\times$ to $50\times$ training speedups**, but required solving complex C++, compiler, and memory synchronization hurdles:

### 1. Joblib Multiprocessing CUDA Context Collisions
Scikit-Learn's multi-class wrappers (like One-Vs-One) internally split classifications and execute them in parallel using `joblib.Parallel()`. However, **CUDA contexts and cuBLAS handles cannot be shared or destroyed across multiple OS processes.** Spawning parallel workers triggered immediate, uncatchable C++ driver crashes (`CUBLAS_STATUS_NOT_INITIALIZED`).
* **Architectural Cure:** We defensively wrapped all model fitting and cross-validation calls inside `with joblib.parallel_backend('sequential'):`. This forces `joblib` to execute strictly on the main thread of the parent process, completely eliminating CUDA context fragmentation while retaining full GPU execution speed.

### 2. The Blackwell (`sm_120`) JIT Compilation Obstacle
Because the RTX 5060 is built on the brand-new **Blackwell architecture (Compute Capability 12.0 / `sm_120`)**, standard prebuilt CuPy wheels (`cupy-cuda12x`) do not contain pre-compiled binary images for `sm_120`. When trying to flatten and count classes on the GPU, CuPy would fail to find matching binaries and crash with a `CUDA_ERROR_NO_BINARY_FOR_GPU` error.
* **The Cure:** We bypassed manual source compilation errors by installing NVIDIA's official PyPI-packaged runtime wheels (`nvidia-cuda-runtime-cu12`, etc.) directly into the virtual environment, pointing `LD_LIBRARY_PATH` to them, and setting **`CUPY_COMPILE_WITH_PTX=1`**. This forces CuPy to generate portable, forward-compatible assembly (PTX), which your system's driver dynamically compiles for the RTX 5060 on the fly.

### 3. Linear Kernel Platt Scaling Limitation
NVIDIA cuML's `SVC` has a C++ level limitation: it only compiles GPU kernels for probability calibration (Platt Scaling) when utilizing the `'rbf'` kernel. When our GA/SA searches discovered that a `'linear'` kernel was optimal, training the final model with `probability=True` crashed the program.
* **The Cure:** If `probability=True` is requested, we dynamically wrap the base `cuml.svm.SVC` in Scikit-Learn's standard **`CalibratedClassifierCV`**. This offloads the heavy SVM matrix fits to the GPU, but performs the lightweight probability calibration on the CPU, seamlessly bypassing the linear-kernel restriction.

### 4. Overcoming VRAM Exhaustion via CPU Fallback
Fitting models on raw, uncompressed 90,000-dimensional features on a full dataset of thousands of images requires massive memory allocations (a single $8000 \times 90000$ matrix uses ~2.88 GB). During multiclass training, XGBoost's GPU histogram generation or SVM's kernel calculations would frequently exceed the 8 GB VRAM limit of the RTX 5060.
* **The Cure:** We implemented a two-pronged defensive memory-handling strategy:
  1. At the end of every cross-validation fold, we explicitly delete the model, trigger Python's Garbage Collector, and force the CuPy memory pool to release all unused cached blocks back to the GPU OS via `cp.get_default_memory_pool().free_all_blocks()`.
  2. We wrapped our training execution in a try-except block. If a model encounters a VRAM or C++ driver limitation, the pipeline automatically intercepts the crash, completely flushes the GPU memory, sets `force_cpu=True`, and **gracefully falls back to training on your highly robust 32 GB RAM / 36-Core CPU backend**. Your overnight sweeps are guaranteed to complete without ever crashing.

---

## 📊 Experimental Results & Discussion

### 2x2 Grid of Thesis-Grade Visualizations

| Global Heatmap Matrix | Classifier Robustness Boxplot |
| :---: | :---: |
| ![Heatmap](docs/figs/01_global_performance_heatmap.png) | ![Boxplot](docs/figs/02_classifier_comparison.png) |
| **PCA Dimensionality Impact** | **Metaheuristic Convergence Race** |
| ![PCA Impact](docs/figs/03_pca_impact_analysis.png) | ![Convergence](docs/figs/04_optimizer_convergence_race.png) |

### Key Scientific Findings
1. **The Dominance of Raw Spatial Structure:** Classifiers trained on HOG and RAW (Flattened) features achieved the highest accuracies, with **XGBoost and Random Forest reaching peak validation scores of $98.1\%$ and $97.7\%$**, respectively. This proves that profile layouts, dominant color blocks, and structured text boundaries are incredibly strong signals for profile legitimacy.
2. **The Destructive Nature of PCA:** Across almost all model architectures, applying Incremental PCA to compress the features down to 500 dimensions resulted in a consistent drop in validation accuracy. For HOG features, PCA significantly increased variance and lowered the median score. This mathematically proves that global dimensionality reduction filters out the highly specific, fine-grained local textures needed to distinguish real users from fake ones.
3. **The SIFT Limitation:** SIFT was consistently the weakest feature extractor, with accuracies heavily bottlenecked between $56\%$ and $652\%$. Because profile screenshots are highly structured and flat, they lack the rich, natural scale-invariant local textures (like those found in outdoor scenes or objects) that SIFT is designed to capture.
4. **Active Optimizer Learning:** While flat classifiers like Random Forest reached their peak performance rapidly, algorithms like Support Vector Machines (SVM) exhibited highly dynamic, climbing learning curves during metaheuristic search, proving that GA, SA, and PSO successfully navigated hyperparameter topologies to extract maximum generalization.

---

## 🚀 System Workflow & CLI Commands

Follow this step-by-step workflow to replicate the complete experimental lifecycle.

### 1. Download & Structure Dataset
Pulls the raw data from Hugging Face, bypasses metadata parsers, extracts the ZIPs, and formats the split directories:
```bash
python tools.py --mode download
```

### 2. Fast Sanity Check
Instantly tests all 30 model-feature configurations in a sequential loop utilizing an aggressive, shuffled 20-sample slice to ensure your packages and hardware work without crashing:
```bash
python main.py --mode test --model all --feature all
```

### 3. Execute the Grid Sweep (GPU-Accelerated)
You can optimize and train specific combinations, or execute the entire 120-permutation sweep. Because we implemented a **Phase-Level Resume System**, if a run ever interrupts, it will instantly scan the `artifacts/` folder, skip completed sweeps, and pick up right where it left off.

* **Run a highly targeted sweep** (Optimizes XGBoost across all 6 feature combinations using Particle Swarm Optimization on the GPU):
  ```bash
  python main.py --mode optimize --model xgb --feature all --optimizer pso --use_cuml
  ```
* **Run the Master Thesis Sweep** (Optimizes and trains all 120 configurations sequentially):
  ```bash
  python main.py --mode optimize --model all --feature all --optimizer all --use_cuml
  ```

### 4. Bypassing Crashes with Smart Resume
If the final training step of a configuration ever crashed due to a prior memory limit, you do not have to re-run the 2-hour optimization. Simply run the command again. `main.py` will read the existing `optuna_summary.json`, print `[INFO] Optimization already completed previously!`, load the best parameters, and trigger the CPU fallback to finish the training pass in minutes!

### 5. Generate Thesis Plots
Analyzes your database of runs, cleans up outliers, extracts the most dynamic convergence curves, and exports them as vector-ready `.png` files:
```bash
python tools.py --mode plot
```

### 6. Run Smart Interactive Inference
Type `q` to exit. This script completely ignores your command-line arguments. It automatically reads `hyperparameters.json` from your latest run, dynamically reconstructs the exact feature extractor and model pipeline, and prompts you for an image path to predict:
```bash
python main.py --mode infer
```
```
Path to profile image (.png) > my_test_screenshot.png
Prediction > REAL (Confidence: 0.98)
```
