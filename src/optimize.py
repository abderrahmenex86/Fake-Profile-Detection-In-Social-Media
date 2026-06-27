import gc
import json
import os

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sko.GA import GA
from sko.PSO import PSO
from sko.SA import SA
from tqdm import tqdm

from src.factory import build_model
from src.models import get_param_space


def safe_float(val):
    if isinstance(val, (list, np.ndarray)):
        return float(val[0]) if len(val) > 0 else 0.0
    return float(val)


def run_tow(objective, dim, lb, ub, max_iter=10, pop_size=8):
    pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
    best_pop, best_score = None, -np.inf
    history = []

    for _ in range(max_iter):
        fit = np.array([-objective(sol) for sol in pop])
        weights = fit / (fit.sum() + 1e-16)

        idx = np.argmax(fit)
        if fit[idx] > best_score:
            best_score = fit[idx]
            best_pop = pop[idx].copy()

        history.append(-best_score)

        new_pop = pop.copy()
        for i in range(pop_size):
            force = np.zeros(dim)
            for j in range(pop_size):
                if i == j:
                    continue
                diff = pop[j] - pop[i]
                force += np.random.rand(dim) * weights[j] * diff
            new_pop[i] = pop[i] + force
        pop = np.clip(new_pop, lb, ub)

    return best_pop, -best_score, history


def run_optimization(args, X_train, y_train, run_dir):
    space = get_param_space(args.model)
    dim = len(space)

    lb, ub, cat_lists = [], [], []
    for p in space:
        if p["type"] == "continuous":
            lb.append(p["bounds"][0])
            ub.append(p["bounds"][1])
            cat_lists.append(None)
        else:
            lb.append(0)
            ub.append(len(p["categories"]) - 1)
            cat_lists.append(p["categories"])

    lb, ub = np.array(lb), np.array(ub)

    if args.optimizer == "ga":
        total_evals = 8 + (8 * 10)
    elif args.optimizer == "pso":
        total_evals = 8 + (8 * 10)
    elif args.optimizer == "sa":
        total_evals = 1 + (20 * 8)
    elif args.optimizer == "tow":
        total_evals = 8 * 10
    else:
        total_evals = 100

    pbar = tqdm(total=total_evals, desc=f"Optimizing {args.model.upper()}", leave=False)

    def decode(solution):
        params = {}
        for i, p in enumerate(space):
            if p["type"] == "continuous":
                val = float(solution[i])
                val = max(p["bounds"][0], min(p["bounds"][1], val))
                params[p["name"]] = val
            else:
                idx = int(round(solution[i]))
                categories = cat_lists[i]
                idx = max(0, min(len(categories) - 1, idx))
                params[p["name"]] = categories[idx]
        return params

    def objective(solution):
        params = decode(solution)

        if getattr(args, "use_cuml", False):
            params["use_cuml"] = True

        if args.model == "xgb" and args.feature == "raw" and not getattr(args, "pca", False):
            params["force_cpu"] = True

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []

        with joblib.parallel_backend("sequential"):
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, X_v = X_train[train_idx], X_train[val_idx]
                y_tr, y_v = y_train[train_idx], y_train[val_idx]

                model = build_model(args.model, **params)

                try:
                    model.fit(X_tr, y_tr)
                except Exception as e:
                    err_msg = str(e).lower()
                    if any(x in err_msg for x in ["memory", "alloc", "cublas", "cuda"]):
                        del model
                        gc.collect()
                        if getattr(args, "use_cuml", False):
                            try:
                                import cupy as cp

                                cp.get_default_memory_pool().free_all_blocks()
                            except ImportError:
                                pass

                        params["force_cpu"] = True
                        model = build_model(args.model, **params)
                        model.fit(X_tr, y_tr)
                    else:
                        raise e

                preds = model.predict(X_v)
                scores.append(accuracy_score(y_v, preds))

                del model
                gc.collect()

                if getattr(args, "use_cuml", False):
                    try:
                        import cupy as cp

                        cp.get_default_memory_pool().free_all_blocks()
                        cp.get_default_pinned_memory_pool().free_all_blocks()
                    except ImportError:
                        pass

        score = float(np.mean(scores))

        pbar.update(1)
        pbar.set_postfix({"Current CV Acc": f"{score:.4f}"})
        return -score

    tqdm.write(f"\n[INFO] Starting {args.optimizer.upper()} optimization for {args.model.upper()}...")

    history = []

    if args.optimizer == "ga":
        opt = GA(func=objective, n_dim=dim, size_pop=8, max_iter=10, lb=lb, ub=ub, prob_mut=0.1)
        best_sol, best_score = opt.run()
        history = [safe_float(f) for f in getattr(opt, "generation_best_Y", [best_score])]

    elif args.optimizer == "pso":
        opt = PSO(func=objective, dim=dim, pop=8, max_iter=10, lb=lb, ub=ub, w=0.8, c1=0.5, c2=0.5)
        opt.run()
        best_sol, best_score = opt.gbest_x, opt.gbest_y
        history = [safe_float(f) for f in getattr(opt, "gbest_y_hist", [best_score])]

    elif args.optimizer == "sa":
        x0 = (lb + ub) / 2
        opt = SA(func=objective, x0=x0, T_max=1.0, T_min=1e-3, L=20, max_iter=8, lb=lb, ub=ub)
        best_sol, best_score = opt.run()

        if hasattr(opt, "best_y_history"):
            history = [-safe_float(f) for f in opt.best_y_history]
        elif hasattr(opt, "generation_best_Y"):
            history = [-safe_float(f) for f in opt.generation_best_Y]
        else:
            history = [-safe_float(best_score)]

    elif args.optimizer == "tow":
        best_sol, best_score, history = run_tow(objective, dim, lb, ub, max_iter=10, pop_size=8)

    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported.")

    pbar.close()

    best_params = decode(best_sol)
    best_score_float = safe_float(-best_score)

    history = [abs(h) for h in history]

    tqdm.write(f"[SUCCESS] Best Accuracy: {best_score_float:.4f}")
    tqdm.write(f"[SUCCESS] Best Params: {best_params}")

    summary = {
        "model": args.model,
        "optimizer": args.optimizer,
        "best_accuracy": best_score_float,
        "best_params": best_params,
        "history": history,
    }

    with open(os.path.join(run_dir, "optuna_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    return best_params
