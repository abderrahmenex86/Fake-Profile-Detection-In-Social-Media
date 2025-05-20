import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sko.PSO import PSO

from experiments.config import (PSO_C1, PSO_C2, PSO_MAX_ITER, PSO_POP_SIZE,
                                PSO_W, RANDOM_SEED)


def optimize(
    module,
    X_train,
    y_train,
    X_val,
    y_val,
    verbose=True,
):
    space = module.param_space()

    dim = len(space)
    lb = []
    ub = []

    cat_category_lists = []

    for p in space:
        if p["type"] == "continuous":
            lb.append(p["bounds"][0])
            ub.append(p["bounds"][1])
            cat_category_lists.append(None)
        elif p["type"] == "categorical":
            lb.append(0)
            ub.append(len(p["categories"]) - 1)
            cat_category_lists.append(p["categories"])

    def decode_solution(solution):
        params = {}
        for i, p in enumerate(space):
            if p["type"] == "continuous":
                params[p["name"]] = solution[i]
            elif p["type"] == "categorical":
                idx = int(round(solution[i]))
                params[p["name"]] = cat_category_lists[i][idx]
        return params

    def fitness(solution):
        params = decode_solution(solution)
        model = module.create_model(params)
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        return -score  # PSO minimizes

    pso = PSO(
        func=fitness,
        dim=dim,
        lb=lb,
        ub=ub,
        pop=PSO_POP_SIZE,
        max_iter=PSO_MAX_ITER,
        w=PSO_W,
        c1=PSO_C1,
        c2=PSO_C2,
    )

    best_sol, best_score = pso.run()
    best_params = decode_solution(best_sol)
    best_model = module.create_model(best_params)
    best_model.fit(X_train, y_train)

    if verbose:
        print(
            f"Best params (PSO): { {k: f"{v:.2e}" if isinstance(v, np.float64) else v for k, v in best_params.items()} }"
        )
        print(f"Best CV accuracy: {-best_score.item():.4f}")

    return best_model, best_params, -best_score
