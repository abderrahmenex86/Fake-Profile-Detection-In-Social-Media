import numpy as np
from sko.SA import SA

from experiments.config import SA_L, SA_MAX_ITER, SA_T_MAX, SA_T_MIN


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
                categories = cat_category_lists[i]
                params[p["name"]] = categories[idx]
        return params

    def fitness(sol):
        params = decode_solution(sol)
        model = module.create_model(params)
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        return -score

    sa = SA(
        func=fitness,
        x0=np.array([(lb[i] + ub[i]) / 2 for i in range(dim)]),
        T_max=SA_T_MAX,
        T_min=SA_T_MIN,
        L=SA_L,
        max_iter=SA_MAX_ITER,
        lb=lb,
        ub=ub,
    )

    best_sol, best_score = sa.run()
    best_params = decode_solution(best_sol)
    best_model = module.create_model(best_params)
    best_model.fit(X_train, y_train)

    if verbose:
        print(
            f"Best params (SA): { {k: f"{v:.2e}" if isinstance(v, np.float64) else v for k, v in best_params.items()} }"
        )
        print(f"Best CV accuracy: {-best_score:.4f}")

    return best_model, best_params, -best_score
