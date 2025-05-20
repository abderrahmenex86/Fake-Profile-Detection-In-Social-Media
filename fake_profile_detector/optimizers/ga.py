import numpy as np
from sko.GA import GA

from experiments.config import GA_MAX_ITER, GA_MUTATION_RATE, GA_POP_SIZE


def optimize(module, X_train, y_train, X_val, y_val, verbose=True):
    space = module.param_space()
    dim = len(space)
    lb = []
    ub = []

    cat_category_lists = []

    for parameter in space:
        if parameter["type"] == "continuous":
            lb.append(parameter["bounds"][0])
            ub.append(parameter["bounds"][1])
            cat_category_lists.append(None)
        elif parameter["type"] == "categorical":
            lb.append(0)
            ub.append(len(parameter["categories"]) - 1)
            cat_category_lists.append(parameter["categories"])

    def decode_solution(solution):
        params = {}
        for i, parameter in enumerate(space):
            if parameter["type"] == "continuous":
                params[parameter["name"]] = solution[i]
            elif parameter["type"] == "categorical":
                idx = int(round(solution[i]))
                categories = cat_category_lists[i]
                params[parameter["name"]] = categories[idx]
        return params

    def fitness(solution):
        params = decode_solution(solution)
        model = module.create_model(params)
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        return score

    ga = GA(
        func=fitness,
        n_dim=dim,
        size_pop=GA_POP_SIZE,
        max_iter=GA_MAX_ITER,
        prob_mut=GA_MUTATION_RATE,
        lb=lb,
        ub=ub,
        precision=1e-5,
    )

    best_sol, best_score = ga.run()

    best_params = decode_solution(best_sol)
    best_model = module.create_model(best_params)

    best_model.fit(X_train, y_train)

    if verbose:
        print(
            f"Best params (GA): { {k: f"{v:.2e}" if isinstance(v, np.float64) else v for k, v in best_params.items()} }"
        )
        print(f"Best CV accuracy: {best_score.item():.4f}")

    return best_model, best_params, best_score
