from . import dgo, ga, lca, pso, random_search, sa, tow, tpe

OPTIMIZERS = {
    "random": random_search.optimize,
    "tpe": tpe.optimize,
    "ga": ga.optimize,
    "pso": pso.optimize,
    "sa": sa.optimize,
    "tow": tow.optimize,
    "lca": lca.optimize,
    "dgo": dgo.optimize,
}
