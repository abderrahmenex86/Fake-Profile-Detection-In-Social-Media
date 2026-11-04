import numpy as np

from fpd.search_spaces import SPACES, Space


def test_space_roundtrip():
    space = Space(SPACES["xgb"])
    rng = np.random.default_rng(0)
    for _ in range(20):
        params = space.decode(space.sample(rng))
        assert set(params) == {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
        }
        assert params["n_estimators"] in [50, 100, 200]
        assert params["max_depth"] in [3, 6, 9, 12]
        assert 0.01 <= params["learning_rate"] <= 0.3
        assert 0.6 <= params["subsample"] <= 1.0
        assert 0.6 <= params["colsample_bytree"] <= 1.0
