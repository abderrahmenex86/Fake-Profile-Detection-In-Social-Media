from fpd.optimizers import OPTIMIZERS
from fpd.search_spaces import SPACES, Space


def test_optimizer_budget():
    space = Space(SPACES["logreg"])

    def objective(params):
        return -abs(params["C"] - 1.0)

    for optimizer in OPTIMIZERS.values():
        result = optimizer(objective, space, 12, 0)
        assert len(result.history) == 12
        assert result.best_params is not None
        assert result.best_score <= 0
