import numpy as np

from baseline import make_baseline_candidates, select_best_baseline


def test_baseline_candidates_cover_requested_improvements():
    names = set(make_baseline_candidates())
    assert "logreg_default" in names
    assert "logreg_balanced" in names
    assert "logreg_balanced_C0.1" in names
    assert "linear_svc_balanced" in names


def test_select_best_baseline_uses_train_only_cv():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 6))
    y = np.array(["Positive", "Negative", "Neutral", "Irrelevant"] * 20)
    X[y == "Positive", 0] += 2
    X[y == "Negative", 0] -= 2

    model, name, results = select_best_baseline(X, y, cv_splits=2)

    assert name in make_baseline_candidates()
    assert len(results) == len(make_baseline_candidates())
    assert model.predict(X[:3]).shape == (3,)
