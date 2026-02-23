import numpy as np

from src.gfn.thresholding import (
    choose_threshold_cost_sensitive,
    choose_threshold_max_f1,
    choose_threshold_youden_j,
)


def test_threshold_strategies_return_finite_threshold_on_simple_case():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    t1 = choose_threshold_max_f1(y_true=y, scores=s)
    t2 = choose_threshold_youden_j(y_true=y, scores=s)
    t3 = choose_threshold_cost_sensitive(y_true=y, scores=s, cost_fp=1.0, cost_fn=2.0)

    assert np.isfinite(t1.threshold)
    assert np.isfinite(t2.threshold)
    assert np.isfinite(t3.threshold)
