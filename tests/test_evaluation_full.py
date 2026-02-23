import numpy as np
import pytest

from src.gfn.evaluation import evaluate_score_only_full


def test_evaluate_score_only_full_single_class_no_crash():
    y = np.array([0, 0, 0, 0, 0])
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    res = evaluate_score_only_full(
        y_true=y,
        scores=s,
        name="single_class",
        n_bootstrap=50,
        ci_alpha=0.05,
        fail_on_constant_score=True,
    )

    assert res.ok is True
    assert res.auc is None
    assert res.prauc is None
    assert res.threshold_max_f1 is not None


def test_evaluate_score_only_full_constant_score_fail_fast():
    y = np.array([0, 1, 0, 1, 0, 1])
    s = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    with pytest.raises(ValueError):
        evaluate_score_only_full(
            y_true=y,
            scores=s,
            name="constant",
            n_bootstrap=10,
            ci_alpha=0.05,
            fail_on_constant_score=True,
        )
