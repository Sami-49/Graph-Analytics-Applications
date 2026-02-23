"""Evaluation protocol: hold-out, CV, bootstrap, cross-dataset.

Scientific correctness requirements:
- AUC must be computed from continuous scores (probabilities / decision function / structural scores).
- If AUC<0.5 for a score-only method, flip orientation (s := -s) and record the flip.
- If scores are constant / near-constant, stop with a clear error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from .thresholding import (
    ThresholdChoice,
    choose_threshold_cost_sensitive,
    choose_threshold_max_f1,
    choose_threshold_youden_j,
)


@dataclass(frozen=True)
class ScoreOnlyResult:
    auc: float
    auc_ci_low: float
    auc_ci_high: float
    f1: float
    f1_ci_low: float
    f1_ci_high: float
    threshold: float
    flipped: bool
    diagnostic: Dict[str, Any]


@dataclass(frozen=True)
class ScoreOnlyFullResult:
    ok: bool
    n: int
    n_pos: int
    n_neg: int

    auc: Optional[float]
    prauc: Optional[float]
    auc_ci_low: Optional[float]
    auc_ci_high: Optional[float]
    prauc_ci_low: Optional[float]
    prauc_ci_high: Optional[float]

    threshold_max_f1: Optional[float]
    threshold_youden: Optional[float]
    threshold_cost: Optional[float]

    metrics_max_f1: Dict[str, float]
    metrics_youden: Dict[str, float]
    metrics_cost: Dict[str, float]

    flipped: bool
    flip_reason: str
    diagnostic: Dict[str, Any]
    curves: Dict[str, Any]


def _check_not_constant_scores(scores: np.ndarray, *, name: str) -> None:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError(f"Score '{name}' has no finite values.")
    if float(np.std(s)) < 1e-10:
        raise ValueError(
            f"Score '{name}' is constant / near-constant (std<1e-10). "
            "Likely causes: feature fallback returning zeros, parsing failures, or a normalization bug."
        )


def _auc_guard_against_hard_labels(y_score: np.ndarray, *, context: str) -> None:
    ys = np.asarray(y_score)
    uniq = np.unique(ys[~pd.isna(ys)])
    # AUC on hard labels is a common scientific bug: it can yield 0.5 with high F1.
    if uniq.size <= 2 and set(map(float, uniq.tolist())).issubset({0.0, 1.0}):
        raise ValueError(
            f"Invalid AUC input in {context}: y_score appears to be hard labels {{0,1}}. "
            "Use predict_proba[:,1] or decision_function (continuous scores)."
        )


def _best_threshold_by_youden(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thr[i])


def _confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn = float(tn)
    fp = float(fp)
    fn = float(fn)
    tp = float(tp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0
    kappa = float(cohen_kappa_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "npv": float(npv),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "mcc": mcc,
        "kappa": kappa,
        "f1": f1,
    }


def evaluate_score_only_full(
    *,
    y_true: np.ndarray,
    scores: np.ndarray,
    name: str,
    n_bootstrap: int,
    ci_alpha: float,
    rng: Optional[np.random.Generator] = None,
    allow_flip_if_auc_below_half: bool = True,
    fail_on_constant_score: bool = True,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
    compute_brier: bool = False,
) -> ScoreOnlyFullResult:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(s) & np.isin(y, [0, 1])
    y = y[mask]
    s = s[mask]

    rng = rng or np.random.default_rng(0)
    diag: Dict[str, Any] = {"name": name, "n": int(y.size), "pos_rate": float(np.mean(y)) if y.size else 0.0}
    curves: Dict[str, Any] = {}

    if y.size < 3:
        return ScoreOnlyFullResult(
            ok=False,
            n=int(y.size),
            n_pos=int((y == 1).sum()),
            n_neg=int((y == 0).sum()),
            auc=None,
            prauc=None,
            auc_ci_low=None,
            auc_ci_high=None,
            prauc_ci_low=None,
            prauc_ci_high=None,
            threshold_max_f1=None,
            threshold_youden=None,
            threshold_cost=None,
            metrics_max_f1={},
            metrics_youden={},
            metrics_cost={},
            flipped=False,
            flip_reason="not_enough_data",
            diagnostic=diag,
            curves={},
        )

    if float(np.std(s)) < 1e-10:
        msg = f"Score '{name}' is constant / near-constant (std<1e-10)."
        if fail_on_constant_score:
            raise ValueError(msg)
        return ScoreOnlyFullResult(
            ok=False,
            n=int(y.size),
            n_pos=int((y == 1).sum()),
            n_neg=int((y == 0).sum()),
            auc=None,
            prauc=None,
            auc_ci_low=None,
            auc_ci_high=None,
            prauc_ci_low=None,
            prauc_ci_high=None,
            threshold_max_f1=None,
            threshold_youden=None,
            threshold_cost=None,
            metrics_max_f1={},
            metrics_youden={},
            metrics_cost={},
            flipped=False,
            flip_reason="constant_score",
            diagnostic={**diag, "error": msg},
            curves={},
        )

    flipped = False
    flip_reason = "none"

    if len(np.unique(y)) < 2:
        # AUC/AP undefined; compute thresholds/metrics anyway.
        auc = None
        prauc = None
        auc_ci = (None, None)
        prauc_ci = (None, None)
    else:
        auc_raw = float(roc_auc_score(y, s))
        s_used = s
        if allow_flip_if_auc_below_half and auc_raw < 0.5:
            flipped = True
            flip_reason = "auc_below_half_flip_score_sign"
            s_used = -s
        auc = float(roc_auc_score(y, s_used))
        prauc = float(average_precision_score(y, s_used))

        fpr, tpr, roc_thr = roc_curve(y, s_used)
        prec, rec, pr_thr = precision_recall_curve(y, s_used)
        curves = {
            "roc": {"fpr": fpr, "tpr": tpr, "thresholds": roc_thr},
            "pr": {"precision": prec, "recall": rec, "thresholds": pr_thr},
        }

        # Bootstrap CIs for AUC/AP
        auc_samples: list[float] = []
        pr_samples: list[float] = []
        for _ in range(int(n_bootstrap)):
            idx = rng.integers(0, y.size, size=y.size)
            yb = y[idx]
            sb = s_used[idx]
            if len(np.unique(yb)) < 2:
                continue
            auc_samples.append(float(roc_auc_score(yb, sb)))
            pr_samples.append(float(average_precision_score(yb, sb)))

        alpha = float(ci_alpha)
        if auc_samples:
            auc_ci = (
                float(np.percentile(auc_samples, 100 * alpha / 2)),
                float(np.percentile(auc_samples, 100 * (1 - alpha / 2))),
            )
        else:
            auc_ci = (None, None)

        if pr_samples:
            prauc_ci = (
                float(np.percentile(pr_samples, 100 * alpha / 2)),
                float(np.percentile(pr_samples, 100 * (1 - alpha / 2))),
            )
        else:
            prauc_ci = (None, None)

    # Thresholds (always computed, even if single-class)
    thr_maxf1: ThresholdChoice = choose_threshold_max_f1(y_true=y, scores=(-s if flipped else s))
    thr_youden: ThresholdChoice = choose_threshold_youden_j(y_true=y, scores=(-s if flipped else s))
    thr_cost: ThresholdChoice = choose_threshold_cost_sensitive(
        y_true=y,
        scores=(-s if flipped else s),
        cost_fp=float(cost_fp),
        cost_fn=float(cost_fn),
    )

    def metrics_at(thr: float) -> Dict[str, float]:
        s_eval = (-s if flipped else s)
        pred = (s_eval >= thr).astype(int)
        m = _confusion_metrics(y, pred)
        if compute_brier:
            if float(np.nanmin(s_eval)) >= 0.0 and float(np.nanmax(s_eval)) <= 1.0:
                m["brier"] = float(brier_score_loss(y, s_eval))
            else:
                m["brier"] = float("nan")
        else:
            m["brier"] = float("nan")
        return m

    m_maxf1 = metrics_at(float(thr_maxf1.threshold))
    m_youden = metrics_at(float(thr_youden.threshold))
    m_cost = metrics_at(float(thr_cost.threshold))

    return ScoreOnlyFullResult(
        ok=True,
        n=int(y.size),
        n_pos=int((y == 1).sum()),
        n_neg=int((y == 0).sum()),
        auc=auc,
        prauc=prauc,
        auc_ci_low=(auc_ci[0] if "auc_ci" in locals() else None),
        auc_ci_high=(auc_ci[1] if "auc_ci" in locals() else None),
        prauc_ci_low=(prauc_ci[0] if "prauc_ci" in locals() else None),
        prauc_ci_high=(prauc_ci[1] if "prauc_ci" in locals() else None),
        threshold_max_f1=float(thr_maxf1.threshold) if np.isfinite(thr_maxf1.threshold) else None,
        threshold_youden=float(thr_youden.threshold) if np.isfinite(thr_youden.threshold) else None,
        threshold_cost=float(thr_cost.threshold) if np.isfinite(thr_cost.threshold) else None,
        metrics_max_f1=m_maxf1,
        metrics_youden=m_youden,
        metrics_cost=m_cost,
        flipped=bool(flipped),
        flip_reason=str(flip_reason),
        diagnostic=diag,
        curves=curves,
    )


def score_only_evaluation(
    *,
    y_true: np.ndarray,
    scores: np.ndarray,
    name: str,
    n_bootstrap: int,
    ci_alpha: float,
    threshold_method: str = "youden",
    rng: Optional[np.random.Generator] = None,
) -> ScoreOnlyResult:
    """Evaluate a continuous structural score (no classifier).

    Returns AUC + F1 (using a learned threshold on the same set) with bootstrap CIs.
    If AUC<0.5 the score is flipped (scores := -scores) and this is reported.
    """

    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(s) & np.isin(y, [0, 1])
    y = y[mask]
    s = s[mask]

    if y.size < 3:
        raise ValueError(f"Not enough labeled examples to evaluate score '{name}'.")
    if len(np.unique(y)) < 2:
        # Undefined ROC/AUC. Keep deterministic neutral AUC.
        return ScoreOnlyResult(
            auc=0.5,
            auc_ci_low=0.5,
            auc_ci_high=0.5,
            f1=0.0,
            f1_ci_low=0.0,
            f1_ci_high=0.0,
            threshold=float("nan"),
            flipped=False,
            diagnostic={"reason": "single_class"},
        )

    _check_not_constant_scores(s, name=name)

    auc_raw = float(roc_auc_score(y, s))
    flipped = False
    s_used = s
    if auc_raw < 0.5:
        flipped = True
        s_used = -s
    auc = float(roc_auc_score(y, s_used))

    if threshold_method == "youden":
        thr = _best_threshold_by_youden(y, s_used)
    else:
        thr = _best_threshold_by_youden(y, s_used)

    pred = (s_used >= thr).astype(int)
    f1 = float(f1_score(y, pred, zero_division=0))

    # Bootstrap CIs
    rng = rng or np.random.default_rng(0)
    f1_samples: list[float] = []
    auc_samples: list[float] = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, y.size, size=y.size)
        yb = y[idx]
        sb = s_used[idx]
        if len(np.unique(yb)) < 2:
            f1_samples.append(0.0)
            auc_samples.append(0.5)
            continue
        thr_b = _best_threshold_by_youden(yb, sb)
        pb = (sb >= thr_b).astype(int)
        f1_samples.append(float(f1_score(yb, pb, zero_division=0)))
        auc_samples.append(float(roc_auc_score(yb, sb)))

    alpha = float(ci_alpha)
    f1_ci_low = float(np.percentile(f1_samples, 100 * alpha / 2))
    f1_ci_high = float(np.percentile(f1_samples, 100 * (1 - alpha / 2)))
    auc_ci_low = float(np.percentile(auc_samples, 100 * alpha / 2))
    auc_ci_high = float(np.percentile(auc_samples, 100 * (1 - alpha / 2)))

    diagnostic: Dict[str, Any] = {
        "auc_raw": auc_raw,
        "auc_used": auc,
        "threshold": thr,
        "threshold_method": threshold_method,
        "n": int(y.size),
        "pos_rate": float(np.mean(y)),
    }
    return ScoreOnlyResult(
        auc=auc,
        auc_ci_low=auc_ci_low,
        auc_ci_high=auc_ci_high,
        f1=f1,
        f1_ci_low=f1_ci_low,
        f1_ci_high=f1_ci_high,
        threshold=thr,
        flipped=flipped,
        diagnostic=diagnostic,
    )


def validate_metric_consistency(*, auc: float, f1: float, context: str) -> Dict[str, Any]:
    """Detect a classic inconsistency: AUC ~ 0.5 but high F1.

    This usually indicates: AUC computed from hard labels, heavy class imbalance,
    or thresholding artifacts.
    """

    diag: Dict[str, Any] = {"context": context, "auc": float(auc), "f1": float(f1)}
    if abs(float(auc) - 0.5) < 0.03 and float(f1) > 0.70:
        diag["warning"] = "AUC≈0.5 but F1 is high: check AUC input (hard-label bug), class imbalance, or thresholding."
    return diag


class Evaluator:
    """Comprehensive evaluation protocol."""

    @staticmethod
    def holdout_evaluation(
        X_train,
        X_test,
        y_train,
        y_test,
        model,
        config,
    ) -> Dict[str, float]:
        """
        Hold-out evaluation with metrics and bootstrapped CI.
        """
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Continuous scores for ROC/AUC (scientific correctness)
        y_score: Optional[np.ndarray] = None
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)

        if y_score is None:
            raise ValueError(
                "Model does not expose predict_proba or decision_function; cannot compute AUC scientifically."
            )

        _auc_guard_against_hard_labels(y_score, context="Evaluator.holdout_evaluation")

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_score) if len(np.unique(y_test)) > 1 else 0.5

        # Bootstrap CI for F1 and AUC
        n_bootstrap = config.evaluation.n_bootstrap
        f1_samples = []
        auc_samples = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(len(y_test), size=len(y_test), replace=True)
            y_test_boot = y_test[idx]
            y_pred_boot = y_pred[idx]
            y_score_boot = y_score[idx]

            if len(np.unique(y_test_boot)) > 1:
                f1_boot = f1_score(y_test_boot, y_pred_boot, zero_division=0)
                auc_boot = roc_auc_score(y_test_boot, y_score_boot)
            else:
                f1_boot = 0.0
                auc_boot = 0.5

            f1_samples.append(f1_boot)
            auc_samples.append(auc_boot)

        alpha = config.evaluation.ci_alpha
        f1_ci_lower = np.percentile(f1_samples, 100 * alpha / 2)
        f1_ci_upper = np.percentile(f1_samples, 100 * (1 - alpha / 2))
        auc_ci_lower = np.percentile(auc_samples, 100 * alpha / 2)
        auc_ci_upper = np.percentile(auc_samples, 100 * (1 - alpha / 2))

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "f1_ci_lower": f1_ci_lower,
            "f1_ci_upper": f1_ci_upper,
            "auc_ci_lower": auc_ci_lower,
            "auc_ci_upper": auc_ci_upper,
        }

    @staticmethod
    def cross_validation(X, y, model, config) -> Dict[str, float]:
        """5-fold stratified cross-validation."""
        skf = StratifiedKFold(
            n_splits=config.evaluation.n_folds,
            shuffle=True,
            random_state=config.evaluation.random_state,
        )

        f1_scores = []
        auc_scores = []

        for train_idx, test_idx in skf.split(X, y):
            X_train_fold = X[train_idx]
            X_test_fold = X[test_idx]
            y_train_fold = y[train_idx]
            y_test_fold = y[test_idx]

            model.fit(X_train_fold, y_train_fold)
            y_pred_fold = model.predict(X_test_fold)

            y_score_fold: Optional[np.ndarray] = None
            if hasattr(model, "predict_proba"):
                y_score_fold = model.predict_proba(X_test_fold)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score_fold = model.decision_function(X_test_fold)

            if y_score_fold is None:
                raise ValueError("Model does not expose predict_proba/decision_function; cannot compute AUC.")
            _auc_guard_against_hard_labels(y_score_fold, context="Evaluator.cross_validation")

            f1 = f1_score(y_test_fold, y_pred_fold, zero_division=0)
            auc = roc_auc_score(y_test_fold, y_score_fold) if len(np.unique(y_test_fold)) > 1 else 0.5

            f1_scores.append(f1)
            auc_scores.append(auc)

        return {
            "f1_mean": np.mean(f1_scores),
            "f1_std": np.std(f1_scores),
            "auc_mean": np.mean(auc_scores),
            "auc_std": np.std(auc_scores),
        }

    @staticmethod
    def evaluate_all_models(
        X_train,
        X_test,
        y_train,
        y_test,
        models: Dict,
        config,
    ) -> pd.DataFrame:
        """Evaluate all models on hold-out set."""
        results = []

        for model_name, model in models.items():
            metrics = Evaluator.holdout_evaluation(
                X_train, X_test, y_train, y_test, model, config
            )
            metrics["model"] = model_name
            results.append(metrics)

        return pd.DataFrame(results)
