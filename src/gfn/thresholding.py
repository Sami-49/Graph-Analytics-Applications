from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    strategy: str
    details: Dict[str, float]


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def choose_threshold_max_f1(
    *,
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    min_specificity: float = 0.05,
) -> ThresholdChoice:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)

    if thresholds is None:
        thresholds = np.unique(s[np.isfinite(s)])
        if thresholds.size == 0:
            return ThresholdChoice(threshold=float("nan"), strategy="max_f1", details={"f1": 0.0})

    best_t = float(thresholds[0])
    best_f1 = -1.0
    best_bacc = -1.0

    # Keep an unconstrained fallback in case all thresholds violate constraints.
    fb_t = float(thresholds[0])
    fb_f1 = -1.0

    for t in thresholds:
        pred = (s >= t).astype(int)
        tn, fp, fn, tp = _confusion_counts(y, pred)
        denom = (2 * tp + fp + fn)
        f1 = float((2 * tp) / denom) if denom > 0 else 0.0

        # unconstrained best (fallback)
        if f1 > fb_f1:
            fb_f1 = f1
            fb_t = float(t)

        # constrained selection to avoid degenerate all-positive thresholds
        specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        bacc = 0.5 * (specificity + recall)

        if float(specificity) < float(min_specificity):
            continue

        if (f1 > best_f1) or (abs(f1 - best_f1) < 1e-12 and bacc > best_bacc):
            best_f1 = f1
            best_bacc = float(bacc)
            best_t = float(t)

    if best_f1 < 0:
        best_t = fb_t
        best_f1 = fb_f1

    return ThresholdChoice(threshold=best_t, strategy="max_f1", details={"f1": float(best_f1)})


def choose_threshold_youden_j(
    *,
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> ThresholdChoice:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)

    if thresholds is None:
        thresholds = np.unique(s[np.isfinite(s)])
        if thresholds.size == 0:
            return ThresholdChoice(threshold=float("nan"), strategy="youden_j", details={"J": 0.0})

    best_t = float(thresholds[0])
    best_j = -np.inf

    for t in thresholds:
        pred = (s >= t).astype(int)
        tn, fp, fn, tp = _confusion_counts(y, pred)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        j = float(tpr - fpr)
        if j > best_j:
            best_j = j
            best_t = float(t)

    return ThresholdChoice(threshold=best_t, strategy="youden_j", details={"J": float(best_j)})


def choose_threshold_cost_sensitive(
    *,
    y_true: np.ndarray,
    scores: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
    thresholds: Optional[np.ndarray] = None,
) -> ThresholdChoice:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)

    if thresholds is None:
        thresholds = np.unique(s[np.isfinite(s)])
        if thresholds.size == 0:
            return ThresholdChoice(
                threshold=float("nan"),
                strategy="cost_sensitive",
                details={"cost": float("inf"), "cost_fp": float(cost_fp), "cost_fn": float(cost_fn)},
            )

    best_t = float(thresholds[0])
    best_cost = float("inf")

    for t in thresholds:
        pred = (s >= t).astype(int)
        tn, fp, fn, tp = _confusion_counts(y, pred)
        cost = float(cost_fp * fp + cost_fn * fn)
        if cost < best_cost:
            best_cost = cost
            best_t = float(t)

    return ThresholdChoice(
        threshold=best_t,
        strategy="cost_sensitive",
        details={"cost": float(best_cost), "cost_fp": float(cost_fp), "cost_fn": float(cost_fn)},
    )
