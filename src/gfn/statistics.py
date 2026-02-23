"""Statistical tests: Mann-Whitney U, Cliff's delta."""
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    dominates = 0
    for xi in x:
        dominates += np.sum(y < xi) - np.sum(y > xi)
    return dominates / (n_x * n_y + 1e-10)


class StatisticalTests:
    """Hypothesis testing and effect sizes."""

    @staticmethod
    def mann_whitney_u_test(real_values: np.ndarray, fake_values: np.ndarray) -> Tuple[float, float]:
        stat, p_value = mannwhitneyu(real_values, fake_values, alternative="two-sided")
        return float(stat), float(p_value)

    @staticmethod
    def cliffs_delta(real_values: np.ndarray, fake_values: np.ndarray) -> float:
        return float(_cliffs_delta(np.asarray(real_values), np.asarray(fake_values)))

    @staticmethod
    def compute_all_tests(scores_df: pd.DataFrame, score_cols: list) -> pd.DataFrame:
        rows = []
        for col in score_cols:
            if col not in scores_df.columns:
                continue
            real = scores_df[scores_df["label"] == 0][col].dropna().values
            fake = scores_df[scores_df["label"] == 1][col].dropna().values
            if len(real) < 2 or len(fake) < 2:
                continue
            stat, p_val = StatisticalTests.mann_whitney_u_test(real, fake)
            delta = StatisticalTests.cliffs_delta(real, fake)
            rows.append({
                "feature": col,
                "mann_whitney_u": stat,
                "mann_whitney_p": p_val,
                "cliffs_delta": delta,
                "significant": "yes" if p_val < 0.05 else "no",
            })
        return pd.DataFrame(rows)
