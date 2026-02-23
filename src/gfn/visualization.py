"""Visualization: ROC, model comparison, ablation, feature distributions, importance."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from pathlib import Path


class Visualizer:
    """Generate publication-quality figures."""

    @staticmethod
    def plot_roc_curves(
        y_test: np.ndarray,
        y_pred_proba_dict: dict,
        output_dir: Path,
    ) -> None:
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))

        for model_name, y_proba in y_pred_proba_dict.items():
            if len(np.unique(y_test)) > 1:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})", linewidth=2)

        plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curves - All Models", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_dir / "roc_all.png", dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_model_comparison(results_df: pd.DataFrame, output_dir: Path) -> None:
        """Barplot: F1 and AUC comparison across models with 95% CI."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        df = results_df.sort_values("f1", ascending=False)

        ax = axes[0]
        x = np.arange(len(df))
        ax.bar(x, df["f1"], color="skyblue", edgecolor="black")
        if "f1_ci_lower" in df.columns and "f1_ci_upper" in df.columns:
            err_lo = df["f1"] - df["f1_ci_lower"]
            err_hi = df["f1_ci_upper"] - df["f1"]
            ax.errorbar(x, df["f1"], yerr=[err_lo, err_hi], fmt="none", color="black", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(df["model"], rotation=45, ha="right")
        ax.set_ylabel("F1 Score")
        ax.set_title("Model Comparison - F1 (with 95% CI)")
        ax.set_ylim([0, 1.1])

        ax = axes[1]
        df2 = results_df.sort_values("roc_auc", ascending=False)
        x2 = np.arange(len(df2))
        ax.bar(x2, df2["roc_auc"], color="lightcoral", edgecolor="black")
        if "auc_ci_lower" in df2.columns and "auc_ci_upper" in df2.columns:
            err_lo = df2["roc_auc"] - df2["auc_ci_lower"]
            err_hi = df2["auc_ci_upper"] - df2["roc_auc"]
            ax.errorbar(x2, df2["roc_auc"], yerr=[err_lo, err_hi], fmt="none", color="black", capsize=3)
        ax.set_xticks(x2)
        ax.set_xticklabels(df2["model"], rotation=45, ha="right")
        ax.set_ylabel("ROC-AUC")
        ax.set_title("Model Comparison - ROC-AUC (with 95% CI)")
        ax.set_ylim([0, 1.1])

        output_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_casdi_vs_others(results_df: pd.DataFrame, output_dir: Path) -> None:
        """CASDI improvement plot vs best baseline."""
        if "casdi" not in results_df["model"].values:
            return

        casdi_f1 = results_df[results_df["model"] == "casdi"]["f1"].values[0]
        casdi_auc = results_df[results_df["model"] == "casdi"]["roc_auc"].values[0]

        baseline_models = results_df[results_df["model"] != "casdi"]
        best_baseline_f1 = baseline_models["f1"].max()
        best_baseline_auc = baseline_models["roc_auc"].max()

        improvement_f1 = ((casdi_f1 - best_baseline_f1) / best_baseline_f1 * 100) if best_baseline_f1 > 0 else 0
        improvement_auc = ((casdi_auc - best_baseline_auc) / best_baseline_auc * 100) if best_baseline_auc > 0 else 0

        fig, ax = plt.subplots(figsize=(10, 6))
        improvements = [improvement_f1, improvement_auc]
        metrics = ["F1 Improvement (%)", "AUC Improvement (%)"]
        colors = ["green" if x >= 0 else "red" for x in improvements]

        ax.bar(metrics, improvements, color=colors, alpha=0.7, edgecolor="black", linewidth=2)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax.set_ylabel("Improvement (%)", fontsize=12)
        ax.set_title("CASDI vs Best Baseline - Performance Gain", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for i, v in enumerate(improvements):
            ax.text(i, v + 1, f"{v:.2f}%", ha="center", fontweight="bold")

        output_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_dir / "casdi_improvement.png", dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_feature_distributions(
        features_df: pd.DataFrame,
        feature_names: list,
        output_dir: Path,
    ) -> None:
        """Distribution plots for key features: real vs fake."""
        n_features = min(6, len(feature_names))
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        for idx, feature_name in enumerate(feature_names[:n_features]):
            if feature_name not in features_df.columns:
                continue

            ax = axes[idx]
            real = features_df[features_df["label"] == 0][feature_name]
            fake = features_df[features_df["label"] == 1][feature_name]

            ax.hist(real, bins=20, alpha=0.6, label="Real", color="blue", edgecolor="black")
            ax.hist(fake, bins=20, alpha=0.6, label="Fake", color="red", edgecolor="black")
            ax.set_xlabel(feature_name, fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.set_title(f"Distribution: {feature_name}", fontsize=11, fontweight="bold")
            ax.legend()
            ax.grid(alpha=0.3)

        output_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_dir / "feature_distributions.png", dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_feature_importance(importance_dict: dict, output_dir: Path) -> None:
        """Feature importance plot from permutation."""
        if not importance_dict:
            return

        # Sort and plot top features
        importance_df = pd.DataFrame(list(importance_dict.items()), columns=["feature", "importance"])
        importance_df = importance_df.sort_values("importance", ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(importance_df["feature"], importance_df["importance"], color="steelblue", edgecolor="black")
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_title("Top 20 Feature Importance (Permutation-Based)", fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        output_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()
