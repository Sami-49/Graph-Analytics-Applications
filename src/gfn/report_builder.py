from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CuratedOutputs:
    tables: Dict[str, Path]
    figures: Dict[str, Path]
    reports: Dict[str, Path]


class ReportBuilder:
    CURATED_TABLES = {
        "dataset_summary.csv",
        "features_clean.csv",
        "univariate_scores.csv",
        "casdi_comparison.csv",
        "generalization.csv",
        "performance.csv",
    }

    CURATED_FIGURES = {
        "fig_pipeline.png",
        "fig_feature_distributions.png",
        "fig_effect_size.png",
        "fig_corr_heatmap.png",
        "fig_univariate_auc.png",
        "fig_roc_best.png",
        "fig_pr_best.png",
        "fig_casdi_ablation_v1.png",
        "fig_casdi_ablation_v2.png",
        "fig_cross_dataset.png",
    }

    CURATED_REPORTS = {
        "summary.md",
        "summary.pdf",
        "audit.md",
    }

    MAIN_FEATURES = [
        "kcore_score",
        "centralization_score",
        "virality_score",
        "community_score",
        "hps_score",
        "spectral_score",
        "casdi_score",
        "casdi_v2_score",
    ]

    @staticmethod
    def validate_features(df: pd.DataFrame, logger) -> None:
        if "spectral_score" in df.columns:
            s = pd.to_numeric(df["spectral_score"], errors="coerce")
            if float(s.std(skipna=True)) < 1e-10:
                raise ValueError(
                    "spectral_score is constant. Likely causes: spectral features returning zeros for all graphs "
                    "(small graphs / failed parsing / LCC extraction) or scoring normalization bug."
                )

        if "community_score" in df.columns and "hps_score" in df.columns:
            a = pd.to_numeric(df["hps_score"], errors="coerce")
            b = pd.to_numeric(df["community_score"], errors="coerce")
            corr = float(a.corr(b)) if a.notna().sum() > 5 and b.notna().sum() > 5 else 0.0
            if corr > 0.95:
                logger.warning(f"hps_score is highly correlated with community_score (corr={corr:.3f}).")

    @staticmethod
    def write_tables(
        *,
        tables_dir: Path,
        dataset_summary: pd.DataFrame,
        features_clean: pd.DataFrame,
        univariate_scores: pd.DataFrame,
        casdi_comparison: pd.DataFrame,
        generalization: pd.DataFrame,
        performance: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Path]:
        tables_dir.mkdir(parents=True, exist_ok=True)

        out: Dict[str, Path] = {
            "dataset_summary.csv": tables_dir / "dataset_summary.csv",
            "features_clean.csv": tables_dir / "features_clean.csv",
            "univariate_scores.csv": tables_dir / "univariate_scores.csv",
            "casdi_comparison.csv": tables_dir / "casdi_comparison.csv",
            "generalization.csv": tables_dir / "generalization.csv",
        }

        dataset_summary.to_csv(out["dataset_summary.csv"], index=False)
        features_clean.to_csv(out["features_clean.csv"], index=False)
        univariate_scores.to_csv(out["univariate_scores.csv"], index=False)
        casdi_comparison.to_csv(out["casdi_comparison.csv"], index=False)
        generalization.to_csv(out["generalization.csv"], index=False)

        if performance is not None and not performance.empty:
            out["performance.csv"] = tables_dir / "performance.csv"
            performance.to_csv(out["performance.csv"], index=False)
        return out

    @staticmethod
    def _save_fig(path: Path, dpi: int = 250) -> None:
        import matplotlib.pyplot as plt

        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=max(200, dpi), bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_feature_distributions(figures_dir: Path, features_clean: pd.DataFrame) -> Path:
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = features_clean.copy()
        if df.empty:
            plt.figure(figsize=(9, 4))
            plt.title("Feature distributions")
            plt.text(0.5, 0.5, "No data", ha="center", va="center")
            out = figures_dir / "fig_feature_distributions.png"
            ReportBuilder._save_fig(out, dpi=250)
            return out

        df = df[df["label"].isin([0, 1])].copy()
        df["label"] = df["label"].map({0: "real", 1: "fake"}).fillna(df["label"].astype(str))

        feats = [c for c in [
            "kcore_score",
            "centralization_score",
            "virality_score",
            "community_score",
            "hps_score",
            "spectral_score",
            "casdi_score",
            "casdi_v2_score",
        ] if c in df.columns]

        n = len(feats)
        if n == 0:
            plt.figure(figsize=(9, 4))
            plt.title("Feature distributions")
            plt.text(0.5, 0.5, "No numeric features available", ha="center", va="center")
            out = figures_dir / "fig_feature_distributions.png"
            ReportBuilder._save_fig(out, dpi=250)
            return out

        ncols = 4
        nrows = int(np.ceil(n / ncols))
        plt.figure(figsize=(4.8 * ncols, 3.6 * nrows))
        for i, col in enumerate(feats, start=1):
            ax = plt.subplot(nrows, ncols, i)
            sns.violinplot(data=df, x="label", y=col, inner=None, cut=0, ax=ax, palette="Set2")
            sns.boxplot(data=df, x="label", y=col, width=0.25, ax=ax, color="white", fliersize=1)
            ax.set_title(col)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(axis="y", alpha=0.2)

        out = figures_dir / "fig_feature_distributions.png"
        ReportBuilder._save_fig(out, dpi=250)
        return out

    @staticmethod
    def plot_pipeline(figures_dir: Path) -> Path:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        boxes = [
            (5, 9, "Dataset (Twitter15/16)"),
            (5, 7.5, "Load cascades + labels"),
            (5, 6, "Build graphs"),
            (5, 4.5, "Compute graph scores\n(baselines + CASDI v1/v2)"),
            (5, 3, "Score-only evaluation\n(ROC/AUC, PR-AUC, F1)"),
            (5, 1.6, "Curated outputs\n(tables, figures, report)"),
        ]
        for x, y, txt in boxes:
            rect = mpatches.FancyBboxPatch(
                (x - 1.7, y - 0.35),
                3.4,
                0.7,
                boxstyle="round,pad=0.08",
                facecolor="#D6EAF8",
                edgecolor="black",
            )
            ax.add_patch(rect)
            ax.text(x, y, txt, ha="center", va="center", fontsize=10)

        for i in range(len(boxes) - 1):
            ax.annotate(
                "",
                xy=(5, boxes[i + 1][1] + 0.35),
                xytext=(5, boxes[i][1] - 0.35),
                arrowprops=dict(arrowstyle="->", color="black"),
            )

        ax.set_title("Pipeline overview", fontweight="bold")
        out = figures_dir / "fig_pipeline.png"
        ReportBuilder._save_fig(out, dpi=250)
        return out

    @staticmethod
    def plot_effect_size(figures_dir: Path, stats_tests: pd.DataFrame) -> Path:
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = stats_tests.copy()
        if df.empty:
            plt.figure(figsize=(8, 4))
            plt.title("Effect sizes (Cliff's delta)")
            out = figures_dir / "fig_effect_size.png"
            ReportBuilder._save_fig(out, dpi=250)
            return out

        df["significant"] = df["p_value"].astype(float) < 0.05

        plt.figure(figsize=(10, 4 + 0.35 * df["feature"].nunique()))
        ax = sns.barplot(data=df, x="cliffs_delta", y="feature", hue="dataset", orient="h")
        ax.axvline(0, color="black", lw=1)
        ax.set_title("Effect size by feature (Cliff's delta)")
        ax.set_xlabel("Cliff's delta")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.2)

        # significance markers (p<0.05)
        # bar order matches df rows in seaborn; use patch index.
        df_plot = df.copy().reset_index(drop=True)
        for i, patch in enumerate(ax.patches):
            if i >= len(df_plot):
                break
            if not bool(df_plot.loc[i, "significant"]):
                continue
            width = float(patch.get_width())
            y = float(patch.get_y() + patch.get_height() / 2)
            x = width + (0.02 if width >= 0 else -0.02)
            ax.text(x, y, "*", va="center", ha="left" if width >= 0 else "right", fontsize=11, weight="bold")

        out = figures_dir / "fig_effect_size.png"
        ReportBuilder._save_fig(out, dpi=250)
        return out

    @staticmethod
    def plot_corr_heatmap(figures_dir: Path, features_main: pd.DataFrame) -> Path:
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = features_main.copy()
        feats = [c for c in ReportBuilder.MAIN_FEATURES if c in df.columns]

        datasets = sorted(df["dataset"].dropna().unique().tolist()) if "dataset" in df.columns else []
        if not datasets:
            datasets = ["all"]
            df["dataset"] = "all"

        ncols = len(datasets)
        fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 6))
        if ncols == 1:
            axes = [axes]

        for ax, d in zip(axes, datasets):
            sub = df[df["dataset"] == d]
            corr = sub[feats].corr() if len(sub) > 3 else pd.DataFrame(np.eye(len(feats)), index=feats, columns=feats)
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, ax=ax)
            ax.set_title(str(d))

        fig.suptitle("Correlation heatmap (main features)", fontweight="bold")
        out = figures_dir / "fig_corr_heatmap.png"
        ReportBuilder._save_fig(out, dpi=250)
        return out

    @staticmethod
    def plot_univariate_auc(figures_dir: Path, univariate_scores: pd.DataFrame) -> Path:
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = univariate_scores.copy() if univariate_scores is not None else pd.DataFrame()
        plt.figure(figsize=(10, 5.5))
        if df.empty or "feature" not in df.columns:
            plt.title("Univariate AUC/PR-AUC")
            plt.text(0.5, 0.5, "No univariate scores", ha="center", va="center")
            out = figures_dir / "fig_univariate_auc.png"
            ReportBuilder._save_fig(out, dpi=250)
            return out

        df = df.sort_values("auc", ascending=False).head(12)
        df_plot = df.melt(id_vars=["feature"], value_vars=[c for c in ["auc", "prauc"] if c in df.columns], var_name="metric", value_name="value")
        sns.barplot(data=df_plot, y="feature", x="value", hue="metric")
        plt.xlim(0.0, 1.0)
        plt.title("Univariate ranking (AUC + PR-AUC)")
        plt.xlabel("score")
        plt.ylabel("")
        plt.grid(axis="x", alpha=0.2)
        out = figures_dir / "fig_univariate_auc.png"
        ReportBuilder._save_fig(out, dpi=250)
        return out

    @staticmethod
    def plot_best_pr(figures_dir: Path, best: Mapping[str, object]) -> Path:
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, average_precision_score

        y_true = np.asarray(best.get("y_true", []), dtype=int)
        y_score = np.asarray(best.get("y_score", []), dtype=float)

        plt.figure(figsize=(6.5, 5.5))
        if y_true.size > 0 and len(np.unique(y_true)) > 1:
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            ap = float(average_precision_score(y_true, y_score))
            plt.plot(rec, prec, lw=2, label=f"AP={ap:.3f}")
        else:
            plt.text(0.5, 0.5, "No labeled data", ha="center", va="center")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR curve (best score)")
        plt.legend(loc="lower left")
        plt.grid(alpha=0.2)

        out = figures_dir / "fig_pr_best.png"
        ReportBuilder._save_fig(out, dpi=250)
        return out

    @staticmethod
    def plot_casdi_ablation_v1(figures_dir: Path, ablation_df: pd.DataFrame) -> Path:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10.5, 4.8))
        df = ablation_df.copy() if ablation_df is not None else pd.DataFrame()
        if df.empty:
            plt.title("CASDI v1 ablation")
            plt.text(0.5, 0.5, "No ablation results", ha="center", va="center")
        else:
            df = df.sort_values("auc", ascending=False)
            sns.barplot(data=df, x="ablation", y="auc", color="#54A24B")
            plt.xticks(rotation=30, ha="right")
            plt.ylim(0.0, 1.0)
            plt.title("CASDI v1 ablation (AUC)")
            plt.ylabel("AUC")
            plt.grid(axis="y", alpha=0.2)

        out = figures_dir / "fig_casdi_ablation_v1.png"
        ReportBuilder._save_fig(out, dpi=250)
        return out

    @staticmethod
    def plot_casdi_ablation_v2(figures_dir: Path, ablation_df: pd.DataFrame) -> Path:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10.5, 4.8))
        df = ablation_df.copy() if ablation_df is not None else pd.DataFrame()
        if df.empty:
            plt.title("CASDI v2 ablation")
            plt.text(0.5, 0.5, "No ablation results", ha="center", va="center")
        else:
            df = df.sort_values("auc", ascending=False)
            sns.barplot(data=df, x="ablation", y="auc", color="#4C78A8")
            plt.xticks(rotation=30, ha="right")
            plt.ylim(0.0, 1.0)
            plt.title("CASDI v2 ablation (AUC)")
            plt.ylabel("AUC")
            plt.grid(axis="y", alpha=0.2)

        out = figures_dir / "fig_casdi_ablation_v2.png"
        ReportBuilder._save_fig(out, dpi=250)
        return out

    @staticmethod
    def plot_model_ranking(figures_dir: Path, performance: pd.DataFrame) -> Path:
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = performance.copy() if performance is not None else pd.DataFrame()
        df = df[df["split"] == "holdout"].copy() if (not df.empty and "split" in df.columns) else df
        if df.empty or ("F1" not in df.columns and "AUC" not in df.columns):
            plt.figure(figsize=(8, 4))
            plt.title("Model ranking")
            msg = "No model results to rank (performance.csv missing/empty or lacks F1/AUC)."
            plt.text(0.5, 0.5, msg, ha="center", va="center")
            plt.axis("off")
            out = figures_dir / "fig_model_ranking.png"
            ReportBuilder._save_fig(out, dpi=250)
            return out

        if "F1" in df.columns:
            df = df.sort_values("F1", ascending=False).head(12)
        else:
            df = df.sort_values("AUC", ascending=False).head(12)
        df["label"] = df["classifier"].astype(str) + " | " + df["feature_set"].astype(str)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        if "F1" in df.columns:
            sns.barplot(data=df, y="label", x="F1", ax=axes[0], color="#4C78A8")
            axes[0].set_title("F1 (holdout)")
            axes[0].set_xlabel("F1")
        else:
            axes[0].set_title("F1 (holdout)")
            axes[0].text(0.5, 0.5, "F1 missing", ha="center", va="center")
            axes[0].set_xlabel("")
        axes[0].set_ylabel("")
        axes[0].grid(axis="x", alpha=0.2)

        if "AUC" in df.columns:
            sns.barplot(data=df, y="label", x="AUC", ax=axes[1], color="#F58518")
            axes[1].set_title("AUC (holdout)")
            axes[1].set_xlabel("AUC")
        else:
            axes[1].set_title("AUC (holdout)")
            axes[1].text(0.5, 0.5, "AUC missing", ha="center", va="center")
            axes[1].set_xlabel("")
        axes[1].set_ylabel("")
        axes[1].grid(axis="x", alpha=0.2)

        out = figures_dir / "fig_model_ranking.png"
        ReportBuilder._save_fig(out, dpi=250)
        return out

    @staticmethod
    def plot_best_roc(figures_dir: Path, best: Mapping[str, object]) -> Path:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        y_true = np.asarray(best["y_true"], dtype=int)
        if "y_score" in best:
            y_score = np.asarray(best["y_score"], dtype=float)
        else:
            y_score = np.asarray(best.get("y_proba", best.get("y_prob", [])), dtype=float)

        plt.figure(figsize=(6.5, 5.5))
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC (best model)")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.2)

        out = figures_dir / "fig_roc_best.png"
        ReportBuilder._save_fig(out, dpi=250)
        return out

    @staticmethod
    def plot_score_contribution(figures_dir: Path, contrib: pd.DataFrame) -> Path:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(9, 5.5))
        if contrib is not None and not contrib.empty and "component" in contrib.columns and "contribution" in contrib.columns:
            df = contrib.copy().sort_values("contribution", ascending=False).head(15)
            sns.barplot(data=df, x="contribution", y="component", color="#E45756")
            plt.title("Score contribution (non-classifier)")
            plt.xlabel("mean |contribution|")
            plt.ylabel("")
            plt.grid(axis="x", alpha=0.2)
        else:
            plt.title("Score contribution (non-classifier)")

        out = figures_dir / "fig_feature_importance.png"
        ReportBuilder._save_fig(out, dpi=250)
        return out

    @staticmethod
    def plot_casdi_ablation(figures_dir: Path, ablation_df: pd.DataFrame) -> Path:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 4.5))
        if not ablation_df.empty:
            df = ablation_df.sort_values("F1", ascending=False)
            sns.barplot(data=df, x="ablation", y="F1", color="#54A24B")
            plt.xticks(rotation=35, ha="right")
            plt.title("CASDI ablation (holdout)")
            plt.ylabel("F1")
            plt.grid(axis="y", alpha=0.2)
        else:
            plt.title("CASDI ablation (holdout)")

        out = figures_dir / "fig_casdi_ablation.png"
        ReportBuilder._save_fig(out, dpi=250)
        return out

    @staticmethod
    def plot_cross_dataset(figures_dir: Path, generalization: pd.DataFrame) -> Path:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(9, 4.5))
        if not generalization.empty:
            df = generalization.copy()
            df["setup"] = df["train"].astype(str) + "→" + df["test"].astype(str)
            sns.barplot(data=df, x="setup", y="F1", color="#B279A2")
            plt.title("Cross-dataset generalization (F1)")
            plt.ylabel("F1")
            plt.grid(axis="y", alpha=0.2)
        else:
            plt.title("Cross-dataset generalization (F1)")

        out = figures_dir / "fig_cross_dataset.png"
        ReportBuilder._save_fig(out, dpi=250)
        return out

    @staticmethod
    def plot_feature_importance(figures_dir: Path, importance: pd.Series) -> Path:
        import matplotlib.pyplot as plt
        import seaborn as sns

        imp = importance.sort_values(ascending=False).head(15)
        plt.figure(figsize=(9, 5.5))
        if not imp.empty:
            sns.barplot(x=imp.values, y=imp.index, color="#E45756")
            plt.title("Permutation importance (top 15)")
            plt.xlabel("importance")
            plt.ylabel("")
            plt.grid(axis="x", alpha=0.2)
        else:
            plt.title("Permutation importance (top 15)")

        out = figures_dir / "fig_feature_importance.png"
        ReportBuilder._save_fig(out, dpi=250)
        return out

    @staticmethod
    def write_summary_md(
        *,
        reports_dir: Path,
        conclusions: List[str],
        figures: Mapping[str, Path],
        best_row: Mapping[str, object],
    ) -> Dict[str, Path]:
        reports_dir.mkdir(parents=True, exist_ok=True)
        md_path = reports_dir / "summary.md"

        fig_lines = []
        ordered = [
            "fig_pipeline.png",
            "fig_feature_distributions.png",
            "fig_effect_size.png",
            "fig_corr_heatmap.png",
            "fig_univariate_auc.png",
            "fig_roc_best.png",
            "fig_pr_best.png",
            "fig_casdi_ablation_v1.png",
            "fig_casdi_ablation_v2.png",
            "fig_cross_dataset.png",
        ]
        for f in ordered:
            p = figures.get(f)
            if p is None:
                continue
            rel = f"../figures/{p.name}"
            fig_lines.append(f"![]({rel})")
            fig_lines.append("")

        lines = [
            "# Summary",
            "",
            "## 1) Objectif",
            "Détecter les fake news via des signaux de propagation (graph mining) sur Twitter15/16.",
            "",
            "## 2) Méthode",
            "Baselines: k-core, centralisation, virality, communities, HPS, spectral. Méthode développée: CASDI (v1) + CASDI_v2 (poids appris sur train).",
            "",
            "## 3) Résultat principal (meilleur score)",
            "",
            f"- method: **{best_row.get('feature_set','')}**",
            f"- dataset: **{best_row.get('dataset','')}**",
            f"- AUC: **{best_row.get('AUC',0):.4f}** (CI [{best_row.get('bootstrap_CI_AUC_low',0):.4f}, {best_row.get('bootstrap_CI_AUC_high',0):.4f}])",
            f"- F1: **{best_row.get('F1',0):.4f}** (CI [{best_row.get('bootstrap_CI_F1_low',0):.4f}, {best_row.get('bootstrap_CI_F1_high',0):.4f}])",
            "",
            "## 4) Conclusions",
            "",
        ]
        for c in conclusions[:8]:
            lines.append(f"- {c}")
        lines.extend(["", "## Figures", ""])
        lines.extend(fig_lines)

        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        pdf_path = reports_dir / "summary.pdf"
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.pyplot as plt

            with PdfPages(pdf_path) as pdf:
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis("off")
                ax.text(0.5, 0.97, "Summary", ha="center", fontsize=16, weight="bold")
                ax.text(0.02, 0.94, md_path.read_text(encoding="utf-8")[:3800], va="top", family="monospace", fontsize=8)
                pdf.savefig(fig)
                plt.close(fig)
        except Exception:
            if pdf_path.exists():
                try:
                    pdf_path.unlink()
                except Exception:
                    pass

        return {"summary.md": md_path, "summary.pdf": pdf_path if pdf_path.exists() else None}

    @staticmethod
    def write_audit_md(*, reports_dir: Path, lines: List[str]) -> Path:
        reports_dir.mkdir(parents=True, exist_ok=True)
        p = reports_dir / "audit.md"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    @staticmethod
    def enforce_curated_only(run_root: Path) -> None:
        tables_dir = run_root / "tables"
        figures_dir = run_root / "figures"
        reports_dir = run_root / "reports"

        if tables_dir.exists():
            for p in tables_dir.iterdir():
                if p.is_file() and p.name not in ReportBuilder.CURATED_TABLES:
                    p.unlink()

        if figures_dir.exists():
            for p in figures_dir.iterdir():
                if p.is_file() and p.name not in ReportBuilder.CURATED_FIGURES:
                    p.unlink()

        if reports_dir.exists():
            for p in reports_dir.iterdir():
                if not p.is_file():
                    continue
                if p.name not in ReportBuilder.CURATED_REPORTS:
                    p.unlink()
