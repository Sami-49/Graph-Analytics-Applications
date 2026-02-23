"""Main CLI orchestrator for full pipeline."""
import logging
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dataclasses import dataclass
from contextlib import contextmanager

from .config import Config
from .paths import DatasetPaths
from .io import LabelParser, CascadeLoader
from .graph_builder import GraphBuilder
from .spectral import SpectralModel
from .kcore import KCoreModel
from .community import CommunityModel
from .centralization import CentralizationModel
from .virality import ViralityModel
from .hps import HybridPropagationScoreModel
from .casdi import CASDIModel, CASDIComponents
from .advanced_features import AdvancedFeatures
from .comparison import GraphMiningComparison
from .evaluation import Evaluator
from .models import ModelBuilder
from .statistics import StatisticalTests
from .visualization import Visualizer
from .reporting import ResultsReporter
from .output_manager import OutputManager, collect_environment_info, try_git_hash
from .output_manager import sha256_file
from .report_builder import ReportBuilder


@dataclass
class StageRuntimes:
    seconds: Dict[str, float]


@contextmanager
def _timed(stage: str, logger: logging.Logger, runtimes: StageRuntimes):
    t0 = time.time()
    logger.info(f"\n[STAGE] {stage} ...")
    try:
        yield
    finally:
        dt = time.time() - t0
        runtimes.seconds[stage] = float(dt)
        logger.info(f"[STAGE] {stage} done in {dt:.2f}s")


class FakeNewsGraphPipeline:
    """Complete pipeline: discovery -> graphs -> features -> evaluation."""

    def __init__(self, config: Config, config_yaml_path: Path):
        self.config = config
        self.config_yaml_path = Path(config_yaml_path)

        config_dict = self.config.to_dict()
        self.om = OutputManager(Path(self.config.output.root), config_dict)
        self.paths = self.om.create_layout()
        self.run_id = self.om.run_id

        self.runtimes = StageRuntimes(seconds={})
        self._init_state()
        self.setup_logging()
        # Curated outputs policy: do not write global pointers/manifests by default.

    def _init_state(self) -> None:
        self.dataset_paths: Optional[DatasetPaths] = None
        self.subsets: List[str] = []
        self.cascade_index: Dict[str, pd.DataFrame] = {}
        self.labels_by_subset: Dict[str, Dict[str, int]] = {}
        self.graphs_by_subset: Dict[str, List[Tuple[str, Any, int]]] = {}
        self.parse_summary_by_subset: Dict[str, Dict[str, Any]] = {}
        self.scores_by_subset: Dict[str, pd.DataFrame] = {}
        self.comparison_by_subset: Dict[str, pd.DataFrame] = {}
        self.corr_by_subset: Dict[str, pd.DataFrame] = {}
        self.results_holdout: Optional[pd.DataFrame] = None
        self.results_cv: Optional[pd.DataFrame] = None
        self.generalization: Optional[pd.DataFrame] = None

    def setup_logging(self) -> None:
        log_file = self.paths.logs_dir / "run.log"
        logging.basicConfig(
            level=getattr(logging, self.config.output.log_level),
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 80)
        self.logger.info("GRAPH-BASED FAKE NEWS DETECTION PIPELINE")
        self.logger.info("=" * 80)

    def _curated_conclusions(self, stats_tests: pd.DataFrame, performance: pd.DataFrame, generalization: pd.DataFrame) -> List[str]:
        concl: List[str] = []
        try:
            # Features: top |delta|
            if not stats_tests.empty:
                s = stats_tests.copy()
                s["abs_delta"] = s["cliffs_delta"].abs()
                top = s.sort_values(["abs_delta", "p_value"], ascending=[False, True]).head(3)
                feats = ", ".join([f"{r.feature} (Δ={r.cliffs_delta:.2f}, p={r.p_value:.2g})" for r in top.itertuples()])
                if feats:
                    concl.append(f"Strongest separability features: {feats}.")
        except Exception:
            pass

        try:
            if not performance.empty:
                hold = performance[performance["split"] == "holdout"].copy()
                if not hold.empty:
                    best = hold.sort_values("F1", ascending=False).iloc[0]
                    concl.append(
                        f"Best model: {best.classifier} + {best.feature_set} on {best.dataset} (F1={best.F1:.3f}, AUC={best.AUC:.3f})."
                    )
        except Exception:
            pass

        try:
            if not generalization.empty:
                g = generalization.copy()
                if "F1" in g.columns:
                    avg = float(g["F1"].mean())
                    concl.append(f"Cross-dataset generalization average F1={avg:.3f} across directions.")
        except Exception:
            pass

        if not concl:
            concl = ["Run completed successfully.", "Outputs are curated to minimize redundancy.", "See summary.md for key findings."]
        return concl[:3]

    def _final_print(self) -> None:
        top_artifacts: List[str] = []
        for rel in [
            self.om.report_path("summary.md"),
            self.om.table_path("univariate_scores.csv"),
            self.om.figure_path("fig_univariate_auc.png"),
        ]:
            if rel.exists():
                top_artifacts.append(str(rel))

        best_model = None
        best_f1 = None
        best_auc = None
        best_ci = None
        uni_path = self.om.table_path("univariate_scores.csv")
        if uni_path.exists():
            try:
                uni = pd.read_csv(uni_path)
                if not uni.empty and "auc" in uni.columns:
                    row = uni.sort_values("auc", ascending=False).iloc[0]
                    best_model = f"score_only + {row.get('feature','')} ({row.get('dataset','')})"
                    best_f1 = float(row.get("f1", 0.0))
                    best_auc = float(row.get("auc", 0.0))
                    best_ci = (
                        float(row.get("f1_ci_low", np.nan)),
                        float(row.get("f1_ci_high", np.nan)),
                        float(row.get("auc_ci_low", np.nan)),
                        float(row.get("auc_ci_high", np.nan)),
                    )
            except Exception:
                pass

        print("\n" + "=" * 80)
        print("RUN COMPLETE")
        print("=" * 80)
        print(f"run_id: {self.run_id}")
        print(f"output: {self.paths.run_root}")
        if top_artifacts:
            print("top_artifacts:")
            for a in top_artifacts:
                print(f"  - {a}")
        if best_model is not None:
            print(f"best_model: {best_model}")
            print(f"best_f1: {best_f1:.4f}")
            if best_auc is not None:
                print(f"best_auc: {best_auc:.4f}")
            if best_ci is not None and not any(np.isnan(x) for x in best_ci):
                print(f"ci_f1: [{best_ci[0]:.4f}, {best_ci[1]:.4f}]")
                print(f"ci_auc: [{best_ci[2]:.4f}, {best_ci[3]:.4f}]")

        # Validation checklist (expected outputs)
        print("validation_checklist:")
        print("  - command_run_all: python -m src.gfn.run --config config.yaml --run-all")
        print("  - command_build_report: python -m src.gfn.run --config config.yaml --build-report")
        print("  - expected_tables: dataset_summary.csv, features_clean.csv, univariate_scores.csv, casdi_comparison.csv, generalization.csv")
        print("  - expected_figures: fig_pipeline.png, fig_feature_distributions.png, fig_effect_size.png, fig_corr_heatmap.png, fig_univariate_auc.png, fig_roc_best.png, fig_pr_best.png, fig_casdi_ablation_v1.png, fig_casdi_ablation_v2.png, fig_cross_dataset.png")
        print("  - expected_reports: summary.md, audit.md (summary.pdf optional)")
        print("=" * 80 + "\n")

    # =========================================================
    # Pipeline stages
    # =========================================================

    def diagnose_data(self) -> None:
        with _timed("diagnose_data", self.logger, self.runtimes):
            self.dataset_paths = DatasetPaths(self.config.dataset.root)
            self.subsets = self.dataset_paths.find_subsets()
            diag = self.dataset_paths.diagnose()

            overview_rows: List[Dict[str, Any]] = []
            for subset in diag.get("subsets_detected", []):
                d = diag.get(subset, {}) if isinstance(diag.get(subset), dict) else {}
                overview_rows.append({
                    "subset": subset,
                    "n_events": d.get("n_events", 0),
                    "n_labeled": d.get("n_labeled", 0),
                    "label_file": d.get("label_file", ""),
                    "example_event_file": d.get("example_event_file", ""),
                    "error": d.get("error", ""),
                })
            # Curated policy: do not write any outputs in diagnose stage.

    def build_graphs(self) -> None:
        with _timed("build_graphs", self.logger, self.runtimes):
            if self.dataset_paths is None:
                self.dataset_paths = DatasetPaths(self.config.dataset.root)
                self.subsets = self.dataset_paths.find_subsets()

            loaded = CascadeLoader.load_all_subsets(self.dataset_paths, self.config)
            for subset_name, (cascade_df, labels) in loaded.items():
                self.cascade_index[subset_name] = cascade_df
                self.labels_by_subset[subset_name] = labels

                graphs: List[Tuple[str, Any, int]] = []
                skipped = {"too_small": 0, "parse_error": 0}
                for event_id in cascade_df["event_id"]:
                    try:
                        tree_file = self.dataset_paths.find_event_file(subset_name, event_id)
                        edges = CascadeLoader.parse_tree_file(tree_file)
                        G = GraphBuilder.build_graph(edges)
                        if G.number_of_nodes() < 2:
                            skipped["too_small"] += 1
                            continue
                        graphs.append((str(event_id), G, int(labels.get(event_id, -1))))
                    except Exception:
                        skipped["parse_error"] += 1
                self.graphs_by_subset[subset_name] = graphs
                self.parse_summary_by_subset[subset_name] = {
                    "n_events_index": int(len(cascade_df)),
                    "n_parsed": int(len(graphs)),
                    "n_skipped": int(skipped.get("too_small", 0) + skipped.get("parse_error", 0)),
                    "skipped": dict(skipped),
                }

            # Curated policy: parsing summary is written later as part of dataset_summary.csv only.

    def compute_features_and_scores(self) -> None:
        with _timed("compute_features_and_scores", self.logger, self.runtimes):
            all_rows: List[pd.DataFrame] = []

            for subset_name, graphs in self.graphs_by_subset.items():
                rows: List[Dict[str, Any]] = []
                casdi_sr: List[float] = []
                casdi_cd: List[float] = []
                casdi_br: List[float] = []
                casdi_ci: List[float] = []
                casdi_vir: List[float] = []

                for event_id, G, label in tqdm(graphs, desc=f"Features ({subset_name})"):
                    basic_stats = GraphBuilder.compute_basic_stats(G)
                    spectral_features = SpectralModel.compute_spectral_features(G)
                    kcore_features = KCoreModel.compute_kcore_features(G)
                    community_features = CommunityModel.compute_community_features(G)
                    centralization_features = CentralizationModel.compute_centralization_features(G)
                    virality_features = ViralityModel.compute_virality_features(G)
                    hps_features = HybridPropagationScoreModel.compute_hps_features(G)
                    adv_features = AdvancedFeatures.compute(G, basic_stats)

                    spectral_radius = float(spectral_features.get("spectral_radius", 0.0))
                    spectral_score = SpectralModel.spectral_score_from_components(
                        spectral_radius=spectral_radius,
                        fiedler_value=float(spectral_features.get("fiedler_value", 0.0)),
                        laplacian_energy=float(spectral_features.get("laplacian_energy", 0.0)),
                    )
                    bridging_ratio = community_features.get("bridging_ratio", 0.0)
                    cent_idx = 0.5 * (
                        centralization_features.get("degree_centralization", 0) +
                        centralization_features.get("pagerank_variance", 0)
                    )

                    rows.append({
                        "subset": subset_name,
                        "event_id": event_id,
                        "label": label,
                        "spectral_score": float(spectral_score),
                        "kcore_score": float(kcore_features.get("max_core_number", 0) * kcore_features.get("core_density", 0)),
                        "community_score": float(community_features.get("community_score", 0)),
                        "centralization_score": float(cent_idx),
                        "virality_score": float(virality_features.get("structural_virality", 0)),
                        "hps_score": float(hps_features.get("hps_score", 0)),
                        "hps_entropy": float(hps_features.get("hps_entropy", 0)),
                        "hps_gini": float(hps_features.get("hps_gini", 0)),
                        **adv_features,
                    })

                    casdi_sr.append(float(spectral_radius))
                    casdi_cd.append(float(kcore_features.get("core_density", 0)))
                    casdi_br.append(float(bridging_ratio))
                    casdi_ci.append(float(cent_idx))
                    casdi_vir.append(float(virality_features.get("structural_virality", 0)))

                if not rows:
                    continue

                scores_df = pd.DataFrame(rows)
                comps = CASDIComponents(
                    spectral_radius=np.array(casdi_sr),
                    core_density=np.array(casdi_cd),
                    bridging_ratio=np.array(casdi_br),
                    centralization_index=np.array(casdi_ci),
                    virality=np.array(casdi_vir),
                )
                c = self.config.casdi
                casdi_scores = CASDIModel.compute_casdi_scores(comps, c.alpha, c.beta, c.gamma, c.delta, c.epsilon)
                for k, arr in casdi_scores.items():
                    scores_df[k] = arr

                # CASDI v2: learn weights on train only (per subset) using logistic regression on robust-z components.
                try:
                    y_all = scores_df["label"].astype(int).to_numpy()
                    labeled_idx = np.where(np.isin(y_all, [0, 1]))[0]
                    train_mask = np.zeros_like(y_all, dtype=bool)
                    if labeled_idx.size >= 10 and len(np.unique(y_all[labeled_idx])) >= 2:
                        from sklearn.model_selection import train_test_split

                        tr_idx, _ = train_test_split(
                            labeled_idx,
                            train_size=float(self.config.evaluation.train_test_split),
                            random_state=int(self.config.evaluation.random_state),
                            stratify=y_all[labeled_idx],
                        )
                        train_mask[tr_idx] = True

                    fit = CASDIModel.fit_casdi_v2_weights(
                        components=comps,
                        y=y_all,
                        train_mask=train_mask,
                        random_state=int(self.config.evaluation.random_state),
                    )
                    v2 = CASDIModel.casdi_v2_scores_from_fit(components=comps, fit=fit)
                    for k, arr in v2.items():
                        scores_df[k] = arr
                except Exception as e:
                    self.logger.warning(f"CASDI_v2 failed for subset {subset_name}: {e}")

                self.scores_by_subset[subset_name] = scores_df
                all_rows.append(scores_df)

            # Curated policy: do not write intermediate tables here.

    def stats_tests(self) -> None:
        with _timed("stats_tests", self.logger, self.runtimes):
            # Curated policy: stats tests are written later as a single curated table.
            return

    def train_and_evaluate(self) -> None:
        with _timed("train_and_evaluate", self.logger, self.runtimes):
            # Curated policy: evaluation is performed in build_curated_outputs().
            return

    def cross_dataset_eval(self) -> None:
        with _timed("cross_dataset_eval", self.logger, self.runtimes):
            # Curated policy: cross-dataset metrics are computed in build_curated_outputs().
            return

    def generate_figures(self) -> None:
        with _timed("generate_figures", self.logger, self.runtimes):
            # Curated policy: figures are generated in build_curated_outputs().
            return

    def build_reports(self) -> None:
        # Curated policy: report is built in build_curated_outputs().
        return

    def build_curated_outputs(self) -> None:
        with _timed("build_curated_outputs", self.logger, self.runtimes):
            # =========================
            # Curated Tables
            # =========================
            # dataset_summary.csv
            ds_rows = []
            for dataset, labels in self.labels_by_subset.items():
                total_events = int(len(self.cascade_index.get(dataset, pd.DataFrame())))
                parsed = int(self.parse_summary_by_subset.get(dataset, {}).get("n_parsed", 0))
                skipped = int(self.parse_summary_by_subset.get(dataset, {}).get("n_skipped", max(total_events - parsed, 0)))
                skip_counts = self.parse_summary_by_subset.get(dataset, {}).get("skipped", {})
                # label counts (fake/real) among parsed
                df = self.scores_by_subset.get(dataset, pd.DataFrame())
                lc = {"fake": int((df.get("label", pd.Series(dtype=int)) == 1).sum()) if not df.empty else 0,
                      "real": int((df.get("label", pd.Series(dtype=int)) == 0).sum()) if not df.empty else 0}
                ds_rows.append({
                    "dataset": dataset,
                    "total_events": total_events,
                    "parsed_events": parsed,
                    "skipped_events": skipped,
                    "skip_reason_counts": json.dumps(skip_counts, sort_keys=True),
                    "label_counts(fake/real)": json.dumps(lc, sort_keys=True),
                })
            dataset_summary = pd.DataFrame(ds_rows)

            # features_clean.csv
            feats_rows: List[pd.DataFrame] = []
            for dataset, df in self.scores_by_subset.items():
                if df is None or df.empty:
                    continue
                cols = {
                    "event_id": "event_id",
                    "dataset": "subset",
                    "label": "label",
                    "kcore_score": "kcore_score",
                    "centralization_score": "centralization_score",
                    "virality_score": "virality_score",
                    "community_score": "community_score",
                    "hps_score": "hps_score",
                    "spectral_score": "spectral_score",
                    "casdi_score": "CASDI_full",
                    "casdi_v2_score": "CASDI_v2_full",
                }
                keep: List[str] = []
                rename: Dict[str, str] = {}
                for out_c, in_c in cols.items():
                    if in_c in df.columns:
                        keep.append(in_c)
                        rename[in_c] = out_c
                if not keep:
                    continue
                sub = df[keep].rename(columns=rename).copy()
                sub["dataset"] = dataset
                feats_rows.append(sub)

            features_clean = pd.concat(feats_rows, axis=0, ignore_index=True) if feats_rows else pd.DataFrame(
                columns=[
                    "event_id",
                    "dataset",
                    "label",
                    "kcore_score",
                    "centralization_score",
                    "virality_score",
                    "community_score",
                    "hps_score",
                    "spectral_score",
                    "casdi_score",
                    "casdi_v2_score",
                ]
            )

            # Basic feature validation (spectral const + HPS-community redundancy warning)
            ReportBuilder.validate_features(features_clean, self.logger)

            # Feature QC: cleanup NaN/inf and drop near-constant features (std < eps).
            qc_warnings: List[str] = []
            qc_dropped: List[str] = []
            eps = 1e-10
            for col in [c for c in ReportBuilder.MAIN_FEATURES if c in features_clean.columns]:
                s = pd.to_numeric(features_clean[col], errors="coerce")
                if not np.isfinite(s.to_numpy(dtype=float)).all():
                    qc_warnings.append(f"QC: {col} had NaN/inf; coerced to numeric and filled NaN with median.")
                    med = float(np.nanmedian(s.to_numpy(dtype=float))) if np.isfinite(np.nanmedian(s.to_numpy(dtype=float))) else 0.0
                    features_clean[col] = s.replace([np.inf, -np.inf], np.nan).fillna(med)
                std = float(pd.to_numeric(features_clean[col], errors="coerce").std(skipna=True))
                if std < eps:
                    qc_warnings.append(f"QC: dropped near-constant feature '{col}' (std<{eps:g}).")
                    qc_dropped.append(str(col))
                    features_clean.drop(columns=[col], inplace=True)

            # Correlation warnings (multicollinearity)
            corr_note = ""
            try:
                feats_for_corr = [c for c in ReportBuilder.MAIN_FEATURES if c in features_clean.columns]
                if len(feats_for_corr) >= 2:
                    corr = features_clean[feats_for_corr].corr().abs()
                    pairs = []
                    for i in range(len(feats_for_corr)):
                        for j in range(i + 1, len(feats_for_corr)):
                            v = float(corr.iloc[i, j])
                            if v > 0.95:
                                pairs.append((feats_for_corr[i], feats_for_corr[j], v))
                    if pairs:
                        pairs = sorted(pairs, key=lambda x: -x[2])
                        corr_note = "QC: extreme correlations |corr|>0.95: " + ", ".join([f"{a}~{b} ({v:.3f})" for a, b, v in pairs[:6]])
                        qc_warnings.append(corr_note)
                        self.logger.warning(corr_note)
            except Exception:
                pass

            # stats_tests (for effect size figure only; not exported as table in PRO pack)
            stats_rows: List[Dict[str, object]] = []
            from scipy.stats import mannwhitneyu
            for dataset in sorted(features_clean["dataset"].unique().tolist()) if not features_clean.empty else []:
                sub = features_clean[(features_clean["dataset"] == dataset) & (features_clean["label"].isin([0, 1]))].copy()
                for feat in [c for c in ReportBuilder.MAIN_FEATURES if c in sub.columns and c not in ["event_id", "dataset", "label"]]:
                    real = pd.to_numeric(sub[sub["label"] == 0][feat], errors="coerce").dropna().values
                    fake = pd.to_numeric(sub[sub["label"] == 1][feat], errors="coerce").dropna().values
                    if len(real) < 2 or len(fake) < 2:
                        continue
                    stat, p = mannwhitneyu(real, fake, alternative="two-sided")
                    dominates = 0
                    for xi in fake:
                        dominates += np.sum(real < xi) - np.sum(real > xi)
                    delta = float(dominates / (len(fake) * len(real) + 1e-10))
                    stats_rows.append({
                        "dataset": dataset,
                        "feature": feat,
                        "mannwhitney_u": float(stat),
                        "p_value": float(p),
                        "cliffs_delta": float(delta),
                        "median_fake": float(np.median(fake)),
                        "median_real": float(np.median(real)),
                    })
            stats_tests = pd.DataFrame(stats_rows)

            # univariate_scores.csv: comprehensive score-only evaluation (AUC, PR-AUC, full confusion metrics)
            from sklearn.metrics import f1_score, roc_auc_score
            from .evaluation import evaluate_score_only_full, score_only_evaluation, validate_metric_consistency

            uni_rows: List[Dict[str, object]] = []
            auc_flips: Dict[str, Dict[str, bool]] = {}
            metric_warnings: List[Dict[str, object]] = []
            rng = np.random.default_rng(int(self.config.evaluation.random_state))

            score_cols = [c for c in [
                "kcore_score",
                "centralization_score",
                "virality_score",
                "community_score",
                "hps_score",
                "spectral_score",
                "casdi_score",
                "casdi_v2_score",
            ] if c in features_clean.columns]

            best_row: Optional[Dict[str, object]] = None
            best_payload: Dict[str, object] = {}

            for dataset in sorted(features_clean["dataset"].unique().tolist()) if (not features_clean.empty and "dataset" in features_clean.columns) else []:
                sub = features_clean[(features_clean["dataset"] == dataset) & (features_clean["label"].isin([0, 1]))].copy()
                if len(sub) < 10:
                    continue
                auc_flips.setdefault(str(dataset), {})

                y = sub["label"].astype(int).to_numpy()
                for sc in score_cols:
                    s = pd.to_numeric(sub[sc], errors="coerce").to_numpy(dtype=float)
                    # New full evaluation (adds PR-AUC, multiple thresholds, and confusion-matrix metrics)
                    full = evaluate_score_only_full(
                        y_true=y,
                        scores=s,
                        name=f"{dataset}:{sc}",
                        n_bootstrap=int(self.config.evaluation.n_bootstrap),
                        ci_alpha=float(self.config.evaluation.ci_alpha),
                        rng=rng,
                        allow_flip_if_auc_below_half=True,
                        fail_on_constant_score=True,
                        cost_fp=float(getattr(self.config.evaluation, "cost_fp", 1.0)),
                        cost_fn=float(getattr(self.config.evaluation, "cost_fn", 1.0)),
                        compute_brier=False,
                    )

                    # Backward-compatible subset used by existing plots/tables
                    auc_flips[str(dataset)][str(sc)] = bool(full.flipped)
                    s_used = -s if bool(full.flipped) else s
                    mask = np.isfinite(s_used) & np.isin(y, [0, 1])

                    # Use max-F1 threshold as the primary reported threshold (more aligned with your spec)
                    thr_main = float(full.threshold_max_f1) if full.threshold_max_f1 is not None else float("nan")
                    row = {
                        "dataset": dataset,
                        "feature": sc,
                        "auc": float(full.auc) if full.auc is not None else float("nan"),
                        "auc_ci_low": float(full.auc_ci_low) if full.auc_ci_low is not None else float("nan"),
                        "auc_ci_high": float(full.auc_ci_high) if full.auc_ci_high is not None else float("nan"),
                        "prauc": float(full.prauc) if full.prauc is not None else float("nan"),
                        "prauc_ci_low": float(full.prauc_ci_low) if full.prauc_ci_low is not None else float("nan"),
                        "prauc_ci_high": float(full.prauc_ci_high) if full.prauc_ci_high is not None else float("nan"),
                        "f1": float(full.metrics_max_f1.get("f1", 0.0)),
                        "threshold": thr_main,
                        "threshold_max_f1": float(full.threshold_max_f1) if full.threshold_max_f1 is not None else float("nan"),
                        "threshold_youden_j": float(full.threshold_youden) if full.threshold_youden is not None else float("nan"),
                        "threshold_cost_sensitive": float(full.threshold_cost) if full.threshold_cost is not None else float("nan"),
                        "flipped": bool(full.flipped),
                        "flip_reason": str(full.flip_reason),
                        # Full metric block at max-F1
                        "accuracy": float(full.metrics_max_f1.get("accuracy", float("nan"))),
                        "balanced_accuracy": float(full.metrics_max_f1.get("balanced_accuracy", float("nan"))),
                        "precision": float(full.metrics_max_f1.get("precision", float("nan"))),
                        "recall": float(full.metrics_max_f1.get("recall", float("nan"))),
                        "specificity": float(full.metrics_max_f1.get("specificity", float("nan"))),
                        "npv": float(full.metrics_max_f1.get("npv", float("nan"))),
                        "fpr": float(full.metrics_max_f1.get("fpr", float("nan"))),
                        "fnr": float(full.metrics_max_f1.get("fnr", float("nan"))),
                        "mcc": float(full.metrics_max_f1.get("mcc", float("nan"))),
                        "kappa": float(full.metrics_max_f1.get("kappa", float("nan"))),
                        # Trade-off metrics at alternative thresholds
                        "f1_youden": float(full.metrics_youden.get("f1", float("nan"))) if full.metrics_youden else float("nan"),
                        "balanced_accuracy_youden": float(full.metrics_youden.get("balanced_accuracy", float("nan"))) if full.metrics_youden else float("nan"),
                        "specificity_youden": float(full.metrics_youden.get("specificity", float("nan"))) if full.metrics_youden else float("nan"),
                        "precision_youden": float(full.metrics_youden.get("precision", float("nan"))) if full.metrics_youden else float("nan"),
                        "recall_youden": float(full.metrics_youden.get("recall", float("nan"))) if full.metrics_youden else float("nan"),

                        "f1_cost": float(full.metrics_cost.get("f1", float("nan"))) if full.metrics_cost else float("nan"),
                        "balanced_accuracy_cost": float(full.metrics_cost.get("balanced_accuracy", float("nan"))) if full.metrics_cost else float("nan"),
                        "specificity_cost": float(full.metrics_cost.get("specificity", float("nan"))) if full.metrics_cost else float("nan"),
                        "precision_cost": float(full.metrics_cost.get("precision", float("nan"))) if full.metrics_cost else float("nan"),
                        "recall_cost": float(full.metrics_cost.get("recall", float("nan"))) if full.metrics_cost else float("nan"),

                        "tn": float(full.metrics_max_f1.get("tn", float("nan"))),
                        "fp": float(full.metrics_max_f1.get("fp", float("nan"))),
                        "fn": float(full.metrics_max_f1.get("fn", float("nan"))),
                        "tp": float(full.metrics_max_f1.get("tp", float("nan"))),
                        "n": int(full.n),
                        "n_pos": int(full.n_pos),
                        "n_neg": int(full.n_neg),
                    }
                    uni_rows.append(row)

                    diag = validate_metric_consistency(
                        auc=float(row["auc"]) if np.isfinite(float(row["auc"])) else 0.5,
                        f1=float(row["f1"]),
                        context=f"score_only:{dataset}:{sc}",
                    )
                    if "warning" in diag:
                        metric_warnings.append(diag)
                        self.logger.warning(str(diag["warning"]))

                    if best_row is None or float(row["auc"]) > float(best_row["auc"]):
                        best_row = dict(row)
                        best_payload = {
                            "dataset": dataset,
                            "feature": sc,
                            "y_true": y[mask],
                            "y_score": s_used[mask],
                        }

            univariate_scores = pd.DataFrame(uni_rows)

            # Convert best_row into previous summary fields expected by ReportBuilder.write_summary_md
            best_summary_row: Dict[str, object] = {}
            if best_row is not None:
                best_summary_row = {
                    "classifier": "score_only",
                    "feature_set": best_row.get("feature", ""),
                    "dataset": best_row.get("dataset", ""),
                    "AUC": best_row.get("auc", 0.0),
                    "bootstrap_CI_AUC_low": best_row.get("auc_ci_low", 0.0),
                    "bootstrap_CI_AUC_high": best_row.get("auc_ci_high", 0.0),
                    "F1": best_row.get("f1", 0.0),
                    "bootstrap_CI_F1_low": best_row.get("f1_ci_low", 0.0),
                    "bootstrap_CI_F1_high": best_row.get("f1_ci_high", 0.0),
                }

            # generalization.csv: score-only cross-dataset with threshold transfer from train
            gen_rows: List[Dict[str, object]] = []
            datasets = sorted(features_clean["dataset"].unique().tolist()) if (not features_clean.empty and "dataset" in features_clean.columns) else []
            if best_row is not None and len(datasets) >= 2:
                best_score = str(best_row.get("feature", ""))
                for train_d, test_d in [(datasets[0], datasets[1]), (datasets[1], datasets[0])]:
                    tr = features_clean[(features_clean["dataset"] == train_d) & (features_clean["label"].isin([0, 1]))].copy()
                    te = features_clean[(features_clean["dataset"] == test_d) & (features_clean["label"].isin([0, 1]))].copy()
                    if len(tr) < 10 or len(te) < 10 or best_score not in tr.columns or best_score not in te.columns:
                        continue

                    y_tr = tr["label"].astype(int).to_numpy()
                    s_tr = pd.to_numeric(tr[best_score], errors="coerce").to_numpy(dtype=float)
                    res_tr_full = evaluate_score_only_full(
                        y_true=y_tr,
                        scores=s_tr,
                        name=f"train:{train_d}:{best_score}",
                        n_bootstrap=int(self.config.evaluation.n_bootstrap),
                        ci_alpha=float(self.config.evaluation.ci_alpha),
                        rng=rng,
                        allow_flip_if_auc_below_half=True,
                        fail_on_constant_score=True,
                    )

                    y_te = te["label"].astype(int).to_numpy()
                    s_te = pd.to_numeric(te[best_score], errors="coerce").to_numpy(dtype=float)
                    s_te_used = -s_te if bool(res_tr_full.flipped) else s_te

                    mask_te = np.isfinite(s_te_used) & np.isin(y_te, [0, 1])
                    y_te2 = y_te[mask_te]
                    s_te2 = s_te_used[mask_te]
                    thr_transfer = float(res_tr_full.threshold_max_f1) if res_tr_full.threshold_max_f1 is not None else float("nan")
                    pred_te = (s_te2 >= thr_transfer).astype(int)
                    f1_te = float(f1_score(y_te2, pred_te, zero_division=0))
                    auc_te = float(roc_auc_score(y_te2, s_te2) if len(np.unique(y_te2)) > 1 else 0.5)

                    gen_rows.append({
                        "train": train_d,
                        "test": test_d,
                        "method": best_score,
                        "threshold_train": thr_transfer,
                        "flipped_train": bool(res_tr_full.flipped),
                        "F1": f1_te,
                        "AUC": auc_te,
                    })
            generalization = pd.DataFrame(gen_rows)

            # casdi_comparison.csv: v1 vs v2 + ablations summary
            casdi_rows: List[Dict[str, object]] = []
            ab_v1_rows: List[Dict[str, object]] = []
            ab_v2_rows: List[Dict[str, object]] = []
            for dataset in sorted(features_clean["dataset"].unique().tolist()) if not features_clean.empty else []:
                sub = features_clean[(features_clean["dataset"] == dataset) & (features_clean["label"].isin([0, 1]))].copy()
                if len(sub) < 10:
                    continue
                y = sub["label"].astype(int).to_numpy()
                v1 = sub["casdi_score"].to_numpy(dtype=float) if "casdi_score" in sub.columns else None
                v2 = sub["casdi_v2_score"].to_numpy(dtype=float) if "casdi_v2_score" in sub.columns else None
                if v1 is not None:
                    r1 = score_only_evaluation(
                        y_true=y,
                        scores=v1,
                        name=f"{dataset}:casdi_score",
                        n_bootstrap=int(self.config.evaluation.n_bootstrap),
                        ci_alpha=float(self.config.evaluation.ci_alpha),
                        rng=rng,
                    )
                    casdi_rows.append({
                        "dataset": dataset,
                        "method": "CASDI_v1",
                        "auc": float(r1.auc),
                        "f1": float(r1.f1),
                        "flipped": bool(r1.flipped),
                    })
                if v2 is not None:
                    r2 = score_only_evaluation(
                        y_true=y,
                        scores=v2,
                        name=f"{dataset}:casdi_v2_score",
                        n_bootstrap=int(self.config.evaluation.n_bootstrap),
                        ci_alpha=float(self.config.evaluation.ci_alpha),
                        rng=rng,
                    )
                    casdi_rows.append({
                        "dataset": dataset,
                        "method": "CASDI_v2",
                        "auc": float(r2.auc),
                        "f1": float(r2.f1),
                        "flipped": bool(r2.flipped),
                    })

                # Ablations from per-subset scores_df (computed earlier)
                src = self.scores_by_subset.get(dataset)
                if src is None or src.empty:
                    continue
                src2 = src[src["label"].isin([0, 1])].copy()
                if len(src2) < 10:
                    continue
                y2 = src2["label"].astype(int).to_numpy()
                for col in [
                    "CASDI_full",
                    "CASDI_minus_spectral",
                    "CASDI_minus_core",
                    "CASDI_minus_bridge",
                    "CASDI_minus_centralization",
                    "CASDI_minus_virality",
                ]:
                    if col not in src2.columns:
                        continue
                    rr = score_only_evaluation(
                        y_true=y2,
                        scores=pd.to_numeric(src2[col], errors="coerce").to_numpy(dtype=float),
                        name=f"{dataset}:{col}",
                        n_bootstrap=int(self.config.evaluation.n_bootstrap),
                        ci_alpha=float(self.config.evaluation.ci_alpha),
                        rng=rng,
                    )
                    ab_v1_rows.append({"dataset": dataset, "ablation": col, "auc": float(rr.auc)})

                for col in [
                    "CASDI_v2_full",
                    "CASDI_v2_minus_spectral",
                    "CASDI_v2_minus_core",
                    "CASDI_v2_minus_bridge",
                    "CASDI_v2_minus_centralization",
                    "CASDI_v2_minus_virality",
                ]:
                    if col not in src2.columns:
                        continue
                    rr = score_only_evaluation(
                        y_true=y2,
                        scores=pd.to_numeric(src2[col], errors="coerce").to_numpy(dtype=float),
                        name=f"{dataset}:{col}",
                        n_bootstrap=int(self.config.evaluation.n_bootstrap),
                        ci_alpha=float(self.config.evaluation.ci_alpha),
                        rng=rng,
                    )
                    ab_v2_rows.append({"dataset": dataset, "ablation": col, "auc": float(rr.auc)})

            casdi_comparison = pd.DataFrame(casdi_rows)
            casdi_ablation_v1 = pd.DataFrame(ab_v1_rows)
            casdi_ablation_v2 = pd.DataFrame(ab_v2_rows)

            # =========================
            # Curated Figures (exact 8)
            # =========================
            figs: Dict[str, Path] = {}
            figs["fig_pipeline.png"] = ReportBuilder.plot_pipeline(self.paths.figures_dir)
            figs["fig_feature_distributions.png"] = ReportBuilder.plot_feature_distributions(self.paths.figures_dir, features_clean)
            figs["fig_effect_size.png"] = ReportBuilder.plot_effect_size(self.paths.figures_dir, stats_tests)
            figs["fig_corr_heatmap.png"] = ReportBuilder.plot_corr_heatmap(self.paths.figures_dir, features_clean)
            figs["fig_univariate_auc.png"] = ReportBuilder.plot_univariate_auc(self.paths.figures_dir, univariate_scores)

            if best_payload:
                figs["fig_roc_best.png"] = ReportBuilder.plot_best_roc(self.paths.figures_dir, {
                    "y_true": best_payload.get("y_true", np.array([0, 1])),
                    "y_score": best_payload.get("y_score", np.array([0.5, 0.5])),
                })
                figs["fig_pr_best.png"] = ReportBuilder.plot_best_pr(self.paths.figures_dir, {
                    "y_true": best_payload.get("y_true", np.array([0, 1])),
                    "y_score": best_payload.get("y_score", np.array([0.5, 0.5])),
                })
            else:
                figs["fig_roc_best.png"] = ReportBuilder.plot_best_roc(self.paths.figures_dir, {"y_true": np.array([0, 1]), "y_proba": np.array([0.5, 0.5])})
                figs["fig_pr_best.png"] = ReportBuilder.plot_best_pr(self.paths.figures_dir, {"y_true": np.array([0, 1]), "y_score": np.array([0.5, 0.5])})

            figs["fig_casdi_ablation_v1.png"] = ReportBuilder.plot_casdi_ablation_v1(self.paths.figures_dir, casdi_ablation_v1)
            figs["fig_casdi_ablation_v2.png"] = ReportBuilder.plot_casdi_ablation_v2(self.paths.figures_dir, casdi_ablation_v2)
            figs["fig_cross_dataset.png"] = ReportBuilder.plot_cross_dataset(self.paths.figures_dir, generalization)

            # Store audit payload for later manifest/audit report generation.
            self._curated_auc_flips = auc_flips
            self._curated_metric_warnings = metric_warnings
            self._curated_best_row = best_summary_row

            # =========================
            # Write curated tables
            # =========================
            tables = ReportBuilder.write_tables(
                tables_dir=self.paths.tables_dir,
                dataset_summary=dataset_summary,
                features_clean=features_clean,
                univariate_scores=univariate_scores,
                casdi_comparison=casdi_comparison,
                generalization=generalization,
                performance=None,
            )

            # =========================
            # Write summary.md (+ optional PDF)
            # =========================
            conclusions = self._curated_conclusions(stats_tests, univariate_scores, generalization)
            rep = ReportBuilder.write_summary_md(
                reports_dir=self.paths.reports_dir,
                conclusions=conclusions,
                figures=figs,
                best_row=best_summary_row,
            )

            # audit.md (minimal, but explicit)
            audit_lines = [
                "# Audit",
                "",
                "## Metrics",
                "- AUC computed from continuous scores (score-only).",
                "- If AUC<0.5, score orientation is flipped and reported in univariate_scores.csv.",
                "",
                "## Feature QC",
            ]

            if qc_warnings:
                for w in qc_warnings[:50]:
                    audit_lines.append(f"- {w}")
            else:
                audit_lines.append("- (no QC warnings)")

            audit_lines.extend([
                "## Warnings",
            ])
            if metric_warnings:
                for w in metric_warnings[:30]:
                    audit_lines.append(f"- {w.get('context','')}: {w.get('warning','')}")
            else:
                audit_lines.append("- (none)")
            ReportBuilder.write_audit_md(reports_dir=self.paths.reports_dir, lines=audit_lines)

            # Enriched manifest.json (hashes + flips + QC + warnings)
            try:
                hashes: Dict[str, str] = {}
                for fn in [
                    "dataset_summary.csv",
                    "features_clean.csv",
                    "univariate_scores.csv",
                    "casdi_comparison.csv",
                    "generalization.csv",
                ]:
                    p = self.paths.tables_dir / fn
                    if p.exists():
                        hashes[f"tables/{fn}"] = sha256_file(p)

                warnings_compact: List[str] = []
                for w in qc_warnings[:30]:
                    warnings_compact.append(str(w))
                for w in metric_warnings[:30]:
                    ww = w.get("warning")
                    if ww:
                        warnings_compact.append(str(ww))

                manifest_extra = {
                    "auc_flips": auc_flips,
                    "qc_dropped_features": qc_dropped,
                    "qc_warnings": qc_warnings[:50],
                    "metric_warnings": metric_warnings[:50],
                    "hashes": hashes,
                    "best": best_summary_row,
                }
                self._write_manifest(stage="pro")
                # merge extra info (overwrite manifest with enriched content)
                try:
                    base = json.loads(self.om.manifest_json.read_text(encoding="utf-8")) if self.om.manifest_json.exists() else {}
                except Exception:
                    base = {}
                base.update(manifest_extra)
                self.om.write_manifest(base)
            except Exception as e:
                self.logger.warning(f"Failed to write enriched manifest: {e}")

            # Curated policy enforcement: remove everything else inside tables/figures/reports
            ReportBuilder.enforce_curated_only(self.paths.run_root)

    def _quality_checks_before_training(self, df_all: pd.DataFrame, feature_cols: List[str]) -> None:
        # D) QUALITY FIXES (MANDATORY CHECKS)
        # 1) constant columns check
        allowed_constant = {"diameter_lcc", "avg_shortest_path_lcc"}
        const_cols = []
        for c in feature_cols:
            if c in allowed_constant:
                continue
            s = pd.to_numeric(df_all[c], errors="coerce")
            if float(s.std(skipna=True)) == 0.0:
                const_cols.append(c)
        if const_cols:
            raise ValueError(f"Constant feature columns detected (std=0). Fix feature extraction: {const_cols[:30]}")

        # 2) spectral_score not constant
        if "spectral_score" in df_all.columns:
            ss = pd.to_numeric(df_all["spectral_score"], errors="coerce")
            if float(ss.std(skipna=True)) < 1e-10:
                raise ValueError(
                    "spectral_score is constant. Likely causes: spectral features returning zeros for all graphs "
                    "(small graphs / failed parsing / LCC extraction) or scoring normalization bug."
                )

        # 3) hps_score too correlated with community_score
        if "hps_score" in df_all.columns and "community_score" in df_all.columns:
            a = pd.to_numeric(df_all["hps_score"], errors="coerce")
            b = pd.to_numeric(df_all["community_score"], errors="coerce")
            corr = float(a.corr(b)) if a.notna().sum() > 5 and b.notna().sum() > 5 else 0.0
            if corr > 0.95:
                self.logger.warning(f"hps_score is highly correlated with community_score (corr={corr:.3f}).")

    # =========================================================
    # Orchestration
    # =========================================================

    def run_diagnose_only(self) -> None:
        self.diagnose_data()
        # Curated policy: diagnose-only does not write outputs.
        self._final_print()

    def run_all(self, build_report: bool = False) -> None:
        np.random.seed(self.config.evaluation.random_state)
        self.diagnose_data()
        self.build_graphs()
        self.compute_features_and_scores()
        self.build_curated_outputs()
        self._final_print()

    def build_report(self) -> None:
        # Regenerate PRO outputs in an isolated run folder.
        self.run_all(build_report=True)

    def _write_manifest(self, stage: str) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        env = collect_environment_info(repo_root / "requirements.txt")
        git_hash = try_git_hash(repo_root)

        subsets = self.subsets or (self.dataset_paths.find_subsets() if self.dataset_paths else [])
        parsed_counts = {k: int(len(v)) for k, v in self.graphs_by_subset.items()}
        artifacts = self.om.list_artifacts()
        manifest = {
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stage": stage,
            "random_seed": int(self.config.evaluation.random_state),
            "dataset_root": str(self.config.dataset.root),
            "subsets": subsets,
            "events_parsed": parsed_counts,
            "parsing_summary": self.parse_summary_by_subset,
            "environment": env,
            "git_hash": git_hash,
            "stage_runtimes_seconds": dict(self.runtimes.seconds),
            "artifacts": artifacts,
        }
        self.om.write_manifest(manifest)

    def legacy_diagnose(self) -> None:
        t0 = time.time()
        self.logger.info("\n[DIAGNOSTIC MODE]")
        paths = DatasetPaths(self.config.dataset.root)
        diag = paths.diagnose()

        self.logger.info(f"Subset folders detected: {diag.get('subsets_detected', [])}")
        for subset in diag.get("subsets_detected", []):
            if subset in diag and isinstance(diag[subset], dict):
                d = diag[subset]
                if "error" in d:
                    self.logger.warning(f"  {subset}: {d['error']}")
                else:
                    self.logger.info(f"  {subset}:")
                    self.logger.info(f"    label_file: {d.get('label_file', 'N/A')}")
                    self.logger.info(f"    n_events: {d.get('n_events', 0)}")
                    self.logger.info(f"    n_labeled: {d.get('n_labeled', 0)}")
                    self.logger.info(f"    example_event_file: {d.get('example_event_file', 'N/A')}")
                    self.logger.info(f"    example_edges_parsed: {d.get('example_edges_parsed', 'N/A')}")

        self.logger.info(f"Diagnose completed in {time.time() - t0:.2f}s")

    def legacy_run_all(self) -> None:
        np.random.seed(self.config.evaluation.random_state)
        t_start = time.time()
        paths = DatasetPaths(self.config.dataset.root)
        all_cascades = CascadeLoader.load_all_subsets(paths, self.config)

        tables_dir = self.out_root / "tables"
        figures_dir = self.out_root / "figures"
        reports_dir = self.out_root / "reports"

        # Write run manifest (single source of truth)
        dataset_counts: Dict[str, int] = {}
        for name, (df, labels) in all_cascades.items():
            dataset_counts[name] = int(len(df))
        manifest_path = self.global_root / "reports" / "run_manifest.json"
        try:
            create_run_manifest(
                self.config,
                dataset_counts=dataset_counts,
                out_path=manifest_path,
                repo_root=Path(__file__).resolve().parents[2],
            )
        except Exception as e:
            self.logger.warning(f"Failed to write run manifest: {e}")
        subset_scores: Dict[str, pd.DataFrame] = {}

        for subset_name, (cascade_df, labels) in all_cascades.items():
            self.logger.info(f"\n{'=' * 80}")
            self.logger.info(f"PROCESSING SUBSET: {subset_name.upper()}")
            self.logger.info(f"{'=' * 80}")

            n_total = len(cascade_df)
            n_labeled = sum(1 for eid in cascade_df["event_id"] if labels.get(eid, -1) in [0, 1])
            self.logger.info(f"Total events: {n_total}, Labeled: {n_labeled}")

            rows: List[Dict[str, Any]] = []
            casdi_sr: List[float] = []
            casdi_cd: List[float] = []
            casdi_br: List[float] = []
            casdi_ci: List[float] = []
            casdi_vir: List[float] = []
            skipped = {"too_small": 0, "error": 0}

            for event_id in tqdm(cascade_df["event_id"], desc=f"Features ({subset_name})"):
                try:
                    tree_file = paths.find_event_file(subset_name, event_id)
                    edges = CascadeLoader.parse_tree_file(tree_file)
                    G = GraphBuilder.build_graph(edges)

                    if G.number_of_nodes() < 2:
                        skipped["too_small"] += 1
                        continue

                    basic_stats = GraphBuilder.compute_basic_stats(G)
                    spectral_features = SpectralModel.compute_spectral_features(G)
                    kcore_features = KCoreModel.compute_kcore_features(G)
                    community_features = CommunityModel.compute_community_features(G)
                    centralization_features = CentralizationModel.compute_centralization_features(G)
                    virality_features = ViralityModel.compute_virality_features(G)
                    hps_features = HybridPropagationScoreModel.compute_hps_features(G)
                    adv_features = AdvancedFeatures.compute(G, basic_stats)

                    spectral_radius = spectral_features.get("spectral_radius", 0.0)
                    spectral_score = SpectralModel.spectral_score_from_components(
                        spectral_radius=float(spectral_radius),
                        fiedler_value=float(spectral_features.get("fiedler_value", 0.0)),
                        laplacian_energy=float(spectral_features.get("laplacian_energy", 0.0)),
                    )

                    bridging_ratio = community_features.get("bridging_ratio", 0.0)
                    cent_idx = 0.5 * (
                        centralization_features.get("degree_centralization", 0) +
                        centralization_features.get("pagerank_variance", 0)
                    )

                    row = {
                        "event_id": event_id,
                        "label": labels.get(event_id, -1),
                        "spectral_score": float(spectral_score),
                        "kcore_score": float(kcore_features.get("max_core_number", 0) * kcore_features.get("core_density", 0)),
                        "community_score": float(community_features.get("community_score", 0)),
                        "centralization_score": cent_idx,
                        "virality_score": virality_features.get("structural_virality", 0),
                        "hps_score": hps_features.get("hps_score", 0),
                        "hps_entropy": hps_features.get("hps_entropy", 0),
                        "hps_gini": hps_features.get("hps_gini", 0),
                        "legacy_hps_score": float(community_features.get("modularity", 0)),
                        **adv_features,
                    }

                    rows.append(row)
                    casdi_sr.append(spectral_radius)
                    casdi_cd.append(kcore_features.get("core_density", 0))
                    casdi_br.append(bridging_ratio)
                    casdi_ci.append(cent_idx)
                    casdi_vir.append(virality_features.get("structural_virality", 0))

                except Exception as e:
                    skipped["error"] += 1
                    self.logger.warning(f"Error {event_id}: {e}")

            self.logger.info(f"Parsed: {len(rows)}, Skipped: {skipped}")

            if not rows:
                self.logger.warning(f"No valid cascades for {subset_name}")
                continue

            scores_df = pd.DataFrame(rows)

            c = self.config.casdi
            components = CASDIComponents(
                spectral_radius=np.array(casdi_sr),
                core_density=np.array(casdi_cd),
                bridging_ratio=np.array(casdi_br),
                centralization_index=np.array(casdi_ci),
                virality=np.array(casdi_vir),
            )
            casdi_scores = CASDIModel.compute_casdi_scores(
                components, c.alpha, c.beta, c.gamma, c.delta, c.epsilon
            )
            for k, arr in casdi_scores.items():
                scores_df[k] = arr

            scores_path = tables_dir / f"model_scores_{subset_name}.csv"
            scores_df.to_csv(scores_path, index=False)
            self.logger.info(f"Saved: {scores_path}")

            unified_path = tables_dir / f"unified_features_{subset_name}.csv"
            scores_df.to_csv(unified_path, index=False)

            labeled_df = scores_df[scores_df["label"].isin([0, 1])].copy()
            if len(labeled_df) < 2:
                continue

            model_cols = [
                "spectral_score", "kcore_score", "community_score",
                "centralization_score", "virality_score", "hps_score"
            ]

            comparison_report = GraphMiningComparison.generate_comparison_report(
                labeled_df[["event_id", "label"] + model_cols]
            )

            corr_after = comparison_report["correlation_matrix"]
            corr_after.to_csv(tables_dir / f"correlation_heatmap_after_{subset_name}.csv")

            if "legacy_hps_score" in labeled_df.columns:
                before_cols = [c for c in model_cols if c != "hps_score"] + ["legacy_hps_score"]
                before_cols = [c for c in before_cols if c in labeled_df.columns]
                corr_before = labeled_df[before_cols].corr()
                corr_before.to_csv(tables_dir / f"correlation_heatmap_before_{subset_name}.csv")
                comparison_report["correlation_before"] = corr_before

            comparison_report["correlation_matrix"].to_csv(tables_dir / f"model_correlation_{subset_name}.csv")
            disc_df = pd.DataFrame(comparison_report["discrimination"]).T
            disc_df.to_csv(tables_dir / f"comparison_report_{subset_name}.csv")

            stat_cols = model_cols + list(adv_features.keys()) + list(casdi_scores.keys())
            stat_cols = [c for c in stat_cols if c in scores_df.columns]
            stats_df = StatisticalTests.compute_all_tests(labeled_df, stat_cols)
            stats_df.to_csv(tables_dir / "statistical_tests.csv", index=False)
            self.logger.info(f"Saved statistical_tests.csv")

            self._run_evaluation(
                labeled_df, subset_name, tables_dir, figures_dir, model_cols,
                adv_features.keys(), casdi_scores.keys()
            )

            self._generate_figures(
                labeled_df, subset_name, figures_dir, model_cols,
                corr_after, comparison_report, adv_features.keys()
            )

            subset_scores[subset_name] = labeled_df

        self._run_cross_dataset(subset_scores, tables_dir, figures_dir)
        self._generate_summary_report(all_cascades, tables_dir, reports_dir)
        self._generate_pipeline_diagram(reports_dir / "pipeline.png")
        self.logger.info(f"\nPipeline complete in {time.time() - t_start:.2f}s")

    def _run_evaluation(
        self,
        labeled_df: pd.DataFrame,
        subset_name: str,
        tables_dir: Path,
        figures_dir: Path,
        model_cols: List[str],
        adv_cols,
        casdi_cols,
    ) -> None:
        X_full = labeled_df.drop(columns=["event_id", "label"], errors="ignore")
        y = labeled_df["label"].values
        config = self.config

        n_splits = int(len(labeled_df) * config.evaluation.train_test_split)
        if n_splits < 2 or len(labeled_df) - n_splits < 2:
            self.logger.warning("Too few samples for evaluation")
            return

        from sklearn.model_selection import train_test_split
        indices = np.arange(len(labeled_df))
        train_idx, test_idx = train_test_split(
            indices, test_size=1 - config.evaluation.train_test_split,
            stratify=y, random_state=config.evaluation.random_state
        )
        X_train_df = labeled_df.iloc[train_idx].drop(columns=["event_id", "label"], errors="ignore")
        X_test_df = labeled_df.iloc[test_idx].drop(columns=["event_id", "label"], errors="ignore")
        y_train, y_test = y[train_idx], y[test_idx]

        casdi_ablations = [
            "CASDI_full", "CASDI_minus_spectral", "CASDI_minus_core",
            "CASDI_minus_bridge", "CASDI_minus_centralization", "CASDI_minus_virality",
        ]

        feature_sets = [
            ("spectral_score", ["spectral_score"]),
            ("kcore_score", ["kcore_score"]),
            ("community_score", ["community_score"]),
            ("centralization_score", ["centralization_score"]),
            ("virality_score", ["virality_score"]),
            ("hps_score", ["hps_score"]),
            ("AdvancedFeatures", [c for c in adv_cols if c in X_train_df.columns]),
            ("CASDI_full", ["CASDI_full"] if "CASDI_full" in X_train_df.columns else []),
            ("CASDI_full_plus_adv", ["CASDI_full"] + [c for c in adv_cols if c in X_train_df.columns]),
        ]

        all_results = []
        y_proba_dict = {}
        best_f1 = -1
        best_model_name = ""
        best_clf = None
        best_X_test = None
        best_cols: List[str] = []

        models = ModelBuilder.get_all_models(config)

        for fs_name, cols in feature_sets:
            cols = [c for c in cols if c in X_train_df.columns]
            if not cols:
                continue

            X_tr = X_train_df[cols].fillna(0).values
            X_te = X_test_df[cols].fillna(0).values

            for clf_name, clf in models.items():
                key = f"{fs_name}_{clf_name}"
                try:
                    metrics = Evaluator.holdout_evaluation(
                        X_tr, X_te, y_train, y_test, clf, config
                    )
                    cv_metrics = Evaluator.cross_validation(X_tr, y_train, clf, config)
                    metrics["cv_f1_mean"] = cv_metrics["f1_mean"]
                    metrics["cv_f1_std"] = cv_metrics["f1_std"]
                    metrics["cv_auc_mean"] = cv_metrics["auc_mean"]
                    metrics["cv_auc_std"] = cv_metrics["auc_std"]
                    metrics["feature_set"] = fs_name
                    metrics["classifier"] = clf_name
                    all_results.append(metrics)

                    clf.fit(X_tr, y_train)
                    proba = clf.predict_proba(X_te)[:, 1]
                    y_proba_dict[key] = proba

                    if metrics["f1"] > best_f1:
                        best_f1 = metrics["f1"]
                        best_model_name = key
                        best_clf = clf
                        best_X_test = X_te
                        best_cols = cols
                except Exception as e:
                    self.logger.warning(f"Eval failed {key}: {e}")

        ablation_rows = []
        for ab_name in casdi_ablations:
            if ab_name not in X_train_df.columns:
                continue
            X_ta = X_train_df[[ab_name]].fillna(0).values
            X_te_ab = X_test_df[[ab_name]].fillna(0).values
            for clf_name, clf in models.items():
                try:
                    m = Evaluator.holdout_evaluation(X_ta, X_te_ab, y_train, y_test, clf, config)
                    ablation_rows.append({
                        "ablation": ab_name,
                        "classifier": clf_name,
                        "f1": m["f1"],
                        "roc_auc": m["roc_auc"],
                    })
                except Exception:
                    pass
        if ablation_rows:
            pd.DataFrame(ablation_rows).to_csv(tables_dir / "casdi_ablation.csv", index=False)

        if all_results:
            res_df = pd.DataFrame(all_results)
            res_df.to_csv(tables_dir / f"evaluation_results_{subset_name}.csv", index=False)

            ranking = res_df.groupby("feature_set").agg({
                "f1": "max", "roc_auc": "max",
                "f1_ci_lower": "min", "f1_ci_upper": "max",
                "auc_ci_lower": "min", "auc_ci_upper": "max",
            }).sort_values("f1", ascending=False)
            ranking.to_csv(tables_dir / "ranking_table.csv")

            if best_clf is not None and best_X_test is not None:
                y_pred_best = best_clf.predict(best_X_test)
                y_proba_best = best_clf.predict_proba(best_X_test)[:, 1]

                fp_mask = (y_pred_best == 1) & (y_test == 0)
                fn_mask = (y_pred_best == 0) & (y_test == 1)
                err_df = labeled_df.iloc[test_idx].copy()
                err_df["pred"] = y_pred_best
                err_df["proba"] = y_proba_best
                err_df["error_type"] = ""
                err_df.loc[fp_mask, "error_type"] = "false_positive"
                err_df.loc[fn_mask, "error_type"] = "false_negative"
                err_rows = err_df[err_df["error_type"] != ""].head(20)
                err_rows.to_csv(tables_dir / "error_analysis.csv", index=False)

                try:
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_test, y_pred_best)
                    plt.figure(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                    plt.title("Confusion Matrix (Best Model)")
                    plt.savefig(figures_dir / "confusion_matrix_best.png", dpi=150)
                    plt.close()
                except Exception:
                    pass

                try:
                    from sklearn.calibration import calibration_curve
                    prob_true, prob_pred = calibration_curve(y_test, y_proba_best, n_bins=10)
                    plt.figure(figsize=(6, 5))
                    plt.plot(prob_pred, prob_true, "s-")
                    plt.plot([0, 1], [0, 1], "k--")
                    plt.xlabel("Mean predicted probability")
                    plt.ylabel("Fraction of positives")
                    plt.title("Calibration Curve (Best Model)")
                    plt.savefig(figures_dir / "calibration_curve_best.png", dpi=150)
                    plt.close()
                except Exception:
                    pass

                try:
                    from sklearn.metrics import precision_recall_curve
                    prec, rec, _ = precision_recall_curve(y_test, y_proba_best)
                    plt.figure(figsize=(6, 5))
                    plt.plot(rec, prec)
                    plt.xlabel("Recall")
                    plt.ylabel("Precision")
                    plt.title("Precision-Recall Curve (Best Model)")
                    plt.savefig(figures_dir / "pr_curve_best.png", dpi=150)
                    plt.close()
                except Exception:
                    pass

                try:
                    from sklearn.inspection import permutation_importance
                    imp = permutation_importance(
                        best_clf, best_X_test, y_test, n_repeats=10, random_state=42
                    )
                    if best_cols and len(imp.importances_mean) >= len(best_cols):
                        imp_dict = {best_cols[i]: float(imp.importances_mean[i]) for i in range(len(best_cols))}
                        Visualizer.plot_feature_importance(imp_dict, figures_dir)
                except Exception:
                    pass

            if y_proba_dict and len(np.unique(y_test)) > 1:
                try:
                    Visualizer.plot_roc_curves(y_test, y_proba_dict, figures_dir)
                except Exception:
                    pass

            res_df["model"] = res_df["feature_set"] + "_" + res_df["classifier"]
            if not res_df.empty:
                try:
                    Visualizer.plot_model_comparison(res_df, figures_dir)
                except Exception:
                    pass

    def _generate_figures(
        self,
        labeled_df: pd.DataFrame,
        subset_name: str,
        figures_dir: Path,
        model_cols: List[str],
        corr_after: pd.DataFrame,
        comparison_report: Dict,
        adv_cols,
    ) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        for col in model_cols:
            if col not in labeled_df.columns:
                continue
        cols = [c for c in model_cols if c in labeled_df.columns]
        if cols:
            fig, axes = plt.subplots(2, 3, figsize=(14, 10))
            axes = axes.flatten()
            for i, col in enumerate(cols[:6]):
                ax = axes[i]
                labeled_df.boxplot(column=col, by="label", ax=ax)
                ax.set_title(col)
            plt.suptitle("")
            plt.tight_layout()
            plt.savefig(figures_dir / "boxplots_scores.png", dpi=150)
            plt.close()

            fig, axes = plt.subplots(2, 3, figsize=(14, 10))
            axes = axes.flatten()
            for i, col in enumerate(cols[:6]):
                ax = axes[i]
                sns.violinplot(x="label", y=col, data=labeled_df, ax=ax)
                ax.set_title(col)
            plt.tight_layout()
            plt.savefig(figures_dir / "violinplots_scores.png", dpi=150)
            plt.close()

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_after.astype(float), annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1)
        plt.title("Correlation Heatmap (After De-redundancy)")
        plt.tight_layout()
        plt.savefig(figures_dir / "correlation_heatmap_after.png", dpi=150)
        plt.close()

        if "correlation_before" in comparison_report:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            sns.heatmap(comparison_report["correlation_before"], annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, ax=axes[0])
            axes[0].set_title("Before (legacy HPS)")
            sns.heatmap(corr_after, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, ax=axes[1])
            axes[1].set_title("After (new HPS)")
            plt.tight_layout()
            plt.savefig(figures_dir / "correlation_heatmap_before_vs_after.png", dpi=150)
            plt.close()

        key_cols = [c for c in model_cols if c in labeled_df.columns][:6]
        if key_cols:
            fig, axes = plt.subplots(2, 3, figsize=(14, 10))
            axes = axes.flatten()
            for i, col in enumerate(key_cols):
                ax = axes[i]
                for lbl, sub in labeled_df.groupby("label"):
                    sub[col].dropna().hist(ax=ax, alpha=0.6, label="Fake" if lbl == 1 else "Real", bins=20)
                ax.set_title(col)
                ax.legend()
            plt.suptitle("Key Feature Distributions (Real vs Fake)")
            plt.tight_layout()
            plt.savefig(figures_dir / "distributions_key_features.png", dpi=150)
            plt.close()

        casdi_ablation = Path(self.config.output.root) / "tables" / "casdi_ablation.csv"
        if casdi_ablation.exists():
            try:
                abd = pd.read_csv(casdi_ablation)
                if "ablation" in abd.columns and "f1" in abd.columns:
                    agg = abd.groupby("ablation")["f1"].agg(["mean", "std"]).reset_index()
                    fig, ax = plt.subplots(figsize=(10, 6))
                    x = np.arange(len(agg))
                    ax.bar(x, agg["mean"], yerr=agg["std"].fillna(0), capsize=3, color="steelblue")
                    ax.set_xticks(x)
                    ax.set_xticklabels(agg["ablation"], rotation=45, ha="right")
                    ax.set_ylabel("F1 Score")
                    ax.set_title("CASDI Ablation Study")
                    plt.tight_layout()
                    plt.savefig(figures_dir / "casdi_ablation.png", dpi=150)
                    plt.close()
            except Exception:
                pass

    def _run_cross_dataset(
        self, subset_scores: Dict[str, pd.DataFrame], tables_dir: Path, figures_dir: Path
    ) -> None:
        names = sorted(subset_scores.keys())
        if len(names) < 2:
            return
        n1, n2 = names[0], names[1]
        df1 = subset_scores[n1]
        df2 = subset_scores[n2]
        common_cols = [c for c in df1.columns if c in df2.columns and c not in ["event_id", "label"]]
        if not common_cols or "label" not in df1.columns or "label" not in df2.columns:
            return

        labeled1 = df1[df1["label"].isin([0, 1])]
        labeled2 = df2[df2["label"].isin([0, 1])]
        if len(labeled1) < 5 or len(labeled2) < 5:
            return

        X1 = labeled1[common_cols].fillna(0).values
        y1 = labeled1["label"].values
        X2 = labeled2[common_cols].fillna(0).values
        y2 = labeled2["label"].values

        models = ModelBuilder.get_all_models(self.config)
        rows = []

        for (train_name, X_tr, y_tr), (test_name, X_te, y_te) in [
            ((n1, X1, y1), (n2, X2, y2)),
            ((n2, X2, y2), (n1, X1, y1)),
        ]:
            for clf_name, clf in models.items():
                try:
                    clf.fit(X_tr, y_tr)
                    from sklearn.metrics import f1_score, roc_auc_score
                    pred = clf.predict(X_te)
                    proba = clf.predict_proba(X_te)[:, 1]
                    f1 = f1_score(y_te, pred, zero_division=0)
                    auc = roc_auc_score(y_te, proba) if len(np.unique(y_te)) > 1 else 0.5
                    rows.append({
                        "train": train_name,
                        "test": test_name,
                        "classifier": clf_name,
                        "f1": f1,
                        "roc_auc": auc,
                    })
                except Exception:
                    pass

        if rows:
            cross_df = pd.DataFrame(rows)
            cross_df.to_csv(tables_dir / "cross_dataset_generalization.csv", index=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            cross_df["setup"] = cross_df["train"] + "→" + cross_df["test"]
            pivot = cross_df.pivot_table(index="setup", columns="classifier", values="f1")
            pivot.plot(kind="bar", ax=ax, rot=45)
            plt.title("Cross-Dataset Generalization (F1)")
            plt.ylabel("F1 Score")
            plt.legend(title="Classifier")
            plt.tight_layout()
            plt.savefig(figures_dir / "cross_dataset_summary.png", dpi=150)
            plt.close()

    def _generate_summary_report(
        self, all_cascades: Dict, tables_dir: Path, reports_dir: Path
    ) -> None:
        lines = [
            "# Summary Report",
            "",
            "## Dataset",
        ]
        for name, (df, labels) in all_cascades.items():
            n = len(df)
            n_labeled = sum(1 for eid in df["event_id"] if labels.get(eid, -1) in [0, 1])
            lines.append(f"- {name}: {n} events, {n_labeled} labeled")

        lines.extend([
            "",
            "## Outputs",
            "- tables/: model_scores, unified_features, correlation, statistical_tests, ranking_table, error_analysis",
            "- figures/: boxplots, violinplots, correlation heatmaps, ROC, model comparison, CASDI ablation",
            "- reports/: summary.md, pipeline.png",
            "",
            "## Key Insights",
            "1. Spectral model fixed: spectral_radius, fiedler_value, laplacian_energy on LCC.",
            "2. HPS redefined as Propagation Heterogeneity: entropy + gini of BFS layer sizes.",
            "3. Community score: modularity + entropy + bridging_ratio.",
            "4. CASDI: robust z-score combination of 5 components with ablations.",
            "5. Advanced features: assortativity, transitivity, diameter, degree_gini, etc.",
        ])
        (reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    def _generate_pipeline_diagram(self, out_path: Path) -> None:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis("off")

            boxes = [
                (5, 9, "Dataset (Twitter15/16)"),
                (5, 7.5, "Load Cascades & Labels"),
                (5, 6, "Build Graphs"),
                (5, 4.5, "Compute Features\n(6 models + Advanced + CASDI)"),
                (5, 3, "Evaluation\n(Holdout, CV, Bootstrap)"),
                (5, 1.5, "Outputs (tables, figures, reports)"),
            ]
            for x, y, txt in boxes:
                rect = mpatches.FancyBboxPatch((x - 1.2, y - 0.3), 2.4, 0.6, boxstyle="round,pad=0.05", facecolor="lightblue", edgecolor="black")
                ax.add_patch(rect)
                ax.text(x, y, txt, ha="center", va="center", fontsize=9)
            for i in range(len(boxes) - 1):
                ax.annotate("", xy=(5, boxes[i + 1][1] + 0.3), xytext=(5, boxes[i][1] - 0.3),
                            arrowprops=dict(arrowstyle="->", color="black"))
            plt.title("Pipeline Flow")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            self.logger.info(f"Saved pipeline diagram: {out_path}")
        except Exception as e:
            self.logger.warning(f"Pipeline diagram failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph-based Fake News Detection")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--run-all", action="store_true", help="Run full pipeline (curated outputs only)")
    parser.add_argument("--build-report", action="store_true", help="Regenerate curated PRO outputs and reports")

    args = parser.parse_args()

    try:
        config = Config.from_yaml(args.config)
    except Exception as e:
        print(f"ERREUR config: {e}")
        sys.exit(1)

    dataset_root = Path(config.dataset.root)
    if not dataset_root.exists():
        print(f"ERREUR: Le dossier dataset n'existe pas: {dataset_root}")
        print("Modifiez 'dataset.root' dans config.yaml pour pointer vers Twitter15_16_dataset")
        sys.exit(1)

    try:
        pipeline = FakeNewsGraphPipeline(config, config_yaml_path=Path(args.config))
    except Exception as e:
        print(f"ERREUR init: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    try:
        if args.run_all:
            pipeline.run_all(build_report=False)
        elif args.build_report:
            pipeline.build_report()
        else:
            parser.print_help()
    except Exception as e:
        print(f"ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
