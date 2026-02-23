# CHANGES.md – Research Quality Upgrade

## Summary
Fixed spectral model bug, de-redundant HPS vs community, implemented CASDI with ablations, added 10 advanced features, full evaluation framework, statistical tests, and rich outputs.

---

## A) Spectral Model Fix

**File: `src/gfn/spectral.py`**
- Replaced implementation with correct, stable computations
- Use undirected LCC before spectral computations
- spectral_radius = largest eigenvalue of adjacency (sparse eigs for n>200, dense otherwise)
- fiedler_value = 2nd smallest eigenvalue of normalized Laplacian (fallback: Laplacian)
- laplacian_energy = sum(|λ_i - avg_degree|) based on Laplacian eigenvalues
- Safeguards: nodes<3 or edges=0 → return 0s; finite floats; LCC for disconnected graphs
- Graceful fallback if scipy not installed

**File: `tests/test_spectral.py`**
- Existing tests already assert non-zero spectral outputs (star, path, complete, two-component)
- No changes needed; tests pass with new implementation

---

## B) De-Redundancy: Community vs HPS

**File: `src/gfn/community.py`**
- community_score already uses modularity + entropy + bridging_ratio (no change)

**File: `src/gfn/hps.py`**
- HPS redefined as Propagation Heterogeneity Score
- BFS layers from root on undirected graph (compute_levels_undirected)
- hps_entropy, hps_gini, hps_score = 0.5*entropy + 0.5*gini

**File: `src/gfn/graph_builder.py`**
- Added `compute_levels_undirected()` for BFS on undirected graph

**File: `src/gfn/run.py`**
- community_score now uses full composite (not just modularity)
- legacy_hps_score = modularity (for before/after correlation comparison)
- Correlation heatmaps: before (legacy_hps) vs after (new hps)

---

## C) CASDI Model and Ablation

**File: `src/gfn/casdi.py`**
- Already implements CASDI formula with robust z-score
- Ablations: CASDI_full, CASDI_minus_spectral, CASDI_minus_core, CASDI_minus_bridge, CASDI_minus_centralization, CASDI_minus_virality

**File: `src/gfn/run.py`**
- Batch CASDI computation from per-event components
- CASDI ablation evaluation; save to outputs/tables/casdi_ablation.csv
- Figure: outputs/figures/casdi_ablation.png

**File: `config.yaml`**
- CASDI weights: alpha, beta, gamma, delta, epsilon (already present)

---

## D) Advanced Graph Features

**File: `src/gfn/advanced_features.py`**
- 10 features: assortativity, transitivity, avg_clustering, diameter_lcc, avg_shortest_path_lcc, degree_gini, pagerank_entropy, kcore_max, depth, breadth
- Safe fallbacks for small graphs
- _entropy edge-case fix

**File: `src/gfn/run.py`**
- Integrated AdvancedFeatures into unified feature table

---

## E) Evaluation & Comparison

**File: `src/gfn/run.py`**
- Hold-out 80/20 stratified
- 5-fold CV
- Bootstrap 95% CI for F1 and ROC-AUC
- Cross-dataset: train Twitter15→test Twitter16, train Twitter16→test Twitter15
- Feature sets: each baseline (1D), AdvancedFeatures, CASDI_full, CASDI_full+adv, CASDI ablations
- 3 classifiers: Logistic Regression, Random Forest, Gradient Boosting

---

## F) Statistics

**File: `src/gfn/statistics.py`**
- Mann-Whitney U test
- Cliff's delta effect size
- compute_all_tests() for feature table

**File: `src/gfn/run.py`**
- Saves outputs/tables/statistical_tests.csv

---

## G) Outputs & Figures

**File: `src/gfn/run.py`**, **`src/gfn/visualization.py`**
- boxplots_scores.png, violinplots_scores.png
- correlation_heatmap_after.png, correlation_heatmap_before_vs_after.png
- roc_all.png, model_comparison.png (with CI), casdi_ablation.png
- distributions_key_features.png, feature_importance.png
- confusion_matrix_best.png, calibration_curve_best.png, pr_curve_best.png
- cross_dataset_summary.png

**File: `src/gfn/run.py`**
- outputs/reports/summary.md
- outputs/reports/pipeline.png (matplotlib flowchart)
- outputs/tables/ranking_table.csv, error_analysis.csv

---

## H) Logging & Diagnose

**File: `src/gfn/run.py`**
- Logging to outputs/logs/run.log
- Diagnose mode: subset folders, label files, example event file, event counts, labeled counts, parsed edges, skipped reasons

**File: `src/gfn/paths.py`**
- Extended diagnose() with n_labeled, example_event_file, example_edges_parsed

---

## I) README Update

**File: `README.md`**
- Corrected spectral formulation (LCC, laplacian_energy formula)
- New HPS definition
- CASDI formula and ablation
- List of outputs (tables, figures, reports)
- How to run on Windows

---

## J) Quality

- Type hints, dataclasses where applicable
- Deterministic seeds (config.evaluation.random_state)
- Defensive programming (try/except, fallbacks)
- Scipy/graphviz graceful fallback

---

## Files Modified
- src/gfn/spectral.py
- src/gfn/hps.py
- src/gfn/graph_builder.py
- src/gfn/community.py (run.py uses community_score)
- src/gfn/run.py (major rewrite)
- src/gfn/casdi.py (no structural change)
- src/gfn/advanced_features.py
- src/gfn/statistics.py
- src/gfn/visualization.py
- src/gfn/paths.py
- src/gfn/config.py (unchanged)
- config.yaml (unchanged)
- README.md

## Files Added
- CHANGES.md

## Commands to Run

```bash
# Diagnose
python -m src.gfn.run --config config.yaml --diagnose

# Full pipeline
python -m src.gfn.run --config config.yaml --run-all
```

## Key Output Locations
- Tables: outputs/tables/
- Figures: outputs/figures/
- Reports: outputs/reports/
- Logs: outputs/logs/run.log
