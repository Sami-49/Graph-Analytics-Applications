# Audit

## Metrics
- AUC computed from continuous scores (score-only).
- If AUC<0.5, score orientation is flipped and reported in univariate_scores.csv.

## Feature QC
- (no QC warnings)
## Warnings
- score_only:twitter16:community_score: AUC≈0.5 but F1 is high: check AUC input (hard-label bug), class imbalance, or thresholding.
