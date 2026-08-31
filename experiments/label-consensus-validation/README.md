# Weak-label consensus validation

This private experiment tested whether combining independently published report-derived weak labels improves label quality on the 58 organizer-labelled gold studies. The gold studies were evaluation-only and were excluded from all fitting.

## Public label sources

- `lixin73/rsna-knee-llm-report-labels-sol56`
- `stevenleehans/rsna-knee-llm-report-labels`
- `pilkwang/rsna-knee-llm-labels`

The source datasets remain subject to their original Kaggle provenance and licenses.

## Results

The strongest individual source was Steven v4 at `0.892707` macro AUC. A predeclared three-source rank mean reached `0.892740`, a negligible `+0.000033`; bootstrap support was not robust (`p_improve=0.54`, 95% interval approximately `[-0.00627, 0.00662]`).

A target-wise nested stratified five-fold weight search was repeated 20 times. It underperformed the Steven v4 baseline on every repeat: mean delta `-0.009813`, `0/20` positive repeats, and a 5th-to-95th percentile interval of `[-0.015556, -0.004239]`.

## Decision

Reject source-level consensus and target-wise weight fitting as a production weak-label rule. No competition submission was produced. Future work should improve report parsing or use physically correct MRI preprocessing and sequence selection, while preserving the gold studies for evaluation only.
