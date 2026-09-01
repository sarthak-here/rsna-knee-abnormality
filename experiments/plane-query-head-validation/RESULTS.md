# Plane-aware target-query head: rejected

The private five-seed validation run completed successfully and did not create a competition submission.

## Result

- Incumbent attention-head ensemble macro AUC: **0.860310**
- Plane-aware target-query ensemble macro AUC: **0.842993**
- Paired delta: **-0.017317**
- Positive seed deltas: **0/5**
- Paired-bootstrap probability of improvement: **0.0264**
- Paired-bootstrap 95% interval: **[-0.035436, 0.000106]**

All predefined acceptance gates failed. The 58 organizer-labelled studies remained evaluation-only (`gold_fit_count = 0`, `gold_eval_count = 58`), and no submission file was produced.

## Decision

Reject the explicit plane/slice transformer and finding-query architecture. Its additional capacity consistently reduced held-out performance, so this branch will not be tuned or submitted. Further work should move away from cached-feature head variants and evaluate a materially different, knee-MRI-specialized representation with verified model provenance and licensing.
