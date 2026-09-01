# Domain-adaptive BYOL: rejected

The private ten-epoch BYOL run completed successfully and did not create a competition submission.

## Result

- Incumbent RadImageNet ensemble macro AUC: **0.860510**
- BYOL-adapted ensemble macro AUC: **0.833554**
- Paired delta: **-0.026956**
- Positive seed deltas: **0/5**
- Paired-bootstrap probability of improvement: **0.0164**
- Paired-bootstrap 95% interval: **[-0.054765, -0.002172]**

The adapted checkpoint SHA-256 was `54cad852295cfb7c17bef2184534ac27db04235cb1f7d484206aed73ffa5aff9`, matching the downloaded artifact. BYOL loss decreased from 0.299218 to 0.035433 without runtime errors, but both weak-validation and gold performance fell for every seed. All 58 organizer-labelled studies remained evaluation-only (`gold_fit_count = 0`, `gold_evaluation_count = 58`), and no submission file was produced.

## Decision

Reject single-slice BYOL domain adaptation of the RadImageNet encoder. The representation-learning objective converged but removed pathology-discriminative signal. Do not tune or submit this branch. The next experiment should change the spatial evidence supplied to the classifier rather than further adapting this encoder without labels.
