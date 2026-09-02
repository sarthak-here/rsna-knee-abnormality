# Adjacent-slice sequence-context SSL: rejected

The private label-free sequence-context validation completed successfully and did not create a competition submission.

## Result

- Incumbent ensemble macro AUC: **0.860451**
- Sequence-context ensemble macro AUC: **0.834912**
- Paired delta: **-0.025539**
- Positive seed deltas: **0/5**
- Paired-bootstrap probability of improvement: **0.0060**
- Paired-bootstrap 95% interval: **[-0.046403, -0.005725]**

The adapted checkpoint matched the recorded 94,347,671-byte artifact and SHA-256 `c900c6c240dad079522e6e9be200b2045635c8d25246cffe414918a02b6a1a43`. All 3,922 fitting studies produced an adjacent-slice pair; no report labels entered pretraining. The fixed weak-validation set contained 427 studies with zero report-group overlap. All 58 organizer-labelled studies remained evaluation-only, and the execution log contained no runtime-error markers.

The InfoNCE objective converged and relative-order loss decreased, but the adapted representation was below the untouched control on every seed. This rejects the sequence-context contract rather than indicating a failed optimization run.

## Decision

Reject this branch. Do not tune its temperature, order-loss weight, epoch count, or target subset against the gold studies, and do not blend or submit it. Continue with a materially different evidence source.
