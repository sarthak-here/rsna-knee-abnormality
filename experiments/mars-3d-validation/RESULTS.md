# MARS 3D foundation model: rejected

The private attributed MARS volumetric validation completed successfully and did not create a competition submission.

## Result

- Incumbent ensemble macro AUC: **0.860510**
- MARS 3D ensemble macro AUC: **0.656439**
- Paired delta: **-0.204071**
- Positive seed deltas: **0/5**
- Paired-bootstrap probability of improvement: **0.0000**
- Paired-bootstrap 95% interval: **[-0.256902, -0.152744]**

The official Apache-2.0 MARS checkpoint matched the pinned 82,121,520-byte artifact and SHA-256 `88388a59d92935a15d30bd41c0e9dbece99ba78db73cb374d44290fd6566ccc3`. Sequence coverage was 100%; 6.62% of plane tokens used the predeclared same-plane fallback, no cross-plane fallback was needed, and no sequence token remained missing. All 58 organizer-labelled studies remained evaluation-only (`gold_fit_count = 0`, `gold_evaluation_count = 58`).

Every seed was strongly negative, and weak-validation AUC was also substantially below the control. This rejects the generic frozen MARS representation for this task; it is not merely an unlucky gold-set result.

## Decision

Reject this MARS 3D branch. Do not blend it, tune its target subset, or submit it. Continue with a materially different candidate that adapts supervised image features end-to-end rather than substituting another frozen generic encoder.
