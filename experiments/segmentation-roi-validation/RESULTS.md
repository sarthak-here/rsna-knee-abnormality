# Segmentation-guided ROI: rejected

The private attributed KneeXNet-guided validation completed successfully and did not create a competition submission.

## Result

- Incumbent ensemble macro AUC: **0.860510**
- Segmentation-ROI hybrid macro AUC: **0.859660**
- Paired delta: **-0.000850**
- Positive seed deltas: **2/5**
- Paired-bootstrap probability of improvement: **0.4094**
- Paired-bootstrap 95% interval: **[-0.008339, 0.006794]**

The checkpoint SHA-256 matched the pinned public artifact. The run attempted 26,442 sagittal ROI masks and accepted 8,340 (31.54%); uncertain masks preserved incumbent tokens. There were no runtime-error markers. All 58 organizer-labelled studies remained evaluation-only (`gold_fit_count = 0`, `gold_evaluation_count = 58`), and no submission was produced.

Per-target deltas were mixed: lateral meniscus and lateral OA improved, while medial meniscus and both other OA targets declined. Because selecting targets after viewing gold results would introduce evaluation-set overfitting, no post-hoc target subset will be tested or submitted.

## Decision

Reject this segmentation-guided ROI branch. Do not tune its mask threshold, crop margin, or target subset against the gold studies. Continue with a materially different representation or supervision source.
