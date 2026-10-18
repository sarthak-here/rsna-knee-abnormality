# ConvNeXt OOF logit-residual screen

This private no-submission experiment preserves the exact published ConvNeXt
fold-0 reader and learns only a bounded correction residual. The residual target
is the logit difference between Steven v4 labels and the authenticated 50/50
Steven/true-OOF labels for exactly 2,610 covered non-gold studies. The remaining
1,739 non-gold studies and all 58 organizer-labelled studies are excluded from
residual fitting.

The representation is fixed before evaluation: masked per-slot means and
standard deviations from the immutable `[4407,6,12,768]` feature cache, followed
by a fixed 384-component sparse random projection fitted without labels.
Five fixed 85% bags fit ridge residuals with `alpha=100`; predicted corrections
are clipped to `[-1,1]` logit. There is no epoch, checkpoint, target, blend, or
hyperparameter search. The frozen four-reader anchor receives a fixed 10% rank
slot from the corrected public reader.

## Predeclared gate

Advance only if every condition passes:

- corrected-reader macro AUC improves over the untouched public head by at
  least `+0.002`;
- fixed 10% residual blend improves over the frozen anchor by at least `+0.001`;
- at least `8/12` target deltas are nonnegative;
- at least `4/5` seed blends improve over the anchor;
- paired bootstrap probability is at least `0.90` and q025 is above `-0.001`;
- all identity, hash, count, finiteness, no-CSV, and no-submission checks pass.

A pass authorizes a separate inference implementation only. Gold labels are
evaluation-only and cannot be used to alter this contract.

## Attribution

- `goodpjw2008/rsna-knee-2-5d-convnext-reader` (Apache-2.0)
- `stevenleehans/rsna-knee-llm-report-labels`

