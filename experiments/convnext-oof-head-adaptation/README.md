# ConvNeXt OOF-corrected head adaptation

This private no-submission experiment tests the strongest unfinished signal in
the repository: authenticated fold-held-out image-teacher correction improved a
fresh ConvNeXt head, but the fresh head remained below the strong published
head. This screen therefore starts from the exact published fold-0 ConvNeXt
reader head instead of random initialization and applies a short, fixed,
low-learning-rate adaptation.

Both arms start from identical public head weights and consume the immutable
fold-0 token cache. The control adapts to Steven v4 report targets; the candidate
uses the fixed 50/50 Steven/true-OOF correction for exactly 2,610 covered
non-gold studies and leaves 1,739 studies unchanged. Five fixed seeds control
dropout and batch order. Training is fixed at five epochs with learning rate
`2e-5`; no checkpoint, epoch, target, or blend is selected.

All 58 organizer-labelled studies are excluded from fitting and evaluated only
after every adapted head is frozen. The notebook cannot create a submission.

## Predeclared gate

Advance only if every condition passes:

- candidate ensemble improves over its paired Steven-control adaptation by at
  least `+0.002` macro AUC;
- candidate ensemble improves over the untouched public fold-0 head by at least
  `+0.002` macro AUC;
- at least four of five candidate-minus-control seed deltas are positive;
- paired bootstrap probability of improvement is at least `0.90`;
- paired bootstrap `q025` is greater than `-0.001`;
- all identity, count, finiteness, hash, no-CSV and no-submission checks pass.

A pass authorizes a separate inference-capable implementation only. It does not
authorize a competition submission by itself.

## Evidence and attribution

- The controlled 2,610-study RadImageNet student screen improved by `+0.018145`
  with `5/5` positive seeds and bootstrap `p=0.993`.
- The full-population public ConvNeXt feature-head screen improved by `+0.003693`
  with `4/5` positive seeds but started from random head weights and remained
  below the published head. This experiment directly addresses that limitation.
- `goodpjw2008/rsna-knee-2-5d-convnext-reader` (Apache-2.0 source/checkpoint)
- `stevenleehans/rsna-knee-llm-report-labels`

