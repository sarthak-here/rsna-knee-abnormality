# Cross-plane token residual screen

## Objective

Add a genuinely different reader to the locked public `0.957` stack without
re-running the image encoder or searching blend weights on the public
leaderboard. The experiment trains a cross-plane target-query Transformer on
the immutable public ConvNeXt token cache and gives it one fixed 10% rank slot
beside the frozen anchor.

The cached representation contains 72 tokens per study (`6` clinical sequence
slots x `12` physical windows x `768` features) for all 4,407 competition
training studies. Unlike the published head, this model first performs
within-sequence attention, then exchanges information across the six sequence
slots, and finally uses one learned query per abnormality. This directly tests
whether cross-plane anatomy/context is the missing complementary signal.

## Supervision and isolation

- Fit set: exactly 4,349 non-gold competition studies.
- Base targets: Steven v4 report probabilities.
- Authenticated image correction: fixed 50/50 report/true-OOF ConvNeXt target
  for the 2,610 studies covered by folds 0-2; the other 1,739 remain unchanged.
- Organizer-labelled studies: all 58 are excluded from fitting, model
  selection, epoch selection, and blend selection.
- Training: five predeclared seeds, fixed 24 epochs, no checkpoint selection.
- Frozen anchor arms: `[r6_flip, raptor384, raptor224, r7]` with weights
  `[0.50, 0.325, 0.175, 0.0]`.
- Candidate: per-target ranks of `0.90 * anchor + 0.10 * cross-plane reader`.
  The weight is immutable and is not searched on organizer gold.

The notebook is a private validation screen. It deletes every CSV and cannot
create a competition submission.

## Predeclared gate

Advance to an inference build only if every condition passes:

- blended macro AUC gain is at least `+0.002` over the frozen anchor;
- at least 8/12 target deltas are nonnegative;
- at least 4/5 individual seed blends improve over the anchor;
- paired bootstrap probability of improvement is at least `0.90`;
- paired bootstrap q025 is greater than `-0.001`;
- all identity, shape, finiteness, isolation, and no-submission checks pass.

Failure closes this exact model/target/blend contract. There will be no
post-hoc target removal, seed selection, blend-weight search, or public-LB
tuning.

## Pinned inputs

- ConvNeXt feature cache SHA-256:
  `7f1986bb30cf92dbbf440b43a5a5933c1a3d13a0bb61d91950457cb0f8e018cc`
- Authenticated OOF prediction SHA-256:
  `60529a0ba00107abd7efd101f70454e7ce89407befbaa4107f24771ee088579e`
- Frozen four-reader gold cache SHA-256:
  `6cdea905549c3df74d94c08a0547a556c9b3a890f6b1d90e432f0b79d3b0c124`

## Attribution

- `goodpjw2008/rsna-knee-2-5d-convnext-reader` (Apache-2.0 source)
- `stevenleehans/rsna-knee-llm-report-labels`
- Heliosli, nartaa, dreaddevelopment, and You WeiLin for the frozen reader
  stack used only as the immutable evaluation anchor.

The frozen Raptor arms are OAI-trained, as already disclosed for the locked
baseline family. This screen creates no submission.
