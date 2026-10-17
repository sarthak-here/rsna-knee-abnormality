# Pairwise-AUC ConvNeXt head

## Motivation

The cross-plane architecture screen proved that extra token mixing alone does
not improve the frozen anchor. The next experiment changes the learning signal
instead of adding more architecture: train the published target-specific
ConvNeXt attention head directly for ranking, which is what the competition's
macro AUC measures.

Existing public and private heads use binary cross-entropy against noisy soft
report targets. That objective spends capacity calibrating probabilities and
treats small uncertain-label differences like confident orderings. This build
uses a weighted soft pairwise-AUC loss: within each batch and target, only pairs
whose corrected target probabilities differ by at least `0.10` contribute, and
each pair is weighted by its target difference. A fixed 15% BCE term preserves
basic calibration while 85% of the objective learns ordering.

## Frozen contract

- Representation: immutable ConvNeXt fold-0 feature cache, shape
  `[4407,6,12,768]`, SHA-256
  `7f1986bb30cf92dbbf440b43a5a5933c1a3d13a0bb61d91950457cb0f8e018cc`.
- Head: exact public `KneeNet.head` implementation from `knee.py`, SHA-256
  `314c2af387a5cbd1d25f192f19f4ecc6d4ae4b30403e92e9d9b3ed2c325880b1`.
- Fit set: exactly 4,349 non-gold competition studies.
- Targets: Steven v4, with fixed 50/50 authenticated true-OOF correction for
  exactly 2,610 studies and 1,739 unchanged studies.
- Objective: `0.85 * weighted_pairwise + 0.15 * BCE`, pair threshold `0.10`.
- Training: five fixed seeds, 20 fixed epochs, no checkpoint selection.
- Evaluation: all 58 organizer-labelled studies are held out until every model
  is frozen.
- Frozen anchor: the four-reader cache at weights `[0.50,0.325,0.175,0]`.
- Candidate: fixed 10% rank blend of the five-seed pairwise reader into the
  frozen anchor. No target-specific or post-hoc weights.
- Output: report, prediction NPZ, and checkpoints only; every CSV is forbidden.

## Promotion gate

The inference path is built only if all conditions pass:

- macro gain at least `+0.002` over the frozen anchor;
- at least 8/12 nonnegative target deltas;
- at least 4/5 seed blends improve;
- bootstrap `p_improve >= 0.90`;
- bootstrap q025 greater than `-0.001`;
- complete identity, isolation, hash, finiteness, and no-CSV checks.

Failure closes this exact objective/head/blend contract without tuning.

## Attribution

- `goodpjw2008/rsna-knee-2-5d-convnext-reader` (Apache-2.0 source)
- `stevenleehans/rsna-knee-llm-report-labels`
- Heliosli, nartaa, dreaddevelopment, and You WeiLin for the frozen evaluation
  anchor. Its Raptor arms are OAI-trained; this private screen cannot submit.
