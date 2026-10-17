# Multi-reader nested blend

## Objective

Build a materially stronger candidate than submission `57197157` (public
`0.957`) without training another weak image encoder. The experiment reuses four
already operational readers:

1. R6 ConvNeXt with horizontal-flip TTA;
2. Raptor CoAtNet-384;
3. Raptor CoAtNet-224;
4. four-fold R7 ConvNeXt-small.

The current locked blend is the immutable anchor. Its reader weights are
`[0.50, 0.325, 0.175, 0.0]`: half R6, while the other half is Raptor with 35%
of that slot assigned to the 224-pixel checkpoint.

## Why this build

The public experiments show that no reader wins uniformly. GP958d's uniform R7
blend scored `0.956`, while Heliosli's per-finding R6/Raptor blend has credible
`0.958` evidence. That makes target-specific reader selection more promising
than another global weight tweak. The additional R7 arm is retained only where
cross-validated evidence supports it.

This is not a public-leaderboard search. Reader weights are selected from the 58
officially labelled studies using nested target-stratified folds. Every outer
prediction is produced with weights selected without that row. The optimizer
penalizes distance from the locked anchor and uses a coarse simplex grid to
limit variance.

## Build sequence

### Phase 1: immutable prediction cache

Create one private, no-submission Kaggle notebook that runs the exact pinned
R6, Raptor-384, Raptor-224, and R7 inference paths on the 58 labelled studies.
It must save:

- `multireader_gold_predictions.npz` with IDs, truth, arm names, and four
  prediction tensors;
- exact notebook, checkpoint, source, and prediction hashes;
- per-reader failure counts and runtime receipts;
- confirmation that no `submission.csv` exists.

The prediction cache is rejected if any reader falls back, any ID is missing or
duplicated, or any probability is non-finite or outside `[0, 1]`.

### Phase 2: nested weight screen

Run `optimize_nested_blend.py` against the immutable cache. For each finding:

- convert each reader to ranks;
- enumerate four-reader simplex weights in increments of `0.10`;
- use five target-stratified outer folds;
- choose weights only on each outer-training partition;
- optimize training AUC minus a fixed L2 penalty from the locked anchor;
- create one complete nested out-of-fold prediction vector;
- derive final deployable weights from the median outer-fold choice, projected
  back to the same grid.

No gold row may influence its own nested prediction.

### Phase 3: predeclared promotion gate

The candidate advances only if all checks pass:

- nested macro AUC improves over the locked anchor by at least `+0.002`;
- at least 8 of 12 findings are non-negative;
- at least 4 of 5 aggregate outer folds improve;
- paired bootstrap probability of improvement is at least `0.90`;
- paired-bootstrap 2.5th percentile is greater than `-0.001`;
- at least one finding assigns positive weight to R7 in at least 3 outer folds;
- all identity, isolation, finiteness, and no-submission checks pass.

Failure closes this branch. There will be no post-hoc target removal, grid
refinement, or public-leaderboard weight tuning.

### Phase 4: hidden inference candidate

Only after the gate passes, copy the frozen final per-finding weights into a
separate inference notebook. That notebook must preserve the four exact reader
paths, fail closed on every missing checkpoint or reader failure, verify hidden
test IDs/order/schema, emit a complete runtime receipt, and reproduce the
preview prediction hash before it can be submitted.

## Immediate deliverables

- [x] Predeclared scientific and runtime contract.
- [x] Fail-closed nested optimizer.
- [x] Exact four-reader 58-study prediction notebook.
- [ ] Private prediction-cache run and audit.
- [ ] Nested gate result.
- [ ] Inference notebook only if the gate passes.

The Raptor checkpoints are OAI-trained. Any eventual submission remains subject
to explicit user approval with that provenance disclosed.

## Launch status

The private notebook implementation was completed and pushed in commit
`79e0710`. Its builder pins both public source notebooks, redirects their exact
reader pipelines to a synthetic test-layout containing only the 58 labelled
studies, adds zero-fallback receipts, retains the four arms in the required NPZ,
runs the unchanged optimizer, and removes every CSV from the output directory.
Local structural generation and optimizer tests passed.

The first Kaggle launch attempt on 17 October 2026 was rejected before a kernel
version was created because the account had reached its `30.00` hour weekly GPU
quota. This is an infrastructure block, not an experiment result. No model code
ran and no scientific gate was evaluated. Launch the exact committed notebook
when GPU quota becomes available; do not alter its contract while waiting.

After quota reset, version 1 stopped before inference because the gold-layout
adapter looked for a nonexistent `train_images` directory. The official raw
DICOM directory is `train_series`, which is also the primary directory expected
by both pinned public readers. The adapter now links that directory as
`gold_eval/test_series`; no preprocessing, model, prediction, or gate logic was
changed.
