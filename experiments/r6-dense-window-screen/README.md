# R6 dense-window localization screen

**Identity correction, 20 October 2026:** the original description below was
wrong about the reference settings. The pinned released R6 manifest sets
`eval_stride=1` and `phys_mm=3.5`; the upstream controller reads those values
from the manifest. Its `.get(..., 3)` expression is a missing-key default, not
the actual deployed stride. This experiment patched the stride to `2`, making
coverage **sparser**, not denser. The local anatomy-centred crops are 100 mm;
140 mm describes a different generic physical cache, not this R6 setting.

The exact four checkpoints, v4 views, flip TTA, side correction and stackers
were retained. The stored stride-2 outputs and rejection remain unchanged.
This correction does not authorize another stride test or reopen the contract.
Evidence and source hashes are in
`../r6-fold0-warmstart/PREPROCESSING_IDENTITY.md`.

The 58 organizer-labelled studies remain evaluation-only. The frozen reference
is the immutable four-reader prediction cache SHA-256
`6cdea905549c3df74d94c08a0547a556c9b3a890f6b1d90e432f0b79d3b0c124`.
The anchor is `[r6_flip, raptor384, raptor224, r7]` at
`[0.50,0.325,0.175,0]`. The candidate replaces only `r6_flip` with the exact
stride-2 output; all weights stay fixed. There is no target, stride, checkpoint,
or blend search, and the notebook cannot submit.

Advance only if every gate passes: macro gain `>=0.001`, at least 8/12
nonnegative target deltas, paired-bootstrap probability positive `>=0.90`,
q025 `>-0.001`, all 58 studies scored with zero fallback, finite probabilities,
exact source/cache identities, and no retained CSV or submission artifact.
Failure closes this exact stride-2 contract without trying another stride on
organizer gold.

The frozen Raptor checkpoints in the evaluation anchor are OAI-trained, as
already disclosed. They are not rerun or changed by this experiment.
