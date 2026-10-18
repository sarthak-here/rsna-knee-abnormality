# R6 dense-window localization screen

This private no-submission screen builds directly on the strongest R6 reader.
It changes only its deterministic evaluation window stride from the published
`3` to `2`, increasing through-plane coverage while preserving the exact four
checkpoints, physical/anatomy-centred v4 views, 140 mm preprocessing, flip TTA,
side correction, state stackers, architecture, and every other inference
setting.

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
