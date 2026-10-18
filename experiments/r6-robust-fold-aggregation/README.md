# R6 robust fold aggregation screen

This is a predeclared, CPU-only, no-submission test built directly on the exact
four-fold R6 logits already captured by the fail-closed uncertainty-router run.
It tests whether the ordinary mean is too sensitive to one fold's extreme
logit. For each study and target, plain/flip logits are first averaged within
each fold; the lowest and highest of the four fold logits are then discarded
and the middle two are averaged. This is the unique four-value 25% trimmed
mean. No target, fold, threshold, or blend weight is selected.

The R6 Effusion and Baker's outputs are preserved exactly because the published
reader replaces their raw model logits with pinned state stackers. The other
ten targets use the robust fold aggregate. The frozen Raptor half remains
`0.65 * rank(Raptor384) + 0.35 * rank(Raptor224)`, and the final candidate keeps
the locked 50/50 R6/Raptor rank blend.

Input is the immutable 58-study cache SHA-256
`92ef1f26200069ae7b3e7e679b59bf30ed3556cf669ccdfe40fccb0cb1046623`.
All organizer labels are evaluation-only. The evaluator cannot emit a CSV or
submit.

Advance only if all gates pass: macro gain `>=0.001`, at least 8/12 target
deltas nonnegative, paired-bootstrap probability positive `>=0.90`, q025
`>-0.001`, exact mean-logit reconstruction for the ten non-stacker targets,
finite probabilities, and complete identity checks. Failure closes this exact
aggregation rule without post-hoc target selection or tuning.

The frozen Raptor checkpoints are OAI-trained, as already disclosed for the
locked baseline family. This local screen creates no new model or submission.

## Terminal result

The exact one-shot evaluator passed all identity and reconstruction checks but
was rejected scientifically. The anchor scored `0.9387940564`; the robust
candidate scored `0.9383339113`, a delta of `-0.0004601451`. Although 9/12
target deltas were nonnegative, bootstrap probability positive was only
`0.38684` and q025 was `-0.00354963`. The ordinary mean-logit reconstruction
maximum absolute error was `7.96e-8`, confirming that the comparison used the
intended public R6 outputs. This exact reducer is closed and must not be tuned
or submitted.
