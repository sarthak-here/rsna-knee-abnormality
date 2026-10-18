# Label-free multireader consensus gate

This one-shot CPU screen tests whether the R7 reader's demonstrated diversity
can complement the locked R6 + Raptor384/224 anchor without fitting any weight
to organizer labels. It consumes the immutable, hash-pinned four-reader
58-study prediction cache.

For each target, predictions are converted to ranks. The frozen anchor remains
`[0.50, 0.325, 0.175, 0]`. A label-free consensus is the per-study median rank
of R6, Raptor384, and Raptor224. R7 receives a study/target-specific slot
`0.10 * exp(-abs(r7-consensus)/0.10)`. The slot is taken proportionally from
the anchor. Thus R7 contributes at most 10% when it agrees with the independent
three-reader consensus and is exponentially suppressed when it is an outlier.
There is no target selection, parameter search, or fallback.

## Predeclared gate

Advance only if every condition passes:

- macro AUC gain over the frozen anchor is at least `+0.001`;
- at least `8/12` target deltas are nonnegative;
- paired bootstrap probability of improvement is at least `0.90`;
- paired bootstrap q025 is greater than `-0.001`;
- all identity, shape, finiteness, and no-submission checks pass.

The organizer labels are read only once after the algorithm is frozen. A failed
screen closes this exact gate; its scale and temperature cannot be tuned.

