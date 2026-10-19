# R6 cross-fitted fold/pass calibration

## Status: invalid design, withdrawn before execution

The 19 October research audit found that cross-fitting the ridge combiner does
not exclude validation studies from the pretrained R6 fold models. This is not
independent OOF evaluation of the complete predictor. The comparator also omits
the deployed Effusion fluid and Baker's state stackers. This contract must not
be run as an advancement gate, used to build hidden inference, or submitted.
No candidate score was observed and no alpha, formula, target, or gate was tuned.
The implementation and original predeclaration below remain preserved as audit
evidence. See `../research-audit-2026-10-19/README.md` for the exact findings.

## Original predeclaration (archived)

This is the falsification test for the hypothesis that the four published R6
folds and their plain/flip passes contain complementary ranking information
that the equal-logit mean discards.

The experiment consumes the immutable full-population R6 cache and does not
change the encoder, checkpoints, preprocessing, stride, side correction,
state/fluid stackers, or flip transform. For each target, one fixed ridge model
(`alpha=100`) combines the eight fold/pass logits. Models are evaluated with
five report-grouped cross-fitted predictions on all 4,349 non-gold studies.
Training targets are the predeclared 50/50 Steven-v4/true-OOF correction for
the 2,610 covered studies and unchanged Steven-v4 targets for the other 1,739.

The 58 organizer-labelled studies are never used for fitting, model selection,
hyperparameter selection, or checkpoint selection. Only after all cross-fitted
predictions and final all-non-gold coefficients are frozen are those 58 studies
evaluated once.

There is no search. The path advances only if every gate passes:

- non-gold cross-fitted macro AUC gain at least `0.003` over equal-logit mean;
- at least 8 of 12 non-gold target deltas are nonnegative;
- at least 4 of 5 report-group fold deltas are positive;
- non-gold paired bootstrap improvement probability at least `0.99` and
  2.5th percentile above zero;
- organizer-gold macro AUC gain at least `0.001` over equal-logit mean;
- at least 8 of 12 organizer-gold target deltas are nonnegative;
- organizer-gold paired bootstrap probability at least `0.90` and 2.5th
  percentile above `-0.001`;
- all identity, count, ordering, finiteness, hash, and no-submission checks.

Failure closes this exact calibration contract. Its alpha, features, folds,
targets, or gates must not be adjusted using organizer gold or leaderboard
feedback. Passing permits an inference implementation; it does not itself
authorize a competition submission.
