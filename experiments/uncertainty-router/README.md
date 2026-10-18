# Fold-uncertainty dynamic router

## Objective

Use information already produced by the strongest operational models instead
of performing another static public-leaderboard reweight. The candidate routes
each study/finding between the R6 four-fold ConvNeXt reader and the Raptor
384/224 pair according to model disagreement.

This is materially different from the rejected per-finding weight searches:
the routing weight varies by study, uses no label during routing, and is fully
predeclared before the 58 evaluation labels are read.

## Immutable inputs

- exact R6 bundle `fe055406fad6a12f`, with plain and horizontal-flip logits from
  all four folds;
- exact Raptor-384 and Raptor-224 probabilities already used by the locked
  stack;
- exact preprocessing and checkpoint hashes from the validated public kernel;
- the 58 organizer-labelled studies only for one terminal evaluation.

The cache must contain exactly 58 unique IDs, truth `[58,12]`, the exact final
R6 probabilities `[58,12]`, aligned R6 logits `[58,4,2,12]` (`plain`, `flip`),
and Raptor probabilities `[58,2,12]`. It must be finite, fail closed, contain
no fallback predictions, and create no submission CSV.

## Fixed router

For each target independently:

1. average each R6 fold's anatomically aligned plain/flip logits for uncertainty
   only; use the exact final R6 probabilities, including its frozen stackers,
   for prediction;
2. define R6 uncertainty as fold-logit standard deviation;
3. define Raptor uncertainty as the absolute logit disagreement between its
   384px and 224px checkpoints;
4. percentile-rank each uncertainty over studies without labels;
5. set the Raptor share to
   `clip(0.50 + 0.20 * (rank(R6 uncertainty) - rank(Raptor uncertainty)),
   0.45, 0.55)`;
6. blend percentile-ranked R6 and Raptor predictions using that per-study
   share, then rank the result.

The Raptor pair remains fixed at `0.65 * rank(384) + 0.35 * rank(224)`. There
is no target selection, coefficient search, calibration fit, or gold-dependent
fallback.

## Promotion gates

Advance only if all checks pass:

- macro AUC gain over the static 50/50 anchor is at least `+0.001`;
- at least 8/12 targets are nonnegative;
- paired bootstrap probability of improvement is at least `0.90`;
- paired-bootstrap q025 is greater than `-0.001`;
- router weights are nonconstant and remain within `[0.45,0.55]`;
- all identity, shape, finiteness, zero-fallback, and no-submission checks pass.

Failure closes this exact formula. Do not tune the coefficient, clipping range,
target subset, or uncertainty definitions after seeing gold results.

## Build sequence

- [x] Predeclare the routing formula and gates.
- [x] Implement a fail-closed evaluator with synthetic tests.
- [ ] Patch the exact 58-study adapter to retain four-fold R6 plain/flip logits.
- [ ] Launch the private no-submission cache and run the evaluator unchanged.
- [ ] Build hidden inference only if every gate passes.

## Launch status

Version 1 was created without the required R6-v1 mirror because the metadata
mistakenly named that public dataset under the notebook owner's account. Kaggle
reported the source as invalid at push time. Version 2 corrects only the dataset
owner from `sarthaksharma14` to `heliosli`; model code, identities, routing
formula, evaluator, and gates are unchanged.

The Raptor checkpoints are OAI-trained and retain their existing provenance and
acknowledgements. No first-place claim is made.
