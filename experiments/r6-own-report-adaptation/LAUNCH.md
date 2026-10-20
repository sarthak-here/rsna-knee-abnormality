# Actual private launch — 20 October 2026

Kaggle accepted **version 1** of
`sarthaksharma14/rsna-knee-r6-own-report-adaptation` and subsequently returned
`KernelWorkerStatus.RUNNING`. This is an actual created run, not a planned job.

Notebook: https://www.kaggle.com/code/sarthaksharma14/rsna-knee-r6-own-report-adaptation

Implementation commit `2e0e868`; rule preregistration `681c8d3`, conservative
synthetic-negation correction before any model result `f89f95d`.
Exact version-1 notebook SHA-256:
`51a90a9a6b5332a93fd8dd345df650ec126d21e9f9772d9bac08b4c007ccb7df`.
Rules SHA-256:
`e10ed7ed45841ab058ffe6926e8bddf01e46a67da2fcf149c6dae0f49569b736`.
Split SHA-256:
`58e14aace2828174c2e96a505630151fa930941051fc531b2a49a8faca2edc81`.

## Verified before launch

- Ten tests pass: rule negation/ambiguity, ignored unknowns, fixed source-fold /
  report-group split, gold parser exclusion, bootstrap/drift gates, padded-state
  KL finiteness. No organizer-gold model score or confirmation loss was inspected.
- Frozen v1.1 rules cover 999 eligible fitting report groups and 303 guarded
  source-fold-0 holdout groups. Selected 192/64/64 groups are disjoint. All 58
  organizer-labelled studies and four exact-report duplicates are excluded.
- Local RTX3060 synthetic GPU graph preflight passed for the **actual float16
  cached representation**. Prefix shape `[32,384,22,22]`; maximum class/state/
  binary logit discrepancies vs full forward were approximately
  `0.000538 / 0.000977 / 0.003906`, below predeclared0.02 tolerance.
- Gradients reach all and only31 final-stage tensors. Local peak allocation
  ~818MB; measured final-stage backward1.752sec. These are not T4 timings or
  accuracy results. No optimizer step occurred in the local preflight.

## What the running notebook does

Prepares the exact original R6 source/medical locator/weights, then measures
eight actual study renders and an actual-device backward before optimization.
If the conservative total projection exceeds the hard three-hour notebook
budget it writes `budget_rejection.json` and does not train. The subprocess also
has a hard deadline; a partial model never passes advancement gates.

If feasible: cache frozen-prefix activations for320 selected studies, train only
the original final stage for three fixed epochs, preserve all rich heads and
state/fluid coefficients, and compare against the untouched same-fold reader.
Only after every tuning gate passes may the sealed64-study confirmation set be
evaluated once. A pass is report-proxy evidence, not independent clinical or
public-leaderboard improvement. No automatic larger training or submission.

## Monitoring and terminal audit

Do not duplicate while queued/running. On terminal state download the exact
version-1 outputs and logs. Require source/input/build/split/checkpoint identities,
finite valid predictions and objectives, only last-stage weights changed, complete
cache/step/epoch receipts, zero reader failures, no CSV/submission, correct MCL
plain-only exception and unchanged postprocessors. Invalid source state slots
are legitimately -infinity before softmax; valid slots and predictions must be
finite. Explicit published anatomy slab-prior paths are logged separately from
reader failures, never silently counted as segmentation successes.

Accept an explicit budget rejection as a stopped experiment, not a model result.
Fix only reproducible runtime/mount/dtype/shape/dependency/receipt issues without
changing rules, split, objective, epochs, LR, windows, budget or gates after
results. Preserve every terminal result. The locked0.957 submission is untouched.
