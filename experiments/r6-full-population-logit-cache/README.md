# R6 full-population fold-logit cache

This private inference-only experiment extends the exact validated R6
flip-TTA reader to all 4,407 competition training studies. It changes no model,
checkpoint, preprocessing, 140 mm physical view, anatomy centring, side
correction, state stacker, evaluation stride, or TTA behavior.
The worker deadline is expanded from the blend notebook's 8.3-hour allowance
to 11 hours because the two Raptor stages are absent and all 4,407 studies are
processed; this is a runtime-only change.

The cache retains, for every study and target:

- the exact final R6 probability `[4407,12]`;
- four-fold plain/flip logits `[4407,4,2,12]`, after the published anatomical
  mirror permutation and MCL plain-only rule;
- exact Unicode study ordering and fail-closed runtime receipts.

No labels are loaded into model inference, no training occurs, and every CSV is
removed after immutable NPZ packaging. This stage cannot submit.

If and only if the cache passes, the next separately predeclared CPU screen
will fit target-wise fold calibration exclusively on non-gold studies using
fixed report-grouped cross-fitting and the authenticated OOF-corrected targets.
All 58 organizer-labelled studies remain terminal evaluation-only. No
calibration or blend parameter may be selected using organizer gold.
