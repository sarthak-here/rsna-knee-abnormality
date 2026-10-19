# R6 full-population fold-logit cache

This private inference-only experiment extends the exact validated R6
flip-TTA reader to all 4,407 competition training studies. It changes no model,
checkpoint, preprocessing, 140 mm physical view, anatomy centring, side
correction, state stacker, evaluation stride, or TTA behavior.
Version 1 proved that a single run cannot finish all 4,407 studies before the
fixed worker deadline: the two workers reached only 1,636 and 1,370 genuine
predictions, then emitted forbidden fallbacks. The repaired runtime contract
uses two deterministic, disjoint positional shards (`row_index mod 2`) of
2,204 and 2,203 studies. Each shard still uses both GPUs, so each worker now
processes about 1,102 studies and remains inside the same 11-hour bound. The
union is exactly the original ordered 4,407-study population. No model,
checkpoint, preprocessing, stride, TTA, or prediction logic changes.

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

`merge_shards.py` fails closed unless both immutable shard receipts report zero
fallbacks, hashes match, each ID list exactly equals its positional slice of
the official training manifest, and the reconstructed arrays have the pinned
shapes, order, finiteness, and probability range.
