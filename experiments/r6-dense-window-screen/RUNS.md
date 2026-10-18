# Run history

## Version 1 — infrastructure failure before inference

Kaggle prepared the exact 58-study evaluation root, then stopped before model
setup or inference because the extracted R6 stage expected the public source
notebook's `NB_T0` timer variable. The compact screen had intentionally omitted
the earlier Raptor cells where that variable was originally initialized.

Version 2 adds only `NB_T0 = time.time()` before the unchanged preparation
cell. The stride-2 setting, model, preprocessing, TTA, checkpoints, anchor,
weights, evaluator, and gates are unchanged.

## Version 2 — complete, integrity pass, scientific rejection

Version 2 completed cleanly in 700.3 seconds with both T4 workers returning
zero, all 58 studies scored, zero fallbacks, finite `[58,12]` probabilities,
the exact anchor/source identities, clean logs, and no retained CSV or
submission artifact. Strict local `allow_pickle=False` validation reproduced
prediction SHA-256
`8681fa0fb16f0f445514c5a1b8201281d391133462fdbfbf1c7359f31af0a203`.

The frozen anchor scored `0.9387940564` macro AUC. Replacing only its R6 arm
with stride-2 inference scored `0.9391694047`, a gain of `+0.0003753484`.
Although 9/12 target deltas were nonnegative, the gain missed the predeclared
`+0.001` threshold; bootstrap probability positive was `0.64636` and q025 was
`-0.00179217`. The macro, bootstrap-probability, and lower-bound gates failed.

Decision: reject and close this exact stride-2 contract. Do not try another
stride on organizer gold, build hidden inference, blend, or submit it.
