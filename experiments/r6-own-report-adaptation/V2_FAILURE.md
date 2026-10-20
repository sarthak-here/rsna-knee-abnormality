# V2: four optimizer updates, then scaled-gradient overflow

Exact downloaded source, split and input identities pass `audit_v2_failure.py`.
See `v2_terminal_receipt.json`. All320 studies cached in frozen order; zero
reported reader failures. Cached temporary feature bytes are not published, so
local validation covers receipt identities rather than rehashing those bytes.
The real-device scaled objective preflight passed with maximum parity error
0.00390625. Full cache progress reached3618 seconds.

`runtime_failure.json` establishes **four completed optimizer steps**, then a
nonfinite gradient on epoch1's fifth study at AMP scale1024, despite finite
objective3.1899046898. No epoch checkpoint, tuning result or confirmation result
exists. This is not a scientific rejection or any evidence of accuracy gain.
The initial-scale fix was insufficient for all real inputs.

## Runtime-only v3 repair

`finite_amp_step` uses bounded dynamic loss-scale backoff and replays the same
study/target with the same CPU/CUDA RNG states. A failed scaled backward takes
**no optimizer step**; the study is retried at half scale. Parameters do not
change until every gradient and its unscaled norm is finite. An intrinsically
nonfinite objective fails immediately; nonfinite gradients at scale1 fail closed.
The hard deadline applies to every retry. The model has no mutable training-mode
BatchNorm buffers in its trainable ConvNeXt stage; all other heads stay eval.
No sample is skipped and no loss, learning rate, batch order, clipping, epoch,
architecture, target, window, gate or three-hour budget is changed.

Thirteen tests pass, including a CPU float16 overflow reproducer that verifies
the identical RNG/input is replayed, no weight changes during retry, and exactly
one clipped optimizer update is ultimately taken. A separate intrinsic infinite
gradient at scale1 is rejected with no weight change. These are numerical unit
tests, not pathology/model-quality evidence. New numerical retry receipts record
each replay and scale. The incomplete v2 weights are not reused or evaluated.
