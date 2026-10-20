# Terminal v3 result: integrity passed; scientific contract REJECTED / CLOSED

Kaggle version3 completed. `audit_terminal.py` verifies downloaded source/input/
split identities, every epoch checkpoint hash, all frozen tensors, optimizer
states, pickle-free predictions and the unchanged gate calculations. See
`v3_terminal_receipt.json`. No tuning/gate changes were made during this audit.

## Measured result

| Fixed measure | Tuning64 | Sealed confirmation64 |
| --- | ---: | ---: |
| Macro report-state NLL reduction | 0.0583852875 | 0.0596624117 |
| Eligible classes (at least5 known cells) | 7 | 8 |
| Nonnegative eligible class gains | 6 | 8 |
| Paired bootstrap positive probability | 1.000 | 1.000 |
| Bootstrap q025 | 0.0361980084 | 0.0360299940 |
| Mean postprocessed probability shift | 0.0088541560 | 0.0090625810 |
| Maximum postprocessed probability shift | 0.2748592496 | **0.4574139863 — FAIL** |
| Every fixed gate passes | Yes | **No** |

Confirmation was opened only after tuning passed every gate. Exactly one of768
confirmation probability cells exceeds the fixed0.30 shift cap: Fracture moves
from0.6299622059 to0.1725482196. This is an instability observation, not proof
that either probability is clinically correct. Do not relax the limit, clip or
blend this prediction after seeing it, select a previous epoch, suppress this
target or otherwise rescue this closed contract.

This establishes improved agreement with the frozen weak report-state labels on
additional-image-gradient-held-out studies. It does **not** establish independent
pathology improvement, whole-pipeline OOF, clinical accuracy or public-AUC gain.
No organizer-gold predictions/evaluation occurred. Locked public0.957 is unchanged.

## Training and integrity

- Three fixed epochs /576 optimizer updates; mean objectives
  0.8329522616,0.7098482345,0.6237621728. Epoch choice was not searched.
- All376 checkpoint keys, shapes and dtypes retained. Only the31
  `enc.stages.3.*` tensors changed; all other heads/prefix/priors/buffers exact.
- All three checkpoint hashes match receipts; optimizer states are finite and
  step counts are192/384/576 for every trainable tensor. RNG/scaler states retained.
- One numerical retry: epoch1 step5, scale1024 overflow, no optimizer update;
  identical study/RNG replay succeeded at512. All subsequent updates completed.
- All320 cache entries match fixed selected IDs/order; zero reported reader
  failures. Temporary prefix bytes were not published and cannot be rehashed
  locally; this limitation is explicit, not a claim of independent pixel replay.
- Both64-study NPZs load with `allow_pickle=False`; IDs, targets, dimensions,
  valid-state masks, padding, normalization and finite[0,1] probabilities pass.
- Gate booleans reproduce exactly. CPU/Kaggle floating reductions agree within
  1e-12 (observed differences about1e-17); no numerical threshold is relaxed.
- Complete training and notebook markers, no traceback, no CSV/submission.
  Nonfatal dependency/nbconvert warnings exist: logs are not warning-free.
- Model report wall time3746.545 seconds (~62.44minutes), within the hard3h cap.

Epoch checkpoint hashes:

1. `688aff41b9dac49d273988ee31020fba481ef7b4091335139fae4d4fa3432dbb`
2. `38af7c6cd09724ee6baa2691f2ea9b56a12b559e8eb498ccc9bb5f83b4137171`
3. `49fa6f6941a0c4bb3f2d28224967673546cf797c395186f9724c38709d361c95`

## Disposition

Close this exact rules/split/last-stage/objective contract. Do not rerun, tune,
target-select, blend, submit or build hidden inference for its checkpoints.
Preserve artifacts and attribution. No larger training is automatically allowed.

The meaningful next research question is whether training can improve trustworthy
pathology supervision without rare-example instability, not whether a different
post-hoc blend rescues this result. Any new contract needs a predeclared method,
measured compute and a genuinely untouched evaluation scope. The two inspected
64-study sets cannot be called sealed/fresh again. No new GPU job is launched
by this terminal result, and no accuracy promise follows from the proxy gain.
