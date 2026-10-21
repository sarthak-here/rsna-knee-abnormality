# Version1 terminal result — REJECTED / CLOSED

Kaggle reported COMPLETE. Downloaded version1 outputs and log, then ran `audit_terminal.py` against the pinned original fold0 checkpoint. The audit passed: exact source/input/split identities, all three checkpoint hashes, finite adapter parameters/optimizer states, 576 optimizer updates, every original model tensor unchanged through fc2.base remapping, pickle-free ordered tuning predictions, all valid state/probability constraints and reproduced metrics/gate booleans. All six adapter tensors changed according to the training report; all original tensors were independently compared at each epoch. No recorded numerical retries. Zero reported reader failures. No CSV/submission, gold evaluation or traceback. Nonfatal warnings remain.

Inclusive reported runtime: 4588.963 seconds (76.483 minutes), under the fixed three-hour cap. Eight-real-study preflight passed before optimization. Temporary GPU prefix-cache bytes were not retained for local rehash or pixel replay; their receipts were validated, not the original cache bytes.

## Frozen tuning gates

- Macro report-state NLL gain **0.0007594828859160144**, below required **0.01**: FAILED.
- Six eligible classes, all six nonnegative gains: passed coverage/class gates.
- Paired study-bootstrap positive fraction **1.0**, q025 **0.0003009353689689071**: passed uncertainty gates.
- Mean probability drift **0.0001993294048555652**, maximum **0.006821766495704651**: passed drift gates.
- Confirmation was NOT opened; no confirmation predictions exist.

This contract is rejected because the magnitude gate failed, despite a small consistent report-proxy improvement. It is not independent pathology accuracy evidence and establishes no gain over the scored0.958 ensemble. No rank/LR/epoch/target/gate changes, rescue blend, hidden inference or competition submission are authorized by this result. Preserve the artifacts and close this exact pilot. Selected submissions57483292/57197157 remain unchanged.

## Checkpoints

1. `22d25d4d232bd1eba6cbea9b5b397036ac5df87728a9fd46da52f839c06869b4`
2. `5a35c571a09feb1a1d7145944491d2b9cc095cea1127492fbfdaf4806c6f6737`
3. `1b488e45ee6f9905241260b79a2f22a588a29ad5cce765d3d2406a65b7f9aa69`

Local immutable download: `.audit/r6-lowrank-v1/output`. Public-safe terminal hashes and full metric receipt: `v1_terminal_receipt.json`; no per-study target/prediction data published.
