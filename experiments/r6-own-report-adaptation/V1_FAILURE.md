# Version 1: AMP runtime failure, not a model result

Kaggle returned ERROR on 20 October 2026. Exact v1 outputs and log are retained
under `.audit/own-report-v1`; their hashes and verified identities are recorded
in `v1_terminal_receipt.json`. Downloaded own-source hashes match v1 launch.

All 320 selected studies were cached in exact split order, with zero reported
reader failures and float16 `[32,384,22,22]` prefixes. Cache progress reached
3077.726 seconds. The eight-study real-GPU budget/parity gate passed, with maximum
logit error 0.00390625 and backward measurement 2.001 seconds. Temporary feature
files were not published, so this audit checks their receipt, not their bytes.

Training then raised `nonfinite gradient, refusing silent scaler-skipped step`.
No epoch checkpoint, tuning report, confirmation report or score was produced.
V1 did not persist per-step progress: an exact completed optimizer-step count is
not provable from its artifacts, and must not be invented. The error occurs after
the finite-objective check. Logs also contain dependency/nbconvert warnings;
they are not warning-free. No CSV/submission was retained.

## Reproduced numerical cause and bounded repair

`diagnose_amp.py` reconstructs the exact released heads and final stage and runs
synthetic backward only, without an optimizer or study inputs. At AMP's default
loss scale65536, the objective2.6764118671 is finite but gradients overflow. On
the identical input/RNG/objective, scale1024 and scale1 produce finite gradients
for all31 trainable tensors. See `amp_diagnostic.json`. This diagnoses a numerical
failure mode; it does not claim bitwise reproduction of the missing real prefix.

Version2 initializes GradScaler at1024, still fails closed on any nonfinite
objective/gradient and never silently skips optimization. The eight-study
preflight now measures the shared **unchanged complete training objective** and
scaled backward and checks every trainable gradient before full caching. Added
runtime-failure receipts record completed steps and scale if it fails again.

Rules, split, architecture, trainable scope, teacher, objective coefficients,
LR, seed, epochs, windows, validation gates and hard inclusive three-hour cap
are unchanged. No tuning or confirmation result was available to influence this
repair. Eleven CPU unit tests pass. This repair is not accuracy progress.
