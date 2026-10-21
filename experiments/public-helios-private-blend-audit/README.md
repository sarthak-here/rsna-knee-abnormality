# Heliosli public 0.958 per-finding blend audit

**Our verified result:** approved exact version1 submission **57483292** completed at public **0.958**, verified22October2026IST. See `SUBMISSION.md` and `terminal_receipt.json`. Approval is resolved; do not request it again or duplicate this submission. The historical pre-approval discussion below is retained as provenance, not current status.

This audit tracks public kernel `heliosli/blend-mine-in-private-lb-luck-to-all-upvoters`, version 1. It is a reweight of the validated R6 flip-TTA + Raptor-384/224 stack and has current public-leaderboard evidence above the locked `0.957` baseline.

## Identity and runtime validation

- Notebook SHA-256: `ad0b97aca3361f16e32b0836655dd68bc2a37ff0cc5a44f6d6f4de07880d986d`.
- Inputs: `heliosli/rsna-knee-r6-frozen-fe055406-w1`, `nartaa/rsna-knee-publication-swa-weights-20261007`, and `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`.
- Official three-row preview schema with 13 columns and three unique IDs.
- Finite probabilities in `[0.33333334, 1.0]`.
- `submission_receipt.json` status `PASS`, 3/3 studies scored, and zero fallbacks.
- Zero Raptor-384, Raptor-224, R6, and combined failures.
- `RAPTOR224_STAGE_DONE`, passing `CONTROLLER_DONE`, and `NOTEBOOK_COMPLETE` are present in the logs.
- Total runtime: `478.8` seconds on T4 x2.
- Preview submission SHA-256: `5dd9febae16c11b76450eecb92bbc67669214f71cdd6edce69ec48a02376d7b4`.

Small immutable receipts are retained under `output-v1`.

## Difference from the locked stack

A source-level comparison against the submitted `0.9576`-title R6 flip-TTA kernel found the model, preprocessing, TTA, Raptor-224 share (`0.35`), and overall 50/50 R6/Raptor split unchanged. The scientific difference is per-finding Raptor weighting:

- Synovitis: `0.80` Raptor;
- Lateral OA: `0.20` Raptor;
- PF OA: `0.60` Raptor;
- Baker's cyst: `0.35` Raptor;
- all other findings: `0.50` Raptor.

The source says these weights were fixed before this run. It also explicitly acknowledges that the Raptor checkpoints use controlled-access OAI data.

## Score evidence and decision

The refreshed full public leaderboard lists team `Heliosli` at displayed public macro AUC **`0.958`**, rank 244, with last submission time `2026-10-15 17:13:12`. The audited kernel completed at `17:03:42`, ten minutes before that submission, providing strong temporal identity evidence that this is the scored contract. Leaderboard snapshot ZIP SHA-256: `e79685ab83ee458f191287b0f7f8406d09cff583e5144ffddacf8eedef82d523`.

This candidate is technically ready for an exact exploratory submission, but it is not submitted from this audit until the user explicitly approves this materially new OAI-trained, per-finding-weight contract after being informed of those facts.
