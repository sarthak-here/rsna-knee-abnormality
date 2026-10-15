# Public 0.957 R6 + Raptor 384/224 reproduction

This experiment tracks the exact scored public kernel `heliosli/0-9573-latest-public-dataset-score-in-one-click`, version 1 (`scriptVersionId=358330886`). Kaggle displays its public score as `0.957`; the notebook title reports `0.9573`.

The candidate preserves the same public 384px/224px Raptor pair used by the locked `0.956` submission but replaces the other half with Heliosli's newer R6 bundle. The blend remains 50% Raptor and 50% Heliosli, with the 224px checkpoint receiving 35% of the Raptor slot.

## Validation

The exact version-1 output was downloaded before submission and passed the predeclared checks:

- official three-row preview schema with 13 columns and three unique study IDs;
- finite probabilities in `[0.33333334, 1.0]`;
- `submission_receipt.json` status `PASS`, three of three studies scored, and zero fallback studies;
- zero Raptor-384, Raptor-224, and Heliosli failures in `blend_receipt.json`;
- `RAPTOR224_STAGE_DONE` and `NOTEBOOK_COMPLETE` present in the logs;
- total blend runtime `424.4` seconds;
- preview submission SHA-256 `56172847949eec2070bd4842f43c3b8f8aeee5ce69460cb3cda0d231c69be26d`.

Small immutable receipts are retained under `output-v1`. Kaggle's API allowed downloading version-1 outputs but returned HTTP 403 when asked to pull the historical source file; the public page and submitted kernel version identify the immutable scored source.

## Submission

- Kaggle submission: `57197157`
- Kernel: `heliosli/0-9573-latest-public-dataset-score-in-one-click`
- Version: `1`
- Initial status: `PENDING`
- Remaining daily submissions after launch: `3`
- Attribution: Heliosli, nartaa, and dreaddevelopment.

Submission `57190953` at `0.956` remains locked until this candidate completes and exceeds it.

## Scored result

Submission `57197157` completed successfully at public macro AUC **`0.957`**. It improves the preceding locked submission `57190953` (`0.956`) by **`+0.001`** and the original `0.944` baseline by **`+0.013`**. Submission `57197157` is therefore promoted as the new locked public baseline. This is public-leaderboard evidence only; final placement still depends on the private leaderboard and competition review.
