# Public 0.9576 R6 flip-TTA + Raptor 384/224 reproduction

This experiment tracks the exact scored public kernel `heliosli/0-9576-latest-public-dataset-score-in-one-click`, version 1 (`scriptVersionId=358332966`). Kaggle displays its public score as `0.957`; the notebook title reports the more precise `0.9576`.

Relative to the locked `0.9573`-title kernel, this candidate keeps the same Raptor 384px/224px pair and blend structure but enables horizontal-flip TTA for Heliosli's R6 ConvNeXt leg (`FLIP_TTA = True`). The final blend remains 50% Raptor and 50% R6, with the 224px checkpoint receiving 35% of the Raptor slot.

## Validation

The exact version-1 output was downloaded before submission and passed the predeclared checks:

- official three-row preview schema with 13 columns and three unique study IDs;
- finite probabilities in `[0.33333334, 1.0]`;
- `submission_receipt.json` status `PASS`, three of three studies scored, and zero fallback studies;
- zero Raptor-384, Raptor-224, R6, and combined failures in `blend_receipt.json`;
- `RAPTOR224_STAGE_DONE`, passing `CONTROLLER_DONE`, and `NOTEBOOK_COMPLETE` present in the logs;
- total blend runtime `441.2` seconds on T4 x2;
- preview submission SHA-256 `1ebfe53cb93ba0b9705724721755b515e64f7703712bc6a1bdddbcd181a7342b`.

The exact inputs were `heliosli/rsna-knee-r6-frozen-fe055406`, `nartaa/rsna-knee-publication-swa-weights-20261007`, and `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`. Small immutable receipts are retained under `output-v1`.

## Submission

- Kaggle submission: `57228245`
- Kernel: `heliosli/0-9576-latest-public-dataset-score-in-one-click`
- Version: `1`
- Initial status: `PENDING`
- Remaining daily submissions after launch: `4`
- Attribution: Heliosli, nartaa, and dreaddevelopment.

Submission `57197157` at displayed public macro AUC `0.957` remains locked until this candidate completes and provides stronger score or rank evidence. No first-place claim is made; final standing depends on the private leaderboard and competition review.
