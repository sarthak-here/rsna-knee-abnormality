# Public GP958d R7 + Raptor 384/224 reproduction

This experiment tracks the exact public kernel `ltcsmokeboi/rsna-knee-gp958d`, version 1. The notebook title reports `gp958d`; Kaggle showed version 1 of 1 and a successful 7m23s T4 x2 preview run.

The candidate is materially distinct from the locked R6 leg: it blends a four-fold R7 ConvNeXt-small reader with the public Raptor 384px/224px pair. The final rank blend is 50% R7 and 50% Raptor, with the 224px checkpoint receiving 35% of the Raptor slot. The Raptor checkpoints are acknowledged by the source as OAI-trained; the user explicitly approved submitting this exact candidate after being informed of that provenance.

## Validation

The exact public version-1 source and output were downloaded before submission and passed strict checks:

- notebook SHA-256 `85c694c903ec6841488b49d5ecbe2838b8fad7db283f1f952c59f48d936e0b24`;
- official three-row preview schema with 13 columns and three unique study IDs;
- finite probabilities in `[0.33333334, 1.0]`;
- `submission_receipt.json` status `PASS`, three of three studies scored, and zero fallback studies;
- zero Raptor-384, Raptor-224, R7, and combined failures in `blend_receipt.json`;
- `RAPTOR224_STAGE_DONE`, passing `CONTROLLER_DONE`, and `NOTEBOOK_COMPLETE` present in the logs;
- total blend runtime `428.0` seconds;
- preview submission SHA-256 `56172847949eec2070bd4842f43c3b8f8aeee5ce69460cb3cda0d231c69be26d`.

Small immutable receipts are retained under `output-v1`.

## Submission

- Kaggle submission: `57238441`
- Kernel: `ltcsmokeboi/rsna-knee-gp958d`
- Version: `1`
- Initial status: `PENDING`
- Remaining daily submissions after launch: `3`
- Attribution: You WeiLin, Heliosli, nartaa, and dreaddevelopment.

Submission `57197157` at displayed public macro AUC `0.957` remains locked until this candidate completes and exceeds it. No first-place claim is made; final standing depends on the private leaderboard and competition review.

## Scored result

Submission `57238441` completed successfully at public macro AUC **`0.956`**. It trails the locked `57197157` baseline (`0.957`) by `0.001`, so it is rejected as a replacement. The R7 result is retained only as diversity evidence; this exact contract will not be promoted, blindly blended, or resubmitted.
