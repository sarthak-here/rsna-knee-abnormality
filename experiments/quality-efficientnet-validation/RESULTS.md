# Quality-filtered EfficientNet-B0 result

## Decision

Rejected. Do not scale, blend, tune, or submit this 24-window EfficientNet-B0
reader contract.

## Evidence

- Kaggle notebook: `sarthaksharma14/rsna-knee-quality-efficientnet-screen`
- Completed version: 2
- Fit studies: `3,479`
- Report-group-isolated weak-validation studies: `870`
- Best weak-validation macro AUC: `0.754442`
- Organizer-gold macro AUC: `0.760454`
- Prior private control: `0.860457`
- Delta versus control: `-0.100003`
- Predeclared gate: organizer-gold AUC at least `0.865457`
- Gold studies used for fitting or model selection: `0`
- Gold studies used for frozen final evaluation: `58`
- Competition submission created: no

## Integrity validation

- All reported losses and AUC values are finite.
- The report contains all 12 planned epochs.
- Checkpoint SHA-256:
  `818db21ae866bf7b0211e31ea108f1328bdc63dbc8ef891ae5b42c1d4f84ccec`
- Official TIMM pretrained-weight SHA-256:
  `827b6e33c2a10291671e2426b902c2fe87a77e88ab6e3ae769cda9c85b9be02e`
- Runtime log contains no traceback, runtime exception, or CUDA OOM.
- No `submission.csv` exists in the completed output.

The locked public baseline remains Kaggle submission `56877854` at `0.944`.
