# Plain ResNet34 result

## Decision

Rejected. Do not scale, blend, tune, or submit this flattened 44-slice
ResNet34 mean/max-pooling contract.

## Evidence

- Kaggle notebook: `sarthaksharma14/rsna-knee-simple-resnet34-screen`
- Completed version: 2
- Fit studies: `3,479`
- Report-group-isolated weak-validation studies: `870`
- Best weak-validation macro AUC: `0.805253`
- Weak-validation gate: `0.870000`
- Organizer-gold diagnostic macro AUC: `0.807047`
- Gold diagnostic floor: `0.860457`
- Gold studies used for fitting or model selection: `0`
- Gold studies used for frozen final evaluation: `58`
- Competition submission created: no

## Integrity validation

- All 14 epoch losses and AUC values are finite.
- Checkpoint SHA-256:
  `f07eec9367b414aa327b090c5cf91649e5d59baaa4571d93eb50e2aa3736f555`
- CC0 ImageNet ResNet34 weight SHA-256:
  `333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef`
- Runtime log contains no traceback, runtime exception, CUDA OOM, or NaN.
- No `submission.csv` exists in the completed output.

The locked public baseline remains Kaggle submission `56877854` at `0.944`.

## Interpretation

The model improved steadily through epoch 11, so optimization worked, but its
ceiling remained far below competitive report-label CV. The flattened Raptor
44-slice representation discards explicit six-series identity and is therefore
not an adequate contract for the simple-CNN hypothesis. The next test must keep
plane, sequence, and slice-position identity rather than tuning this model.
