# Slot-aware ResNet34 result

## Decision

Rejected. Do not scale, tune, blend, or submit this five-slot ResNet34
mean/max-pooling contract.

## Evidence

- Kaggle notebook: `sarthaksharma14/rsna-knee-slot-aware-resnet34-screen`
- Completed version: 1
- Fit studies: `3,479`
- Report-group-isolated weak-validation studies: `870`
- Best weak-validation macro AUC: `0.797059`
- Weak-validation gate: `0.870000`
- Organizer-gold diagnostic macro AUC: `0.795129`
- Gold diagnostic floor: `0.860457`
- Gold studies used for fitting or model selection: `0`
- Gold studies used for frozen final evaluation: `58`
- Competition submission created: no

## Integrity validation

- The five fixed slot ranges exactly matched the predeclared contract:
  `[0:12]`, `[12:22]`, `[22:30]`, `[30:36]`, and `[36:44]`.
- All 14 epoch losses and AUC values are finite.
- Checkpoint SHA-256:
  `760cd5223faf8818e89f0ded71d0e39818164a3ae5de9e8cc7d71d9d806e5a5c`
- CC0 ImageNet ResNet34 weight SHA-256:
  `333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef`
- Runtime log contains no traceback, runtime exception, CUDA OOM, or NaN.
- No `submission.csv` exists in the completed output.

The locked public baseline remains Kaggle submission `56877854` at `0.944`.

## Interpretation

Slot preservation did not help: it scored below the rejected flat contract
(`0.797059` versus `0.805253` weak AUC). The limitation is therefore not merely
cross-slot windows. Do not spend more GPU time on this ResNet34/Raptor-corpus
family. The next supervision experiment should use a genuinely stronger,
different OOF image teacher rather than another pooling or routing change.
