# ConvNeXtV2 domain-pool screen result

## Decision

Rejected. The public RSNA-trained ConvNeXtV2-Tiny backbone did not pass the
predeclared private screen and must not be scaled, blended, or submitted using
this reader contract.

## Evidence

- Kaggle notebook: `sarthaksharma14/rsna-knee-convnextv2-domain-pool-screen`
- Completed version: 4
- Three gold-study AUCs: `0.816303`, `0.836123`, `0.823675`
- Ensemble gold AUC: `0.827406`
- Prior private control: `0.860457`
- Delta versus control: `-0.033051`
- Required gate: ensemble AUC at least `0.865457` and at least two of three
  seeds above `0.860457`
- Gate result: failed
- Gold studies used for fitting: `0`
- Gold studies used for final evaluation: `58`
- Competition submission created: no

## Integrity validation

- Feature array: shape `(4407, 6, 10, 768)`, `float16`, all finite
- Feature SHA-256:
  `8d5882017cb69f88418acd94dbb44d31e8dd23ad3c6e22d10c43fb5bf7d186a2`
- Slot mask: shape `(4407, 6)`, `float32`, binary
- Study manifest: `4,407` rows and `4,407` unique study IDs
- Head SHA-256:
  `3e8e05d8445a27abbb7018e6cd7a783be53260c665161f12b40f6ab2433475ef`
- Public checkpoint SHA-256:
  `4680dff05e3556eebbf07e84ca824ab6f1fe8db442dc9cc3948e18d226f008c3`
- Source: `yaolinghan/rsna-knee-pool-models`, Apache-2.0
- Runtime log contained no traceback, exception, CUDA OOM, or error marker
- No `submission.csv` exists in the completed output

The locked public baseline remains Kaggle submission `56877854` at `0.944`.
