# CPU asset search — 19 October 2026

The cancelled R6 shard jobs were not polled or restarted. No GPU job or
competition submission was launched by this search.

## Completed source-manifest checks

The public dataset listing for `heliosli` returned six datasets. The released
R6/R7 bundles are inference packages; no separate R6 target/cache release was
listed under that owner during this check. This is an owner-scoped observation,
not a claim that no relevant release exists anywhere.

Two large public file inventories were enumerated to exhaustion using Kaggle's
authenticated listing API, 200 entries per page, following every next-page token:

| Dataset | Files | Pages | Result |
| --- | ---: | ---: | --- |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | 837 | 5 | No matching original per-study R6 target table, state-target table, pool UID manifest or OOF artifact located |
| `jessky/rsna-knee-local-cv-cnx50` | 956 | 5 | Public README and manifest identify a different five-fold distillation recipe, not the original R6 training assets |

The first 20 or 200 files are not a complete manifest: both responses supplied
continuation tokens. The counts above include all pages, including vendored
dependencies. No checkpoint or image dataset was downloaded for this check.

## Local-CV five-fold package

Only `README.md` (747 bytes) and `manifest.json` (1,442 bytes) were downloaded
from `jessky/rsna-knee-local-cv-cnx50` into the ignored `.audit` directory.

- README SHA-256: `d0cd07836cbcf56129132c2b96cb3251b44dc2577cd827864a7da6a6644fdb0d`
- Manifest SHA-256: `cdd4d3e393921020b45b1ddd30b06418120ffc7462b00c05b8ece8eea15f38b2`
- Model family: `cnxS6s_d50`, ConvNeXt-Small, five report-hash folds.
- Recipe: `labels_distill50`, 256px crop from 288px pixels, 48 windows,
  ten epochs, seed zero, encoder LR 0.00015 and head LR 0.0005.
- Companion branch: CoAtNet g95 from `jessky/rsna-knee-local-cv-cand`.
- Its README explicitly discloses that second-order leakage from the external
  OOF teacher has not been ruled out.

This package supplies five checkpoints but does not resolve the original R6
target lineage or the independent-evaluation requirement. No score above the
locked 0.957 baseline was established by this inspection. Do not launch a GPU
reproduction merely because the checkpoints are available. The unresolved
teacher exclusion is a validity prerequisite, not a tunable model setting.

## Decision

The original R6 training-asset prerequisite remains unresolved. Retain the
cancelled cache/calibration closure and 0.957 baseline. These results justify
no new GPU expenditure and do not constitute a model improvement.

Public source locations:

- https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004
- https://www.kaggle.com/datasets/jessky/rsna-knee-local-cv-cnx50
