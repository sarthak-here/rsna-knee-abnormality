# ConvNeXt-MIL public reproduction result

## Outcome

- Kaggle submission: `56877854`
- Submitted artifact: exact private notebook version `1`
- Public score: **`0.944`**
- Previous locked baseline: `0.943` (`56827069`)
- Decision: promoted as the new validated public baseline
- Scored: 6 October 2026

## Provenance and attribution

The experiment is a byte-identical private reproduction of the public notebook
`goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944`. Its SHA-256 is
`941802c45fbc21a763c259d54d35f5fe1600f3a269a39d6773edcbf4185f457e`.
No source cell or blend constant was changed.

The added reader is Good P. J. W.'s independently trained 2.5D ConvNeXt-Tiny
MIL reader from `goodpjw2008/rsna-knee-2-5d-convnext-reader`. Kaggle metadata
declares that dataset Apache-2.0. The notebook preserves its full community
credits, including the public DINOv2, A5, RadImageNet, Raptor, CoAtNet, report
label, preprocessing, and checkpoint contributors named in the source.

## Strict validation

Before submission, the completed preview output was checked against the official
`sample_submission.csv`:

- exact 13-column schema, column order, row count, test IDs, and ID order;
- unique study IDs and finite probabilities bounded to `[0, 1]`;
- exact implementation of the predeclared 70% public-stack / 30% ConvNeXt-reader
  rank blend;
- completion receipt status `COMPLETE`;
- 20 DINO members, five A5 folds, four CoAt family members, and four Raptor views;
- no inner rerank, score-recovery path, partial ensemble, fallback submission,
  traceback, CUDA OOM, or reader failure;
- all three public ConvNeXt checkpoints completed at weight `0.30`;
- public-stack preview SHA-256
  `8f0c5e7b2b561538dae55439a4d2e9173141096da3defd9bc0a4bac351eef37a`.

The exact validated notebook version was submitted rather than the three-row
local preview CSV, allowing Kaggle's code-competition runner to execute the same
pipeline against the hidden test set. The resulting `0.944` is leaderboard
evidence; it does not establish first place by itself.
