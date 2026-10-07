# ConvNeXt OOF teacher audit

This private, inference-only audit tests whether the public ConvNeXt reader can
provide legitimate out-of-fold image supervision. It reproduces the author's
report-group fold assignment and runs each available checkpoint only on its
declared held-out fold. The three public checkpoints therefore cover folds
0–2; folds 3–4 remain uncovered and are not synthesized.

Every public source file and checkpoint is SHA-256 pinned before execution.
All organizer-labelled studies are excluded. The notebook outputs only an OOF
prediction table and an integrity report; it cannot create a competition
submission.

## Attribution

- `goodpjw2008/rsna-knee-2-5d-convnext-reader`, Apache-2.0
- ConvNeXt-Tiny ImageNet-22k/1k initialization through the published reader

Passing this audit establishes coverage and provenance only. It does not prove
that teacher-corrected targets improve the locked `0.944` ensemble and does not
authorize a submission.

