# Slot-aware ResNet34 validation

This private screen follows the failed flat 44-slice ResNet34 experiment. The
public Raptor corpus has five fixed clinical slots, but the rejected reader
sampled across the concatenated array and could form windows across slot
boundaries. This experiment preserves sagittal fat-suppressed, sagittal
non-fat-suppressed, coronal fat-suppressed, coronal non-fat-suppressed, and
axial identity.

Four quality-filtered adjacent 2.5D windows are sampled independently from each
slot. A CC0 ImageNet ResNet34 encodes every window. Spatial average pooling and
per-slot mean/max pooling feed a plain MLP; there is no attention. Training uses
plain BCE on Steven v4 soft labels and a report-hash GroupKFold split.

All 58 organizer-labelled studies remain excluded from fitting, splitting,
checkpoint selection, and hyperparameter selection. The notebook cannot create
a competition submission. Passing requires weak-validation macro AUC at least
`0.87` and frozen organizer-gold diagnostic AUC at least `0.860457`, and would
permit only multi-fold private validation.

## Attribution

- `pytorch/resnet34` — CC0-1.0 ImageNet weights
- `dreaddevelopment/knee-raptor-corpus`
- `dreaddevelopment/knee-raptor-corpus-ext`
- `stevenleehans/rsna-knee-llm-report-labels`

## Result

Version 1 completed and passed artifact-integrity checks, but the candidate was
rejected. Best weak-validation macro AUC was `0.797059`; the frozen checkpoint
scored `0.795129` on organizer gold. Preserving the five fixed slots did not
recover the flat reader's already inadequate ceiling. No submission was made.
See `RESULTS.md`.
