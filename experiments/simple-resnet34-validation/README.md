# Plain ResNet34 validation

This private one-fold experiment directly tests the current public competition
hypothesis that compact 224px CNNs with simple pooling can be more robust than
complex attention heads under noisy report-derived supervision.

The model uses the CC0 `pytorch/resnet34` ImageNet weights, 24 quality-filtered
adjacent 2.5D windows from the attributed Apache-2.0 Raptor 44-slice corpus,
simple spatial average pooling followed by study-level mean-plus-max pooling,
and plain BCE on Steven v4 soft labels. There is no attention module and no
positive-class reweighting.

Report hashes define the GroupKFold split. All 58 organizer-labelled studies
are excluded from fitting, splitting, checkpoint selection, and hyperparameter
selection. They are evaluated only after the checkpoint is frozen. The notebook
cannot create a competition submission.

The predeclared one-fold gate requires weak-validation macro AUC at least
`0.87` and organizer-gold diagnostic AUC no lower than `0.860457`. Passing only
permits multi-fold private validation; it does not authorize submission.

## Attribution

- `pytorch/resnet34` — CC0-1.0 ImageNet weights
- `dreaddevelopment/knee-raptor-corpus`
- `dreaddevelopment/knee-raptor-corpus-ext`
- `stevenleehans/rsna-knee-llm-report-labels`
