# Quality-filtered EfficientNet-B0 validation

This private one-fold screen tests a compact end-to-end convolutional reader
that is materially different from the frozen-feature probes and six-slot
readers already rejected.

The experiment uses the public Apache-2.0 Raptor 44-slice corpus and its 2.5D
window contract, but replaces CoAtNet with the public TIMM EfficientNet-B0. It
predeclares a simple image-quality rule based on slice contrast and gradient
energy, removes the lowest-quality quartile, samples 24 adjacent-slice windows,
and applies intensity, noise, and small erasing augmentation. A target-specific
attention head pools windows independently for each finding.

Steven v4 supplies soft training targets. Report hashes define a group-isolated
weak-label fold. All 58 organizer-labelled studies are excluded from training,
splitting, checkpoint selection, and hyperparameter selection and are evaluated
only after the best weak-validation checkpoint is frozen. The notebook cannot
create a competition submission.

The one-fold gate is organizer-gold macro AUC `>= 0.865457`. Passing would only
justify a multi-fold private validation; it would not authorize submission.

## Attribution

- `dreaddevelopment/knee-mri-training-the-twelve-finding-model`
- `dreaddevelopment/knee-raptor-corpus`
- `dreaddevelopment/knee-raptor-corpus-ext`
- `timm/tf-efficientnet/PyTorch/tf-efficientnet-b0/1`
- `stevenleehans/rsna-knee-llm-report-labels`

The design is motivated by current competition reports that compact 224px
models remain competitive when slice quality, augmentation, regularization, and
weak-label handling are treated carefully. Those claims are not treated as
evidence; this notebook exists to test the hypothesis privately.
