# Full-population ConvNeXt OOF target validation

This private paired experiment tests whether the OOF-teacher supervision gain
survives when both arms fit all 4,349 non-gold studies. The control uses Steven
v4 labels everywhere. The candidate applies the fixed 50/50 Steven/ConvNeXt mix
only to the 2,610 studies with true fold-held-out teacher predictions and leaves
the remaining 1,739 labels unchanged.

Both arms share authenticated frozen RadImageNet features, architecture,
initialization, optimizer, batch order, five seeds, and fixed 20 epochs. No
checkpoint is selected. All 58 organizer-labelled studies are evaluation-only.
The notebook cannot create a submission.

Passing requires at least +0.005 ensemble macro AUC, at least four positive
seeds, paired-bootstrap p-improve at least 0.95, and positive q025.

## Attribution

- `goodpjw2008/rsna-knee-2-5d-convnext-reader`, Apache-2.0
- `stevenleehans/rsna-knee-llm-report-labels`
- RadImageNet ResNet-50 from the attributed public reproduction assets

