# ConvNeXt OOF-corrected student validation

This private paired experiment tests the public reader author's fixed 50/50
report/image target hypothesis without creating a submission. Both arms use the
same authenticated frozen RadImageNet features, the same 2,610 non-gold studies,
the same attention head, initialization, optimizer, batch order, and fixed 20
epochs. The only difference is supervision: Steven v4 labels versus a 50/50
average of Steven v4 and true fold-held-out ConvNeXt predictions.

No weak-validation or gold label selects a checkpoint. All 58 organizer-labelled
studies are absent from fitting and used only for the frozen paired diagnostic.
Passing requires at least +0.005 ensemble macro AUC, at least four positive seeds,
bootstrap p-improve at least 0.95, and positive q025. Passing would permit further
private validation only, not a competition submission.

## Attribution

- `goodpjw2008/rsna-knee-2-5d-convnext-reader`, Apache-2.0
- `stevenleehans/rsna-knee-llm-report-labels`
- RadImageNet ResNet-50 checkpoint from the attributed public reproduction assets

