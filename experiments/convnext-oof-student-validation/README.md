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

## Result

Version 1 passed every predeclared gate. Across five seeds, the corrected-target
student improved over its identically trained control every time. The ensemble
macro AUC increased from `0.829413` to `0.847557` (`+0.018145`); paired bootstrap
`p_improve` was `0.993` with a positive 95% lower bound of `0.003315`. This is
strong evidence that the OOF teacher improves supervision under this controlled
contract. It does not show that the student beats the locked `0.944` competition
ensemble, so no submission was created.
