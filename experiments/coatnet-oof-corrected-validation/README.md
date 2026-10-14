# OOF-corrected CoAtNet-384 private screen

This experiment combines two independently established signals without using the 58 organizer-labelled studies for fitting or checkpoint selection:

- the public Raptor CoAtNet-384 target-specific attention-MIL training implementation from `hukumjailer/rsna-knee-consensus-raptor-training`;
- the authenticated fold-held-out ConvNeXt teacher predictions produced by `experiments/convnext-oof-teacher-audit` (SHA-256 `60529a0ba00107abd7efd101f70454e7ce89407befbaa4107f24771ee088579e`).

Training targets are Steven v4 report labels for all non-gold studies. For the 2,610 studies with legitimate fold-held-out teacher predictions, the target is the predeclared 50/50 Steven/teacher average; the remaining 1,739 targets are unchanged. A report-grouped fold selects the checkpoint. The 58 organizer-labelled studies are evaluated once only after the checkpoint is frozen.

This notebook is private validation only and deletes any submission CSV. The pass gate is organizer-gold macro AUC >= 0.90 with all identity, coverage, and isolation checks satisfied. A pass permits scaling or building inference; it is not itself a submission.

Version 1 established the full data path but Kaggle cancelled it after three epochs because each CoAtNet-384 epoch required about 3.7 hours on T4. It also exposed that the inherited gold-only AUC helper converted soft selection targets to integers, producing an undefined metric. Version 2 preserves the model, data, targets, split, and evaluation isolation, uses the predeclared two-epoch runtime-bounded screen, and computes selection AUC against a fixed 0.5 threshold on soft report targets.

