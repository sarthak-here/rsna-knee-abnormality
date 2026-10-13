# OOF-corrected CoAtNet-384 private screen

This experiment combines two independently established signals without using the 58 organizer-labelled studies for fitting or checkpoint selection:

- the public Raptor CoAtNet-384 target-specific attention-MIL training implementation from `hukumjailer/rsna-knee-consensus-raptor-training`;
- the authenticated fold-held-out ConvNeXt teacher predictions produced by `experiments/convnext-oof-teacher-audit` (SHA-256 `60529a0ba00107abd7efd101f70454e7ce89407befbaa4107f24771ee088579e`).

Training targets are Steven v4 report labels for all non-gold studies. For the 2,610 studies with legitimate fold-held-out teacher predictions, the target is the predeclared 50/50 Steven/teacher average; the remaining 1,739 targets are unchanged. A report-grouped fold selects the checkpoint. The 58 organizer-labelled studies are evaluated once only after the checkpoint is frozen.

This notebook is private validation only and deletes any submission CSV. The pass gate is organizer-gold macro AUC >= 0.90 with all identity, coverage, and isolation checks satisfied. A pass permits scaling or building inference; it is not itself a submission.

