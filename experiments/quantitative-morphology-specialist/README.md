# Quantitative morphology specialist

This is a predeclared, no-submission screen of a materially different reader. It
uses the validated 288 px / 140 mm / 12-slice physical cache and extracts fixed
intensity, texture, bilateral-symmetry, regional, through-plane, and paired-
sequence-difference measurements. It does not train another image encoder.

The model is a fixed five-seed ExtraTrees ensemble trained on exactly 4,349
non-gold competition studies. Targets use the authenticated fixed 50/50 blend of
Steven v4 and true fold-held-out ConvNeXt predictions for exactly 2,610 studies;
the other 1,739 retain Steven v4. All 58 organizer-labelled studies are excluded
from feature/model decisions and fitting and are evaluated only after freezing.

The frozen anchor is `[r6_flip, raptor384, raptor224, r7]` with weights
`[0.50, 0.325, 0.175, 0]`. The candidate assigns the morphology reader a fixed
10% rank-residual slot. No per-target selection or post-result tuning is allowed.

Advance only if all gates pass:

- macro AUC gain >= 0.002;
- at least 8/12 target deltas are nonnegative;
- at least 4/5 seed blends improve;
- paired bootstrap improvement probability >= 0.90;
- paired bootstrap q025 > -0.001;
- exact identities, hashes, counts, finite predictions, and no CSV artifact.

If rejected, this exact contract is closed. The notebook cannot submit.
