# BiomedCLIP target-attention residual screen

This private no-submission experiment tests the frozen, independently pretrained
Microsoft BiomedCLIP visual representation produced by the validated physical
feature cache. The reader is a small target-specific attention head over the 24
fixed tokens (six clinical slots × four 2.5D windows). The visual tower remains
frozen; no image encoder is fitted on competition labels.

Five fixed seeds train for 30 fixed epochs on exactly 4,349 non-gold studies.
Targets use the authenticated fixed 50/50 Steven-v4/true-OOF correction for
2,610 studies and leave 1,739 unchanged. All 58 organizer-labelled studies are
excluded from fitting and evaluated only after every model is frozen.

The frozen anchor is `[r6_flip, raptor384, raptor224, r7]` with weights
`[0.50, 0.325, 0.175, 0]`. The candidate gives the BiomedCLIP reader a fixed 10%
rank-residual slot. No target-specific blend, checkpoint selection, early
stopping, or post-result tuning is permitted.

Advance only if all gates pass: macro gain >= 0.002, at least 8/12 nonnegative
targets, at least 4/5 positive seed blends, bootstrap p >= 0.90, bootstrap q025
> -0.001, and all identity/finiteness/no-submission checks.
