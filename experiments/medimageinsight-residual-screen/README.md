# MedImageInsight target-attention residual screen

This private no-submission experiment evaluates the validated frozen
MedImageInsight feature cache (SHA-256
`2f8344d247185a7faca711057ef17ce4cd15df540dd3181501c2dfc85e18922a`).
The MRI-capable medical encoder stays frozen. A small target-specific attention
head sees the 24 fixed tokens (six clinical slots × four 2.5D windows), each
with 1,024 features.

Five fixed seeds train for 30 fixed epochs on exactly 4,349 non-gold studies.
Targets use the authenticated fixed 50/50 Steven-v4/true-OOF correction for
exactly 2,610 covered studies and leave 1,739 unchanged. All 58
organizer-labelled studies are excluded from fitting and evaluated only after
every seed is frozen.

The frozen anchor is `[r6_flip, raptor384, raptor224, r7]` at
`[0.50, 0.325, 0.175, 0]`. The candidate gives the MedImageInsight reader one
fixed 10% rank-residual slot. There is no target, checkpoint, epoch, blend, or
post-result search.

Advance only if every gate passes: macro gain >= 0.002, at least 8/12
nonnegative targets, at least 4/5 positive seed blends, bootstrap p >= 0.90,
bootstrap q025 > -0.001, and all identity/finiteness/no-submission checks.
