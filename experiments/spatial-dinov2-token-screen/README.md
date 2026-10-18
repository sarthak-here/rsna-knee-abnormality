# Spatial DINOv2 token residual screen

This private, no-submission experiment tests a representation change that the
rejected global-feature DINOv2 screen did not preserve: coarse spatial anatomy
inside every 2.5D window.

The frozen DINOv2-S encoder consumes the immutable `288 px / 140 mm / 12 slice`
physical cache. Instead of reducing every window to a CLS/mean vector, its
`20 x 20` patch map is pooled to a fixed `4 x 4` grid. The resulting tensor is
`[4407, 6, 4, 16, 384]`: study, sequence slot, depth window, spatial cell and
feature channel. A small target-query head can therefore learn different
anatomical evidence locations for each of the twelve findings while retaining
slot, depth, row and column identity.

The scientific contract is frozen before organizer-gold evaluation:

- exact physical-cache and study-manifest hashes;
- immutable Meta DINOv2-S Kaggle model source version 1;
- exactly 4,349 non-gold fitting studies;
- fixed authenticated 50/50 Steven/true-OOF correction for exactly 2,610
  covered studies and 1,739 unchanged studies;
- five fixed seeds, 20 fixed epochs, one fixed architecture;
- frozen four-reader anchor `[r6_flip, raptor384, raptor224, r7]` at
  `[0.50, 0.325, 0.175, 0]`;
- fixed 10% rank-residual slot;
- all 58 organizer-labelled studies are evaluation-only;
- no checkpoint, epoch, target, blend-weight or public-leaderboard search;
- no CSV or submission capability.

Advance only if all gates pass: macro gain at least `0.002`, at least `8/12`
nonnegative targets, at least `4/5` positive seed blends, paired-bootstrap
`p >= 0.90`, `q025 > -0.001`, and every identity/finiteness/integrity check.

