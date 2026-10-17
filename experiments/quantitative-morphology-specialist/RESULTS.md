# Version 1 result: rejected

The private no-submission run completed successfully and passed all operational
integrity checks, but failed every scientific advancement gate.

- Frozen anchor macro AUC: `0.9325528441`
- Morphology reader macro AUC: `0.6944669670`
- Fixed 10% residual macro AUC: `0.9304399100`
- Delta versus anchor: `-0.0021129341`
- Positive seed blends: `0/5`
- Nonnegative target deltas: `5/12`
- Bootstrap improvement probability: `0.1634`
- Bootstrap q025: `-0.0065713084`

The run used exactly 4,349 non-gold fit studies, 2,610 authenticated corrected
targets, 1,739 unchanged targets, and 58 evaluation-only gold studies. The
physical-cache, manifest, OOF, and anchor hashes matched their pinned values.
The downloaded prediction package contained 58 unique IDs, 12 targets, five
complete seed predictions, finite probabilities in `[0,1]`, clean logs, and no
CSV artifact.

The reader did provide isolated positive deltas for Lateral Meniscus, Lateral
OA, PF OA, Effusion, and Synovitis, but those observations are terminal evidence,
not permission for post-hoc target selection. This exact contract is closed: do
not tune, blend, build inference for, or submit it.
