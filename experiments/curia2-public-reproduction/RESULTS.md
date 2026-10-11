# Curia-2 public reproduction result

## Outcome

Rejected. Kaggle submission **57073748** completed with public score **0.899**, well below the locked **0.944** baseline from submission **56877854**. The Curia-2 candidate must not be promoted, tuned against the public leaderboard, or blended solely from this score.

## Reproduction identity

- Kaggle notebook: `sarthaksharma14/rsna-knee-curia-2-private-reproduction`, version 4
- Scientific contract: public Curia-2 V64 branch, five heads, selected 80/20 V52/Curia blend
- Upstream implementation: `prvsiyan/rsna-knee-read-the-report-then-the-knee`
- Encoder: Raidium Curia-2, research-only RAIL-M, non-clinical research benchmark
- Encoder SHA-256: `403a02e27531d2858ecd1e9b1ec2d5ea7bfa909f10ff0d9e8416090a6a8c96ef`
- Preview submission SHA-256: `6c16b159361dd203cabe5f6348fbd0f1dfe952f0fb2837b65182ad5e38cc52fa`
- Code submission: `57073748`
- Public score: `0.899`

## Validation evidence before submission

The offline preview run completed with receipt status `COMPLETE` and audit status `CANDIDATE_SELECTED`. It verified official schema and preview ID order, three unique preview IDs, finite probabilities in `[0.3333333333, 1.0]`, five complete training heads, five test heads, alpha `0.20`, all five positive outer folds, nested diagnostic AUC `0.852098`, final descriptive diagnostic AUC `0.852206`, exact source/model/license identities, and a submission file hash matching both receipt and audit.

These private diagnostics did not transfer to the hidden competition test set. The leaderboard result is therefore the decisive evidence for this inference contract.

## Decision

Keep submission `56877854` at public `0.944` as the locked baseline. Do not retry this exact Curia-2 preprocessing/model/blend contract or tune its blend weight against the public leaderboard. Curia-2 may be retained only as documented negative diversity evidence.
