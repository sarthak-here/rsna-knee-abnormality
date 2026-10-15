# Public GP958e audit

This audit reviews public kernel `ltcsmokeboi/rsna-knee-gp958e`, version 1, published after the exact GP958d contract scored `0.956` in submission `57238441`.

## Identity and runtime checks

- Notebook SHA-256: `1848ff39cd0c03e8b763d03a207404116f97b6065df156b20809c853b1cf5058`.
- The preview output has the official three-row, 13-column schema and three unique IDs.
- All probabilities are finite in `[0.33333334, 1.0]`.
- The receipt reports `PASS`, 3/3 studies scored, and zero fallbacks.
- Blend receipts report zero Raptor-384, Raptor-224, R7, and combined failures.
- Required completion markers are present; total runtime is `445.4` seconds.
- Preview submission SHA-256: `1199c2d767b5a53219d777b6041593cc392f56eff09228bda88d68c90d5f3cc8`.

## Scientific difference

A source-level comparison against GP958d found only one prediction-contract change: `A224` was changed from `0.35` to `0.0`, removing the 224px checkpoint from the Raptor half of the final rank blend. The model, R7 predictions, Raptor-384 predictions, 50/50 R7/Raptor weighting, and all preprocessing remain unchanged. The notebook still spends runtime computing Raptor-224, but its final weight is zero.

The full public leaderboard snapshot downloaded after publication lists team `You WeiLin` / `ltcsmokeboi` at displayed score `0.956`, rank 664, last submission time `2026-10-15 14:13:42`; it contains no scored evidence for the newly published GP958e variant. Snapshot ZIP SHA-256: `9e37c9505023a509313403e2829d49772575d33b85b6adb08ed65d0fc62a1c6b`.

## Decision

**Rejected without submission.** The `gp958e` title is not leaderboard evidence, and the only change is a post-result blend-weight endpoint adjustment after GP958d scored `0.956`. Submitting it would be blind public-leaderboard weight tuning rather than a validated model improvement. The locked baseline remains submission `57197157` at displayed public macro AUC `0.957`.
