# Public 0.956 exact reproduction

This experiment preserves version 1 of `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` byte-for-byte and changes only Kaggle kernel metadata so it can be run privately under this account.

- Upstream notebook SHA-256: `6406e92300011e48f0534ce74a91f5b7a0629b3771168ab7275f5584f884f36d`
- Upstream script version: `357929858`
- Upstream public score: `0.956`
- Upstream preview runtime: 7m 37s on two T4 GPUs
- Added stage: the public 224px CoAtNet checkpoint receives 35% of the Raptor slot; the resulting Raptor slot remains 50% of the final Raptor/R7 rank blend.

The public preview outputs were independently downloaded before launch. They confirm `R224_STAGE rc=0 ok=True`, zero Raptor failures, zero R7 failures, three exact preview IDs, finite predictions, and a completed blend receipt.

The current competition rules permit external data and models when they are publicly and reasonably accessible at minimal cost. The nartaa dataset is public and free, and its 13 October 2026 model-use terms expressly permit non-commercial research, RSNA Knee competition use, adaptation, and redistribution of the weights or adapted versions with attribution. The reproduction retains the upstream OAI/NDA acknowledgement and all model/source attribution. The Apache-2.0 notebook license applies to the public code; attached models retain their own terms.

This private run is a packaging and identity check before any submission. Submission is allowed only if the exact 224 stage runs, required outputs and receipts are present, all official preview IDs and columns match, predictions are finite and within `[0,1]`, there are no fallback studies, and the notebook runtime remains valid.
