# Public 0.956 notebook provenance audit — 14 October 2026

## Candidate

`varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956`

- Notebook version: 1 (`scriptVersionId=357929858`)
- Downloaded notebook SHA-256: `6406e92300011e48f0534ce74a91f5b7a0629b3771168ab7275f5584f884f36d`
- Reported public score: `0.956`
- Reported change: add the otherwise-unused 224-pixel CoAtNet checkpoint at 35% of the Raptor slot inside a 50/50 Raptor/R7 rank blend.

## Provenance findings

The claimed lift is not an independent competition-only checkpoint. The notebook explicitly identifies both the 384-pixel Raptor anchor and the added 224-pixel checkpoint as members of `nartaa/rsna-knee-publication-swa-weights-20261007`. Its markdown includes the controlled-access OAI/NDA acknowledgement, and its first inference stage loads `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`.

The attached nartaa dataset metadata confirms that both checkpoints used 2,399 OAI knees for masked external supervision. Kaggle reports the dataset license as `other`. The publisher added model-use permission for non-commercial competition use on 13 October 2026, but explicitly states that no open-data license is applied to OAI data and that OAI access remains governed by NDA terms. This does not establish the generally accessible source-data provenance required by this project's competition-rule gate.

The other half is not independently clean enough to rescue the candidate:

- `heliosli/rsna-knee-compact-r7cnxfsss` is marked license `other` and describes itself as a private R7 bundle.
- `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` is marked license `other`, explicitly grants no new open-source license, and bundles SKM-TEA/KneeXNet auxiliary weights under their original restrictions.
- The notebook's Apache-2.0 license covers the notebook code, not these attached checkpoint datasets.

The implementation also retains fallback behavior: the 224 stage may be skipped on timeout or failure, and model failures can be replaced by another model's rank or `0.5`. That makes the displayed notebook score insufficient as a strict identity receipt for the exact intended three-leg blend without inspecting the hidden rerun logs.

## Decision

Rejected for reproduction, submission, and blending. The score is credible leaderboard evidence for the published stack, but the gain depends directly on OAI-trained weights and two `other`-licensed/private checkpoint bundles. No GPU run or Kaggle submission was launched. The locked submission remains `56877854` at public `0.944`.

The useful research takeaway is architectural rather than asset-level: a lower-resolution, tighter-field-of-view checkpoint can add rank diversity to a high-resolution reader. Any follow-up must reproduce that diversity with competition-only training data and a fully auditable checkpoint.
