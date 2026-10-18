# Public 0.9586-title kernel audit — identity correction

Audited `heliosli/this-is-best-public-dataset-new-version-0-9586` version 1.

## Correction: this is not a new model contract

The page URL pins `scriptVersionId=358332966`. That is the exact script version
already submitted as `57228245` and documented under
`experiments/public-09576-reproduction`. The renamed page did not identify a
new notebook version. The compact bundle, Raptor checkpoints, component hashes,
preprocessing, architecture, and 224 share are unchanged.

The only observed output change comes from a mutable upstream dependency that
changed five per-finding blend weights. Treating the renamed title and refreshed
output as a materially new kernel was an identity-check failure. Submission
`57336570` is therefore a duplicate-family reweight, not model progress.

## Exact receipts

- Notebook SHA-256:
  `cb0e1eac33a00f96b7d1bcc0e9914f7eb045af2bc9cd01bc290b2213e7f17ddd`
- Preview submission SHA-256:
  `5e9ba665f3e30642e18dbb555eb7cbd16ab8a659ffb5f7226c72aaac44c9df1b`
- Official preview shape: `3 x 13`, with three unique official test IDs
- Prediction range: `[0.3333333333, 1.0]`, all finite
- Submission receipt: `PASS`, `3/3` scored, zero fallback studies
- Reader failures: zero Raptor384, zero Raptor224, zero compact-reader failures
- Runtime: `443.1` seconds total on two T4 GPUs
- Compact reader: four-fold ConvNeXt-small `E_lrprior`, flip TTA, state/fluid
  stackers, pinned component and model hashes

The refreshed blend is 50% compact reader and 50% Raptor by default, with the
224px checkpoint receiving 35% of the Raptor slot. Its changed per-finding
Raptor weights are Medial OA `0.50`, Lateral OA `0.15`, PF OA `0.65`,
Synovitis `0.85`, and Baker's `0.30`; all other findings use `0.50`.

## Independent evidence

The changed external weights were evaluated once, without fitting, against the
immutable 58-study four-reader cache. The earlier 50/50 R6/Raptor contract
scored macro AUC `0.9387940564`; the refreshed weights scored `0.9383513313`, a
delta of `-0.0004427250`. This is negative independent evidence. Do not reuse,
tune, or promote these weights as an algorithmic improvement.

The Raptor384/224 checkpoints are the public nartaa assets documented as
OAI-trained. The compact reader is Heliosli's public bundle. The submission was
launched before the shared script-version identity was recognized. Kaggle
exposes no cancellation action for the pending code submission through the
available API. Its terminal score is retained only as an audit receipt; it
cannot support a breakthrough claim, and the exact contract must not be
submitted again.
