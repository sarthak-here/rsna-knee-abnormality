# Public 0.9586-title kernel audit

Audited `heliosli/this-is-best-public-dataset-new-version-0-9586` version 1.
No submission was made by this audit.

## Exact identity and receipts

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

The exact blend is 50% compact reader and 50% Raptor by default, with the 224px
checkpoint receiving 35% of the Raptor slot. Per-finding Raptor weights are
Medial OA `0.50`, Lateral OA `0.15`, PF OA `0.65`, Synovitis `0.85`, and
Baker's `0.30`; all other findings use `0.50`.

## Score evidence and caveat

The public kernel title explicitly reports `0.9586`. The full leaderboard
snapshot downloaded at `2026-10-18T03:13:04Z` lists Heliosli at displayed
`0.958`, rank 159, with last submission `2026-10-17 19:46:59`. The kernel ran
at `2026-10-17 04:21:38`, so the public API does not prove a tight one-to-one
temporal binding between this exact version and the unrounded `0.9586` claim.
Operationally, however, the exact public version is fully runnable and its
preview receipts are clean.

## Provenance and decision gate

The Raptor384/224 checkpoints are the public nartaa assets documented as
OAI-trained. The compact reader is Heliosli's public bundle. This materially new
per-finding contract must not be submitted without the user's explicit approval
after those facts and the score-identity caveat are disclosed. If approved,
submit exact version 1 with Heliosli, nartaa, dreaddevelopment, and the upstream
compact-reader contributors attributed; promote only if its actual score exceeds
the locked `0.957` submission 57197157.
