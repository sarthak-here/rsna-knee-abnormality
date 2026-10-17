# S6 fifth-arm audit

## Decision

Reject and close the S6 fifth-arm experiment before GPU use. Do not treat S6 as
a standalone replacement for submission `57197157`: its source-faithful public
score is `0.950`, below the locked `0.957` stack. More importantly, a later
public direct comparison added S6 at weight `0.30` beside the R6 recipe and
scored `0.957`, versus `0.958` for the corresponding recipe without S6.

S6 is still scientifically useful as a diversity arm because it differs from
the locked family: it is a ConvNeXt-small reader using anatomy-V-Net-placed
`v4` views, 320-pixel input, 288-pixel crops, three slices, and per-finding
spatial plus window-attention MIL. Its public dataset states that it was
trained on 4,263 competition studies using report-derived JEV-blend soft
labels, with all 58 organizer-labelled studies excluded.

## Immutable identities

- source notebook: `laimerbotrolenk/rsna-four-model-ensemble`, version 1
- notebook SHA-256:
  `50ded2186d86e172e85a8928e8b93b3cdd0c9112117d2267ac059e651793a003`
- decoded embedded S6 runner SHA-256:
  `2e8c4dca6773588ee7a4246ba048e667c1e2516be1ec43a8b6e0269b5fa62c96`
- weights dataset: `nartaa/rsna-knee-s6fall-weights`
- checkpoint: `E_prior_s42_fall.pt`
- checkpoint SHA-256:
  `0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12`
- anatomy checkpoint SHA-256:
  `5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e`
- `compact_infer.py` SHA-256:
  `34f433fe72b732543813165c3397ec80cef01b0d49b25b7b22668714f0b0ef4a`
- checkpoint manifest SHA-256:
  `ed3c6d8d2e69ae08c4a97eab3c860a9ec20c5c69484e35de6a3119d396d731cd`
- run config SHA-256:
  `36e49cf0c8db97e7f995482933911f018c32fca8c5933b4d64399c128c19c886`
- completion receipt SHA-256:
  `acc99ed61393404feaa4f84a2e20ce27ad9f55a257b8e0d0760e83e105a20c6a`

The dataset metadata is tagged `other`, while its bundled README explicitly
grants CC BY-NC 4.0 for the weights and preserves third-party terms. The model
README explicitly says no OAI data was used. Attribution is due to nartaa,
Heliosli, Desai et al., stevenleehans, pilkwang, riadmohamed42, RSNA, and the
competition data contributors.

## CPU-side integrity audit

`audit_embedded_s6.py` safely parses the notebook AST, decodes but never
executes the embedded runner, and fail-closes on the pinned identities and
scientific constants. It checks the all-data checkpoint, 4,263-study count,
gold exclusion declaration, geometry, public score, source hashes, hard-error
receipt path, and the absence of a positive S6 weight in the public four-model
wrapper (`W_A=0`, `W_B=1` inside S6; the outer notebook currently gives S6
zero weight). This prevents a title-only or silently changed S6 asset from
entering a future experiment.

## Terminal evidence

The four-reader nested experiment completed after the quota reset but failed
its predeclared robustness gates despite a positive descriptive gain. Before
spending more GPU, the newer public notebook
`goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957` supplied the direct S6
A/B result above. Its latest version 7 removes S6 and instead uses all 58 gold
studies to choose aggressive per-finding weights; it was unscored at audit time
and also permits silent fallback for its optional reader.

That evidence reverses the earlier provisional plan to retain S6 as a fifth
arm. No S6 cache, nested optimization, or submission will be launched. Reopen
only if genuinely independent evidence—not organizer-gold or public-leaderboard
weight selection—shows that an exact, zero-fallback S6 contract adds value.
