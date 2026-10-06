# Public frontier reproducibility audit — 6 October 2026

## Purpose

The validated public baseline is Kaggle submission `56877854` at `0.944`.
Before spending GPU quota or creating another competition submission, the
highest-ranked public notebooks were audited for a genuinely reproducible
candidate above that score.

## Audited notebooks

### `pjmathematician/rsna-knee-d4-blend`

- Downloaded notebook SHA-256:
  `6fd1a3d61a15ad5922972fb6589d372a88d428442950d21740f304ae77fa6b72`.
- The public metadata contains blank dataset-source entries.
- Its trailing `OUR LEG` resolves `rsna-knee-eff6-assets` and
  `rsna-knee-ens14-assets`, but neither asset is named among the attached
  public dataset sources.
- The leg is blended into the public output at weight `0.45`.

Decision: rejected as a source-faithful reproduction candidate because the
claimed improvement depends on assets not available through the notebook's
public metadata.

### `yamadan96/rsna-knee-d4-public0946`

- Downloaded notebook SHA-256:
  `35547f8a28ed09fc68acefa6d23f12fa0bfd6a959f54fdadfdfc3dc3c32768f8`.
- Its own opening correction states that the public pipeline scores `0.943`,
  not `0.946`.
- It attributes the parent's `0.946` result to a private ConvNeXt component
  that is absent from the public metadata.

Decision: useful confirmation of the public frontier, but not an improvement
candidate.

### `aastikrajan15/knee-s75-w50`

- Downloaded notebook SHA-256:
  `ce6ea0f8b64d9ea50808f8fa36a588fc760fbc4aa1dcc14eb1f83d5469c0de8a`.
- Public metadata again contains a blank dataset-source entry.
- The notebook runs an additional embedded `ours_s10` model stage and blends
  it with the public stack at weight `0.50`.
- The required additional asset is therefore not independently identifiable
  from the public metadata.

Decision: rejected until the model asset, provenance, and license are publicly
resolvable and independently verifiable.

### `kolyaflexcu/rsna-knee-speedy947`

- Downloaded notebook SHA-256:
  `6f658d22bc1c8285a093614cae5a32dca9bce79274957b1810901fb4afbac86d`.
- Despite the notebook slug, its own documentation identifies the reproduced
  public score as `0.942`.

Decision: rejected as a score-improvement candidate.

## Conclusion

No audited public notebook supplied a fully reproducible, publicly attributed
candidate with verified evidence above the locked `0.944` baseline. No Kaggle
submission was launched from this audit. Future work must introduce a
materially distinct public model or a fully public checkpoint rather than
retesting unavailable private legs or blindly changing ensemble weights.
