# R6 preprocessing identity correction — 20 October 2026

The exact released manifest, not parser defaults or earlier experiment prose,
defines these settings:

| Setting | Released value | Meaning |
|---|---:|---|
| `size` | 384 | Rendered slice side in pixels |
| `crop` | 352 | Model crop side in pixels |
| `eval_stride` | 1 | Window-centre stride multiplier |
| `phys_mm` | 3.5 | Through-plane physical sampling step, not crop field of view |
| `views` | v4 | Full field plus anatomy-centred local views |
| local crop | 100 mm | `px_cache.LOCAL_MM`; full-field view is also retained |

Pinned manifest SHA-256:
`6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648`.
Pinned `px_cache.py` SHA-256:
`b13e38a7ad3bd496f8027e02209653e224b0926db4f0853eeb0d076a2d6e2632`.

The preserved public controller reads
`manifest.get('eval_stride', 3)` and `manifest.get('phys_mm', 0.0)`.
The literal `3` is a fallback default; because these manifest keys exist, it
does not describe the deployed value. This was checked in preserved notebook
`this-is-best-public-dataset-new-version-0-9576.ipynb`, SHA-256
`cb0e1eac33a00f96b7d1bcc0e9914f7eb045af2bc9cd01bc290b2213e7f17ddd`.
The historical screen's `patch_stride_two()` replaces the entire manifest
lookup with literal `2`. Therefore its “stride 3 to 2 / denser coverage”
description was false. It tested a sparser replacement for released stride 1.
Its failed advancement decision remains closed; no retest or gold-driven
choice of stride follows from this finding.

Similarly, the generic 288px/140mm cache is not the released R6 pipeline.
The new CPU fixture harness reads stride/physical step directly from the pinned
manifest. The release's side classifier and existing SKM-TEA anatomy proposal
weights have also been recovered and loaded strictly on CPU:

- `side_cls.pt`: SHA-256
  `ab623e2b619db4917639220fed15202e8c8aedd07c8e298a66a2afc0f153650a`;
  21,278,913 parameters, existing R6 bundle, dataset license field `other`.
- `anatomy.pt`: SHA-256
  `5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e`;
  112,496,037 bytes, 9,362,768 parameters, from the existing deployed asset
  package `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`, license field
  `other`. Despite that package name, this asset is the released SKM-TEA VNet
  localization helper, not a newly approved OrthoFoundation-MSK-5M reader.

No new permission, controlled-access dataset use, pathology accuracy gain or
training authorization is inferred from these identities. Rich target lineage,
independent evaluation scope, final pixel/normalization parity and a measured
compute budget remain separate gates.
