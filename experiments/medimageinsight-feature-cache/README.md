# MedImageInsight physical feature cache

This private no-label/no-submission experiment extracts a materially distinct
medical-image representation from Microsoft's MedImageInsight DaViT encoder.
The exact public Kaggle model is
`thelionai/medimageinsight/PyTorch/default/1` (MIT), with checkpoint SHA-256
`0453e28ee7d1585796ee74f34a6b7ec3af198b63f19052e46a3ade2a5d68ea22`.

The immutable physical cache contributes four fixed, non-overlapping 2.5D
windows for each of six clinical sequence slots. The frozen encoder produces a
unit-normalized 1024-dimensional embedding for every occupied window. No labels,
organizer-gold identities, anchor predictions, CSVs, or submission capability
are present in this stage.

Required output contract:

- exactly 4,407 unique Unicode study IDs in immutable manifest order;
- features `[4407,6,4,1024]` float16 and masks `[4407,6]` uint8;
- exactly 23,089 occupied sequence slots, finite normalized embeddings, and
  zeros only for missing slots;
- pinned checkpoint/config/loader/HDF5/manifest hashes and complete receipt;
- no CSV and no submission artifact.

If this cache passes strict local validation, the separately predeclared screen
will train a fixed five-seed target-attention head on 4,349 non-gold studies,
with 2,610 authenticated OOF-corrected targets, 1,739 unchanged targets, all 58
organizer-labelled studies evaluation-only, and a fixed 10% rank-residual slot
against the frozen four-reader anchor.
