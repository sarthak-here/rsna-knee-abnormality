# Public ConvNeXt reader feature cache

This private extraction job builds a reusable study-level window feature package
from the public Apache-2.0 ConvNeXt reader. It pins checkpoint fold 0 and the
author's source hashes, then applies the exact canonical DICOM preprocessing,
six clinical slots, twelve deterministic evaluation windows, 256px views, and
FP16 inference used by the published reader.

The cache contains 72 ConvNeXt tokens per study plus slot masks and normalized
slice positions for all 4,407 training studies. It performs no fitting, reads no
organizer labels, and cannot create a submission. A separate paired experiment
will compare Steven v4 targets with legitimate OOF-corrected targets on this
stronger representation.

## Attribution

- `goodpjw2008/rsna-knee-2-5d-convnext-reader`, Apache-2.0

## Result

Version 2 completed and passed strict full-file validation. The cache contains
features shaped `[4407, 6, 12, 768]` in FP16, 21,334 occupied slots, no empty
studies, no zero occupied slots, and finite features and positions. The
363,928,367-byte artifact has SHA-256
`7f1986bb30cf92dbbf440b43a5a5933c1a3d13a0bb61d91950457cb0f8e018cc`.
All checkpoint/source hashes and arguments matched. No submission was created.
