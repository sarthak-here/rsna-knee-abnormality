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

