# Result: rejected

Private notebook `sarthaksharma14/rsna-knee-convnext-reader-mirror-validation`
version 1 completed successfully and produced no submission.

- Plain public ConvNeXt fold-0 reader: `0.909258` macro AUC
- Mirrored view alone: `0.906921`
- Fixed 50/50 plain/mirror average: `0.910321`
- Delta: `+0.001063`
- Bootstrap `p_improve`: `0.6666`
- Bootstrap 95% interval: `[-0.003861, 0.005881]`

The candidate failed every confidence gate except remaining finite and
leakage-free: delta was below `+0.005`, `p_improve` was below `0.95`, q025 was
negative, and absolute AUC was below `0.914`. All 58 organizer-labelled studies
were evaluation-only. The checkpoint and source hashes matched, predictions
were finite in `[0,1]`, logs were clean, and no submission artifact existed.

Do not scale or submit this mirror transform on the public ConvNeXt reader.
