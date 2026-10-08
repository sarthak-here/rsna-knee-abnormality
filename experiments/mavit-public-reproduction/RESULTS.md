# Execution status

## Version 1: reproducible validation-only failure

The public model successfully loaded and predicted all three preview studies
with zero inference failures. The notebook then rejected its own staged CSV
because the strict check compared parsed decimal values as float64 against the
original float32 array bit-for-bit. The generated values were valid; the check
was incorrectly typed.

Asset discovery also recursively scanned the mounted competition tree, causing
about 25 minutes of unnecessary startup latency.

Version 2 changes no model, preprocessing, or probability contract. It:

- discovers the pinned public dataset directly and never scans competition DICOMs;
- validates CSV values by casting the parsed table back to the original float32 type.

The exact checkpoint, source hashes, raw-probability output, and strict
zero-fallback requirement remain unchanged.

## Version 2: validated and submitted

Version 2 completed with the exact pinned public assets, zero fallbacks, zero
failures, valid official preview schema and IDs, finite probabilities in
`[0,1]`, and a clean completion receipt. Exact notebook version 2 was submitted
to the competition as submission `56949917` with full Hengck23/MAViT
attribution. Its public score is pending.
