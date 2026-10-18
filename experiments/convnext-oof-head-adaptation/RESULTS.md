# Version 1 result: rejected

The private paired five-seed run completed successfully and passed every
operational integrity check, but it failed three of the five predeclared
scientific advancement gates.

- Untouched public fold-0 head macro AUC: `0.9092582160`
- Steven-control adaptation macro AUC: `0.9120261741`
- OOF-corrected adaptation macro AUC: `0.9129608527`
- Candidate minus paired control: `+0.0009346785` (gate: `>= +0.002`)
- Candidate minus untouched public head: `+0.0037026366`
- Positive paired seed deltas: `5/5`
- Bootstrap improvement probability: `0.7560` (gate: `>= 0.90`)
- Bootstrap q025: `-0.0017557617` (gate: `> -0.001`)

Strict validation confirmed 58 unique Unicode evaluation-only IDs, twelve
binary targets, five complete control and candidate prediction tensors, finite
probabilities in `[0,1]`, exact 4,349/2,610/1,739/58 counts, zero gold fitting,
clean execution logs, and no CSV or submission artifact. The prediction NPZ
SHA-256 is `b0725f06eb418b13f374122028f61e7605f7d48c0b04161dbe7fdd5abe7f56cc`;
the report SHA-256 is
`6845f981d827cc630859cc827a7fb3d76785cd7f3b050d944d72ee32a60353f5`.

The correction signal is directionally consistent across all seeds, but its
paired effect is too small and uncertain to justify inference construction or
submission. This exact adaptation contract is closed. Do not tune its epochs,
learning rate, seeds, target subset, or checkpoint using organizer-gold results.
