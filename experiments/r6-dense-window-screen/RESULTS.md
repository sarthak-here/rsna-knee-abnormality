# Terminal result

R6 stride-2 inference completed with full operational integrity but did not
clear the advancement gate.

| Measure | Result | Gate |
|---|---:|---:|
| Frozen anchor macro AUC | 0.9387940564 | reference |
| Stride-2 candidate macro AUC | 0.9391694047 | reference |
| Candidate minus anchor | +0.0003753484 | >= +0.001 |
| Nonnegative targets | 9 / 12 | >= 8 / 12 |
| Bootstrap probability positive | 0.64636 | >= 0.90 |
| Bootstrap q025 | -0.00179217 | > -0.001 |

Integrity passed: exact R6 source and anchor identities, all 58 unique ordered
evaluation-only studies, zero fallbacks, both worker exit codes zero, finite
bounded predictions, stride-2 runtime marker, clean logs, no CSV, and no
submission capability. Prediction SHA-256 is
`8681fa0fb16f0f445514c5a1b8201281d391133462fdbfbf1c7359f31af0a203`.

This exact contract is rejected and closed. Another evaluation stride must not
be selected using organizer-gold results.
