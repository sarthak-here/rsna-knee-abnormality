# Supervised final-stage adaptation: rejected

The private leakage-safe end-to-end adaptation completed successfully and did not create a competition submission.

## Result

- Incumbent ensemble macro AUC: **0.860457**
- Adapted ensemble macro AUC: **0.831551**
- Paired delta: **-0.028907**
- Positive seed deltas: **0/5**
- Paired-bootstrap probability of improvement: **0.0004**
- Paired-bootstrap 95% interval: **[-0.046557, -0.012440]**

The adapted checkpoint matched the recorded 94,347,023-byte artifact and SHA-256 `bf9b8f923f2a882c7d69b84a797fe99a115fe24f883fff4be53b19d618c4ff10`. The fixed split contained 3,923 fitting and 426 weak-validation studies with zero report-group overlap. All 58 organizer-labelled studies remained evaluation-only, and the execution log contained no runtime-error markers.

Weak-validation AUC improved during the fixed three adaptation epochs, but the frozen adapted representation was below the control on every gold-evaluation seed. The result rejects weak-label supervised final-stage adaptation under this contract.

## Decision

Reject this branch. Do not tune adaptation depth, learning rate, epochs, or target subsets against the 58 gold studies, and do not blend or submit it. Continue with a materially different evidence source rather than further fitting the image encoder to noisy report labels.
