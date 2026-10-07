# Result

The full-population paired target experiment was rejected.

- Control ensemble gold macro AUC: `0.8413796`
- Corrected-target ensemble gold macro AUC: `0.8483561`
- Delta: `+0.0069765`
- Positive seeds: `4 / 5`
- Paired-bootstrap `p_improve`: `0.8706`
- Paired-bootstrap 95% interval: `[-0.0055199, 0.0201912]`
- Fit studies per arm: 4,349
- OOF-corrected studies: 2,610
- Unchanged studies: 1,739
- Gold studies used for fitting: 0
- Submission artifacts: none

The point estimate and seed-count gates passed, but bootstrap confidence and
the lower-bound gate failed. Partial correction is therefore not reliable
enough to justify training an inference submission from this contract. The
locked public baseline remains `0.944`.
