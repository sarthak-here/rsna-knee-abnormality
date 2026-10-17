# Version 1 result: rejected

The private five-seed run completed successfully and passed every operational
integrity check, but failed every scientific advancement gate.

- Frozen anchor macro AUC: `0.9325528441`
- BiomedCLIP reader macro AUC: `0.7275269703`
- Fixed 10% residual macro AUC: `0.9297649173`
- Delta versus anchor: `-0.0027879269`
- Positive seed blends: `0/5`
- Nonnegative target deltas: `1/12`
- Bootstrap improvement probability: `0.0720`
- Bootstrap q025: `-0.0068746676`

Strict validation confirmed all five checkpoint hashes, 58 unique Unicode gold
IDs, twelve targets, five complete seed-prediction tensors, finite probabilities
in `[0,1]`, exact feature/model/OOF/anchor identities, 4,349/2,610/1,739/58
counts, clean logs, and no CSV or submission artifact.

This frozen BiomedCLIP target-attention contract is closed. Do not tune its head,
epochs, residual weight, or target subset, and do not build inference or submit it.
