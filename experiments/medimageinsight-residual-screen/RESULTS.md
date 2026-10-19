# Version 1 result: rejected

The fixed five-seed screen completed with full operational integrity but failed
every scientific advancement gate.

- frozen anchor macro AUC: `0.9325528441`
- MedImageInsight reader macro AUC: `0.7450981569`
- fixed 10% candidate macro AUC: `0.9294880521`
- candidate minus anchor: `-0.0030647920`
- positive seed blends: `0/5`
- nonnegative target deltas: `3/12` (Synovitis was numerically unchanged)
- bootstrap improvement probability: `0.0431`
- bootstrap q025: `-0.0070333964`

Strict validation confirmed 58 unique pickle-free evaluation-only IDs, five
complete prediction tensors and checkpoint hashes, exact
4,349/2,610/1,739/58 counts, finite probabilities, clean logs, and no CSV or
submission artifact.

Decision: reject and close this exact MedImageInsight target-attention residual
contract. Do not tune its head, epochs, seeds, target subset, or blend weight;
do not build inference or submit it.
