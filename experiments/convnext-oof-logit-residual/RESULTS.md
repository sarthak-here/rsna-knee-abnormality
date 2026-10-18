# Version 2 result: rejected

The repaired private run completed successfully and passed every operational
integrity check, but failed every predeclared scientific advancement gate.

- Untouched public-head macro AUC: `0.9092582160`
- Corrected residual-reader macro AUC: `0.8849722686`
- Reader delta: `-0.0242859474`
- Frozen anchor macro AUC: `0.9325528441`
- Fixed 10% candidate macro AUC: `0.9316838892`
- Candidate delta: `-0.0008689549`
- Positive seed blends: `0/5`
- Nonnegative target deltas: `5/12`
- Bootstrap improvement probability: `0.2379`
- Bootstrap q025: `-0.0033893693`

Strict validation confirmed 58 unique Unicode evaluation-only IDs, twelve
binary targets, five complete bag predictions, exact
4,349/2,610/1,739/58 counts, pinned feature/checkpoint/source/anchor hashes,
finite probabilities in `[0,1]`, clean logs, and no CSV or submission. The
prediction NPZ SHA-256 is
`17ffcc48d60d3b62b6c529d2e0ef03c2e5f9ff6fab903f46f37f5c7f52ccf21a`;
the report SHA-256 is
`e188a8d3a7bcc11f2edd51bbf792d5d208b8090865bc3552d2eaca5a2711790b`.

This exact residual-distillation contract is closed. Do not tune its projection,
ridge penalty, bagging, clipping, target subset, or blend weight, and do not
build inference or submit it.

