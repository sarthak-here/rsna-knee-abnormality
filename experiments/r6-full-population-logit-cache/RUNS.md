# Run history

## Version 1 — failed, 2026-10-19

The single full-population run reached the fixed 11-hour worker deadline.
Worker 0 produced 1,636 genuine predictions and 568 deadline fallbacks;
worker 1 produced 1,370 genuine predictions and 833 deadline fallbacks. The
collector correctly failed closed with `R6 logit identity drift`; no cache was
accepted. This is a runtime partitioning failure, not a model failure.

The repair preserves the scientific contract and partitions the immutable
ordered population into two deterministic positional shards. Both shard jobs
must complete with zero fallbacks, and their IDs must be disjoint with an exact
4,407-study union before any calibration work begins.
