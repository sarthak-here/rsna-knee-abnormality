# Frozen scored0.958 control for follow-on comparisons

Reference: approved, completed submission57483292, exact Heliosli kernel `blend-mine-in-private-lb-luck-to-all-upvoters` version1, script358751289; source SHA `ad0b97aca3361f16e32b0836655dd68bc2a37ff0cc5a44f6d6f4de07880d986d`. Attribution: Heliosli, nartaa, dreaddevelopment; upstream Raptor OAI provenance remains acknowledged. This comparator does not change any model, preprocessing, TTA, checkpoint, stacker or weight.

The source's **zero-reader-failure** prediction formula is:

`w[c]*(0.65*average_tie_percentile_rank(Raptor384)+0.35*average_tie_percentile_rank(Raptor224))+(1-w[c])*average_tie_percentile_rank(R6)`.

Raptor weights are0.20 LateralOA,0.60 PFOA,0.80 Synovitis,0.35 Baker's,0.50 elsewhere. R6 inputs must already include source flip/MCL handling and state/fluid stackers. This is neither rank-after-raw-probability-blending nor ordinal tie-breaking. Results are cast to float32 exactly as in the published final cell. The comparator rejects invalid values, shape/order mismatches and source Raptor0.5 fallback sentinels; it does not implement fallback blending or reproduce a degraded hidden run.

Exact `/1` public-preview reader outputs were freshly downloaded without executing publisher code. `control.py` recomputes the blend and matches all36 preview score cells **bitwise after float32 conversion**, maximum error0. Input hashes and target order are recorded in `parity_receipt.json`. Three unit tests cover average ties, row-permutation equivariance, invalid-reader rejection and constant-score ties.

```powershell
python -m unittest discover -s experiments/frozen-0958-control -p test_control.py
python experiments/frozen-0958-control/control.py .audit/approved-helios958-v1 --receipt experiments/frozen-0958-control/parity_receipt.json
```

This is three-study preview arithmetic parity only, not independent pathology evidence, gold evaluation, hidden-output replay, a newly trained model or an additional public-score gain. No labels read, AUC computed, model fitting, GPU use or competition submission. It provides the correctly frozen control for a separately justified candidate; it does not authorize new GPU expenditure or revive failed contracts. Final picks57483292/57197157 remain unchanged. Do not tune this formula or use repeated gold/LB comparisons to fit its weights.
