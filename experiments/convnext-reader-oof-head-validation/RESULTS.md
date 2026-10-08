# Result

The ConvNeXt-reader OOF-corrected head experiment was rejected.

- Original public fold-0 head gold AUC: `0.9092582`
- Fresh Steven-control ensemble gold AUC: `0.8972215`
- OOF-corrected ensemble gold AUC: `0.9009144`
- Corrected minus control: `+0.0036930`
- Corrected minus original public head: `-0.0083438`
- Positive seeds: `4 / 5`
- Paired-bootstrap `p_improve`: `0.7702`
- Paired-bootstrap 95% interval: `[-0.0059782, 0.0142947]`
- Gold studies used for fitting: 0
- Submission artifacts: none

The absolute-AUC and positive-seed gates passed, but the minimum delta,
bootstrap-confidence, lower-bound, and original-reader comparison did not.
This rejects replacing the published reader head with the partial-coverage OOF
target recipe. The locked public baseline remains `0.944`.
