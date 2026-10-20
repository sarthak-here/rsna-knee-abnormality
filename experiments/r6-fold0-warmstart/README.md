# R6 fold-0 strict reconstruction — CPU gate, 20 October 2026

## Result

The exact released **knee-trained** R6 fold-0 checkpoint now reconstructs
successfully on CPU. This is a real weight-loading/implementation result, not
an accuracy improvement or permission to launch training.

- Verified exact SHA-256 for the 200,379,391-byte fold-0 checkpoint, published
  state specification, and attributed trainer before executing the loader.
- All **376** checkpoint tensors load strictly into the complete network.
  Spatial attention, extent head, synovitis decomposition, report-state and
  binary heads are retained. Every loaded tensor equals its checkpoint value.
- A deliberate missing classification-head tensor is rejected. No backbone-only
  load, `strict=False` allowance, or fabricated source-pool manifest is used.
- An untouched reference and a copied candidate give bitwise-identical
  synthetic head outputs, including report-state/binary logits, before training.
- Parameter selection enables only `enc.stages.3.*`: 31 tensors, **15,470,592**
  parameters of 50,064,966 total. All existing heads remain frozen and intact.
  Tests verify this operation changes no stored weight or buffer.
- Local dependency/source versions are recorded in `strict_load_receipt.json`.
  timm 1.0.25 is taken from the preserved inference environment; key source-file
  hashes are retained in the receipt rather than silently importing latest code.
- No GPU use, optimizer, study predictions, labels, gold scoring or submission.

The untouched baseline is the exact loaded checkpoint, not a freshly initialized
head. Freezing its head weights does not guarantee outputs remain unchanged
after encoder adaptation; that must be measured in the eventual paired test.

## Reproduce

The checkpoint and state specification were downloaded from the existing
attributed R6 release `heliosli/rsna-knee-r6-frozen-fe055406-w1` into ignored
`.audit/r6-strict-load`. No new OAI dataset or controlled-access asset was used.
The dataset lists its license as `other`; this reconstruction does not invent
new permission for redistribution or a materially new use.

```powershell
python experiments/r6-fold0-warmstart/strict_reconstruct.py --checkpoint .audit/r6-strict-load/E_lrprior_s0_f0.pt --state-spec .audit/r6-strict-load/state_spec.json --trainer .audit/rsna-knee-blend-code/r6src/compact_train.py --timm-root .audit/frontier-2026-10-17/goodpjw958/output/_own_env --out experiments/r6-fold0-warmstart/strict_load_receipt.json
python -m unittest discover -s experiments/r6-fold0-warmstart -p "test_*.py"
```

## Remaining gates — not solved by this load test

1. Raw-DICOM -> native decoding -> published anatomy-centred v4 windows ->
   normalization/cropping and physical metadata must match the deployed worker.
   The generic 288px/12-slice cache used by video experiments is **not** a
   substitute for this R6 pipeline. The receipt deliberately records raw-pixel
   parity as false. The current synthetic test starts after the encoder.
2. Synthetic parity for flip permutation/MCL exception, state readouts, and
   fluid/state coefficient application now passes (details below). End-to-end
   real-study parity, including measuring fluid volumes from pixels, remains
   untested. A raw-logit gain must not be called a gain over the deployed stack.
3. Define evidence-based supervision and budget before an optimizer step. The
   original likelihood-ratio/state target table remains missing; Steven/older
   reader labels are not proven stronger. This loader is not an unchanged-source
   reproduction and does not bypass the original trainer's warm-start checks.
4. Use only the source fold-0 model for its guarded holdouts. The independent
   CPU audit gives 3,317 candidate fitting studies and 977 report-guarded
   image-gradient-held-out studies, conditional on published lineage. Fix any
   tuning/confirmation subdivision before evaluation. Source pool-wide priors
   prevent describing this as fully independent whole-pipeline OOF.
5. Measure bounded paired-control throughput before any Kaggle GPU launch;
   persist checkpoints/optimizer/RNG and stop if the proposed budget is not
   feasible. Do not revive the cancelled full-population cache/calibration jobs.

Attribution: Heliosli's released R6 package and bundled trainer/inference source,
with their preserved upstream credits. The scored 0.957 baseline is untouched.

## CPU postprocessing gate

`postprocess.py` pins the existing inference worker and both published stacker
specifications, then preserves the exact deployed TTA and readout ordering.
`audit_postprocess.py` extracts the original worker's scoring/stacker function
bodies as an independent reference and compares deterministic **synthetic**
fixtures. It never runs the worker's main routine, writes CSVs or predicts studies.

- Trainer/inference mirror transforms, inverse mirror, class/binary permutations
  and report-state column layouts match.
- Sixteen fixtures verify per-pass logit/state/binary assembly, including the
  MCL plain-only exception. Results match the source bitwise.
- Ninety-six fixtures verify final probability assembly for the published
  Effusion fluid stacker and Baker's state stacker. They cover absent, zero,
  positive, negative and NaN volume inputs: the source's explicit missing-volume
  branch is preserved, not confused with an unseen-reader fallback. Maximum
  absolute probability difference from source is **0.0**; no source fallback
  branch was triggered in these fixtures.
- Invalid nonfinite logits are rejected rather than silently retaining an
  uncorrected prediction. No coefficients are fitted or changed.

```powershell
python experiments/r6-fold0-warmstart/audit_postprocess.py --worker .audit/rsna-knee-blend-code/r6src/compact_infer.py --trainer .audit/rsna-knee-blend-code/r6src/compact_train.py --bundle-dir .audit/r6-strict-load --out experiments/r6-fold0-warmstart/postprocess_receipt.json
```

The scope is deliberately narrower than complete model/pixel parity. Synthetic
fold counts test arithmetic only; no other fold's checkpoint was loaded or used
to evaluate fold-0 holdouts. These tests neither authenticate stacker fitting
lineage nor establish an independent accuracy score. The original load receipt's
`postprocessor_parity_tested=false` describes that earlier test; the separate
postprocessing receipt records this later synthetic gate precisely.
