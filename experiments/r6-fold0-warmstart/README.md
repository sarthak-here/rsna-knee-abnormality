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

## Actual preview DICOM native-encoding gate

The official competition exposes individual public-preview DICOM files, so a
bounded fixture could be fetched without any training archive or GPU job.
`fetch_preview_series.py` downloaded 34 axial preview DICOMs (62,719,196 bytes),
plus the two small official test metadata CSVs, under the workspace directory
`C:/Users/sarthak/Desktop/codex/r6p`. The downloader refuses candidates exceeding
80 files/96 MiB or extending past the first listing-page boundary. No training
labels, new external MRI dataset or patient report was downloaded.

`audit_native_preview.py` verifies each raw-file hash and compares the published
training/native path with the inference worker's memory-saving implementation:

- Full-header versus header-only reads produce identical physical slice order.
- Full versus reread canonical pixel arrays and in-plane spacings match exactly
  for all 34 DICOMs, with no slice silently omitted by the ordering code.
- Whole-volume training quantization versus slice-wise inference quantization
  yields the same `[34,960,960]` uint16 bytes, display window, extrema and stated
  quantization metadata. This fixture is Explicit VR Little Endian; it does not
  establish parity for every compressed transfer syntax.
- Native-array SHA-256 is
  `0ec597d62f7368a6768c7aac2e5b3b1143eb19325d2f209541940165e34309c1`.
  Source identities and pydicom 3.0.1 are pinned. Dependencies are installed only
  into `.audit/r6-cpu-deps`, without changing the user's global environment.

```powershell
python experiments/r6-fold0-warmstart/audit_native_preview.py --fixture C:/Users/sarthak/Desktop/codex/r6p --code-root .audit/rsna-knee-blend-code/cloud_code --probe-worker .audit/rsna-knee-blend-code/r6src/probe_infer.py --deps .audit/r6-cpu-deps --out experiments/r6-fold0-warmstart/native_preview_receipt.json
```

**Scope:** one real preview series, before study-level laterality correction;
not a complete study, no anchor model, no anatomy-centred v4 rendering and no
pixel-derived fluid-volume validation yet. No model predictions, optimizer or
GPU were used. Do not mark the whole preprocessing gate complete or use this
small uncompressed fixture to forecast training time across the competition.

## Manifest-pinned real v4 study gate

`PREPROCESSING_IDENTITY.md` corrects the earlier stride and physical-spacing
description: released R6 uses stride **1**, a **3.5 mm through-plane step**,
384px rendered slices, 352px model crops, and **100 mm local views** plus full
field. Do not copy the generic cache's 140 mm setting into this reader.

`fetch_preview_series.py --study-uid` extends the existing raw fixture to one
complete official preview study, refusing more than eight listing pages,
400 files or 512 MiB. It reuses existing DICOMs, keeps the earlier series receipt
untouched, and writes a separate immutable whole-study receipt. This acquires
neither training labels nor controlled-access MRI data.

`audit_preview_v4.py` is CPU-only before importing torch. It pins the release
manifest, renderer/trainer, anatomy locator and side classifier. It invokes the
original worker's real study-side/geometry/anchor/rendering path and compares
its window tensors with the original training `Studies(mode='eval')` loader
over the captured rendered cache. No disease model or optimizer runs.
It fails on omitted series, missing/out-of-field anchors or runtime-error
fallbacks, and explicitly distinguishes accepted segmentation anchors from
the source's geometric slab-prior branch. A matching fixture is not evidence
that every protocol, augmentation, encoder normalization or positive-fluid
branch has been validated.

```powershell
python experiments/r6-fold0-warmstart/audit_preview_v4.py --fixture C:/Users/sarthak/Desktop/codex/r6p --fixture-receipt C:/Users/sarthak/Desktop/codex/r6p/study_fixture_064d58efe1af.json --out experiments/r6-fold0-warmstart/preview_v4_receipt.json
```

The terminal `preview_v4_receipt.json` now records a **partial implementation
pass**, not complete preprocessing certification:

- All five official series / 284 DICOM files (436,378,362 raw bytes) were
  present and hash-checked. The prior 34-file fixture receipt was preserved.
- The deployed loader and original training evaluation loader produced
  bitwise-identical `[574,3,384,384]` uint8 windows and `[574,9]` metadata.
- Explicit left-knee DICOM tags were accepted consistently with sagittal
  position; this fixture did **not** exercise image-classifier correction.
- Both sagittal series are non-fluid-sensitive. The segmentation model found
  zero qualifying tissue mass, so the original code explicitly selected its
  **geometry-derived slab prior**, not a substituted field-centre anchor.
  All five projected anchors were in field. A positive segmentation anchor
  therefore remains untested.
- The source correctly reported `no fluid-sensitive sagittal series`; actual
  positive fluid-volume measurement remains untested, not silently counted as
  a successful measurement.
- CPU runtime including auxiliary model loading was 29.925 seconds. No GPU,
  disease-network forward, labels, optimizer, predictions or submission ran.

The complete preview fixture receipt SHA-256 is
`836d62c9343f36a164d7ddaf1500e833809aef7a5bae47dfc6a05758e158c8cf`.
Do not redownload or repeat this completed fixture gate. A fluid-sensitive
sagittal preview fixture is needed for the remaining positive anatomy/fluid
branches; model-crop/encoder-normalization parity also remains separate.
This result does not authorize a GPU probe or solve supervision/evaluation.

## Positive anatomy/fluid fixture and real encoder-input gate

`preview_fluid_v4_receipt.json` adds a **different** complete official preview
study with fluid-sensitive sagittal sequences: five series, 135 DICOMs,
91,028,286 bytes. Its immutable raw receipt is
`C:/Users/sarthak/Desktop/codex/r6p/study_fixture_6de38b5f7bf9.json`, SHA-256
`94a0184c47431e76c7b271ea8a599a9be27742d18df8bfec2c11f0e0d4a87e4e`.
No earlier fixture was redownloaded or earlier completed check repeated.

- Original training-evaluation and deployed loaders match bitwise on real
  `[510,3,384,384]` uint8 windows and `[510,9]` metadata.
- The published locator accepts two sagittal candidates: method
  `skm_tea_joint_avg2`, with a patella anchor and all five projections in field.
  No geometric slab-prior or field-centre substitution was used for this study.
- With no explicit laterality tags, the published side classifier was actually
  exercised: score `+7.833`, retaining the inferred right side. This tests its
  **retain** branch, not the forced-flip/re-render branch.
- The exact source D2 pixel heuristic returns finite nonnegative volumes:
  Effusion channel `1.1076069875 mL`, Baker's channel `9.4348099974 mL`.
  Two fluid-sensitive sagittal series were available. These are source
  heuristic outputs, **not** validated clinical segmentations or disease scores.
- CPU execution including auxiliary model loading took 40.287 seconds. No
  pathology reader, optimizer, labels, GPU or submission was used.

`input_parity.py` and `input_parity_receipt.json` separately verify the previously
untested **352px crop and normalization**, using the first study's retained
rendered cache rather than repeating raw decoding/anchor tests. The complete
window hash was checked against the earlier immutable receipt before sampling.
One actual window per slot/view combination gives 15 real views. AST-extracted
original training `center()` and worker crop assignment give bitwise-equal
`[1,15,3,352,352]` crops. The exact fold-0 checkpoint is loaded strictly; a
forward pre-hook intercepts normalized pixels **before** encoder computation.
The intercepted `[32,3,352,352]` channels-last tensor (including the source's
zero-window padding) matches the preserved mean/std formula bitwise, SHA-256
`f0e958322f8b9c6c43cce6e5b181886d4f8a52dcd521a6dc25fcbf2a384c19de`.
Neither the encoder nor disease heads execute. The optional
`audit_preview_v4.py --check-model-input` integrates this same check for future
fixtures; do not rerun the completed fixtures merely to regenerate receipts.

**Scope still limited:** no forced side-flip/re-render fixture, all-protocol or
all-compression coverage, stochastic training augmentation parity, full disease
forward or clinical accuracy proof. Original rich supervision, source-prior /
postprocessor exposure, the five-study pool discrepancy, independent evaluation
scope and bounded compute remain unresolved before a training launch. These
implementation gates do not establish a better model or justify GPU spending.
