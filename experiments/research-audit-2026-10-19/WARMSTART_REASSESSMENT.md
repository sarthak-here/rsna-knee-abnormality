# Reassessing direct knee-checkpoint fine-tuning — 20 October 2026

## A narrower conclusion than “missing pool IDs make adaptation impossible”

That statement was too broad. The local attributed R6 trainer matches **both**
the released manifest's SHA-256 and trainer MD5:

- SHA-256 `04df7e2e154b2c2521696908f272940c63ce17e9b6337b85847fee64504eeaba`
- MD5 `8622009e3db0414124d5614ed252d7a2`

The source defines fold membership as
`int(sha1(uid).hexdigest()[:8], 16) % 4`, not the full SHA integer.
Its training rows for fold k explicitly exclude fold k. If the published
checkpoint/trainer lineage is correct, a missing list of the 4,344 pool IDs
does not prevent deriving a conservative **image-gradient exclusion** set:
all official IDs assigned to k were excluded whether or not they were in the
original pool. No invented pool manifest or bypass of the original warm-start
checks is needed to establish that narrower claim.

The manifest remains the evidence for which trainer produced the checkpoint;
this is not independent authentication of the author's original execution.

## Reproduced CPU counts

Run from the repo root:

```powershell
python experiments/research-audit-2026-10-19/audit_fold_exclusions.py --train .audit/competition-metadata/train.csv --manifest .audit/r6-w1/manifest.json --trainer .audit/rsna-knee-blend-code/r6src/compact_train.py
```

All 58 gold IDs and four non-gold exact-report duplicates of gold are excluded.
A validation report group is retained only when **every official study in that
group** hashes to the same held-out source fold; this removes cross-fold report
duplicates the original UID split would otherwise expose.

| Fixed source checkpoint | Candidate fitting rows outside its fold | Eligible UID-hash holdout | Report-guarded image holdout |
|---|---:|---:|---:|
| fold 0 | 3,317 | 1,028 | 977 |
| fold 1 | 3,212 | 1,133 | 1,095 |
| fold 2 | 3,228 | 1,117 | 1,068 |
| fold 3 | 3,278 | 1,067 | 1,029 |

Fold 0's sorted guarded holdout-ID digest is
`6fe5e0679ff0096d7e4c03d2337df06f599ba223ed660d5f89c9d7d6c017cdae`.
These are metadata counts, not evaluated model results. The five-study pool
difference is not explained by this calculation and must not be presented as
reconciled.

## What this enables

The next sensible *implementation target* is a single R6 fold-0 transfer probe,
not generic video pretraining, all-fold logits, or an immediate four-fold run:

1. Load exact fold-0 checkpoint
   `59cc88a197bc80c7fa5faa89e69ae7bec04f69cab5259a23a0d0ceb9431e53f6`
   with the published architecture/state spec and strict keys. Keep one
   untouched reference copy. Reproduce its input normalization and prediction
   path before taking any optimizer step.
2. Preserve existing heads, state/extent semantics and preprocessing. Freeze
   early ConvNeXt stages and initially train only the last stage at a low LR.
   The recipe's `freeze_blocks=16` flag applies to its OFL backbone, not this
   ConvNeXt; it must not be mistaken for evidence the released R6 has 16 frozen
   blocks. This is continuation from knee-trained weights, unlike the recent
   generic VideoMAE/Swin experiment.
3. Fit only outside the source fold. Split guarded holdout report groups into
   tuning and sealed confirmation before measuring losses. Never use the other
   three R6 folds on this holdout: they may have fitted its images.
4. Fix the new supervision/objective and compute budget **before** the probe.
   Missing original targets prevent calling a different-label continuation an
   unchanged-source reproduction; they do not make every new transfer objective
   mechanically impossible. Conversely, replacing stronger rich labels with
   Steven probabilities is not an established improvement. Do not use an
   existing cross-fold competition teacher without entire-holdout exclusion.
5. Evaluate the same complete output path before/after adaptation; do not
   silently drop or refit deployed state/fluid postprocessors. Establish their
   fitting exposure separately before treating the result as independently OOF.

This is not a ready-to-launch contract: strict checkpoint reconstruction, exact
pixel/preprocessing parity, the supervision decision, independent evaluation
scope, and measured bounded compute remain implementation gates. Do not fake
`pool_uids.json` or loosen the source trainer's warm-start assertions to proceed.

## What it does not solve

The trainer sets class bias from mean priors across the full source pool
(`bias0 = logit(col(pool, 'prior', ...).mean(0))`). Published state priors also
use pool information. Thus image-gradient exclusion is **not** a certificate
that the entire upstream pipeline excluded every validation label/statistic.
Report-derived validation targets remain proxy labels, not independently
annotated pathology truth; the repeatedly inspected organizer set is not fresh.
The old full-population calibration path stays closed. No +0.01, competitive
accuracy gain, GPU launch or submission follows from this CPU audit alone.
