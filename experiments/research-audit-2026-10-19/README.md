# Research audit — 19 October 2026

## Verdict

The current evidence does **not** establish a path to a public +0.01 improvement
within four days. Our own training has not produced a validated replacement for
submission 57197157 at public 0.957. The gains from 0.944 to 0.956 to 0.957 came
from attributed public inference contracts. They are real scored improvements,
but they do not demonstrate that our training strategy is working.

The audit examined the repository's result records (44 RESULTS.md documents),
the exact R6 manifest and trainer/inference source, the calibration evaluator,
and existing immutable prediction caches. No new model was fitted or submitted.
Run `python experiments/research-audit-2026-10-19/audit_models.py` from the repo
root to reproduce the cache calculations. It requires the existing local
artifacts under `.audit/` and the preserved uncertainty-router output.

## Evidence by model family

Gold AUCs below are descriptive on the same small organizer set, **not public
leaderboard scores**. Different anchors must not be compared as if identical.

| Family / contract | Measured result | Implication |
| --- | --- | --- |
| Public ConvNeXt MIL reproduction | public 0.944 | Strong released reader, successful reproduction |
| Public R7/Raptor224/384 contract | public 0.956 | Released stack supplied the +0.012 improvement |
| Public R6/Raptor224/384 contract | public 0.957 | Current locked baseline |
| Later GP958d R7/Raptor contract | public 0.956 | No replacement improvement |
| MAViT reproduction | public 0.935 | Architecturally different, insufficient as replacement |
| Curia-2/V52 reproduction | public 0.899 despite +0.0101 local diagnostic | A local gain against a weaker control did not establish a competitive model |
| OOF-corrected RadImageNet student, 2,610 rows | 0.829413 -> 0.847558 (+0.018145) | Correction helps this particular weaker student |
| Same partial correction extended to all 4,349 | 0.841380 -> 0.848356 (+0.006977), uncertain | Does not establish transfer to R6 |
| Fresh ConvNeXt heads | original public head 0.909258; corrected head 0.900914 | Rebuilding the head lost performance |
| Public ConvNeXt head adaptation | 0.912026 control -> 0.912961 corrected (+0.000935) | Small paired effect; no evidence of +0.01 on a strong reader |
| ConvNeXt logit correction residual | 0.909258 -> 0.884972 | Correction mechanism failed |
| Cross-plane token head | reader 0.893531; tested blend -0.001317 | Did not improve its tested anchor |
| BiomedCLIP head | reader 0.727527; tested blend -0.002788 | Large strength gap |
| MedImageInsight head | reader 0.745098; tested blend -0.003065 | Large strength gap |
| Spatial DINOv2 head | reader 0.749965; tested blend -0.001166 | Retaining a patch grid did not fix the tested representation |
| Morphology ExtraTrees | reader 0.694467; tested blend -0.002113 | Not competitive under the tested features |
| Pairwise AUC head | reader 0.901438; tested blend -0.001429 | Objective change did not help the tested blend |
| End-to-end DINOv2 / RadImageNet / ConvNeXt screens | gold 0.669819 / 0.828038 / 0.817358 | These specific recipes failed; end-to-end training was already attempted |
| Runtime-bounded CoAtNet-384 | gold 0.677085 after two epochs | Under this budget the recipe was far from competitive |
| MARS 3D / high-res DINOv2 / ConvNeXtV2 domain pool | gold 0.656439 / 0.752404 / 0.827406 | More pretrained encoders did not solve the training gap |
| BYOL / sequence SSL / supervised representation adaptation | deltas -0.026956 / -0.025539 / -0.028907 | All degraded their tested controls |
| Segmentation ROI / report latent / entity supervision | deltas -0.000850 / -0.000942 / -0.000433 | No reliable gain in tested forms |
| R6 stride-2 | tested blend +0.000375 | Below gate and uncertain; not a breakthrough |
| R6 uncertainty router / robust reducer | deltas -0.001119 / -0.000460 | No evidence for those reducers |

These closed contracts remain closed. This audit is not permission to tune their
weights, choose targets retrospectively, or resubmit them.

## Four defects in the current research argument

### 1. Cross-fitting a calibrator does not undo base-model training exposure

The R6 manifest records four folds and a 4,344-study training pool. The shipped
trainer computes its folds as `sha1(uid) % K` in `compact_train.py`; the proposed
calibrator instead uses five `GroupKFold` splits and supplies all eight fold/pass
logits for each study. For original pool members, three of the four base folds
were trained on that study. Holding the study out of the ridge fit does not hold
it out of those encoders. The result cannot be described as independent OOF
evaluation of the whole predictor.

Also, the authenticated ConvNeXt table proves that a teacher did not fit the
individual row it predicts. That alone does not prove that every teacher used to
construct *training* targets excluded the entire new outer validation group.
Pseudo-label and student fold lineage must be audited together.

**Decision:** do not use the existing calibration gates to approve inference or
submission. A full-population cache remains potentially useful for diagnostics,
but it is not a leakage-free validation set and does not establish a need to
finish these GPU jobs at any cost. Both repaired shards were QUEUED when checked
during this audit; no cancellation was issued by this read-only review.

### 2. The calibration comparator drops deployed postprocessing

`r6-crossfit-fold-calibration/evaluate.py` loads final R6 probabilities but defines
the comparator as `sigmoid(logits.mean(axis=(1,2)))`. R6 applies a fluid stacker
for Effusion and a state stacker for Baker's after that averaging. The new
candidate also predicts directly from the eight raw class logits.

On the preserved router cache this changes 116 of 696 probability cells, with
maximum absolute difference 0.912865. Deployed R6 gold AUC is 0.936579; the bare
head is 0.937692 on that cache. This is a comparator mismatch, regardless of
which happens to score higher on these 58 studies. A gain against the bare head
would not establish a gain over deployed R6 plus Raptor.

### 3. Several residual screens used a different anchor from the scored stack

The public zero-failure blend is
`0.50*rank(R6) + 0.325*rank(Raptor384) + 0.175*rank(Raptor224)`.
Several residual screens instead formed
`rank(0.50*R6 + 0.325*Raptor384 + 0.175*Raptor224)`.
Ranking after combining raw probabilities is not the same operation.

Recalculation on the exact same multireader cache (SHA-256
`6cdea905549c3df74d94c08a0547a556c9b3a890f6b1d90e432f0b79d3b0c124`):

| Comparator on that cache | Gold macro AUC |
| --- | ---: |
| Raw-probability blend used by residual screens | 0.9325528441 |
| Published average-tie rank formula | 0.9387134632 |
| Old nested optimizer's ordinal-rank formula | 0.9388184192 |

The ordinal-rank optimizer breaks ties by row order, unlike the published
average-tie rank formula. Prior negative residual results still reject their
tested contracts, but cannot be presented as exact paired tests against the
locked deployment. These comparator discrepancies also explain why supposedly
frozen anchor AUCs differ across reports; a single cache SHA is insufficient to
pin the evaluation formula.

### 4. The proposed R6 retraining recipe does not reproduce R6's supervision

The actual manifest records ConvNeXt-Small, 352px crops from a 384px v4 cache,
176 training windows per study, four folds, 16 epochs, and
`E_lrprior=lr:lr_prior:w`. The source supports weighted likelihood-ratio targets,
report-state supervision and additional heads; deployed state/fluid stackers
consume more than twelve class probabilities. Its target file is
`probe_targets_opusv11b_rr6_plr6_cdr.csv`.

The proposed 50/50 Steven/old ConvNeXt target table supplies twelve probabilities
and is not that target file. There is no paired R6 evidence that it is better.
Replacing rich supervision from the stronger R6 recipe with predictions from a
weaker reader could remove useful information; that is an inference to test,
not an established correction. The original 4,344 pool and our 4,349 non-gold
population also differ by five studies and must be reconciled explicitly.

The source already contains a strict whole-model `--warm-start` mechanism that
requires matching folds, pool IDs, architecture and state categories. Promising
a new five-fold warm start from these four-fold checkpoints would violate those
conditions and contaminate new holdouts. The actual source targets, state-target
table, pool UID manifest and training cache must be located and authenticated
before claiming a reproducible control fold. They have not been established by
the current full-population inference job.

A live read-only `kaggle datasets files` inspection of
`heliosli/rsna-knee-r6-frozen-fe055406-w1` during this audit returned exactly
nine files: four fold checkpoints, `manifest.json`, `side_cls.pt`,
`state_spec.json`, `state_stacker.json` and `fluid_stacker.json`. That bundle
contains neither original per-study target table, `pool_uids.json`, nor the
pixel training cache. Filename searches of the current repo (including ignored
audit artifacts) and `audit-09576` found no exact original target file or pool
manifest. This does not prove that the author has released none elsewhere; it
does prove that the present inference bundle alone is insufficient to run the
promised training reproduction.

## Validation limits

The 58 studies have been inspected repeatedly while deciding which model family
to try next. Excluding them from individual gradient updates is valuable, but
does not make them a fresh independent test after dozens of adaptive decisions.
Five seeds share the same studies and labels; they measure initialization
sensitivity, not five independent datasets. Bootstraps quantify uncertainty on
these studies, not protection against this accumulated selection.

The inherited Raptor source explicitly says its checkpoints were selected on
the organizer-gold studies. Our own evaluator's `gold_fit_count=0` is therefore
not evidence that the complete upstream stack never used them. Raptor remains
a useful scored baseline, but gold comparisons involving it require that
qualification. No new data-policy conclusion is inferred here.

Likewise, a high AUC against thresholded Steven/teacher pseudo-labels measures
agreement with that label recipe. It is not independent evidence that pathology
classification improved. A +0.01 gold AUC gain is not +0.01 public AUC.

## Decision and next expenditure

1. Preserve 57197157 and the negative results. Do not count the current caches,
   new commits, or proxy AUC as a new competitive model.
2. Withhold the proposed calibration promotion. Require held-out provenance
   across the base model and target teacher, and compare the exact deployed
   prediction formula before any new promotion experiment is run.
3. Perform the CPU asset audit first: locate the original R6 target/state files,
   4,344 pool IDs, preprocessing cache and per-fold initialization; reconcile
   missing studies and reconstruct the original four-fold exclusion map.
4. If those assets are available, the next GPU expenditure is one bounded,
   resumable, paired continuation of the same source fold: unchanged-source
   control versus one justified supervision change. Measure throughput before
   selecting a schedule. Persist model/optimizer/RNG state and intermediate
   predictions so a deadline does not erase the run's value.
5. If source supervision or a valid independent test cannot be established,
   the honest outcome is that R6 retraining is not currently a four-day build.
   Do not substitute another generic frozen head to keep the queue busy.

The earlier numerical probabilities for success and the statement that R6
retraining was the highest-probability route were subjective and unsupported by
this asset/validation audit. They should not be used for resource decisions.
There is currently no measured basis for a +0.01 promise.

## Source records

- `experiments/public-0956-reproduction/README.md`
- `experiments/public-09573-reproduction/README.md`
- `experiments/public-gp958d-reproduction/README.md`
- `experiments/convnext-oof-student-validation/RESULTS.md`
- `experiments/convnext-oof-full-population-validation/RESULTS.md`
- `experiments/convnext-oof-head-adaptation/RESULTS.md`
- `experiments/coatnet-oof-corrected-validation/output-v2/coatnet_oof_corrected_report.json`
- `experiments/multireader-nested-blend/optimize_nested_blend.py`
- `experiments/biomedclip-residual-screen/build_notebook.py`
- `experiments/r6-crossfit-fold-calibration/evaluate.py`
- `.audit/r6-w1/manifest.json` (SHA-256
  `6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648`)
- `.audit/rsna-knee-blend-code/r6src/compact_train.py`
- `.audit/rsna-knee-blend-code/r6src/compact_infer.py`
- `C:/Users/sarthak/Desktop/codex/audit-09576/0-9576-latest-public-dataset-score-in-one-click.ipynb`
