# Data-first reconstruction and fitting-only EDA — 22 October2026

This implements the user's request to examine the successful recipe's datasets, cleaning, EDA and supervision rather than iterate another small model tweak. It does not claim model progress or an improved score. Aggregate results are reproducible with `python experiments/data-supervision-eda-2026-10-22/audit_data.py`. No raw report, study IDs, per-study targets or predictions are published.

## What the three attached datasets actually supply

| Attachment in exact scored0.958 notebook | Function | Not evidence of |
|---|---|---|
| heliosli/rsna-knee-r6-frozen-fe055406-w1 | Four trained R6 folds, manifest, side classifier, state specification and stackers | New training examples or original per-study rich targets |
| nartaa/rsna-knee-publication-swa-weights-20261007 | Trained Raptor224/384 checkpoints | A directly accessible new external-image cohort |
| heliosli/rsna-orthofoundation-fold0-epoch7-20261004 | Retained support-code/dependency/localization assets used by existing inference | Permission to obtain/use new OrthoFoundation or OAI data |

The public source says the Raptor checkpoints were retrained using blended weak labels and OAI examples. The linked author write-up reports4349competition studies plus2399OAI knees, unknown finding cells masked rather than treated as negative,16epochs, backbone/head learning rates separated, study-gradient accumulation, native crop geometry, and selected epoch averaging. Those are author-reported recipe results, not controlled evidence each ingredient individually caused the score. It explicitly says the58gold studies were reused for selection. Source: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792 . No new external data downloaded here.

The exact R6 manifest independently specifies4344studies,4folds,16epochs,176training windows,384render/352crop,physical3.5mm steps,100mm anatomy-centred local views and the positive-LR/cell-weight/state supervision route. The trainer excludes blanked non-knee/otherwise invalid report targets, absent rendered studies, review holdouts and fold-3 exclusions according to its original metadata. It does NOT treat a raw probability table as an interchangeable LR target table. The original final per-study targets/pool remain unauthenticated locally; do not claim an exact training reproduction.

## New measured fitting-only findings

To preserve evaluation scope, the script excludes every report group containing a gold study or ANY source-fold0 member BEFORE calling the frozen parser. It parses3199fitting report groups covering3228studies. Guard test injects a spy parser and proves gold, gold duplicates and fold0 groups are never parsed. It does not score any model or generate new targets.

- **1841/3199 reports (57.55%) lack a heading recognized by our literal parser.** That parser returns no labels for those reports. This measures failure of our heading vocabulary, NOT that the reports are invalid or contain no findings.
- Only**2875/38388finding cells (7.49%)** receive known states from the current conservative rules. Unknown cells stay unknown, not negative.
- Medial/Lateral/PF OA and Contusion have**zero**direct supervision by construction: those entities are absent from the parser, not absent disease in the dataset.
- Fracture:30known report groups, of which27none,2old/healed and just**1acute**. MCL:342known,338normal and4complete tears. Scaling this same parser to the full fitting cohort does not create broad positive/severity supervision.
- Multi-entity clauses are discarded wholesale. Among entity-clause counts, this removes191ACL,192MCL and132Synovitis clauses (not independent studies or known-correct labels). Sentence/entity-local evidence handling is a concrete development target.
- Every fitting study has metadata in sagittal/coronal/axial planes, but202/3228lack fluid-sensitive sagittal metadata. Those are protocol flags, not proven unusable pixels. They argue for missing-protocol handling, not silently dropping202studies.
- Multiple studies share exact normalized reports. The fitting cohort3228studies/3199groups means study-level random splits would not guarantee report-group separation. This does not establish duplicated image pixels or patient identity.

## Specific successor engineering target

Rebuild the supervision dataset FIRST, not another encoder title search: multilingual section detection; finding-local evidence spans; separate current findings from clinical question/history/postoperative state; preserve negated, uncertain and absent evidence separately; coarse/partial labels where detailed R6 categories cannot be justified; add explicit OA compartment and bone-injury coverage rather than inventing fine-grained grades. Deduplicate/report-group before folds; retain protocol missingness and confidence masks. No automatic imputation of negatives or pseudo-grades.

Before any new fitting, require an independently reviewed report-extraction fixture covering all12findings, multilingual/negation/history/uncertainty/multiple-entity cases, a provenance-pinned per-cell evidence/confidence ledger, adequate positive support and a separately frozen untouched evaluation scope. These are weak-report quality checks, NOT independent image-pathology truth. New parser performance has not been measured here. No new training launched and no closed pilot's gates, labels or inspected confirmation sets were changed.

The already-approved exploratory adapter inference remains a separate active submission attempt; do not interrupt or duplicate it because of this EDA. Preserve selected0.958and0.957final entries and all attribution/OAI-NDA/model terms. Broad research direction does not authorize new gated external data access.
