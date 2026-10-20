# R6 own-report last-stage adaptation — new private hypothesis

User authorized building our own fine-tuning experiment on 20 October 2026.
This is NOT reproduction of R6's unavailable original supervision. It does not
reuse the rejected soft-v2/Steven/teacher tables or read organizer-gold reports.
No publisher contact or competition submission is authorized by this experiment.

## Frozen first pilot contract (before coverage or losses)

- Exact released fold-0 R6 checkpoint; all heads, priors, state/fluid stackers
  preserved. Only `enc.stages.3` (15,470,592 parameters) may change. Other source
  folds are not used. No fabricated pool or upstream warm-start assertion bypass.
- New `report_rules.py` literal multilingual rules supervise only explicit,
  unambiguous report **states**, not fabricated binary pathology probabilities.
  No mention/uncertainty/mixed-entity clauses/conflicting states are masked.
  Original fine-grained category indices are retained. OA and contusion remain
  unsupervised by the new rules. The rules are frozen before coverage inspection.
- Exclude all 58 gold studies and four non-gold exact-report duplicates, without
  extracting any gold report. Fit only outside original SHA1-prefix fold 0.
  Retain holdout groups only if every member hashes to fold 0: 977 source-image
  held-out studies before report-rule coverage filtering. Source pool-wide
  priors/postprocessor lineage prevents calling this whole-pipeline OOF.
- Select a deterministic bounded subset: 192 report groups for fitting, 64 for
  tuning and 64 sealed confirmation, each with at least one explicit state.
  Stable report-group hashes, no study overlap. Insufficient coverage stops the
  pilot, never silently broadens a rule or changes the split after results.
- Original v4 anatomy/side pipeline, 384 rendered / 352 crop, physical step
  3.5mm, released stride1. Pick 32 fixed evenly spaced evaluation windows per
  study for this cheap pilot; plain and original anatomical flip-TTA retained.
  This sampled-window pilot is NOT full deployed-reader inference parity.
- Cache only the frozen prefix (stem + stages0–2). Adapt the actual final
  convolutional stage and pass its maps through all untouched original heads.
  This is not another frozen-feature head. Verify cached split vs full network
  execution on the training GPU before training; no silent feature fallback.
- One fixed seed20261020, 3 epochs, AdamW LR1e-5 / WD0.01, one study per step,
  gradient clipping1.0. No optimizer, epoch, checkpoint, target or blend search.
  Heads remain in eval mode; only the final stage enters train mode.
- Loss = per-class balanced masked report-state CE +0.5 class KL +0.5 state KL
  +0.1 binary KL to the untouched **same fold-0** reader on identical windows.
  Distillation preserves existing behavior; it is not independent truth or a
  corrected-label teacher. New fitting never uses holdout reports/images.
- Hard wall budget3h including setup/cache/evaluation; measure 8 actual study
  renders and GPU final-stage steps first. Reject before optimization if the
  conservative projected budget does not fit. Persist every completed epoch,
  optimizer/RNG and source/input/checkpoint hashes. No CSV or submission output.

## Once-only advancement gates

Fixed final checkpoint vs untouched reader on the identical sampled windows.
At least4 state classes need at least5 observed tuning cells each. Macro state
NLL improvement >=0.01; at least4 nonnegative class gains; paired study-bootstrap
positive probability >=0.90 and q025>-0.01. Frozen postprocessed class probability
drift mean<=0.05 and max<=0.30. Only after all tuning gates pass, open sealed
confirmation exactly once and require the same gates. No gold evaluation in this
pilot. A pass permits a separately budgeted next experiment, not submission.

This evaluates agreement with our weak report rules on additional-gradient-held-
out images, not independently adjudicated pathology or a public-AUC improvement.
The existing 0.957 baseline and closed experiments remain untouched. Preserve
Heliosli/upstream attribution, source model terms, and OAI/NDA acknowledgement.
