# R6 own-report last-stage adaptation — new private hypothesis

User authorized building our own fine-tuning experiment on 20 October 2026.
This is NOT reproduction of R6's unavailable original supervision. It does not
reuse the rejected soft-v2/Steven/teacher tables or read organizer-gold reports.
No publisher contact or competition submission is authorized by this experiment.

Runtime status: v1 failed during AMP backward after caching all320 studies;
see `V1_FAILURE.md` and `v1_terminal_receipt.json`. The numerical-only v2 repair
uses initial AMP scale1024 instead of default65536 and checks scaled complete
objective gradients at preflight. Scientific contract and three-hour cap remain
fixed. No model improvement or evaluation result was obtained from v1.

V2 subsequently failed on its fifth training study after four completed updates,
despite passing preflight; no epoch or evaluation result. See `V2_FAILURE.md`
and `v2_terminal_receipt.json`. The v3 runtime-only repair backs off AMP scaling
and replays the same study/RNG without skipping any optimizer update, failing
closed at scale1. All scientific parameters and hard3h cap remain unchanged.

Prelaunch v1.1 correction: synthetic counterexamples showed that “no tear”
cannot establish a fine-grained normal state, “no large effusion” cannot mean
none, and a negator after an entity may refer to another finding. These cells
are now masked or scope-corrected. This semantic fix was made before any
training, validation loss, prediction or gold inspection, not tuned to coverage
or a model score. The corrected rules are re-frozen before split regeneration.

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
  Prefix storage is float16; test the actual rounded cache, with maximum
  absolute logit error<=0.02 on valid source state slots. Invalid padded state
  slots remain source-defined -infinity, not treated as runtime failures.
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
