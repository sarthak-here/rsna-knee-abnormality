# Operational follow-up

On 19 October, both R6 logit shard notebooks were still QUEUED. They have not
been cancelled. Any completed cache may be validated and retained as diagnostic
data, but it cannot establish independent full-pipeline OOF performance.

The previous instruction to run `r6-crossfit-fold-calibration/evaluate.py` as
an advancement test is withdrawn because of the documented design defects.
This is a pre-execution validity decision, not a scientific score rejection.
No replacement formula or retrospective gate search is authorized by it.

Further GPU retries or model launches are not justified until the original R6
training assets and fold lineage are authenticated and the resource budget is
measured. CPU research into those assets can continue. The locked public
submission remains 57197157 at 0.957. No score improvement is claimed.

## User-approved cancellation

Later on 19 October, the user explicitly approved cancelling both queued shard
jobs. Kaggle's Active Events UI showed both version-1 runs as `Cancelled` and
zero active events. Independent CLI checks returned
`KernelWorkerStatus.CANCEL_ACKNOWLEDGED` for both
`sarthaksharma14/rsna-knee-r6-full-logits-shard-0` and
`sarthaksharma14/rsna-knee-r6-full-logits-shard-1`.

The research automation was updated to stop polling, retrying or relaunching
this closed cache/calibration path. CPU asset/provenance research remains in
scope; this cancellation does not alter the locked scored baseline.
