# Submission57520075 — terminal hidden execution failure, no score

The authenticated Kaggle UI shows **Notebook Threw Exception** for exact version1/script361088023. The API returns `status=COMPLETE` with `errorDescription` and no public/private scores. COMPLETE alone is not success: future monitoring must inspect errorDescription as well as scores/status.

The only exposed error in the submission details panel and full API submission object is:

> Your notebook hit an unhandled error while rerunning your code. Note that the hidden dataset can be larger/smaller/different than the public dataset

No detailed traceback, failed line, hidden worker receipt or hidden inference output was exposed in these inspected surfaces. The public notebook's logs are the successful3-study preview, not hidden-test logs. The CLI competitions logs command is for simulation episodes, not this medical code competition. Do not present those preview logs as a hidden failure trace.

## What is and is not established

Preview integrity passed, but this did not establish hidden-set runtime/robustness. The stricter zero-reader-fallback condition, the original8.3h deadline, full input-tree checkpoint search, compilation, mount behavior and hidden data differences are candidate failure surfaces, **not proven causes**. The elapsed wall time alone does not isolate queueing from execution. Do not infer an accuracy regression, numeric instability, CUDA/OOM or specific timeout from this generic error.

The bounded checkpoint-discovery repair remains only a tested local build. It is not proof it would fix this hidden failure. No new version, GPU job or repeat submission has been launched. The permitted research window has ended and there is no justified reproducible hidden repair yet. No blind resubmission or relaxation of scientific/integrity gates.

## Final protection

At diagnosis, the live authenticated **Selected** filter showed**2/2**, containing the exact0.958Helios submission57483292and original0.957submission57197157by their unique descriptions. No final-selection checkbox was changed. The failed adapter has no score and is not promoted. The earlier general All-filter view transiently displayed0/2 during loading; the loaded Selected-filter result is the authoritative verification.

Preserve all preview/training/source artifacts, attribution/model terms/OAI-NDA acknowledgement and unrelated changes. Our own training has not demonstrated a competition-score gain; best verified public result remains the attributed0.958 reproduction. Stop polling this terminal submission. Any future repair needs an exact reproducible failure or additional detailed error evidence and appropriate execution authority/time; do not conceal missing cause information.
