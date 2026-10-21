# Build and launch boundary

The user explicitly requested "so mf build it" on22October2026IST. New private rank8 adaptation pilot, not a replay of the closed full-stage experiment. The contract in README was written before new split construction or any new model evaluation. No additional data, paid API, publisher contact, gold evaluation, public leaderboard tuning or competition submission is included.

## Built artifacts

- Exact notebook SHA256: `6c94ff834fcf66ef7ac7d495ce575dd7693e27c21fb36dbecebfb9a3131ba19b`.
- New split SHA256: `55322d1981812d21b40ad800ecca15eedf231a6e7b119793ebb82a6611a4d9fd`.
- Generated trainer SHA256: `20d811e8fc40e7eeb0328a15f87398fc7b793c62ef227e0d61ae62f5d1286420`.
- Rules unchanged SHA256: `e10ed7ed45841ab058ffe6926e8bddf01e46a67da2fcf149c6dae0f49569b736`.
- New192/64/64report groups; ALL320old selected groups excluded; all58gold reports excluded before extraction. Conditional source-fold0 image-gradient holdout only, not independent pathology truth. Fit has only4known fracture states and no new directOA/contusion supervision, so no claim of reliable all-target pathology learning follows.
- All12CPUtests passed. New synthetic GPU preflight:6adapter gradient tensors finite atAMPscale1024,3nonzero up-projections initially (down-projection gradients are initially zero by zero-init design), initial full-head outputs bitwise equal, every original tensor preserved,92,160parameters trainable. Local forward/backward6.066seconds, adapter backward0.619seconds, peak CUDA676,910,080bytes on RTX3060Laptop; no local study optimizer updates. This is not Kaggle throughput.

The builder generates a private notebook and split locally. They are ignored by git to avoid publishing per-study report-state targets/evaluation groups; identities and source code are versioned. Rebuild with `python experiments/r6-lowrank-spatial-adaptation/build_notebook.py` using existing pinned local assets; inspect `build_receipt.json`. Do not alter the frozen representation/split/rules/objective/LR/epochs/rank/gates after observing new results.

## Launch

**BUILT, NOT LAUNCHED.** Initial `kaggle kernels push` attempt was rejected by the approval checker before process creation: the user requested building the candidate but had not explicitly authorized this exact new private Kaggle upload/run. No version1 acceptance or job status exists. Do not retry, use UI/another route, duplicate, or claim RUNNING/QUEUED until explicit user approval of `sarthaksharma14/rsna-knee-r6-lowrank-spatial-adaptation` and its hard3h private GPU run. Payload includes project/published model code, generated anonymous study IDs and report-rule state targets, and references to the same existing competition/model datasets. No raw reports, credentials, new external data or competition submission are intended. Ask once for this scoped upload/run permission; do not repeat unchanged notifications.

Implementation and tests are pushed in `f0413b8`. Final selected57483292/57197157 remain protected. The previous own-report pilot's upload approval is not reused for this materially new job.

Target private notebook: `sarthaksharma14/rsna-knee-r6-lowrank-spatial-adaptation`, version1 if accepted. No launch is claimed until Kaggle accepts it and its status is verified. Hard3h cap includes setup/cache/evaluation;8actual-study measured preflight must fit conservative projection before any update. No silent increase. Preserve failure/update-count/AMP-retry receipts and exact-version outputs, including optimizer/RNG checkpoints. Close on a budget rejection or scientific gate failure. No automatic submission even after a proxy pass.
