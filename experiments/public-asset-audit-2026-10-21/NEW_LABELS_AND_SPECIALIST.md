# New public releases: static provenance audit, 21 October 2026

Decision: neither release currently supplies independently validated advancement evidence. No labels were adopted, no checkpoint downloaded, and no model/source code executed. No GPU or competition submission occurred. Locked submission 57197157 remains 0.957.

## LLM ensemble weak labels

Source: https://www.kaggle.com/datasets/kiritohayate95/rsna-knee-weak-label-final (updated 2026-10-21 04:19:46 UTC).

Only `blend_report.json` downloaded; SHA-256 `434378cb66e233cb805aa0653a6ab6565fe1321976e56b18340a298224a8df15`. Original retained at `.audit/weak-final-static/blend_report.json`. Inventory also lists targets.csv, sources_aligned.csv and subsets.csv; those were not downloaded.

The report credits ten source streams from Steven, Yunus, Laymond, pilkwang, lixin73, the publisher's MedGemma/Mistral/Qwen runs, and a public rule parser. It explicitly evaluates seven blending procedures against 58 organizer-labelled studies. `chosen` is `mean_all`, whose procedure-level `uses_gold` is false and reported gold AUC is 0.902. This does NOT certify a gold-independent final label artifact: all twelve entries under `calibration` explicitly say `slope_from: gold`. Source prompts, candidate selection history, unknown-state treatment, calibration application and artifact construction are not authenticated by this small report. The CLI reports the dataset license as unknown.

Do not misrepresent this as the earlier 35-candidate/36%-coverage label release; it is a distinct release with a distinct calibration concern. Nor does this audit prove that every raw constituent label is unusable. Uncalibrated source labels would require their own lineage and terms audit plus an independently frozen evaluation contract. No such adoption or GPU launch is authorized by this finding.

## Raptor meniscus/fracture specialist

Source: https://www.kaggle.com/datasets/subsh2004/rsna-knee-raptor-specialist (updated 2026-10-21 03:02:09 UTC).

Downloaded only:

- `specialist_training_summary.json`, SHA-256 `35bad1bde3969e6e4012aa813f9f878f8ef8e8e5b64e542537da050ad5da727f`.
- `rsna-knee-specialist-trainer.log`, SHA-256 `4b8ca3e7f4dac27b6705f71404ad3ae399e4d568cb85bfe51ed9155ddca483e8`.

Originals retained in `.audit/specialist-static/`. Inventory advertises a 292,881,923-byte checkpoint; its bytes/hash were NOT inspected. Dataset metadata says CC0, which does not replace upstream model/data-use obligations.

Published log identifies initialization as `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` from `rsna-knee-publication-swa-weights-20261007`, the existing OAI-trained Raptor family. It reports 4,349 fit studies, 58 gold validation studies, `weak_labels_calibrated.csv` from `subsh2004/rsna-knee-calibrated-labels`, 384px inputs, eight training windows, 109 frozen early backbone tensors, and 294 trainable backbone plus eight head tensors. These are publisher log claims, not independently verified checkpoint reconstruction or training provenance.

The summary explicitly records `best_gold_auc` and `best_epoch: 3`. Gold macro AUC is 0.8758556893 before adaptation, then 0.9006311164, 0.9083220668 and 0.9084652691 across three epochs. Thus its claimed increase is training-proxy/selected-gold evidence, not a measured public-leaderboard gain or untouched pathology confirmation. Comparisons to our full four-reader baseline would additionally require identical inference/postprocessing and populations; these numbers alone do not establish superiority or inferiority to that stack.

This is genuine backbone fine-tuning rather than a static blend, but gold-selected checkpoint evidence and unauthenticated calibrated supervision do not justify importing or submitting it. No gold re-evaluation, target selection or specialist blend was performed here. A full source/fold/label audit and untouched evaluation would be needed before any new experimental contract.

## Scope

These are static inspections of public publisher artifacts, not clinical or runtime validation. Existing failed experiments remain closed; no closed gate was relaxed. Public catalog title/recency is not treated as score evidence.
