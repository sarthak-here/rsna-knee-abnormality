# Public frontier audit — 17 October 2026

## Current leaderboard frontier

The complete public leaderboard snapshot downloaded at the 17 October heartbeat
has SHA-256
`e71640bb872087695d3c85dd1a76a32f592098b048873c68958fe45c928836be`.
Its displayed frontier has moved materially:

- rank 1 `CentralyDS`: `0.968`;
- ranks 2–4: `0.967`;
- locked submission `57197157`: `0.957`.

The leading teams do not currently expose an exact public inference contract
that can be reproduced from the newly listed kernels or datasets. The gap is
therefore real, but a leaderboard score alone does not provide a buildable
candidate.

## Newly inspected public releases

### `laimerbotrolenk/rsna-four-model-ensemble`

- Source notebook SHA-256:
  `50ded2186d86e172e85a8928e8b93b3cdd0c9112117d2267ac059e651793a003`.
- Adds an S6 reader to R7 and the Raptor 384/224 pair.
- The notebook itself explicitly says V33 has **no measured leaderboard score**.
- Laimer Botrolenk's team is currently displayed at `0.958`, but its last
  submission occurred roughly eleven hours after this kernel ran. That timing
  does not identify V33 as the scored contract.

Decision: retain as a possible fifth-arm source, but do not spend a submission
or claim `0.958` for V33 without an exact score receipt. Its S6 arm can be added
to a private nested screen only after the current four-reader cache is run.

### `ltcsmokeboi/rsna-knee-gp958e`

- Refreshed source SHA-256:
  `1848ff39cd0c03e8b763d03a207404116f97b6065df156b20809c853b1cf5058`.
- The team now displays `0.958`, but its last submission was approximately
  fifteen hours after the public GP958e run.
- GP958e still changes only the Raptor-224 blend endpoint relative to GP958d;
  no exact score receipt links the public source to the later team score.

Decision: no longer dismiss the family as permanently unscored, but it is not
stronger than the audited Heliosli `0.958` candidate and still lacks exact score
identity. Do not submit it as a new frontier candidate.

### `riadmohamed42/rsna-knee-dino-5fold-weights`

- Contains five 91.5 MB `dino_best_fold*.pt` files.
- No complete public training/preprocessing source accompanies the weights.
- The author's current public score is `0.949`.

Decision: reject as a replacement or diversity run. Five folds and a dataset
upload are not evidence of a competitive or reproducible reader.

### `dkim54/rsna-knee-models-v11i`

- CC BY-NC 4.0 release with ten ConvNeXt-small 384px checkpoints and public
  `dataset.py`, `model.py`, `infer_lib.py`, and `models.json`.
- Source hashes:
  - `models.json`: `2719714fef4aecdba7455bd5895069a524f9c3a91a3751c756a1688b3c4f1e3e`;
  - `model.py`: `94e4421a500a2742f5380bbd2d5d66ce748ffc90aab24d107fa91b33f3e6ed8a`;
  - `dataset.py`: `36e960434083605082c54b1996c54d51e36a6ccf6a72ecfac324aec1238637f1`;
  - `infer_lib.py`: `76dd275d1161025db5d4d6b67ae439587db3e7afc594388f856abe174ec4efac`.
- The public dataset code explicitly replaces weak labels with the 58 gold
  labels during training rather than preserving them as evaluation-only.
- The author's current public score is `0.947`.

Decision: reject. It is fully inspectable but materially below the locked score
and does not offer clean independent validation evidence.

### `haungjiawei/rsna-knee-train-v2-code-and-labels`

The release contains source and weak labels but no trained checkpoint. The
author's current score is `0.891`. It is useful as implementation reference,
not as a frontier candidate.

### `goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958`

- Latest public notebook SHA-256:
  `a906df9328b6a53d7e0fe4e18962c655867c8617c54e6745a8af647033b2399c`.
- The notebook says versions 25 and 26 scored `0.958`; the current version 27
  restores version 25’s recipe. It adds a three-fold 2.5D ConvNeXt-Tiny MIL
  reader at very small per-target weights on top of the already audited
  R6-flip + Raptor384/224 per-finding blend.
- The reader is materially distinct and its Apache-2.0 dataset publishes
  training, preprocessing, inference source, and three checkpoints. Source
  hashes are:
  - `infer.py`: `39d4517cc33c4bb1e4a03cfb49a0673a8789055de8c3cc0957c6e3035f6be4af`;
  - `knee.py`: `314c2af387a5cbd1d25f192f19f4ecc6d4ae4b30403e92e9d9b3ed2c325880b1`;
  - `preprocess.py`: `68601ad84ec3a0f030b5c2d37cc0d657cd3e12b17f7130e7385fba9648d324b6`;
  - `train.py`: `e41b26488ea5516c113f473c35f2391e0e97dc693b53d076f81b8953f169db2b`.
- Standalone public evidence is only `0.929`. More importantly, the public
  trainer evaluates all 58 organizer-labelled studies after every epoch.
  Although the saved checkpoint is the final EMA rather than an explicitly
  gold-selected checkpoint, this repeated monitoring means the 58-study set is
  not a clean independent promotion screen for this reader.
- The notebook’s Part B catches every exception and silently restores the base
  submission. The public output contains `_own.csv`, showing that the preview
  reader ran, but there is no fail-closed hidden receipt binding that leg to the
  scored version.

Decision: do not submit version 27 and do not add this reader to the predeclared
58-study nested screen. It offers no displayed gain over the simpler Heliosli
`0.958` candidate, and its gold-monitoring history invalidates that screen as
independent evidence. Retain the source only as architecture reference.

### `heliosli/this-is-best-public-dataset-new-version-0-9576`

- Refreshed notebook SHA-256:
  `cb0e1eac33a00f96b7d1bcc0e9914f7eb045af2bc9cd01bc290b2213e7f17ddd`.
- Preview execution is healthy: three official IDs, zero Raptor/R6 failures,
  zero R6 fallbacks, successful completion receipts, and final submission
  SHA-256
  `5e9ba665f3e30642e18dbb555eb7cbd16ab8a659ffb5f7226c72aaac44c9df1b`.
- It is not the already audited scored Heliosli contract. It changes the
  per-finding Raptor weights again (`0.85` Synovitis, `0.15` Lateral OA,
  `0.65` PF OA, `0.30` Baker’s) and adds the same tiny 2.5D reader leg.
- Heliosli’s leaderboard row still points to the 15 October scored submission;
  this refreshed kernel ran on 16 October. Its `0.9576` title therefore does
  not establish an exact score for these new weights.

Decision: reject as an unscored post-result reweight. The previously audited
Heliosli `0.958` version remains the only exact candidate in this family worth
an approval decision.

### `prvsiyan/the-bee-s-knees-final-rsna-push`

- Notebook SHA-256:
  `07a50c3e6e5af6616cd6cd41a647adb40d30141e33f80cf9822e1cf8f8c41d3e`.
- This is a carefully pinned canary for a global `55/45` R6/Raptor blend,
  preserving the 35% Raptor-224 share inside the Raptor slot. Its output
  receipt passes exact preview control, has zero reader failures, and records
  candidate submission SHA-256
  `ff4e63cb26dcc316dbb73eb3f8830cb9bfdaaab029223c5d8f53d454dcf7375c`.
- The receipt explicitly says `COMPLETE_BLEND55_NO_OFFICIAL_SCORE`,
  `submission_sent=false`, and `official_score=null`. It is a three-study
  runtime canary, not validation evidence.
- The author’s identifiable leaderboard row remains `0.957`; the public
  canary therefore supplies no evidence that the `55/45` reweight improves the
  locked submission.

Decision: reject as an unscored global endpoint tweak. Preserve its strong
receipt engineering as a reference, but do not spend a slot on the weight
change without non-public-leaderboard validation.

### Leading-team release check

The leaderboard snapshot’s top four teams display `0.968`, `0.967`, `0.967`,
and `0.967`. Public-kernel searches for their listed accounts (`centralyds`,
`tomirol`, `azamat1ch`, and `sggpls`) found no RSNA Knee source or checkpoint
release. Their scores establish the frontier but do not expose a reproducible
candidate.

## Result

No newly released exact public contract has credible scored evidence above
`0.958`. The two newest apparent `0.958` paths either add a gold-monitored weak
reader or change target weights after the last identifiable score. The best
active work remains:

1. run the committed four-reader nested screen when GPU quota resets;
2. submit the already audited Heliosli `0.958` contract only after explicit
   user approval;
3. consider the S6 reader as a fifth nested arm only if it can produce an exact,
   zero-fallback 58-study cache and improve the nested gate.

No competition submission was created by this audit.
