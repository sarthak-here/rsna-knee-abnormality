# Public frontier audit — 17 October 2026

## Current leaderboard frontier

The refreshed complete public leaderboard archive downloaded at 07:31 UTC on
17 October has SHA-256
`6f0cc01cbdc0d0f3f826ad95026ca0251a9c97f741d74e5cbf7871202426fad1`.
Its displayed frontier has moved materially:

- rank 1 `RB & Matheus & Tucker & Tony`: `0.969`;
- ranks 2–3: `0.968`;
- rank 4: `0.967`;
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

### `heliosli/this-is-best-public-dataset-new-version-0-9586`

- Refreshed notebook SHA-256:
  `cb0e1eac33a00f96b7d1bcc0e9914f7eb045af2bc9cd01bc290b2213e7f17ddd`.
- Preview execution is healthy: three official IDs, zero Raptor/R6 failures,
  zero R6 fallbacks, successful completion receipts, and final submission
  SHA-256
  `5e9ba665f3e30642e18dbb555eb7cbd16ab8a659ffb5f7226c72aaac44c9df1b`.
- It is not the already audited scored Heliosli contract. It changes the
  per-finding Raptor weights again (`0.85` Synovitis, `0.15` Lateral OA,
  `0.65` PF OA, `0.30` Baker’s) and adds the same tiny 2.5D reader leg.
- The author renamed the slug/title from `0.9576` to `0.9586` without changing
  the notebook: the refreshed source SHA-256 remains exactly the same.
- Heliosli’s leaderboard row has its last submission at
  `2026-10-16 20:44:08`, while this kernel completed at
  `2026-10-17 04:21:38`. The leaderboard event therefore predates the kernel
  by more than seven hours and cannot be its score receipt. The new `0.9586`
  title is title-only evidence.

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

The refreshed top four teams display `0.969`, `0.968`, `0.968`, and `0.967`.
Their member accounts are now exactly identified from the full leaderboard:
`rashmibanthia`, `tomirol`, `tonylica`, `tuckerarrants`, `azamat1ch`,
`matthewliew`, `centralyds`, and `sggpls`. Fresh per-user kernel and dataset
queries found no newly released RSNA Knee source or checkpoint behind those
scores. Tony Li's only public competition kernels/assets remain the older DINO
RadImageNet releases, not the team's 17 October `0.969` contract. The scores
establish the frontier but do not expose a reproducible candidate.

### `dkim54/rsna-knee-blend-code`

- The 16 October upload is a source-only delivery bundle. Its archive SHA-256
  is `563d9663fccaca21357bd5ec771571d15663aea07c985d56ce275c37c6c381f8`;
  it contains no checkpoints or prediction files.
- The manifest pins Heliosli's S6 and R6 implementations. Verified full hashes
  include S6 `compact_infer.py`
  `34f433fe72b732543813165c3397ec80cef01b0d49b25b7b22668714f0b0ef4a`,
  R6 `compact_infer.py`
  `8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d`,
  and launcher
  `650028b7aaebfe01c8ce66a9fddc00516a64d6e6062e1b77cd29a8830b24d269`.
- The launcher adds fail-closed S6 anchor checks, worker reinitialization after
  setup failure, and optional pixel hashing. Those guards are useful for a
  future exact S6 cache because they prevent silent field-centre scoring when
  the SKM-TEA anchor bank fails to load.
- Kaggle labels the dataset license as `other`, and the bundle provides neither
  the required S6/R6 weights nor a score receipt. The dataset owner's public
  kernel list exposes no matching scored notebook.

Decision: retain the fail-closed launcher as implementation evidence for the
already planned S6 fifth-arm audit, but do not treat this source-only bundle as
a runnable or scored candidate. No GPU or submission slot is justified.

### `goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957`

- Latest public notebook SHA-256:
  `4cdb68479c362408f82621ae6ff0759c6e04951d04c2de4ffed10fb47014e797`.
- The public write-up records the relevant direct comparison: versions 4 and 5
  added S6 at weight `0.30` to the R6 recipe and scored `0.957`, while the
  corresponding recipe without S6 scored `0.958`. This is negative evidence
  for S6 beside R6, not support for launching a fifth-arm GPU screen.
- Despite the notebook slug, latest version 7 drops S6. It instead sets
  per-finding blend weights from all 58 organizer-labelled studies, explicitly
  describes the setting as aggressive and likely to overfit, and was still
  unscored when audited.
- Its optional 2.5D reader is wrapped in a broad exception handler that restores
  the public base stack on failure, so the output does not fail closed on the
  materially distinct arm.

Decision: reject version 7 and close the planned S6 fifth-arm path. The only
direct public A/B evidence shows a `-0.001` change, while the latest replacement
is organizer-gold-tuned, unscored, and permits silent fallback. No GPU run or
submission is justified.

### Refreshed `nartaa/rsna-knee-s6fall-weights`

- The 17 October dataset update publishes the exact S6 training source,
  production argv, frozen pool, soft targets, manifests, and hashes. The
  training README SHA-256 is
  `f86f136760ccef8afe2414e4e1693f4570fb29e707a5022a8dfaed95b7201b9c`;
  its full source checksum manifest is
  `0cb063bfc58d64b89bd65e6aa7da8c7946b27c8c3188227d651e4778aef0be6b`.
- This resolves the earlier source-availability question but does not improve
  the scientific evidence. The release confirms that the `4,263`-study run
  omitted 86 otherwise eligible competition studies only because their local
  DICOM copies had been deleted to save disk space; it was not a quality-filter
  decision.
- The selected epoch-16 S6 checkpoint remains the same public `0.950` model,
  and the direct R6-plus-S6 comparison above remains `0.001` worse than R6
  without S6.

Decision: retain the newly complete source as reproducibility evidence, but do
not reopen the rejected S6 GPU path. A source release cannot override the
negative measured blend evidence.

### `shardulsalodkar/rana-knee-abnormality-v2`

- Notebook SHA-256:
  `31f6d315b0c8b0ae7d32fcfff1362e3cdaea7e70d4ce18f0d38efc9386d3a141`.
- This is a CPU preprocessing shard that converts selected DICOM series to
  zipped native 3D volumes. It contains no model, checkpoint, validation
  metric, inference contract, or submission output.
- Its current run selects studies `400:1000`; failures are merely warned about
  and the ZIP is still created without failed series, so it is not yet a
  fail-closed immutable training corpus.

Decision: monitor RANA only if a model and independently validated checkpoint
are released. The present volume builder is infrastructure, not a candidate.

### `neronrx/mycelis-rsna-knee-4fold-hive`

- New CC0 bundle with four 200 MB ConvNeXt-small 384px checkpoints, per-fold
  logs and prediction CSVs, plus shared model/inference source. The shared
  `knee.py` SHA-256 is
  `dbcb103e916b1050d4bad11c91cdee7933fe66163d4374f03513fc3d0f23f248`.
- The architecture is materially distinct: six sequence slots, eight to twelve
  3-slice windows per slot, ConvNeXt-small tokens, a two-layer transformer and
  target-specific attention pooling.
- Each public fold file includes exactly 58 organizer-gold predictions in
  addition to 870 pseudo-label validation rows. Aligning those immutable gold
  predictions to the independently preserved official truth gives fold macro
  AUCs `0.584348`, `0.586904`, `0.588210`, and `0.580780`; averaging all four
  folds reaches only `0.601871`.
- The logs report strong pseudo-label validation AUC (`~0.91–0.93`) but confirm
  the same severe gold transfer failure. Training also initializes from an
  external `E_lrprior_s0_f0.pt`, while the bundle omits the complete trainer,
  target-generation provenance and that initialization checkpoint's identity.

Decision: reject before downloading the large checkpoints or spending GPU.
This is useful negative evidence that high pseudo-label validation does not
predict organizer-label transfer; it is not competitive diversity for the
locked `0.957` stack.

### `kritikabenjwal/rsna-knee-labels-soft-v2`

- New Apache-2.0 table release covering all 4,407 studies. The target-table
  SHA-256 is
  `a07f0b972e837cf3fb5a46ba83f9ccb934294112446f19af1e028be91bdcfe4f`;
  the state-table SHA-256 is
  `03a4e9a748885974f87e036aeba26ee128e65ad4d2653669ff5051a6f6129aba`.
- The 4,349 non-gold rows have complete `prior` probabilities in `[0.01,
  0.9556]`, and all 58 gold rows are blank in the target table. However, every
  `strict` target is blank, every fold is `0`, and every `review_holdout` flag
  is `0`; the release therefore contains no usable grouped split or independent
  validation contract.
- No public notebook currently identifies itself as the exact generator or
  consumer of this v2 dataset. The author's public Qwen lineage explicitly
  tests and iterates report-state prompts on all 58 organizer-gold reports
  (`rsna-knee-qwen-labeler-full` SHA-256
  `155262c9d6503b4efe85c4eef7c57dc7201fe5669403575a0d7830b2652e8897`),
  while the later MIL notebook repeatedly reports gold AUC during training
  (`rsna-knee-labelrefine-mil` SHA-256
  `ea4f74250bf31d7ec9a1b4cd21daa600f451982be722dd25301effe5a0ab388c`).

Decision: do not use the new targets for training or candidate selection. A
license and complete ID coverage do not establish how probabilities were
calibrated, whether v2 was selected using gold feedback, or how to reproduce a
leakage-safe fold. Reconsider only with exact generation source, frozen mapping
provenance and independent validation evidence.

### `shardulsalodkar/rana-knee-abnormality-v4`

- Notebook SHA-256:
  `a0b05bb641b9856ee7c4f92adc1032c1f93174f810aacd910df61ad6fb989fc6`.
- V4 proposes resizing each native 3D series to `96x96x96` and extracting
  pooled SwinUNETR encoder features from an `ssl_pretrained_weights.pth`
  checkpoint. Every executable cell is commented out; the public run creates
  no embeddings, model, validation result, receipt or submission.
- The external SSL checkpoint and three input datasets are blank in downloaded
  metadata, with no pinned model identity or training-data provenance.

Decision: retain the volumetric Swin idea as architecture research only. V4 is
an inactive sketch, not reproducible evidence and not ready for GPU screening.

### `kiyoshiohno/rsna-knee-template`

- Refreshed notebook SHA-256:
  `66fbf17addfacbaca201153700220102ce8dda636d6e2b4851a8ac5150a7c66e`.
- The notebook contains no inference implementation. Its only code searches
  Kaggle inputs for a manifest declaring `bundle_id=rsna-knee-local-logic` and
  dynamically executes `rsna_knee_pipeline.py` from that dataset.
- The required logic dataset appears as a blank/private source in downloaded
  metadata. No runner hash, model contract, score receipt or public source
  binds the current template to a candidate; changing the attached dataset can
  silently change all preprocessing, models and blending while the notebook
  hash stays fixed.

Decision: reject as an opaque launcher. Do not infer reproducibility or score
identity from the public template or its attached public model datasets.

## 22:55 UTC refresh

### Leading-team release check

Fresh per-user Kaggle kernel and dataset queries were repeated for every member
of the teams currently occupying public ranks 1--4: `rashmibanthia`, `tomirol`,
`tonylica`, `tuckerarrants`, `azamat1ch`, `matthewliew`, `centralyds`, and
`sggpls`. Seven accounts still expose no matching RSNA Knee artifact. Tony Li's
only matching notebook and dataset remain the September DINO/RadImageNet
release already known to the project. There is still no public source or model
identity for the `0.969`/`0.968`/`0.967` contracts.

### `lavinwins/top-1-gold`

- Source SHA-256:
  `8fd35f21d4280c04e30c49a0e9347cb16edcd004eaeef4454ce6c44ddfd41561`.
- The notebook proposes a competition-only 24-channel sagittal/coronal
  ConvNeXt-Tiny trained on newly mined report labels, but its checked-in
  configuration is `SMOKE_TEST=True`.
- The public version has no executed code cells, output, checkpoint, OOF
  receipt, leaderboard score, or immutable prediction artifact. Its own text
  reports only `0.6896` gold AUC for the earlier report-label miner and presents
  `0.80+` as an expectation rather than a result.

Decision: retain the multi-plane channel-stack idea as architecture reference,
but do not spend GPU on this exact unexecuted contract. Our stronger fixed OOF
targets and multiple independently screened readers already show that a novel
architecture without transfer evidence is not enough.

### Other newest public notebooks

- `masumhuq/rsna-knee-v7-fixed-fusion` (SHA-256
  `28cee6c9101ed85ce1a61689addb1b6182c9c5b6f1c39c92970ccf8d592ee587`)
  explicitly labels its weights "58-study tuned" and expects only
  `0.952--0.955`; reject.
- `masumhuq/rsna-knee-v8-58-study-weights` (SHA-256
  `0e9ee7b698c65c783e8ef082d8c0ba85e1de546dbcd3906b9eafcf6e976ebd46`)
  is another gold-tuned OAI/Raptor fusion with no executed output; reject.
- `yuvrajlamba01/rsna-knee-lupi-multi-plane-mil-sota-blend-0-957`
  (SHA-256
  `b1c5f83277ad00ebaf0189a84c443fe144701b8731664c5464293285343a4403`)
  is an unexecuted repackaging of the same R6/Raptor/OAI family and exposes no
  evidence above `0.957`; reject as a frontier candidate.
- The refreshed `laimerbotrolenk/rsna-four-model-ensemble` remains byte-identical
  to SHA-256
  `50ded2186d86e172e85a8928e8b93b3cdd0c9112117d2267ac059e651793a003`,
  unexecuted, and explicitly unscored; the earlier S6 rejection still applies.
- `toghrulaliyev/rsna-knee-aze-team-dd6d24` (SHA-256
  `9ba1feafec6ec133f2cac2c1dd0b15266c46f84aa5c5f1ffe8d00cb949ff1ae1`)
  is a 164-cell unexecuted exploratory notebook with no pinned checkpoint,
  validation receipt, or scored inference contract; reject.

This refresh produced no candidate that justifies a GPU or submission slot.

## Result

No newly released exact public contract has credible scored evidence above
`0.958`. The two newest apparent `0.958` paths either add a gold-monitored weak
reader or change target weights after the last identifiable score. The
four-reader nested screen has now completed and failed its uncertainty gates,
and the public S6 A/B result closes the planned fifth-arm extension. The best
active work remains:

1. submit the already audited Heliosli `0.958` contract only after explicit
   user approval;
2. audit or build a materially distinct reader with independent evidence rather
   than tuning this rejected family.

No competition submission was created by this audit.

## 18 October refresh

### `yeshawang/rsna-20261013-top6-velciraptor`

- Public notebook SHA-256:
  `ac05affa2fbe7b8a9f72120769abcbb14493511fc8c3d2bd7741cde02b19ad9c`.
- The author's team `66game` now displays public `0.960`; its latest
  submission was at `2026-10-18 07:11:59`, after the notebook's listed run at
  `04:23:11`.
- The downloadable notebook is nevertheless unexecuted: all cells have null
  execution counts, its output contains only the notebook-render log, and it
  emits no submission, model receipt, or checkpoint identity.
- Its final inference cell requires a private/inaccessible
  `rsna-resnet2027-single` dataset containing `inference.py` and
  `resnet50.0000.pt`. The newly public `yeshawang/model231` contains only a
  1.61 GB `latest_epoch.pth`, has license `unknown`, and provides no source or
  mapping to that private contract. `yeshawang/model412` is the already closed
  S6 family, not the missing ResNet reader.

Decision: the `0.960` leaderboard row is meaningful frontier evidence, but the
public release does not expose a reproducible contract. Do not infer that the
unexecuted notebook produced the score and do not reconstruct the private model
from an unrelated filename.

### `notfu4y/rsna-knee-deterministic-report-svmri-infer`

- Public notebook SHA-256:
  `530c4322fdda55e751effa616e9e2c5bb4a6d7b3c3594d1d1524d4020a1b8f57`.
- The public preview completed five DINOv2 folds over the three official test
  IDs in 0.3 minutes with a valid `3 x 13` submission, no decode errors and
  pinned pipeline signature
  `6ce65ee374911b43993b30b8ba7531c9e6f6eb96d22c51a08e28113ec6e790d5`.
- The required checkpoint directory is a private/inaccessible
  `notfu4y/rsna-artifacts` dataset. Kaggle's public output includes only the
  three-study features, image cache and submission receipt, not the five
  checkpoints or manifest.
- The author's displayed public score remains `0.956`, dated three days before
  this notebook run, so it is neither evidence above the locked `0.957` nor an
  exact score receipt for the release.

Decision: retain its fail-closed preprocessing and attention-head source as an
implementation reference, but do not submit or claim the inaccessible
checkpoint ensemble. The next private screen must change the representation
materially and train only on the project's frozen non-gold target contract.

### `shardulsalodkar/rana-knee-abnormality-v4`

V4 remains an unexecuted 96-cube SwinUNETR embedding sketch. Its public output
contains only a render log, and it still exposes no model, validation result or
submission contract. The prior rejection remains unchanged.

## Reusable artifact audit

`audit_kernel_artifacts.py` now makes future CPU-side public-kernel checks
repeatable. It binds the downloaded notebook, metadata, preview submission,
receipts, and logs by SHA-256; verifies the official 13-column schema, exact
three preview IDs and order, unique IDs, finite `[0,1]` probabilities, public
offline competition metadata, completion markers, and absence of fatal log
markers. It explicitly records that a passing artifact audit does **not**
verify a leaderboard score. The tool executes no notebook code and creates no
competition submission.
