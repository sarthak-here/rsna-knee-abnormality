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

## Result

No newly released exact public contract has credible scored evidence above
`0.958`. The best active work remains:

1. run the committed four-reader nested screen when GPU quota resets;
2. submit the already audited Heliosli `0.958` contract only after explicit
   user approval;
3. consider the S6 reader as a fifth nested arm only if it can produce an exact,
   zero-fallback 58-study cache and improve the nested gate.

No competition submission was created by this audit.
