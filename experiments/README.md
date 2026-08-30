# Gold-set inference search

The private Kaggle experiment evaluates inference settings against the 58 radiologist-labeled studies while leaving the production notebook untouched.

## Tested variants

- 24, 42, and 56 slice windows with the standard checkpoint
- 42 windows with the SWA checkpoint
- Horizontal-flip inference for both checkpoints
- Optuna-selected two-candidate rank blends

## Results

| Configuration | Macro AUC |
| --- | ---: |
| Existing 42-window baseline | 0.91269 |
| Standard checkpoint, 24 windows | 0.91662 |
| Standard checkpoint, 56 windows | 0.91683 |
| Fixed 50/50 blend of 24 and 56 windows | 0.91718 |
| Nested Optuna out-of-fold blend | 0.91639 |

The fixed blend improved macro AUC by 0.00449 and beat the baseline in 91.7% of 1,500 paired bootstrap samples. Flip inference reduced accuracy, and the SWA checkpoint did not improve on the selected standard-checkpoint views.

The experiment source is in `optuna-gold-search/`. Generated predictions and model weights are intentionally excluded from Git.

## Expanded window search

`optuna-window-search/` caches backbone features and evaluates window count, left and right trimming, sampling density, and phase. It completed 1,250 Optuna trials covering 1,194 distinct configurations.

- Existing feature-cache baseline: 0.91269
- Best all-gold configuration: 0.92006
- Nested out-of-fold result: 0.91216
- Out-of-fold delta: -0.00052

The apparent all-gold improvement did not generalize, so it was rejected and never promoted to production.

## Diverse swarm validation

`swarm-0936-validation/` validates an attributed public ensemble combining DINO, RadImageNet, MaxSpan, WideDense, and dual CoAtNet predictions. The visible test completed every branch, verified checkpoint fingerprints, and produced a valid submission in 163 seconds. This branch was submitted separately because its public reference score is 0.936, materially above the 0.924 single-checkpoint baseline.

## Confidence-gated ensemble validation

`confidence-gate-validation/` evaluates the accessible DINOsaur V5.2 tri-expert ensemble while preserving the original public notebook attribution. The Kaggle run completed successfully with 20 DINO members, five DINOv3 folds, V18 calibration, MaxSpan v5, and WideDense v4/CoAtNet branches. Its generated `submission.csv` passed strict validation against the official test IDs, row order, 13-column schema, finite-value checks, and probability bounds.

The visible three-row test set is too small for a meaningful confidence gate, so the notebook safely defers the gate there and activates it only during hidden scoring. The exact completed notebook version was submitted as Kaggle submission `55894162`; its public score is pending.

An OrthoDiffusion residual candidate was also investigated but rejected because its required fine-tuned specialist checkpoints are private and unavailable. No unverifiable result from that branch was submitted.
