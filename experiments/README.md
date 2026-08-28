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
