# Result

The private five-seed global report-latent experiment completed successfully and was rejected.

| Metric | Control | Report-supervised |
|---|---:|---:|
| Five-seed ensemble gold macro AUC | 0.861690 | 0.860748 |
| Delta |  | -0.000942 |

Only one of five seeds improved. Paired bootstrap support was inconclusive and centred slightly below zero: `p_improve=0.3688`, mean delta `-0.000971`, and 95% interval `[-0.006735, 0.004786]`.

The leakage contract held. Each seed fitted its TF-IDF vocabulary and SVD only on its non-gold training report groups; no weak-validation or gold report was transformed. No organizer label entered fitting, stopping, or selection, and no competition submission was created.

Decision: reject coarse whole-report latent reconstruction. Its near-zero effect suggests that global report semantics are too diffuse for twelve anatomically distinct findings. Any follow-up must use predeclared target-specific entity/sentence alignment rather than tuning this loss weight after seeing gold results.
