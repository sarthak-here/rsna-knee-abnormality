# Research history

This log records completed, rejected, and in-progress experiments. A result is not described as an improvement unless it beats the locked public leaderboard baseline of `0.936`.

## Validated baseline

- Diverse DINO, RadImageNet, and CoAtNet swarm: `0.936` public score.
- Confidence-gated variant, submission `55894162`: `0.936`; tied and therefore not promoted as an improvement.

## E11 residual validation

- Public source: `antoinegg1/rsna-knee-e11-diverse-heads-v20`.
- Initial residual candidate, submission `55900501`: `0.936`; tied the baseline.
- Extended residual candidate, submission `55904749`: `0.935`; below the baseline.
- Decision: rejected. No further residual-weight or post-processing variants of the same model are permitted without materially new evidence.

## OrthoDiffusion exploration

- Investigated an orthopedic diffusion foundation model as a materially different specialist.
- The exploratory notebook and retained upstream attribution are preserved for reproducibility.
- Decision: not promoted and not submitted. Required specialist assets were not accepted as a verified public, reproducible improvement path.

## MedicalNet 3D validation

- Private Kaggle notebook: `sarthaksharma14/rsna-knee-medicalnet-3d-validation`, version 2.
- Backbone: MedicalNet/Med3D 3D ResNet-50 pretrained on 23 medical datasets.
- Training: 4,349 report-derived weak-label studies; all 58 organizer-labelled studies excluded from fitting.
- Five independently seeded heads were evaluated against the public RadImageNet E9 OOF screening control.
- Screening control macro AUC: `0.854324`.
- MedicalNet-only macro AUC: `0.643037`.
- Fixed 20% MedicalNet blend macro AUC: `0.847129`.
- Delta: `-0.007195`; positive seeds: `0/5`; bootstrap improvement probability: `0.051`; bootstrap 2.5th percentile: `-0.016108`.
- Decision: rejected. It failed every promotion gate and produced no competition submission.

## Repository policy

The repository remains private during the active competition. Publishing is deferred until the official competition deadline has passed and tracked files have been checked for secrets, generated artifacts, and attribution.
