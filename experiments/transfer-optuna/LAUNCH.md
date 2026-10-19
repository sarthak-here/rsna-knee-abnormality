# Private launch — 20 October 2026 IST

Implementation and preflight pushed in commit `2c9d676`.

Kaggle accepted version 1:
<https://www.kaggle.com/code/sarthaksharma14/rsna-knee-video-transfer-optuna-private-screen>.
The CLI resolved the notebook title to this slug (including `private`), rather
than the initially requested metadata slug. The local builder/metadata is
corrected to the actual created identity; **do not push again merely to fix
that warning**, or create a duplicate run.

No trained model score is available at launch. Five CPU tests and both local
strict-weight synthetic GPU checks passed; no predictive trial was run locally.
The launched job first repeats the synthetic preflight on its actual GPU. If
the measured combined baseline cost exceeds two hours, it must stop without
training rather than extend the budget. Otherwise it executes up to eight
Optuna trials with one-epoch backbone adaptation, halfway median pruning,
report-group-held-out proxy BCE, and a two-hour active-search deadline.

Required evidence: exact source/weight/input identities; `contract.json` with
2,637 fit / 856 tune / 852 sealed confirmation / 58 excluded gold / four excluded
gold-report duplicates; actual GPU `preflight.json`; environment versions;
SQLite trial history; finite proxy losses; epoch/checkpoint identities;
`screen_report.json` or explicit fail-closed `budget_rejection.json`; complete
logs and no generated CSV/submission. No organizer-gold or confirmation scoring
is authorized inside the Optuna objective. A proxy winner is not a pathology
improvement or an advancement over locked public 0.957.

OrthoFoundation-MSK-5M remains excluded: new OAI-trained asset, model terms not
established, no exact candidate approval. The new VideoMAE release is used only
for private noncommercial research under CC-BY-NC-4.0, not cleared by this launch
for a competition/prize submission.
