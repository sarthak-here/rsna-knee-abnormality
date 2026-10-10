# Curia-2 V64 GPU private validation

## Outcome

The source-faithful Curia-2 V64 branch passed its predeclared nested OOF diversity gate against the embedded V52 RadImageNet control. This is private validation evidence only; no competition submission was produced or authorized by this experiment.

| Metric | Result |
|---|---:|
| V52 RadImageNet gold OOF macro AUC | 0.841993 |
| Curia-2 standalone gold OOF macro AUC | 0.811234 |
| Nested 80/20 blend macro AUC | 0.852098 |
| Nested gain over V52 | +0.010105 |
| Final descriptive 80/20 blend macro AUC | 0.852206 |
| Final descriptive gain | +0.010213 |
| Positive-alpha outer folds | 5/5 |
| Selected alpha | 0.20 |
| `selection_supported` | true |

Outer-fold alpha choices were `[0.20, 0.25, 0.25, 0.25, 0.25]`. The 58 organizer-labelled studies were used only for the frozen nested OOF comparison, not encoder/head fitting or checkpoint selection.

## Execution and integrity

Private Kaggle notebook: `sarthaksharma14/rsna-knee-curia-2-gpu-private-validation`, version 5.

The notebook completed Curia-2 feature extraction and all five report-group folds:

- weak-label OOF macro AUC: 0.81036
- organizer-gold OOF macro AUC: 0.81123
- OOF rows: 4,407 unique studies
- fold counts: 882 / 882 / 881 / 881 / 881
- organizer-gold rows: 58
- submission CSVs remaining: 0 (cleanup removed 10 generated artifacts)

Pinned identities:

- Curia-2 model: `403a02e27531d2858ecd1e9b1ec2d5ea7bfa909f10ff0d9e8416090a6a8c96ef`
- encoder config: `9b22dcc40c81d8e3d536978cb81db049e9940ea6b02b3bfc815e1e781d57f252`
- preprocessor config: `469b3bed97af2b8923002a11a6fe4500f34bcae39a49c5822ae15cefa15702e5`
- image processor source: `74a4eb5e27539686965327c4e8293648a552777452a51eb1ccac956067cc236b`
- official Curia Research-Only RAIL-M license: `db9f5568c810d75d934ce5a6d092fb911caf18fb4df757fa823f0c6681fae143`
- five-fold head artifact: `cf33d120efd810cd0cf3ee1021e2929e7ef7423f30a71f529d1ce4d28d47d8ec`
- OOF CSV: `8c6e47638e85dddc64003156bf931477a62cba6b9f3c61dafa785fa592f783f9`

## Audit recovery

The source notebook's final audit cell stopped after training because two legacy public-family audit files were no longer mounted: `exact_public_family_oof.npz` and `expert_oof_all.csv`. Neither file contributes model predictions to the V64 gate. The gate was independently reproduced from the immutable V64 OOF output, the source-embedded V52 expert UIDs/predictions, the official `train.csv` gold labels, and the exact published fold/alpha logic. The reproduction returned the metrics above and satisfied every predeclared numerical gate.

The next allowed step is an inference-capable private reproduction using these exact five heads and alpha 0.20, followed by strict hidden-runtime, schema, test-ID, provenance, and license validation. It is not yet a leaderboard submission.


## Version 6 full-path confirmation

Private notebook version 6 completed with `status=CANDIDATE_SELECTED` and independently reproduced the same OOF gate:

- `selection_supported=true`
- selected alpha: 0.20
- nested macro AUC: 0.852098
- final descriptive macro AUC: 0.852206
- five trained heads and five positive-alpha outer folds
- preview test: 3/3 official IDs, 64 available slice tokens, five-head inference
- selected preview artifact SHA-256: `6c16b159361dd203cabe5f6348fbd0f1dfe952f0fb2837b65182ad5e38cc52fa`
- fallback V52 preview SHA-256: `eb4b5d6f7aaa874edcc9f418672267436cab26e2df78b71d4d6b3940c8162e90`
- cleanup removed 17 submission CSV artifacts; none remained in notebook output

This confirms the complete preprocessing, five-fold training, Curia-2 test encoding, five-head inference, and 80/20 rank-blend path on the public preview. A separate private reproduction must retain and strictly validate its final `submission.csv` before any Kaggle submission.
