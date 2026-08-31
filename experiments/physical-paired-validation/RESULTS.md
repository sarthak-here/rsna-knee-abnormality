# Result

The private five-seed paired experiment completed successfully and the canonical physical preprocessing arm was rejected.

| Metric | Incumbent | Canonical |
|---|---:|---:|
| Five-seed ensemble gold macro AUC | 0.860221 | 0.774359 |
| Delta |  | -0.085862 |

Every seed was negative, ranging from `-0.070456` to `-0.097890`. Paired bootstrap support was decisively negative: `p_improve=0.0`, mean delta `-0.085908`, and 95% interval `[-0.133417, -0.041033]`.

The leakage contract held: zero organizer-labelled studies entered fitting, validation, early stopping, or model selection; all 58 gold studies were evaluation-only. No competition submission was created.

Decision: do not pursue this combination of fixed physical-millimetre slice positions, robust MAD normalization, geometry-first laterality, and strict clinical sequence selection. The incumbent representation was substantially stronger on both weak validation and gold evaluation.
