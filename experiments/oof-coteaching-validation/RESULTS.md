# Result

The private nested OOF co-teaching experiment completed successfully and was rejected.

| Metric | Control | Co-taught student |
|---|---:|---:|
| Five-seed ensemble gold macro AUC | 0.860510 | 0.857406 |
| Delta |  | -0.003104 |

Only one of five seeds improved. Paired bootstrap support was negative: `p_improve=0.1512`, mean delta `-0.003157`, and 95% interval `[-0.009452, 0.002954]`.

The nested contract held. Every correction came from a group-held-out fixed-epoch image teacher trained only inside the corresponding non-gold outer-training partition. Gold studies never entered teacher fitting, OOF generation, correction, validation, stopping, or selection. No submission was created.

Decision: reject this conservative target-correction and reliability-weighting contract. The next branch must change the image head's capacity to represent plane and slice structure rather than continue modifying report-derived targets.
