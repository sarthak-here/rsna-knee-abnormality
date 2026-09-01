# Result

The private five-seed target-specific sentence-alignment experiment completed successfully and was rejected.

| Metric | Control | Entity-supervised |
|---|---:|---:|
| Five-seed ensemble gold macro AUC | 0.860510 | 0.860077 |
| Delta |  | -0.000433 |

Two of five seeds improved. Paired bootstrap support remained inconclusive and centred below zero: `p_improve=0.4142`, mean delta `-0.000479`, and 95% interval `[-0.005295, 0.004510]`.

Every seed fitted twelve independent target text spaces only on its non-gold training report groups. Weak-validation and gold reports were not transformed, no organizer label entered fitting or selection, and no competition submission was created.

Decision: reject target-specific TF-IDF/SVD reconstruction. Both global and entity-level text reconstruction are now closed research branches unless a materially different text encoder or learning objective is justified independently of the gold outcomes.
