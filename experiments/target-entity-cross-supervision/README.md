# Target-specific report cross-supervision

This private follow-up replaces rejected whole-report alignment with predeclared finding-specific sentence alignment. Each finding's image attention output receives an auxiliary training-only target derived exclusively from report sentences mentioning that anatomy or pathology. Evaluation remains image-only.

The cached image features are the authenticated output of the prior private run, avoiding another DICOM decode. Every seed independently fits each target's TF-IDF/SVD representation only on its non-gold training report groups. Weak-validation and gold reports are not transformed; all 58 organizer-labelled studies remain final evaluation-only.

This fine-grained formulation is motivated by entity-level medical image-language methods such as MedKLIP. The experiment preserves public RadImageNet and Steven v4 attribution and cannot create a competition submission.
