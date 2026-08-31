# Report-image cross-supervision

This private experiment tests whether report semantics can reduce weak-label noise without requiring reports at inference. A frozen official RadImageNet ResNet-50 produces image tokens. The control and candidate use identical classifier paths; the candidate alone receives a fixed training-only cosine objective toward a TF-IDF/SVD report representation.

Every text transform is fitted on the 4,349 non-gold studies only. Report-duplicate groups remain isolated between weak training and validation. The 58 organizer-labelled studies never enter text fitting, image fitting, validation, stopping, or selection, and are used only for final image-only evaluation across five deterministic seeds.

The formulation follows the image-report cross-supervision principle established by REFERS and later medical vision-language work, using a deliberately lightweight text representation to keep the ablation attributable. Public RadImageNet and Steven v4 label provenance is preserved. The notebook cannot create a competition submission.
