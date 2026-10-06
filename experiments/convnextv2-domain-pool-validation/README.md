# ConvNeXtV2 domain-pool validation

Private screen of the public Apache-2.0 `yaolinghan/rsna-knee-pool-models`
ConvNeXtV2-Tiny checkpoint. The experiment loads the exact pinned checkpoint
with `weights_only=True`, removes its published linear pooling head, freezes the
RSNA-trained image backbone, and trains the attributed study-level
slot/depth-Transformer reader contract on Steven v4 weak labels.

All 58 organizer-labelled studies are excluded from fitting, splitting,
stopping, and model selection. They are used only for the frozen three-seed
comparison. The notebook cannot create a competition submission.

The screen requires an ensemble gold AUC of at least `0.865457` and at least
two of three seeds above the `0.860457` private control before any scaling work.

The direct Kaggle Model attachment failed during worker provisioning before
Python emitted logs. For execution, the exact pinned checkpoint is mirrored in
the private dataset `sarthaksharma14/rsna-knee-convnextv2-domain-checkpoint`;
`checkpoint-dataset/SOURCE.md` records its original author, Apache-2.0 license,
source model reference, and SHA-256. The binary is intentionally git-ignored.
