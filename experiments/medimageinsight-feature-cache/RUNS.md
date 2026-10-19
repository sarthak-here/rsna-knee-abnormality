# Run log

## Version 1 — failed before model construction

Kaggle stopped at the fail-closed checkpoint identity gate. The original pin
came from the identically sized Hugging Face LFS artifact
(`0453e28e...ea22`), but Kaggle model version
`thelionai/medimageinsight/PyTorch/default/1` contains different bytes.

The exact Kaggle model version was downloaded independently. Its 2,464,060,700
byte checkpoint hashes to
`5eeda63bf616a61664bc95b2c09d3b3d7125209e635678bd3f5f324e9bdb1414`;
the config and wrapper retain their original pinned hashes. Version 2 changes
only this source-specific checkpoint identity. No representation, image,
window, preprocessing, output, or validation contract changed.

## Version 2 — complete

Version 2 completed the frozen extraction in 7,979.4 seconds. Strict local
validation with `allow_pickle=False` passed: 4,407 unique Unicode IDs,
`[4407,6,4,1024]` float16 features, `[4407,6]` masks, exactly 23,089 occupied
slots, finite values, occupied-vector norms from 0.999839 to 1.000165, zero
missing-slot features, clean logs, and no CSV or submission artifact.

Output SHA-256:
`2f8344d247185a7faca711057ef17ce4cd15df540dd3181501c2dfc85e18922a`.
