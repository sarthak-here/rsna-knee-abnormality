# Technology stack

## MRI and data

Python, NumPy, pandas, pydicom, DICOM physical geometry, native decoder/quantizer integration, PyTorch transforms and anatomy-centred rendering. Experiment-specific HDF5 and NumPy caches hold pixels/features locally. Real CPU fixtures validate limited protocols, not every compression syntax or clinical end-to-end parity.

## Learning and models

PyTorch, CUDA, timm and family-specific libraries in each notebook. Explored families include CoAtNet, ConvNeXt/ConvNeXtV2, DINO/DINOv2, RadImageNet/ResNet, EfficientNet, Swin3D, VideoMAE, medical vision encoders and volumetric MRI readers. An archived proposal is not proof of completed training or competition eligibility.

R6 adaptation preserves original heads/priors/postprocessors. Full-stage training unfreezes the final encoder stage; rank-8 training instead adds zero-initialized updates inside three fc2 projections, with six matrices trainable. Frozen prefixes are cached in float16. Fixed pilots use AdamW, masked balanced state cross-entropy, reference KL terms, clipping and AMP. Numerical retries replay the same study and CPU/CUDA RNG without skipping an optimizer update.

## Evaluation

scikit-learn metrics/estimators in individual experiments, paired study bootstrap, report-group guards, source-fold lineage audits, report-state NLL/BCE proxies and bounded Optuna studies with SQLite persistence/pruning. Organizer-set AUC, report-proxy loss and competition scores are distinct measurements.

## Infrastructure

Private Kaggle GPU notebooks, Kaggle API/CLI, Git/GitHub, PowerShell, Python tests, notebook builders, SHA-256/MD5 receipts and finite tensor/probability checks. Versions are recorded where experiment receipts pin them; there is no single environment shared by all historical notebooks. Nonfatal dependency warnings occurred.

The final inference runtime repair replaces a recursive global input-tree scan with bounded checkpoint mount discovery. Model/blend settings are unchanged. The original hidden exception cause remains unknown.
