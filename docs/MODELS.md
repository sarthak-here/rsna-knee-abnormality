# Model and experiment catalogue

This index covers all 83 experiment directories tracked at release preparation. It includes original training, attributed reproductions, feature caches, audits, proposals and rejected experiments—not 83 successfully trained models. Names containing a score or “OOF” describe historical experiment titles, not an independently verified result; read the terminal results and lineage limitations.

## Key original training

- [R6 rank-8 spatial adapters](../experiments/r6-lowrank-spatial-adaptation/RESULTS.md): six new matrices;576 updates; failed the frozen proxy improvement threshold. Final user-directed inference tied public0.958 and scored private0.942.
- [R6 full final-stage adaptation](../experiments/r6-own-report-adaptation/RESULTS.md):31 original tensors trained;576 updates; improved report-state loss but failed confirmation probability-drift cap.
- [Swin3D / VideoMAE transfer](../experiments/transfer-optuna/RESULTS.md): three completed trials, one pruned, four skipped before training. Swin trial3 selected on tuning proxy only; no sealed confirmation/pathology gain.
- [Research audit](../experiments/research-audit-2026-10-19/README.md): consolidated outcomes from earlier image-head, end-to-end, SSL, localization and residual experiments.

## Complete directory index

| Experiment | Contract / documentation | Terminal result record |
| --- | --- | --- |
| `biomedclip-feature-cache` | [BiomedCLIP physical feature cache](../experiments/biomedclip-feature-cache/README.md) | [RESULTS.md](../experiments/biomedclip-feature-cache/RESULTS.md) |
| `biomedclip-residual-screen` | [BiomedCLIP target-attention residual screen](../experiments/biomedclip-residual-screen/README.md) | [RESULTS.md](../experiments/biomedclip-residual-screen/RESULTS.md) |
| `coatnet-oof-corrected-validation` | [OOF-corrected CoAtNet-384 private screen](../experiments/coatnet-oof-corrected-validation/README.md) | See directory records; no uniform RESULTS.md |
| `coatnet2-public-gold-reproduction` | [Public CoAtNet-2 gold reproduction](../experiments/coatnet2-public-gold-reproduction/README.md) | See directory records; no uniform RESULTS.md |
| `confidence-gate-validation` | [confidence-gate-validation](../experiments/confidence-gate-validation/) | See directory records; no uniform RESULTS.md |
| `convnext-mil-public-reproduction` | [Public ConvNeXt-MIL reproduction](../experiments/convnext-mil-public-reproduction/README.md) | [RESULTS.md](../experiments/convnext-mil-public-reproduction/RESULTS.md) |
| `convnext-oof-full-population-validation` | [Full-population ConvNeXt OOF target validation](../experiments/convnext-oof-full-population-validation/README.md) | [RESULTS.md](../experiments/convnext-oof-full-population-validation/RESULTS.md) |
| `convnext-oof-head-adaptation` | [ConvNeXt OOF-corrected head adaptation](../experiments/convnext-oof-head-adaptation/README.md) | [RESULTS.md](../experiments/convnext-oof-head-adaptation/RESULTS.md) |
| `convnext-oof-logit-residual` | [ConvNeXt OOF logit-residual screen](../experiments/convnext-oof-logit-residual/README.md) | [RESULTS.md](../experiments/convnext-oof-logit-residual/RESULTS.md) |
| `convnext-oof-student-validation` | [ConvNeXt OOF-corrected student validation](../experiments/convnext-oof-student-validation/README.md) | [RESULTS.md](../experiments/convnext-oof-student-validation/RESULTS.md) |
| `convnext-oof-teacher-audit` | [ConvNeXt OOF teacher audit](../experiments/convnext-oof-teacher-audit/README.md) | [RESULTS.md](../experiments/convnext-oof-teacher-audit/RESULTS.md) |
| `convnext-reader-feature-cache` | [Public ConvNeXt reader feature cache](../experiments/convnext-reader-feature-cache/README.md) | [RESULTS.md](../experiments/convnext-reader-feature-cache/RESULTS.md) |
| `convnext-reader-mirror-validation` | [ConvNeXt reader anatomical-mirror validation](../experiments/convnext-reader-mirror-validation/README.md) | [RESULTS.md](../experiments/convnext-reader-mirror-validation/RESULTS.md) |
| `convnext-reader-oof-head-validation` | [ConvNeXt reader OOF-corrected head validation](../experiments/convnext-reader-oof-head-validation/README.md) | [RESULTS.md](../experiments/convnext-reader-oof-head-validation/RESULTS.md) |
| `convnextv2-domain-pool-validation` | [ConvNeXtV2 domain-pool validation](../experiments/convnextv2-domain-pool-validation/README.md) | [RESULTS.md](../experiments/convnextv2-domain-pool-validation/RESULTS.md) |
| `crossplane-token-residual` | [Cross-plane token residual screen](../experiments/crossplane-token-residual/README.md) | [RESULTS.md](../experiments/crossplane-token-residual/RESULTS.md) |
| `curia2-gpu-validation` | [Curia-2 GPU private validation](../experiments/curia2-gpu-validation/README.md) | [RESULTS.md](../experiments/curia2-gpu-validation/RESULTS.md) |
| `curia2-public-reproduction` | [curia2-public-reproduction](../experiments/curia2-public-reproduction/) | [RESULTS.md](../experiments/curia2-public-reproduction/RESULTS.md) |
| `data-supervision-eda-2026-10-22` | [Data-first reconstruction and fitting-only EDA — 22 October2026](../experiments/data-supervision-eda-2026-10-22/README.md) | See directory records; no uniform RESULTS.md |
| `dinosaur-v5-public-baseline` | [DINOsaur V5 public baseline reproduction](../experiments/dinosaur-v5-public-baseline/README.md) | See directory records; no uniform RESULTS.md |
| `dinosaur-v55-public-reproduction` | [dinosaur-v55-public-reproduction](../experiments/dinosaur-v55-public-reproduction/) | [RESULTS.md](../experiments/dinosaur-v55-public-reproduction/RESULTS.md) |
| `discussion-derived-audit-2026-09-02` | [discussion-derived-audit-2026-09-02](../experiments/discussion-derived-audit-2026-09-02/) | [RESULTS.md](../experiments/discussion-derived-audit-2026-09-02/RESULTS.md) |
| `domain-byol-validation` | [Domain-adaptive BYOL validation](../experiments/domain-byol-validation/README.md) | [RESULTS.md](../experiments/domain-byol-validation/RESULTS.md) |
| `e11-residual-validation` | [e11-residual-validation](../experiments/e11-residual-validation/) | See directory records; no uniform RESULTS.md |
| `end-to-end-convnext-validation` | [End-to-end ConvNeXt-Tiny validation](../experiments/end-to-end-convnext-validation/README.md) | [RESULTS.md](../experiments/end-to-end-convnext-validation/RESULTS.md) |
| `end-to-end-dinov2-validation` | [End-to-end DINOv2 validation](../experiments/end-to-end-dinov2-validation/README.md) | [RESULTS.md](../experiments/end-to-end-dinov2-validation/RESULTS.md) |
| `end-to-end-radimagenet-validation` | [End-to-end RadImageNet validation](../experiments/end-to-end-radimagenet-validation/README.md) | [RESULTS.md](../experiments/end-to-end-radimagenet-validation/RESULTS.md) |
| `final-selection-2026-10-21` | [Final selection, 21 October 2026](../experiments/final-selection-2026-10-21/README.md) | See directory records; no uniform RESULTS.md |
| `frozen-0958-control` | [Frozen scored0.958 control for follow-on comparisons](../experiments/frozen-0958-control/README.md) | See directory records; no uniform RESULTS.md |
| `highres-dinov2-validation` | [High-resolution DINOv2 validation](../experiments/highres-dinov2-validation/README.md) | [RESULTS.md](../experiments/highres-dinov2-validation/RESULTS.md) |
| `highres-pixel-cache` | [High-resolution physical pixel cache](../experiments/highres-pixel-cache/README.md) | See directory records; no uniform RESULTS.md |
| `label-consensus-validation` | [Weak-label consensus validation](../experiments/label-consensus-validation/README.md) | See directory records; no uniform RESULTS.md |
| `label-source-head-validation` | [Paired label-source image-head validation](../experiments/label-source-head-validation/README.md) | [RESULTS.md](../experiments/label-source-head-validation/RESULTS.md) |
| `mars-3d-validation` | [MARS 3D foundation-model validation](../experiments/mars-3d-validation/README.md) | [RESULTS.md](../experiments/mars-3d-validation/RESULTS.md) |
| `mavit-public-reproduction` | [Public MAViT 288 reproduction](../experiments/mavit-public-reproduction/README.md) | [RESULTS.md](../experiments/mavit-public-reproduction/RESULTS.md) |
| `medicalnet-3d-validation` | [MedicalNet 3D held-out validation](../experiments/medicalnet-3d-validation/README.md) | See directory records; no uniform RESULTS.md |
| `medimageinsight-feature-cache` | [MedImageInsight physical feature cache](../experiments/medimageinsight-feature-cache/README.md) | [RESULTS.md](../experiments/medimageinsight-feature-cache/RESULTS.md) |
| `medimageinsight-residual-screen` | [MedImageInsight target-attention residual screen](../experiments/medimageinsight-residual-screen/README.md) | [RESULTS.md](../experiments/medimageinsight-residual-screen/RESULTS.md) |
| `multireader-consensus-gate` | [Label-free multireader consensus gate](../experiments/multireader-consensus-gate/README.md) | [RESULTS.md](../experiments/multireader-consensus-gate/RESULTS.md) |
| `multireader-nested-blend` | [Multi-reader nested blend](../experiments/multireader-nested-blend/README.md) | See directory records; no uniform RESULTS.md |
| `oof-coteaching-validation` | [Nested OOF co-teaching validation](../experiments/oof-coteaching-validation/README.md) | [RESULTS.md](../experiments/oof-coteaching-validation/RESULTS.md) |
| `optuna-gold-search` | [optuna-gold-search](../experiments/optuna-gold-search/) | See directory records; no uniform RESULTS.md |
| `optuna-window-search` | [optuna-window-search](../experiments/optuna-window-search/) | See directory records; no uniform RESULTS.md |
| `orthodiffusion-residual-validation` | [OrthoDiffusion exploration](../experiments/orthodiffusion-residual-validation/README.md) | See directory records; no uniform RESULTS.md |
| `pairwise-auc-head` | [Pairwise-AUC ConvNeXt head](../experiments/pairwise-auc-head/README.md) | [RESULTS.md](../experiments/pairwise-auc-head/RESULTS.md) |
| `physical-paired-validation` | [Paired physical preprocessing validation](../experiments/physical-paired-validation/README.md) | [RESULTS.md](../experiments/physical-paired-validation/RESULTS.md) |
| `physical-preprocessing-audit` | [Physical preprocessing audit](../experiments/physical-preprocessing-audit/README.md) | [RESULTS.md](../experiments/physical-preprocessing-audit/RESULTS.md) |
| `plane-query-head-validation` | [Plane-aware target-query validation](../experiments/plane-query-head-validation/README.md) | [RESULTS.md](../experiments/plane-query-head-validation/RESULTS.md) |
| `public-0956-reproduction` | [Public 0.956 exact reproduction](../experiments/public-0956-reproduction/README.md) | See directory records; no uniform RESULTS.md |
| `public-09573-reproduction` | [Public 0.957 R6 + Raptor 384/224 reproduction](../experiments/public-09573-reproduction/README.md) | See directory records; no uniform RESULTS.md |
| `public-09576-reproduction` | [Public 0.9576 R6 flip-TTA + Raptor 384/224 reproduction](../experiments/public-09576-reproduction/README.md) | See directory records; no uniform RESULTS.md |
| `public-09586-audit` | [Public 0.9586-title kernel audit — identity correction](../experiments/public-09586-audit/README.md) | See directory records; no uniform RESULTS.md |
| `public-asset-audit-2026-10-21` | [Newly updated public training assets — static triage, 21 October 2026 IST](../experiments/public-asset-audit-2026-10-21/README.md) | See directory records; no uniform RESULTS.md |
| `public-frontier-audit-2026-10-06` | [Public frontier reproducibility audit — 6 October 2026](../experiments/public-frontier-audit-2026-10-06/README.md) | See directory records; no uniform RESULTS.md |
| `public-frontier-audit-2026-10-14` | [Public 0.956 notebook provenance audit — 14 October 2026](../experiments/public-frontier-audit-2026-10-14/README.md) | See directory records; no uniform RESULTS.md |
| `public-frontier-audit-2026-10-17` | [Public frontier audit — 17 October 2026](../experiments/public-frontier-audit-2026-10-17/README.md) | See directory records; no uniform RESULTS.md |
| `public-gp958d-reproduction` | [Public GP958d R7 + Raptor 384/224 reproduction](../experiments/public-gp958d-reproduction/README.md) | See directory records; no uniform RESULTS.md |
| `public-gp958e-audit` | [Public GP958e audit](../experiments/public-gp958e-audit/README.md) | See directory records; no uniform RESULTS.md |
| `public-helios-private-blend-audit` | [Heliosli public 0.958 per-finding blend audit](../experiments/public-helios-private-blend-audit/README.md) | See directory records; no uniform RESULTS.md |
| `quality-efficientnet-validation` | [Quality-filtered EfficientNet-B0 validation](../experiments/quality-efficientnet-validation/README.md) | [RESULTS.md](../experiments/quality-efficientnet-validation/RESULTS.md) |
| `quantitative-morphology-specialist` | [Quantitative morphology specialist](../experiments/quantitative-morphology-specialist/README.md) | [RESULTS.md](../experiments/quantitative-morphology-specialist/RESULTS.md) |
| `r6-adapter-exploratory-submission` | [Explicit exploratory submission after failed proxy gate](../experiments/r6-adapter-exploratory-submission/README.md) | See directory records; no uniform RESULTS.md |
| `r6-crossfit-fold-calibration` | [R6 cross-fitted fold/pass calibration](../experiments/r6-crossfit-fold-calibration/README.md) | See directory records; no uniform RESULTS.md |
| `r6-dense-window-screen` | [R6 dense-window localization screen](../experiments/r6-dense-window-screen/README.md) | [RESULTS.md](../experiments/r6-dense-window-screen/RESULTS.md) |
| `r6-fold0-warmstart` | [R6 fold-0 strict reconstruction — CPU gate, 20 October 2026](../experiments/r6-fold0-warmstart/README.md) | See directory records; no uniform RESULTS.md |
| `r6-full-population-logit-cache` | [R6 full-population fold-logit cache](../experiments/r6-full-population-logit-cache/README.md) | See directory records; no uniform RESULTS.md |
| `r6-lowrank-spatial-adaptation` | [Rank8 adapters inside the strong R6 image reader — new no-submission pilot](../experiments/r6-lowrank-spatial-adaptation/README.md) | [RESULTS.md](../experiments/r6-lowrank-spatial-adaptation/RESULTS.md) |
| `r6-own-report-adaptation` | [R6 own-report last-stage adaptation — new private hypothesis](../experiments/r6-own-report-adaptation/README.md) | [RESULTS.md](../experiments/r6-own-report-adaptation/RESULTS.md) |
| `r6-robust-fold-aggregation` | [R6 robust fold aggregation screen](../experiments/r6-robust-fold-aggregation/README.md) | See directory records; no uniform RESULTS.md |
| `report-image-cross-supervision` | [Report-image cross-supervision](../experiments/report-image-cross-supervision/README.md) | [RESULTS.md](../experiments/report-image-cross-supervision/RESULTS.md) |
| `research-audit-2026-10-19` | [Research audit — 19 October 2026](../experiments/research-audit-2026-10-19/README.md) | See directory records; no uniform RESULTS.md |
| `s6-fifth-arm-audit` | [S6 fifth-arm audit](../experiments/s6-fifth-arm-audit/README.md) | See directory records; no uniform RESULTS.md |
| `segmentation-roi-validation` | [Segmentation-guided ROI validation](../experiments/segmentation-roi-validation/README.md) | [RESULTS.md](../experiments/segmentation-roi-validation/RESULTS.md) |
| `sequence-context-validation` | [Adjacent-slice sequence-context validation](../experiments/sequence-context-validation/README.md) | [RESULTS.md](../experiments/sequence-context-validation/RESULTS.md) |
| `simple-resnet34-validation` | [Plain ResNet34 validation](../experiments/simple-resnet34-validation/README.md) | [RESULTS.md](../experiments/simple-resnet34-validation/RESULTS.md) |
| `slot-aware-resnet34-validation` | [Slot-aware ResNet34 validation](../experiments/slot-aware-resnet34-validation/README.md) | [RESULTS.md](../experiments/slot-aware-resnet34-validation/RESULTS.md) |
| `spatial-dinov2-token-screen` | [Spatial DINOv2 token residual screen](../experiments/spatial-dinov2-token-screen/README.md) | [RESULTS.md](../experiments/spatial-dinov2-token-screen/RESULTS.md) |
| `supervised-adaptation-validation` | [Supervised RadImageNet adaptation validation](../experiments/supervised-adaptation-validation/README.md) | [RESULTS.md](../experiments/supervised-adaptation-validation/RESULTS.md) |
| `swarm-0936-validation` | [swarm-0936-validation](../experiments/swarm-0936-validation/) | See directory records; no uniform RESULTS.md |
| `target-entity-cross-supervision` | [Target-specific report cross-supervision](../experiments/target-entity-cross-supervision/README.md) | [RESULTS.md](../experiments/target-entity-cross-supervision/RESULTS.md) |
| `training-signal-audit` | [Fitting-signal audit — no new model or GPU job](../experiments/training-signal-audit/README.md) | See directory records; no uniform RESULTS.md |
| `transfer-optuna` | [Transfer-learning comparison — 19 October 2026](../experiments/transfer-optuna/README.md) | [RESULTS.md](../experiments/transfer-optuna/RESULTS.md) |
| `uncertainty-router` | [Fold-uncertainty dynamic router](../experiments/uncertainty-router/README.md) | See directory records; no uniform RESULTS.md |

## Weights and reproducibility

The catalogue publishes architecture/code locations and aggregate evidence, not private medical targets or unrestricted weights. Original checkpoints stay in ignored local/private Kaggle storage unless redistribution rights and a separate release audit permit publication. See [release safety](PUBLIC_RELEASE.md). Source hashes, initialization lineage and final checkpoint hashes are retained in experiment receipts.

No model is represented as independently pathology-validated, first-place, or guaranteed to improve the competition score.
