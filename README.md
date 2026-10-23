# RSNA Knee Abnormality Detection

Knee MRI research, model adaptation and audited Kaggle inference for the [RSNA Knee Abnormality Detection competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection). This archive combines attributed public-model reproductions with our own supervised, low-rank, video-transfer and multi-plane experiments.

**Best recorded public score: 0.958.** That result reproduces an attributed public ensemble; it is not evidence that our training improved the published models. Our final rank-8 adapter submission completed with **public 0.958 / private 0.942**, tying the displayed public control. No private improvement over the control has been established.

## Models and measured outcomes

| Model / experiment | Implementation | Result and limitation |
| --- | --- | --- |
| R6 + Raptor384/224 control | Reproduced knee-trained readers, anatomy-centred windows, state/fluid stackers and per-finding rank blend | Public **0.958**, submission57483292; attributed reproduction |
| R6 rank-8 spatial adapters | Six adapter matrices /92,160 parameters inside three final-stage MLP projections; original tensors frozen | Three epochs /576 updates; proxy NLL gain0.000759 below0.01 gate. Exploratory submission57535170: public **0.958**, private **0.942** |
| R6 final-stage warm start |31 trainable tensors /15,470,592 parameters, original heads preserved | Three epochs /576 updates; confirmation report-state NLL gain0.059662, but maximum probability drift0.457414 exceeded0.30; rejected |
| Swin3D Tiny transfer | Adapted final stage with multi-sequence pooling and bounded Optuna screen | Selected trial3 tuning proxy BCE **0.567161**, default0.576836; no pathology or competition gain established |
| VideoMAE Base transfer | Adapted final transformer blocks in the same private screen | Tuning proxy BCE0.600935; private noncommercial research only, not competition/prize-use clearance |
| ConvNeXt MIL / R7-Raptor reproductions | Released inference contracts | Recorded public0.944 /0.956; upstream models, not our original training |
| Other research families | CoAtNet, DINOv2, RadImageNet, ResNet34, 3D readers, medical vision features, cross-plane attention, report supervision and segmentation ROI | Failed, rejected and implementation-only branches are included in the catalogue; not all are trained or submitted models |

See the [complete experiment catalogue](docs/MODELS.md), [technology stack](docs/TECHNOLOGY.md), [research audit](experiments/research-audit-2026-10-19/README.md) and [final adapter result](experiments/r6-adapter-exploratory-submission/V2_SUBMISSION.md). Historical documents retain their original baselines; this overview supersedes older descriptions of the current best score.

## Inference architecture

The final control combines R6 with Raptor384 and Raptor224. It selects multi-series DICOM inputs, orders slices using physical geometry, localizes anatomy and preserves original knee-reader preprocessing/postprocessing. R6 uses384px windows,352px model crops,3.5mm through-plane steps, stride1, full-field views and100mm local views. Anatomical flip/TTA includes the source MCL exception. These are not the older root CoAtNet pipeline's140mm/64-slice settings.

Raptor384/224 percentile ranks are mixed65/35, then blended with R6 using frozen per-finding weights and average-tie ranks. The exploratory adapter replaces only R6fold0 with the final epoch3 adapter; folds1–3, Raptor, preprocessing, TTA and state/fluid stackers are unchanged.

## Included research

- Model/adaptation Python code, notebook builders and available attributed inference notebooks.
- Experiment contracts, asset SHA-256 identities, runtime receipts, synthetic tests and aggregate result reports.
- DICOM/preprocessing parity checks, strict checkpoint loading, cache audits, supervision EDA and failure analyses.

Historical code is research material, not a single supported package. The root inference entrypoint is an older CoAtNet baseline, **not** the0.958 serving contract.

## Running and reproducibility

Follow each experiment's README and frozen build/launch receipts. Dataset mounts, dependencies and checkpoints differ; there is no universal install command. Kaggle access and restricted assets must be obtained independently under their original terms. Tests requiring ignored local checkpoints/caches cannot run from a source-only clone.

Do not launch historical notebooks blindly: some contain closed contracts or account-specific references. This repository does not automatically train, submit or change final selections.

## Data, weights and attribution

This is a research-code archive, not a medical dataset or unrestricted checkpoint distribution. Raw DICOMs, reports, per-study identifiers/targets/splits, prediction tables, credentials and private weight payloads must not be included in a public release. See [public-release safety](docs/PUBLIC_RELEASE.md).

The scored0.958 control is attributed to [Heliosli's public notebook](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters), incorporating upstream R6/Raptor work credited in its source, including nartaa and dreaddevelopment. Preserve all original notices, model terms and OAI/NDA acknowledgements. Existing SKM-TEA localization is distinct from OrthoFoundation-MSK-5M, which was not approved as a new reader. This archive grants no blanket license over third-party assets.

## Targets and limitations

ACL, MCL, medial/lateral meniscus, medial/lateral/patellofemoral OA, effusion, synovitis, Baker's cyst, contusion and fracture.

Report-derived labels are weak supervision, not independent pathology truth. Source pool-wide priors mean warm-start evaluation is conditional additional-image-gradient holdout, **not whole-pipeline OOF**. Repeatedly inspected organizer sets are not fresh validation. Implementation parity is not accuracy progress. This project is not clinically validated and must not be used for diagnosis or treatment.
