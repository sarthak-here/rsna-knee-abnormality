# MARS 3D foundation-model validation

This private experiment compares the incumbent 2D RadImageNet representation with the official MARS 3D Swin Transformer, a volumetric MRI foundation model pretrained on 336,476 multi-sequence MRI volumes.

Attribution and provenance:

- MARS: “Large-scale multi-sequence pretraining for generalizable MRI analysis in versatile clinical applications,” *Nature Biomedical Engineering* (2026).
- Official repository: `zqiuak/MARS`, Apache-2.0 license.
- Official checkpoint SHA-256: `88388a59d92935a15d30bd41c0e9dbece99ba78db73cb374d44290fd6566ccc3`.

The candidate encodes one fluid-sensitive 32×96×96 volume from each sagittal, coronal, and axial plane, then trains the same kind of lightweight target-attention head used by the control. All 58 organizer-labelled studies are excluded from fitting, splitting, weak validation, stopping, and model selection. The notebook cannot submit.

Acceptance requires +0.005 ensemble macro AUC, at least four of five positive seeds, paired-bootstrap probability of improvement of at least 0.95, and positive q025. Passing would authorize only a separate private comparison with the locked 0.936 ensemble.
