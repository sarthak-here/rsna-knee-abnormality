# Segmentation-guided ROI validation

This private experiment tests whether anatomical localization improves the incumbent MRI representation. It uses the publicly released KneeXNet-2.5D ResNet34 U-Net to find cartilage and meniscus foreground in sagittal MRI stacks. Confident foreground boxes produce anatomy-focused RadImageNet tokens; uncertain masks fall back to the untouched incumbent token. A fixed hybrid uses ROI predictions only for medial/lateral meniscus and OA targets.

Public model attribution:

- M. Sanogo et al., “KneeXNet-2.5D: a clinically-oriented and explainable deep learning framework for MRI-based knee cartilage and meniscus segmentation,” *npj Health Systems* (2026), DOI: 10.1038/s44401-026-00072-5.
- Source and model: `pitthexai/Knee_MRI_Segmentation_2.5D`, commit `f2452feb602f2bd9e790550c8a5a07f62e7d6e55`.
- Checkpoint SHA-256: `0c2b5fe87f86e78c7d90e9dfc3de1307aaef57c36d44ebb1251eecdcffc3c663`.
- The authors release the code and weights publicly for research, educational, and academic use. This experiment is private research validation and makes no submission.

All 58 organizer-labelled studies are excluded from fitting, splitting, weak validation, stopping, and model selection. Acceptance requires +0.005 ensemble macro AUC, at least four of five positive seeds, paired-bootstrap probability of improvement of at least 0.95, and positive q025. Passing would authorize only a separate private comparison with the locked 0.936 ensemble.
