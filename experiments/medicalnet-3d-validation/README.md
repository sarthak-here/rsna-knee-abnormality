# MedicalNet 3D held-out validation

This private experiment tests a materially different volumetric encoder without making a competition submission.

- Backbone: Tencent MedicalNet / Med3D 3D ResNet-50, pretrained on 23 public medical datasets.
- Training labels: public report-derived soft labels, with all 58 organizer-labelled studies removed before fitting.
- Validation: untouched 58-study gold set, five independent head seeds, paired bootstrap comparison, and a fixed 20% blend with the public RadImageNet E9 OOF arm.
- Screening gate: at least +0.005 macro AUC over that public control, four of five seeds positive, bootstrap probability of improvement at least 0.95, and a positive 2.5th-percentile delta.

Passing this screen only permits a later comparison with the locked 0.936 ensemble. It does not permit a leaderboard submission.

## Attribution

- Sihong Chen, Kai Ma, and Yefeng Zheng, *Med3D: Transfer Learning for 3D Medical Image Analysis*.
- Tencent MedicalNet, MIT license.
- Public MedicalNet weights packaged by Royce Playz.
- Public RSNA knee report labels published by Steven Lee Hans, with upstream contributors credited in that dataset.
