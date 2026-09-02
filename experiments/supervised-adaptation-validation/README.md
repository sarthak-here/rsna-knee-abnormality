# Supervised RadImageNet adaptation validation

This private experiment fine-tunes the final RadImageNet ResNet-50 stage on multi-plane knee-MRI study bags. A report-group-isolated non-gold weak-validation split is fixed before fitting. All 58 organizer-labelled studies remain evaluation-only, and the notebook cannot create a submission.

Acceptance requires +0.005 macro AUC, at least four of five positive seeds, paired-bootstrap probability of improvement of at least 0.95, and positive q025.
