# Domain-adaptive BYOL validation

This private experiment tests whether label-free adaptation of the pinned RadImageNet ResNet-50 to the competition's knee-MRI domain improves the incumbent image representation.

The BYOL stage uses only the 4,349 non-gold studies, a fixed predeclared configuration, and no labels. The 58 organizer-labelled studies are excluded from representation learning, classifier fitting, weak validation, stopping, and model selection. Both control and candidate use identical attention heads and report-group-isolated weak-label splits across five seeds. The notebook cannot create a competition submission.

Acceptance requires all of the following:

- ensemble macro-AUC delta of at least +0.005;
- at least four of five positive seed deltas;
- paired-bootstrap probability of improvement of at least 0.95;
- positive paired-bootstrap 2.5th percentile.

Passing this screen would only authorize a separate private comparison with the locked 0.936 ensemble, not a competition submission.
