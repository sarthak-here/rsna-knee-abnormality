# Result: rejected after one-fold screen

The private run completed successfully and its checkpoint SHA-256 was verified.
It trained every DINOv2-S block on 3,479 non-gold studies and used 870
report-group-isolated studies for checkpoint selection. The 58 organizer-labelled
studies were evaluation-only.

- Best weak-validation macro AUC: **0.699018**
- Organizer-gold macro AUC: **0.669819**
- Prior frozen RadImageNet control ensemble: **0.860457**
- Difference from that control: **-0.190638**

This exact full-unfreeze, 224px, JEV-labelled recipe was rejected. It does not justify
five-fold training and produced no competition submission. The monotonic but low
learning curve suggests either destructive whole-backbone adaptation or a supervision
mismatch; the next experiment isolates label-source quality on fixed authenticated
features before spending more GPU time.
