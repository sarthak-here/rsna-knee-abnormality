# Result: rejected after one-fold screen

The private one-fold run completed successfully and the checkpoint hash was verified.
It trained on 3,479 non-gold studies, validated on 870 report-group-isolated non-gold
studies, and used all 58 organizer-labelled studies only for final evaluation.

- Best weak-validation macro AUC: **0.819546**
- Organizer-gold macro AUC: **0.828038**
- Prior frozen RadImageNet control ensemble: **0.860457**
- Difference from that control: **-0.032419**

The full end-to-end ResNet-50 did not justify a five-fold run. No competition
submission was produced. The result rejects scaling this exact RadImageNet/JEV/224px
recipe; it does not establish that all end-to-end fine-tuning is ineffective.
