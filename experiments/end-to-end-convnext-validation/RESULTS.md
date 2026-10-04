# Result: rejected after one-fold screen

The private ConvNeXt-Tiny run completed successfully and its checkpoint hash was
verified. It used Steven v4 supervision, 3,479 fitting studies, 870 report-group-isolated
validation studies, and kept all 58 organizer-labelled studies evaluation-only.

- Best weak-validation macro AUC: **0.820395**
- Organizer-gold macro AUC: **0.817358**
- Prior frozen RadImageNet control ensemble: **0.860457**
- Difference from control: **-0.043099**

This 224px/9-slice ConvNeXt recipe was rejected and was not submitted. The next step
changes the information content rather than trying another backbone on the same cache:
a 288px, 15-slice, 140mm physical cache.
