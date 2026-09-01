# Nested OOF co-teaching validation

This private experiment uses cross-fold image/report agreement to alter weak supervision without organizer labels. For every outer seed, five fixed-epoch image teachers generate predictions only for held-out non-gold report groups. Confident teacher/report disagreements receive a small predeclared correction; all examples receive bounded reliability weights. A fresh image-only student is compared with an identically trained uncorrected control.

The authenticated cached RadImageNet features avoid image re-encoding. All 58 organizer-labelled studies are absent from teacher fitting, OOF generation, correction, validation, stopping, and selection, and are used only for final paired evaluation. No competition submission can be created.
