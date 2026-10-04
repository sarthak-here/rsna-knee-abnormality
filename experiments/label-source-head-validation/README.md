# Paired label-source image-head validation

This private experiment isolates supervision quality from image representation. It
trains identical attention heads on the same authenticated frozen RadImageNet features,
using either Steven v4 soft labels or the CC0 JEV hybrid soft labels. Five deterministic
report-group-isolated splits are evaluated on the 58 organizer-labelled studies, which
are excluded from every fit and validation decision.

The notebook cannot create a competition submission. Its purpose is to choose the label
source for the next expensive representation-learning experiment without confounding
that decision with architecture or preprocessing changes.
