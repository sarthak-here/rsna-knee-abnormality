# Adjacent-slice sequence-context validation

This private experiment adapts the final RadImageNet stage without report labels. It learns adjacent-slice consistency with a symmetric InfoNCE loss and through-plane order with a relative-position objective. A fixed report-group-isolated non-gold validation split is excluded from pretraining; all 58 organizer-labelled studies remain evaluation-only. The notebook cannot submit.

Acceptance requires +0.005 macro AUC, at least four of five positive seeds, paired-bootstrap probability of improvement at least 0.95, and positive q025.
