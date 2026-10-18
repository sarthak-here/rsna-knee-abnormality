# Run history

## Version 1 — infrastructure failure before inference

Kaggle prepared the exact 58-study evaluation root, then stopped before model
setup or inference because the extracted R6 stage expected the public source
notebook's `NB_T0` timer variable. The compact screen had intentionally omitted
the earlier Raptor cells where that variable was originally initialized.

Version 2 adds only `NB_T0 = time.time()` before the unchanged preparation
cell. The stride-2 setting, model, preprocessing, TTA, checkpoints, anchor,
weights, evaluator, and gates are unchanged.
