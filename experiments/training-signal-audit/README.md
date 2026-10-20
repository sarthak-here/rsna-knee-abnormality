# Fitting-signal audit — no new model or GPU job

The closed own-report adaptation improved proxy losses but failed its fixed
confirmation stability gate. This audit checks a separate, narrower question:
what supervision actually entered its fitting objective? It reads **only the
192 existing fit reports**, reconstructs their already frozen state targets,
and calculates the exact coefficients in the unchanged original loss. No model,
predictions, gold reports or tuning/confirmation reports are loaded/parsed.

Run from repo root:

```powershell
python experiments/training-signal-audit/audit_signal.py
python -m unittest discover -s experiments/training-signal-audit -p "test_*.py"
```

## Material findings

- Only536 known fitting cells across8 of12 findings. The3 OA targets and
  Contusion had no direct report supervision, only preservation/distillation.
- Fracture: **5 known cells:4 none,1 old/healed,0 acute**.
- MCL:60 known cells,58 normal and2 complete tears; intermediate categories absent.
- Synovitis:12 known cells,11 positive and1 none.
- The original class-balanced CE coefficient is
  `192 / (8 * known_class_cells)`: fracture4.8 per cell versus medial-meniscus
  0.193548 per cell, a **24.8-fold difference**. Every supervised class has equal
  total coefficient24 per epoch despite very different independent support.

These are loss coefficients, **not measured gradient magnitudes** and not a
causal attribution of the confirmation failure. A strong source classifier may
learn something useful from negatives, but this fitting sample cannot establish
better acute-fracture discrimination from its new labels. The already observed
proxy improvement does not remove that limitation.

## Evaluation scope inventory, not a fresh-test certificate

Metadata-only grouping finds844 guarded source-fold0 report groups not selected
for the previous pilot. This includes reports without usable literal-state
labels, unlike the303 *covered* groups in the old split preparation. It is not
844 usable pathology-labelled validation studies. No labels/losses from these
unused reports were inspected by this audit. Only an aggregate count and group
digest are saved, not raw reports or unused study identifiers.

“Not selected in this pilot” does **not** prove globally untouched evaluation:
prior study exposure across other experiments and source pool-wide priors still
need a lineage audit. Both inspected64-study sets remain consumed, never sealed
again. No candidate/holdout is selected from this inventory here.

## Disposition

Do not rescue the closed model by suppressing fracture, changing its weights,
clipping/blending or choosing a previous epoch. No such change is implemented.

Before another training run, the useful implementation target is a fitting-only
supervision-quality/support check that identifies what can actually improve,
including missing positive states, and a genuinely untouched evaluation scope.
Any successor must be a separately declared contract rather than a retuned
version of the consumed confirmation test. New label generation, external data,
paid APIs or a new GPU job are not authorized by this audit. Preserve all terms,
attribution and the0.957 baseline; no score/accuracy improvement is claimed.
