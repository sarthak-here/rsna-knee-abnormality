# Version 1 terminal audit — 20 October 2026 IST

The private job completed, not crashed. The earlier user-facing completion
summary is now backed by downloaded checkpoint hashes and parameter-finiteness
checks, exact source/asset identities, reconstructed report-group splits,
read-only SQLite trial history, and the final no-submission log marker.
`terminal_audit.json` contains the reproducible machine-readable findings.

| Trial | Family | Result | Tuning soft-label BCE (lower is better) |
|---|---|---|---:|
| 0 | Swin3D Tiny, default LR 3e-5 | One epoch complete | 0.5768359076 |
| 1 | VideoMAE Base, default LR 3e-5 | One epoch complete | 0.6009348336 |
| 2 | Swin3D Tiny, LR 1.3349e-5 | Pruned halfway | 0.6157098250 (partial only) |
| 3 | Swin3D Tiny, LR 8.3827316486e-5, WD 0.0001267288158 | One epoch complete, selected | 0.5671608386 |
| 4–7 | Proposed configurations | Skipped before training: insufficient remaining measured budget | None |

Thus three trials finished, one was stopped early, and four did not train.
Do not describe this as eight trained models or a converged architecture search.
Active trial time was 6,378.321 seconds (106.31 minutes), below the 7,200-second
cap. The completion marker occurred at log-relative 8,568.251 seconds (142.80
minutes), including setup/download/preflight/input checks outside active search.

A CPU diagnostic computed only from fitting labels gives a constant per-target
prevalence predictor BCE of 0.6051387939 on the same tuning rows. The selected
Swin improves this proxy loss by 0.0379779553 and the default Swin by
0.0096750690. This does show useful report-proxy learning, but cannot establish
pathology accuracy, macro AUC, or improvement over the scored 0.957 stack.

## Artifact checks

- All three complete-trial checkpoint bytes match SQLite SHA-256 receipts;
  every floating model-state tensor is finite, epoch is one, parameters match
  its trial, and every checkpoint binds the exact contract hash.
- Downloaded `models.py`, `search.py`, assets and GPU preflight match their
  recorded hashes. Both pretrained release identities match the pinned source.
- Recomputed partitions match every ordered ID: 2,637 fit, 856 tune, 852 sealed
  confirmation; 58 gold and four gold-report duplicates excluded. Neither gold
  nor confirmation was scored.
- All recorded objective values are finite. Trial states are COMPLETE/PRUNED,
  never FAILED. The selected trial agrees with the final report.
- Full 28-entry remote inventory contained no CSV/submission. Logs end with
  `TRANSFER_OPTUNA_COMPLETE_NO_SUBMISSION`; no fatal Python traceback.
- Logs are **not warning-free**: pip reports dependency conflicts with unused
  Gradio/Diffusers packages. These did not abort training, but must be disclosed
  rather than calling the environment clean. Do not rerun this search merely
  to remove those warnings. An isolated runtime is preferable for future work.

Selected checkpoint SHA-256:
`2274760cb880d2450b630593233daeb5ed287a0c5aa4a608eb72d59ec71bd89c`.
The exact bytes remain ignored under `.audit/transfer-optuna-output-v1` and on
the private version-1 notebook; large weights are not committed to Git.

## Decision

Freeze trial 3's configuration. No additional Optuna trials, target selection,
organizer-gold scoring, automatic full-data training or submission is justified
by these tuning losses. A future advancement requires the predeclared once-only
sealed-confirmation comparison and independent pathology evidence. VideoMAE
remains private noncommercial research, not competition/prize-use clearance.
The public 0.957 baseline is unchanged.

The user's newer direction is direct fine-tuning of a strong **knee-trained**
checkpoint. The source-fold exclusion audit in
`experiments/research-audit-2026-10-19/WARMSTART_REASSESSMENT.md` (commit ea6b794)
is the CPU-side starting point for that work; it does not authorize a new GPU
run before strict reconstruction, preprocessing parity, supervision and compute
gates. Do not poll or rerun this completed video job.

## Reproduce the audit

```powershell
python experiments/transfer-optuna/audit_result.py --root .audit/transfer-optuna-output-v1 --train .audit/competition-metadata/train.csv --labels .audit/transfer-optuna-models/labels/llm_labels_v4_blend.csv --out experiments/transfer-optuna/terminal_audit.json
```
