# Training blocker, not another preprocessing gate — 20 October 2026

Preprocessing checks are complete for the documented fixtures. They have not
produced a new competitive model. No more generic fixture collection is needed
to establish that the original supervision/validation contract remains missing.

## Concrete findings

The released state specification embeds exact parent-file identities:

| Original artifact | Embedded MD5 |
|---|---|
| `probe_targets_opusv11b_rr6.csv` | `4f6d49ee3de5a9914c1101192e97cf4a` |
| `state_targets_opusv11b_rr6.csv` | `b5940451a9fe177d7fedcc81b16ecb4b` |
| `state_spec_opusv11b.json` | `59954e61e55c00e2d9505a633203f99e` |

The deployed manifest instead names the later target table
`probe_targets_opusv11b_rr6_plr6_cdr.csv`; the parent's digest must not be
misrepresented as authentication of that later table.

`audit_supervision.py` reads the actual source/spec and three local label
alternatives. None matches an embedded original digest. The soft-v2 probability
table has 4,407 rows, but lacks **all 24** original likelihood-ratio/cell-weight
columns. Its default eligible target pool is 4,349, not the released 4,344.
That calculation does not include source pixel-cache availability. The local
3,325-ID `bleg-cpu-audit` pool belongs to an older `E_prior` run and cannot be
used to fabricate the R6 source pool.

Crucially, original `lr_loss` calls `torch.log(lr)`: these targets are positive
**likelihood ratios**, not probabilities and not already-log ratios. Supplying
an unrelated probability column under the original LR objective would change
its meaning. Converting or choosing a different loss would be a new hypothesis,
not unchanged-source reproduction or an established improvement.

The five-study discrepancy remains unresolved: source filtering can exclude
blanked reports, review holdouts, fold -3 and unavailable rendered studies.
The count alone does not identify which five studies were absent.

## Fresh publisher inventory check

The public publisher inventory listed six releases on this check. In addition
to the previously audited nine-file `rsna-knee-r6-frozen-fe055406-w1`, direct
Kaggle file listings checked these predecessor releases:

- `heliosli/rsna-knee-r6-frozen-fe055406`: 9 files;
- `heliosli/rsna-knee-compact-r6cnxfsss`: 7 files;
- `heliosli/rsna-knee-compact-v1`: 8 files.

They contain checkpoints, specifications/stackers, manifests and (in compact-v1)
a README—not the original per-study target/state tables or training pool.
No weight downloads, GPU jobs or submissions were launched. Exact filename web
searches also did not recover the files; lack of search results is not proof
that no other source exists. Do not repeat these unchanged inventory checks as
model progress.

## Decision

Stop treating more preprocessing commits as progress toward a better score.
Original-source reproduction is not ready. A different fine-tuning objective
is not proved impossible, but must be explicitly justified with preserved rich
head semantics, defensible source-fold holdouts and a bounded paired test.
Report-label agreement would remain a proxy, not independent pathology truth.
Do not launch on unrelated weaker labels simply to make a GPU job appear.

The actionable asset request, if asking the publisher is authorized, is the
final CDR target table, matching state-target table, exact pool UID list and
construction/holdout lineage. No message has been sent on the user's behalf.
The scored 0.957 baseline and all closed experiments remain untouched.

Reproduce the metadata audit without a GPU:

```powershell
python experiments/r6-fold0-warmstart/audit_supervision.py
```
