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

## Do not resurrect the closed soft-v2 label route

The schema audit above is **not** permission to train from soft-v2. The earlier
`public-frontier-audit-2026-10-17/README.md` already rejected that release:
its exact v2 generator/mapping is not pinned, and the author's public lineage
iterates report-state prompts on all 58 organizer-gold reports. This is a
separate provenance problem from the missing LR columns. Excluding gold rows
from a new optimizer or adopting a new loss does not undo upstream selection.

An additional CPU inspection checked the actual non-gold state vocabulary
against the frozen R6 head, without scoring predictions or making targets.
The vocabularies are not drop-in compatible: e.g. soft-v2's
`trace_or_small` does not identify which of R6's `none_or_trace` and `small`
states applies; `macerated_or_postop` is not an authenticated alias for
`macerated_or_destroyed`; OA `moderate` does not supply the size/grade detail
in the original fine-grained state categories. Even matching names do not
establish identical report extraction or definitions.

Partial-label learning could in principle retain a set of possible fine-grained
states instead of inventing a single category. That mathematical alternative
does **not** authenticate this source's generation/selection lineage or supply
independent pathology evidence. It is therefore not a justified GPU candidate
with these closed inputs. No mappings, training targets, loss tuning or models
were generated. The reproducible audit records per-finding vocabulary mismatch
counts and links the earlier provenance decision; do not repeat this route as
another experiment while awaiting publisher-contact approval.

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
