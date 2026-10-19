# Public BLEG fold-asset audit — 19 October 2026

## Scope and decision

CPU metadata/source-lineage inspection only. No cancelled R6 job was polled or
restarted, no GPU notebook was launched, no checkpoint was downloaded, and no
organizer-gold prediction was evaluated. The locked baseline remains 0.957.

The newly inspected BLEG packages publish useful run metadata and partial OOF
files, but do not resolve the missing original R6 training contract or provide
independent evidence for launching a new reader. Keep this candidate on hold.

## Public packages inspected

- `kritikabenjwal/rsna-knee-bleg-s0-f01`: seed-zero fold 0 and 1 checkpoints,
  partial OOF arrays, gold-prediction JSONs, pool manifests and training logs.
- `sarthakgupta1801/rsna-knee-bleg-s0-f23`: seed-zero fold 2 and 3 counterparts.
- `xhaitanya/rsna-knee-bleg-s1-f01`: seed-one fold 0 and 1 counterparts.

Only the first package's fold-zero `run_config.json`, `train.log` and
`pool_uids.json` were downloaded. All three local files are in the ignored
`.audit/bleg-cpu-audit` directory. The other packages were inspected by public
file listings; their checkpoint bytes and array values have not been audited.

## Exact downloaded identities

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `run_config.json` | 2,183 | `16e55136947ecb0d7d8ae70224dd0948b04b46e02c317eec65f8438428e7d6ae` |
| `train.log` | 3,145 | `b3528d6ae6b41fda700dea14dab54029766eb16fec6983e46b1bbcef75ce4347` |
| `pool_uids.json` | 226,100 | `3e7d560edddfdf883c0b65df11cf557f2cae36f300e46c72e1cad2eea3642022` |

## What the actual configuration says

- ConvNeXt-Small; 352px crop; 3.5mm spacing; four folds; seed zero.
- Training arm `E_prior`, two epochs, 32 training windows, batch four.
- Warm start `/tmp/warm_r6`, scale 0.5; encoder LR 0.00015 after scaling.
- Targets are explicitly
  `kritikabenjwal/rsna-knee-labels-soft-v2/labels_soft_v2_targets.csv`.
- State targets are `/tmp/state_targets_blank.csv`.
- Pool has exactly 3,325 unique IDs, confirmed from the downloaded JSON;
  the log lists hash-fold counts `[786, 843, 874, 822]` and 58 gold studies.
- Thus the pool is 1,024 rows short of the 4,349 non-gold competition population;
  the reason for the omitted rows is not established by these three artifacts.
- The log records zero known state/binary supervision cells and zero positive
  or negative extent-supervision cells. Architectural state/extent heads are
  present, but that does not mean their original supervision was preserved.
- Warm-start log records strict loading of 376 tensors with checkpoint MD5
  `e267b556b0fb1631f5d208c6c813cab1`. The complete source checkpoint identity
  and authentic pre-training pool/fold exclusion chain remain unverified.
- The log reports epoch loss 0.5488 then 0.4705, and cumulative times 2,192s
  then 4,274s. It contains no OOF accuracy or independently labelled result.

## Why this is not sufficient evidence

The prior source audit already put `labels-soft-v2` on hold: no exact frozen
generation/calibration provenance was available, and the author's earlier
label-prompt workflow iterated on organizer-gold reports. The BLEG configuration
now identifies an actual consumer of that label release, improving consumer
lineage, but does not resolve the label-generation question.

Partial OOF filenames do not prove that the warm-start encoder excluded their
rows during its earlier training. A source pool configured to match a new pool
is not itself evidence of the original checkpoint's training exclusions.
Conversely, this audit does not assert leakage occurred: the required proof
has not yet been established. No independent scored evidence above 0.957 was
established by this inspection.

Before any launch, require the exact source checkpoint and original pool/fold
manifest, an explanation of the 1,024 omitted studies, frozen label-generation
provenance, complete source/weights identities and model-use terms, and an
independent validation path. Do not tune or evaluate the released gold JSONs to
select weights or targets.

## Other fresh search results

`haungjiawei/rsna-knee-train-v2-code-and-labels` still lists source, duplicate
groups, excluded-gold IDs and a weak-label table but no model checkpoint. This
does not resolve the original R6 supervision. `abdullahmathar/rsna-knee-argon4-soft-labels`
lists only one soft-label CSV; a filename alone does not establish target
generation or independence. Neither release justifies GPU spending from the
evidence inspected here.
