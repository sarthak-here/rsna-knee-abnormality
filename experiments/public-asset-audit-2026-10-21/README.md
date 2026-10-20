# Newly updated public training assets — static triage, 21 October 2026 IST

Fresh Kaggle API discovery sorted by dateRun/updated, not notebook titles or
votes. No GPU, untrusted source execution, label adoption or submission occurred.
Prior rejected contracts/checkpoints remain closed; locked0.957 is unchanged.

## Distinct V-JEPA release: not launch-qualified

[nurikw3/rsna-knee-vjepa21-checkpoints](https://www.kaggle.com/datasets/nurikw3/rsna-knee-vjepa21-checkpoints)
was updated19 October. Inventory contains five `f0_best.pt` through `f4_best.pt`
files, each363835140bytes, plus `finetune_model.py`, `infer.py` and vendored source.
Only the small source/provenance/license files were downloaded. Their exact
SHA256s are in `vjepa_static_receipt.json`; weights were not downloaded/hashed.

The released class enables the video encoder for training and has a spatial/
slice transformer finding head and layer-decayed optimizer groups. This is
materially different source from R6/Raptor, but a class definition does not
authenticate how the released checkpoints were actually trained. Training
labels, fold groups, selection history, compute and exact score remain unproven.

**Input integrity blocker:** `infer.py:55` passes a list of decoded DICOM headers
into `_ordered_datasets`, whose line36 calls `pydicom.dcmread` again on each
item. This branch is used for series longer than24 slices. The public
[pydicom API](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.filereader.dcmread.html)
expects a path/readable file or buffer, not an already decoded dataset. Static
control flow indicates this can fail; the outer lines104–105 catch the exception
and replace the series with `None`. If every view is missing, lines110–111 mark
one zero-filled view as present, allowing a prediction rather than failing closed.
This is a source-derived failure path, not a measured full-DICOM runtime result.

`audit_vjepa_source.py` checks the double-read AST and explicit fallback source;
it never imports or executes downloaded files. Its synthetic test distinguishes
the problematic decoded-object list from the ordinary file-list call.

The upstream manifest hashes files but names ref`main`, not a commit. Its MIT
code-license file and dataset CC0 metadata are not proof of all pretrained-weight
terms. No launch is justified until source/checkpoint identities, reader parity,
failure receipts, model terms, independent evaluation and compute are established.
Do not spend a GPU run merely because the dataset advertises five folds.

## Other training-asset leads

[haungjiawei/rsna-knee-train-v2-code-and-labels](https://www.kaggle.com/datasets/haungjiawei/rsna-knee-train-v2-code-and-labels):
downloaded README SHA256
`9b681e75d4e0741cefac12f1e38a7bf9749dd76bc2f0a5c97dcabab286c440ff`.
It describes DINOv2-small final-six-block tuning, but also gold58 gradient
weight5 and gold-based stopping. It is not a drop-in independent validation
recipe. Its weak-label table was not downloaded or used. No exact score verified.

[bliverrigpdoge/rsna-knee-trainkit](https://www.kaggle.com/datasets/bliverrigpdoge/rsna-knee-trainkit):
inventory has source, consensus/teacher tables and initialization weights; no
authenticated original R6 LR/state target filename or pool lineage established.
Search-list size22bytes was misleading: listed files include ~200MB weights.
No label/training adoption follows from this inventory.

[sarthakgupta1801/rsna-knee-px-s1](https://www.kaggle.com/datasets/sarthakgupta1801/rsna-knee-px-s1):
inventory includes index/meta/offset/side files and a17922813726byte JPEG bin.
It is not evidence of R6-native parity. The search result's size0 is not an empty
dataset certificate. No huge pixel cache was downloaded.

## Reproduce

```powershell
python experiments/public-asset-audit-2026-10-21/audit_vjepa_source.py
python -m unittest discover -s experiments/public-asset-audit-2026-10-21 -p "test_*.py"
```

This narrows a real new-reader lead and identifies a concrete fail-closed blocker.
It is research evidence, not new model accuracy or authorization for a new job.
Preserve attribution, source terms and OAI/NDA acknowledgement as applicable.
