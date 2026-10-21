"""Build a separately frozen rank8 pilot from pinned, audited runtime helpers."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "r6-own-report-adaptation"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, text):
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("runtime transform anchor mismatch: " + old[:70])
    return text.replace(old, new, 1)


def training_source():
    if sha(OLD / "train.py") != "c5287355e800e2f255ca37873038f49342f27705782bdeb1b646b65d288fea66":
        raise ValueError("audited runtime source drift")
    text = (OLD / "train.py").read_text(encoding="utf-8")
    text = replace_once(text, "SEED = 20261020", "SEED = 20261022\nfrom adapters import install, assert_frozen")
    text = replace_once(text,
        '    candidate.enc.stages[3].requires_grad_(True)\n    names = [n for n, p in candidate.named_parameters() if p.requires_grad]\n    if len(names) != 31 or any(not n.startswith("enc.stages.3.") for n in names):\n        raise ValueError("trainable parameter scope changed")',
        '    names = install(candidate)\n    assert_frozen(reference, candidate)\n'
        '    initial_adapters = {n: candidate.state_dict()[n].detach().clone() for n in names}')
    text = replace_once(text, 'candidate.eval(); candidate.enc.stages[3].train()', 'candidate.eval()  # Frozen source including DropPath; linear adapters have no train/eval state.')
    text = replace_once(text,
        '    changed = [k for k, v in reference.state_dict().items() if not torch.equal(v, candidate.state_dict()[k])]\n    if any(not k.startswith("enc.stages.3.") for k in changed):\n        raise RuntimeError("frozen head/prefix/buffer changed")',
        '    adapter_names = assert_frozen(reference, candidate)\n'
        '    changed = [n for n in adapter_names if not torch.equal(initial_adapters[n], candidate.state_dict()[n])]\n'
        '    if not changed:\n'
        '        raise RuntimeError("no adapter updated after training")')
    text = replace_once(text, '"gold_evaluation": False,',
        '"gold_evaluation": False, "architecture": "final_stage_fc2_rank8", "adapter_rank": 8, "trainable_parameters": 92160,')
    text = replace_once(text, '("train.py", "prepare_splits.py", "report_rules.py")',
        '("train.py", "prepare_splits.py", "report_rules.py", "adapters.py", "excluded_previous_split.json")')
    text = replace_once(text, '            teacher = tuple(v.float().detach() for v in tail(reference, early, meta))',
        '            teacher = tuple(v.float().detach() for v in tail(reference, early, meta))\n'
        '            if i == 0:\n'
        '                initial_candidate = tuple(v.float() for v in tail(candidate, early, meta))\n'
        '                if not all(torch.equal(x, y) for x, y in zip(teacher, initial_candidate)):\n'
        '                    raise RuntimeError("zero-init adapter changed initial reader output")')
    text = replace_once(text, '"state_dict": candidate.cpu().state_dict(), "optimizer": optimizer.state_dict(),',
        '"state_dict": candidate.cpu().state_dict(), "adapter_rank": 8, "initial_checkpoint_sha256": sha(checkpoint), "optimizer": optimizer.state_dict(),')
    text = replace_once(text, '"training": training_log, "changed_tensors": changed, "reader_failures": 0,',
        '"training": training_log, "changed_tensors": changed, "original_tensors_unchanged": True, "adapter_parameters": 92160, "reader_failures": 0,')
    text = replace_once(text, 'OWN_REPORT_ADAPTATION_COMPLETE_NO_SUBMISSION', 'R6_LOWRANK_ADAPTATION_COMPLETE_NO_SUBMISSION')
    text = replace_once(text, '"""Bounded own-report adaptation of actual R6 final encoder stage. No submission."""',
        '"""New rank8 spatial-backbone adaptation; original tensors remain frozen. No submission."""')
    compile(text, "generated_rank8_training", "exec")
    return text


def build():
    previous = json.loads((OLD / "splits.json").read_text())
    excluded = {"source_sha256": sha(OLD / "splits.json"),
                "report_groups": sorted(r["report_group"] for k in ("fit", "tune", "confirmation") for r in previous[k])}
    put(HERE / "excluded_previous_split.json", json.dumps(excluded, indent=2) + "\n")
    (HERE / "report_rules.py").write_bytes((OLD / "report_rules.py").read_bytes())
    put(HERE / "train.py", training_source())
    sys.path.insert(0, str(HERE))
    from prepare_splits import prepare
    repo = HERE.parents[1]
    split = prepare(repo / ".audit/competition-metadata/train.csv", repo / ".audit/r6-strict-load/state_spec.json")
    if split["status"] != "READY":
        raise ValueError("new split coverage rejection; do not launch")
    put(HERE / "splits.json", json.dumps(split, indent=2) + "\n")
    original = json.loads((OLD / "own-report-adaptation.ipynb").read_text(encoding="utf-8"))
    if sha(OLD / "own-report-adaptation.ipynb") != "93ee2bf92236862d89736adbd330b4081d00e10da9799ab22b6c4929940b1bbe":
        raise ValueError("audited asset notebook changed")
    source = lambda cell: "".join(cell["source"])
    stage = source(original["cells"][2])
    launch = source(original["cells"][3]).replace("OWN_REPORT", "R6_LOWRANK").replace("own_report_", "r6_lowrank_")
    files = {n: (HERE / n).read_text(encoding="utf-8") for n in
             ("adapters.py", "train.py", "prepare_splits.py", "report_rules.py", "splits.json", "excluded_previous_split.json")}
    for name in ("postprocess.py", "strict_reconstruct.py"):
        files[name] = (HERE.parent / "r6-fold0-warmstart" / name).read_text(encoding="utf-8")
    setup = "import time, json, sys\nfrom pathlib import Path\nNB_T0=time.time()\nOWN=Path('/kaggle/working/r6_lowrank_adaptation')\nOWN.mkdir(exist_ok=False)\n"
    setup += "FILES=" + repr(files) + "\nfor name, content in FILES.items():\n    (OWN/name).write_text(content)\n"
    for code in (setup, stage, launch):
        compile(code, "private_rank8_cell", "exec")
    code_cell = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(True)}
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": original["metadata"], "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# R6 rank8 spatial adaptation\nPrivate report-proxy pilot; no submission. Original R6/source/localization attribution: Heliosli and bundled upstream authors. Preserve model terms and OAI/NDA acknowledgement. No new external data or OrthoFoundation reader.\n"]},
        code_cell(setup), code_cell(stage), code_cell(launch)]}
    path = HERE / "r6-lowrank-adaptation.ipynb"
    put(path, json.dumps(notebook, indent=1))
    metadata = json.loads((OLD / "kernel-metadata.json").read_text())
    metadata.update(id="sarthaksharma14/rsna-knee-r6-lowrank-spatial-adaptation",
                    title="RSNA Knee R6 Lowrank Spatial Adaptation", code_file=path.name)
    put(HERE / "kernel-metadata.json", json.dumps(metadata, indent=2) + "\n")
    receipt = {"notebook_sha256": sha(path), "split_sha256": sha(HERE / "splits.json"),
               "files_sha256": {n: hashlib.sha256(v.encode()).hexdigest() for n, v in files.items()},
               "excluded_previous_groups": len(excluded["report_groups"]),
               "counts": {k: len(split[k]) for k in ("fit", "tune", "confirmation")},
               "fit_state_counts": split["fit_state_counts"], "submission_launcher_included": False,
               "training_launched": False}
    put(HERE / "build_receipt.json", json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    build()
