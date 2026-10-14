import json
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / "upstream_train_knee.py").read_text(encoding="utf-8-sig")


def replace_once(old, new):
    global source
    if source.count(old) != 1:
        raise RuntimeError(f"source contract drift for: {old[:80]!r}")
    source = source.replace(old, new)


replace_once(
    '    vds = StudyWindows(HERE, gold_ids, id2row, labels, a.res, a.k_eval, train=False, norm=a.norm, **_roi_kw)',
    '    selection_ids = oof_ids if oof_ids else gold_ids\n'
    '    if set(selection_ids) & set(gold_ids):\n'
    '        raise RuntimeError("organizer gold entered checkpoint selection")\n'
    '    vds = StudyWindows(HERE, selection_ids, id2row, labels, a.res, a.k_eval, train=False, norm=a.norm, **_roi_kw)')
replace_once(
    '        for j, name in enumerate(LAB):\n'
    '            if len(set(Y[:, j].astype(int))) > 1:\n'
    '                aucs[name] = float(roc_auc_score(Y[:, j], P[:, j]))',
    '        for j, name in enumerate(LAB):\n'
    '            hard = (Y[:, j] >= 0.5).astype(np.int8)\n'
    '            if len(set(hard)) > 1:\n'
    '                aucs[name] = float(roc_auc_score(hard, P[:, j]))')
source = source.replace('"gold_auc"', '"selection_auc"')
source = source.replace('best_gold_auc', 'best_selection_auc')
source = source.replace('GOLD macro-AUC', 'SELECTION macro-AUC')
source = source.replace('gold-val', 'gold-eval-only')
source = source.replace('ids=np.array(gold_ids)', 'ids=np.array(selection_ids)')

prep = (root / "prepare_targets.py").read_text(encoding="utf-8")
eval_gold = (root / "evaluate_gold.py").read_text(encoding="utf-8")
run = '''
import subprocess, sys
cmd = [sys.executable, "-u", "train_knee.py",
       "--arch", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
       "--res", "384", "--epochs", "2", "--bs", "4", "--k", "12", "--k_eval", "24",
       "--grad_ckpt", "--topk", "2", "--workers", "2", "--tag", "oofcorr_coatnet384_f0",
       "--labels", "/kaggle/working/labels_oof_corrected.parquet",
       "--folds", "5", "--fold", "0", "--fold_file", "/kaggle/working/folds.json"]
print("RUN", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
'''
cells = [
    {"cell_type": "markdown", "metadata": {}, "source": [
        "# OOF-corrected CoAtNet-384 private screen\n",
        "Competition-only one-fold validation. The 58 organizer labels are evaluation-only. No submission capability.\n",
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
     "source": ["!pip install -q timm\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
     "source": (["from pathlib import Path\n", "Path('train_knee.py').write_text("]
                + [repr(source)] + [", encoding='utf-8')\n"])},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": prep.splitlines(True)},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": run.splitlines(True)},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
     "source": (["Path('evaluate_gold.py').write_text("] + [repr(eval_gold)] +
                [", encoding='utf-8')\n", "subprocess.run([sys.executable, '-u', 'evaluate_gold.py'], check=True)\n"])},
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
(root / "coatnet-oof-corrected-validation.ipynb").write_text(json.dumps(nb), encoding="utf-8")

