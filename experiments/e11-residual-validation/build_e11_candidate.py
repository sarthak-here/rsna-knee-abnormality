"""Build an attributed E11 residual candidate from the validated 0.936 notebook."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


E11_SHA256 = "df98518d0a8f7cc506b4b806da03c3fca18c8f8f743e813c5401759b45e0f1fe"
E13_SHA256 = "ad9f19af73bfdf4e49263c0e45060dc3cb239e1195039b26dc8c0a3a6bcd1a8a"


def cell_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def set_cell_text(cell: dict, text: str) -> None:
    cell["source"] = text


def replace_once(notebook: dict, old: str, new: str, label: str) -> None:
    matches = []
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        if old in cell_text(cell):
            matches.append(index)
    if len(matches) != 1:
        raise RuntimeError(f"{label}: expected one match, found {len(matches)}")
    index = matches[0]
    set_cell_text(notebook["cells"][index], cell_text(notebook["cells"][index]).replace(old, new, 1))


def replace_last(notebook: dict, old: str, new: str, label: str) -> None:
    locations = []
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        text = cell_text(cell)
        start = 0
        while True:
            found = text.find(old, start)
            if found < 0:
                break
            locations.append((index, found))
            start = found + 1
    if not locations:
        raise RuntimeError(f"{label}: no match")
    index, offset = locations[-1]
    text = cell_text(notebook["cells"][index])
    set_cell_text(notebook["cells"][index], text[:offset] + new + text[offset + len(old) :])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    notebook = copy.deepcopy(json.loads(args.source.read_text(encoding="utf-8")))

    replace_once(
        notebook,
        f"_RAD_E13_HEADS_SHA256 = '{E13_SHA256}'\n",
        (
            f"_RAD_E13_HEADS_SHA256 = '{E13_SHA256}'\n"
            f"_RAD_E11_PUBLIC_HEADS_SHA256 = '{E11_SHA256}'\n"
        ),
        "checkpoint hash",
    )

    old_files = (
        "    files = {_RAD_ENCODER_SHA256: ASSET / 'resnet-50-radimagenet-marwan/ResNet50.pt', "
        "_RAD_REFERENCE_HEADS_SHA256: ASSET / 'rsna-knee-e9-radimagenet-heads-v15/v52_radimagenet_heads.pt', "
        "_RAD_E13_HEADS_SHA256: ASSET / 'kernel-sources/rsna-knee-e13-train/rsna_rad_e11/v52_e11_heads.pt'}"
    )
    new_files = (
        "    files = {_RAD_ENCODER_SHA256: ASSET / 'resnet-50-radimagenet-marwan/ResNet50.pt', "
        "_RAD_REFERENCE_HEADS_SHA256: ASSET / 'rsna-knee-e9-radimagenet-heads-v15/v52_radimagenet_heads.pt', "
        "_RAD_E13_HEADS_SHA256: ASSET / 'kernel-sources/rsna-knee-e13-train/rsna_rad_e11/v52_e11_heads.pt', "
        "_RAD_E11_PUBLIC_HEADS_SHA256: ASSET / 'rsna-knee-e11-diverse-heads-v20/v52_e11_heads.pt'}"
    )
    replace_once(notebook, old_files, new_files, "checkpoint inventory")

    loader_anchor = "@_rad_torch.inference_mode()\ndef _rad_encode"
    public_loader = """def _rad_load_e11_public_heads(device):
    heads_path = _rad_find_file('v52_e11_heads.pt', _RAD_E11_PUBLIC_HEADS_SHA256)
    payload = _rad_torch.load(heads_path, map_location='cpu', weights_only=False)
    expected = {'version': 'e11-radimagenet-resnet50-diverse-1', 'targets': _RAD_LABELS, 'encoder_sha256': _RAD_ENCODER_SHA256, 'slots': [list(slot) for slot in _RAD_E11_SLOTS], 'crop_mm': _RAD_E11_CROP_MM, 'img': _RAD_E13_IMG, 'slices_per_plane': _RAD_E13_CACHE_SLICES, 'feature': 'global_average_pool'}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f'public E11 head contract drift for {key}')
    folds = payload.get('folds')
    if not isinstance(folds, list) or len(folds) != 5:
        raise RuntimeError('public E11 bundle requires exactly five heads')
    if sorted((int(record.get('fold', -1)) for record in folds)) != list(range(5)):
        raise RuntimeError('public E11 fold identity drift')
    heads = []
    for record in folds:
        head = _RadHead().to(device).eval()
        head.load_state_dict(record['state_dict'], strict=True)
        heads.append(head)
    return (heads, str(heads_path))

@_rad_torch.inference_mode()
def _rad_encode"""
    replace_once(notebook, loader_anchor, public_loader, "public E11 loader")

    replace_once(
        notebook,
        "    e13_heads, _ = _rad_load_e13_heads(device)\n",
        (
            "    e13_heads, _ = _rad_load_e13_heads(device)\n"
            "    e11_public_heads, _ = _rad_load_e11_public_heads(device)\n"
        ),
        "head loading",
    )

    pass2_anchor = (
        "    pass2_probability = _rad_np.mean(_rad_np.stack(pass2_predictions), axis=0)\n"
        "    pass2_rank = _rad_rank_columns(pass2_probability)\n"
    )
    pass2_new = pass2_anchor + (
        "    e11_public_predictions = [\n"
        "        _rad_predict_head(head, features, token_mask, device)\n"
        "        for head in e11_public_heads\n"
        "    ]\n"
        "    e11_public_probability = _rad_np.mean(\n"
        "        _rad_np.stack(e11_public_predictions), axis=0\n"
        "    )\n"
        "    e11_public_rank = _rad_rank_columns(e11_public_probability)\n"
        "    del e11_public_predictions, e11_public_probability\n"
    )
    replace_once(notebook, pass2_anchor, pass2_new, "shared-feature E11 inference")

    save_anchor = (
        "    _rad_validate(\n"
        "        final,\n"
        "        expected_ids,\n"
        "    )\n\n"
        "    final.to_csv(\n"
        "        primary,\n"
        "        index=False,\n"
        "    )"
    )
    vote_block = (
        "    # Public E11 residual: retain only targets stable across 100 outer-fold runs.\n"
        "    _e11_residual_alpha = {\n"
        "        'ACL': 0.35,\n"
        "        'Medial OA': 0.25,\n"
        "        'Lateral OA': 0.80,\n"
        "        'PF OA': 0.50,\n"
        "        'Synovitis': 0.50,\n"
        "        'Contusion': 0.60,\n"
        "    }\n"
        "    _final_rank = _rad_rank_columns(final[_RAD_LABELS].to_numpy())\n"
        "    for _target, _alpha in _e11_residual_alpha.items():\n"
        "        _index = _RAD_LABELS.index(_target)\n"
        "        final[_target] = (\n"
        "            (1.0 - _alpha) * _final_rank[:, _index]\n"
        "            + _alpha * e11_public_rank[:, _index]\n"
        "        )\n"
        "    final[_RAD_LABELS] = _rad_rank_columns(final[_RAD_LABELS].to_numpy())\n"
        "    globals()['E11_RESIDUAL_APPLIED'] = tuple(sorted(_e11_residual_alpha))\n"
        "    print(\n"
        "        '[E11 residual] applied to: ' + ', '.join(sorted(_e11_residual_alpha)),\n"
        "        flush=True,\n"
        "    )\n\n"
        + save_anchor
    )
    replace_last(notebook, save_anchor, vote_block, "post-calibration residual")

    notebook["cells"].insert(
        1,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": (
                "## Public E11 residual validation\n\n"
                "This private validation fork preserves the attributed 0.936 swarm and adds "
                "the public antoinegg1/rsna-knee-e11-diverse-heads-v20 RadImageNet heads. "
                "Those heads use complementary non-fat-suppressed planes and are evaluated on "
                "features already computed by the parent notebook. Only targets supported by "
                "grouped OOF validation and 100 randomized outer-fold checks receive a vote. "
                "The source dataset declares CC-BY-NC-SA-4.0.\n"
            ),
        },
    )
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    notebook.setdefault("metadata", {})["e11_residual_validation"] = {
        "source_notebook": "cemalata/rsna-knee-swarm-0-936-repro",
        "source_public_head": "antoinegg1/rsna-knee-e11-diverse-heads-v20",
        "source_public_head_sha256": E11_SHA256,
        "targets": ["ACL", "Medial OA", "Lateral OA", "PF OA", "Synovitis", "Contusion"],
        "offline_protocol": "extended grouped OOF ladder plus 100 randomized five-fold meta-validation runs",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
