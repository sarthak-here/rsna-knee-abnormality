"""Audit the embedded S6 runner without executing untrusted notebook code."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import zlib
from pathlib import Path


EXPECTED_NOTEBOOK_SHA256 = (
    "50ded2186d86e172e85a8928e8b93b3cdd0c9112117d2267ac059e651793a003"
)
EXPECTED_RUNNER_SHA256 = (
    "2e8c4dca6773588ee7a4246ba048e667c1e2516be1ec43a8b6e0269b5fa62c96"
)
EXPECTED_MODEL_SHA256 = (
    "0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12"
)
EXPECTED_ANATOMY_SHA256 = (
    "5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e"
)
EXPECTED_INFER_SHA256 = (
    "34f433fe72b732543813165c3397ec80cef01b0d49b25b7b22668714f0b0ef4a"
)
EXPECTED_MANIFEST_SHA256 = (
    "ed3c6d8d2e69ae08c4a97eab3c860a9ec20c5c69484e35de6a3119d396d731cd"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def assignment(tree: ast.AST, name: str):
    matches = []

    def collect(target: ast.expr, value) -> None:
        if isinstance(target, ast.Name):
            if target.id == name:
                matches.append(value)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            if not isinstance(value, (tuple, list)) or len(target.elts) != len(value):
                raise RuntimeError("literal destructuring mismatch")
            for child, child_value in zip(target.elts, value):
                collect(child, child_value)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        for target in targets:
            collect(target, value)
    if len(matches) != 1:
        raise RuntimeError(f"expected one literal assignment for {name}, got {len(matches)}")
    return matches[0]


def decode_runner(notebook: Path) -> tuple[bytes, str]:
    raw_notebook = notebook.read_bytes()
    if sha256_bytes(raw_notebook) != EXPECTED_NOTEBOOK_SHA256:
        raise RuntimeError("source notebook SHA-256 mismatch")
    package = json.loads(raw_notebook)
    source_cells = [
        "".join(cell.get("source", []))
        for cell in package.get("cells", [])
        if "S6_SRC_B64" in "".join(cell.get("source", []))
    ]
    if len(source_cells) != 1:
        raise RuntimeError(f"expected one S6 source cell, got {len(source_cells)}")
    outer_tree = ast.parse(source_cells[0])
    encoded = assignment(outer_tree, "S6_SRC_B64")
    runner = zlib.decompress(base64.b64decode(encoded, validate=True))
    if sha256_bytes(runner) != EXPECTED_RUNNER_SHA256:
        raise RuntimeError("decoded S6 runner SHA-256 mismatch")
    return runner, source_cells[0]


def audit(notebook: Path, manifest_path: Path) -> dict:
    runner, wrapper = decode_runner(notebook)
    text = runner.decode("utf-8")
    tree = ast.parse(text)
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("checkpoint manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    model = assignment(tree, "B_MODEL")
    geometry = assignment(tree, "B_GEOMETRY")
    source_sha = assignment(tree, "SOURCE_SHA")
    anatomy_file, anatomy_sha = assignment(tree, "ANATOMY_FILE"), assignment(
        tree, "ANATOMY_SHA256"
    )
    w_a, w_b = assignment(tree, "W_A"), assignment(tree, "W_B")

    checks = {
        "model_sha256": model["sha256"] == EXPECTED_MODEL_SHA256,
        "train_studies": model["expect_train"] == 4263,
        "all_data_checkpoint": model["file"] == "E_prior_s42_fall.pt",
        "anatomy_file": anatomy_file == "anatomy.pt",
        "anatomy_sha256": anatomy_sha == EXPECTED_ANATOMY_SHA256,
        "compact_infer_sha256": source_sha.get("compact_infer.py")
        == EXPECTED_INFER_SHA256,
        "geometry": geometry
        == {
            "backbone": "convnext_small",
            "size": 320,
            "crop": 288,
            "views": "v4",
            "slices": 3,
            "phys_mm": 3.5,
            "eval_stride": 1,
            "chunk": 192,
            "item_timeout": 900,
        },
        "s6_runner_is_solo": (w_a, w_b) == (0.0, 1.0),
        "hard_error_receipt": "k1_error.json" in text,
        "success_receipt": "k1_receipt.json" in text,
        "submission_cleanup": "def remove_submission()" in text,
        "gold_exclusion_declaration": manifest.get("gold58_in_training") is False,
        "public_score_declaration": "0.950 public" in text,
        "manifest_public_score": manifest.get("public_lb") == 0.950,
        "manifest_model_identity": manifest.get("sha256") == EXPECTED_MODEL_SHA256,
        "outer_wrapper_discards_s6_csv": "_s6_solo_discarded.csv" in wrapper,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"S6 audit failed: {failed}")

    return {
        "status": "AUDIT_PASS",
        "notebook": str(notebook),
        "notebook_sha256": sha256_file(notebook),
        "decoded_runner_sha256": sha256_bytes(runner),
        "checkpoint_sha256": model["sha256"],
        "checkpoint_manifest_sha256": sha256_file(manifest_path),
        "anatomy_sha256": anatomy_sha,
        "compact_infer_sha256": source_sha["compact_infer.py"],
        "train_studies": model["expect_train"],
        "gold_studies_in_training": False,
        "public_auc_declared": 0.950,
        "geometry": geometry,
        "checks": checks,
        "runner_executed": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.notebook, args.manifest)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
