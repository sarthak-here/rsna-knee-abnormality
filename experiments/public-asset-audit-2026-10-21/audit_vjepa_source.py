"""Static AST inspection only: never import or execute downloaded model code."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]/'.audit/vjepa21-static'


def double_read_sites(source):
    tree = ast.parse(source)
    helpers = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if any(isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)
                   and x.func.attr == 'dcmread' for x in ast.walk(node)):
                helpers[node.name] = node
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id not in helpers:
            continue
        if node.args and isinstance(node.args[0], ast.ListComp):
            expression = node.args[0].elt
            if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Attribute) and expression.func.attr == 'dcmread':
                found.append({'caller_line': node.lineno, 'helper': node.func.id,
                              'helper_line': helpers[node.func.id].lineno,
                              'decoded_datasets_passed_to_reader_helper': True})
    return found


def audit(root):
    infer = (root/'infer.py').read_text()
    sites = double_read_sites(infer)
    assert sites == [{'caller_line': 55, 'helper': '_ordered_datasets', 'helper_line': 35,
                      'decoded_datasets_passed_to_reader_helper': True}]
    assert 'except Exception:\n                    stack = None' in infer
    assert 'if not present.any():\n        present[0] = True' in infer
    model = (root/'finetune_model.py').read_text()
    assert 'self.set_encoder_trainable(True)' in model and 'layer_decay=0.8' in model
    paths = [root/x for x in ('infer.py','finetune_model.py','source_manifest.json','LICENSE')]
    manifest = json.loads((root/'source_manifest.json').read_text())
    return {'dataset': 'nurikw3/rsna-knee-vjepa21-checkpoints', 'scope': 'STATIC_SOURCE_ONLY',
            'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'double_read_sites': sites, 'silent_reader_exception_to_missing_view': True,
            'all_missing_inputs_force_one_zero_view_present': True,
            'five_checkpoint_inventory_seen': True, 'weights_downloaded_or_hashed': False,
            'downloaded_code_executed': False, 'gpu_used': False, 'submission_created': False,
            'distinct_backbone_source': 'V-JEPA2.1 with spatial/slice transformer finding head',
            'encoder_trainable_in_released_class': True, 'training_execution_proven': False,
            'upstream_ref': manifest['ref'], 'upstream_commit_pinned': False,
            'inference_parity_verified': False, 'training_folds_and_targets_authenticated': False,
            'exact_public_score_verified': False, 'qualified_for_gpu_launch': False,
            'source_code_license_file': 'MIT; dataset metadata declares CC0, not proof of pretrained-weight terms',
            'pydicom_contract_reference': 'https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.filereader.dcmread.html'}


if __name__ == '__main__':
    result = audit(ROOT)
    (HERE/'vjepa_static_receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
