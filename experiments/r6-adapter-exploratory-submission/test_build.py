import hashlib
import json
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

import torch
from build_notebook import build, ROOT, EPOCH_SHA, BASE_SHA


class BuildTests(unittest.TestCase):
    def test_patch_and_original_cells(self):
        nb, _, receipt = build()
        base = json.loads((ROOT / '.audit/final24h-helios-exact/blend-mine-in-private-lb-luck-to-all-upvoters.ipynb').read_text(encoding='utf-8'))
        for old, new in ((2, 3), (3, 4), (4, 5), (5, 6), (6, 7)):
            self.assertEqual(base['cells'][old]['source'], nb['cells'][new]['source'])
        self.assertEqual(receipt['base_notebook_sha256'], BASE_SHA)
        self.assertEqual(receipt['epoch3_sha256'], EPOCH_SHA)
        self.assertIn("'--max-fallback-fraction', '0.0'", ''.join(nb['cells'][9]['source']))
        self.assertIn('assert not r384_fail.any()', ''.join(nb['cells'][10]['source']))
        self.assertIn("A224 = 0.35", ''.join(nb['cells'][10]['source']))
        for c in nb['cells']:
            if c['cell_type'] == 'code' and not ''.join(c['source']).lstrip().startswith('%%'):
                compile(''.join(c['source']), '<notebook>', 'exec')

    def test_actual_setup_patch(self):
        nb, _, _ = build()
        source = ROOT / '.audit/rsna-knee-blend-code/r6src/compact_infer.py'
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            (dest / source.name).write_bytes(source.read_bytes())
            code = ''.join(nb['cells'][8]['source']).replace("Path('/kaggle/input')", repr(str(ROOT / '.audit/r6-lowrank-v1/output')))
            # Path object is required before rglob, not a plain string literal.
            code = code.replace(repr(str(ROOT / '.audit/r6-lowrank-v1/output'))+'.rglob', 'Path('+repr(str(ROOT / '.audit/r6-lowrank-v1/output'))+').rglob')
            env = dict(Path=Path, hashlib=hashlib, json=json, SRC=dest, WORK=dest,
                SOURCE_SHA={'compact_infer.py': hashlib.sha256(source.read_bytes()).hexdigest()})
            exec(code, env)
            patched = (dest / source.name).read_text()
            self.assertEqual(patched.count("if p.name == 'E_lrprior_s0_f0.pt':"), 1)
            compile(patched, '<patched-worker>', 'exec')
            self.assertEqual(env['_exploratory']['modified_fold_only'], 0)

    def test_real_final_checkpoint_loader(self):
        _, hook, _ = build()
        sys.path[:0] = [str(ROOT / 'experiments/r6-lowrank-spatial-adaptation'),
            str(ROOT / '.audit/rsna-knee-blend-code/r6src'),
            str(ROOT / '.audit/frontier-2026-10-17/goodpjw958/output/_own_env')]
        import compact_train as trainer
        torch.set_num_threads(2)
        assets = ROOT / '.audit/r6-strict-load'
        sd = torch.load(assets / 'E_lrprior_s0_f0.pt', map_location='cpu', weights_only=True)
        model = trainer.Net('convnext_small', None, crop=352,
            state_spec=json.loads((assets / 'state_spec.json').read_text()), in_chans=3,
            spatial=True, extent=True, syn_decomp=True)
        model.load_state_dict(sd, strict=True)
        epoch = ROOT / '.audit/r6-lowrank-v1/output/r6_lowrank_adaptation/results/epoch_3.pt'
        code = textwrap.dedent(hook).replace('__ADAPTER_PATH__', repr(str(epoch)))
        exec(code, dict(torch=torch, m=model, sd=sd, p=assets/'E_lrprior_s0_f0.pt'))
        final = torch.load(epoch, map_location='cpu', weights_only=True)['state_dict']
        self.assertEqual(set(final), set(model.state_dict()))
        self.assertTrue(all(torch.equal(t, model.state_dict()[k]) for k, t in final.items()))


if __name__ == '__main__':
    unittest.main()
