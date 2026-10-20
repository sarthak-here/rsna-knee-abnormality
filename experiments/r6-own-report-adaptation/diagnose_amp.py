"""Synthetic exact-head AMP scaling diagnostic; no optimizer or study data."""
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


def main():
    repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(repo / '.audit/rsna-knee-blend-code/r6src'),
                   str(repo / '.audit/frontier-2026-10-17/goodpjw958/output/_own_env')]
    import compact_train as trainer
    from train import tail
    torch.set_num_threads(2); torch.manual_seed(20261020)
    assets = repo / '.audit/r6-strict-load'
    spec = json.loads((assets / 'state_spec.json').read_text())
    model = trainer.Net('convnext_small', None, crop=352, state_spec=spec,
                        in_chans=3, spatial=True, extent=True, syn_decomp=True)
    model.load_state_dict(torch.load(assets/'E_lrprior_s0_f0.pt', weights_only=True), strict=True)
    model.eval().requires_grad_(False).cuda()
    model.enc.stages[3].requires_grad_(True).train()
    early = torch.randn(32, 384, 22, 22, device='cuda', dtype=torch.float16)
    meta = torch.zeros(32, 9, device='cuda'); meta[:, 1] = 1; meta[:, 3] = 1.5
    meta[:, 4:6] = 1; meta[:, 8] = 32
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        ref = tuple(x.float() for x in tail(model, early, meta))
    state_rng = torch.cuda.get_rng_state()
    results = []
    for scale in (65536., 1024., 1.):
        model.zero_grad(set_to_none=True); torch.cuda.set_rng_state(state_rng)
        with torch.autocast('cuda', dtype=torch.float16):
            z, sl, bl = tail(model, early, meta)
        ce = F.cross_entropy(sl[0, :4].float(), torch.zeros(4, device='cuda', dtype=torch.long))
        kl_z = F.binary_cross_entropy_with_logits(z.float(), ref[0].sigmoid()) - F.binary_cross_entropy_with_logits(ref[0], ref[0].sigmoid())
        logp = sl.float().log_softmax(-1).masked_fill(~model.state_valid[None], 0.)
        kl_s = F.kl_div(logp, ref[1].softmax(-1), reduction='none').sum(-1).mean()
        kl_b = F.binary_cross_entropy_with_logits(bl.float(), ref[2].sigmoid()) - F.binary_cross_entropy_with_logits(ref[2], ref[2].sigmoid())
        loss = ce + .5*kl_z + .5*kl_s + .1*kl_b
        (loss*scale).backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        results.append({'scale': scale, 'loss': float(loss.detach()),
                        'all_gradients_finite': all(bool(torch.isfinite(g).all()) for g in grads),
                        'gradient_tensors': len(grads)})
    report = {'synthetic_only': True, 'optimizer_steps': 0, 'results': results}
    if results[0]['all_gradients_finite'] or not all(r['all_gradients_finite'] for r in results[1:]):
        raise RuntimeError('AMP scaling diagnosis was not reproduced')
    Path(__file__).with_name('amp_diagnostic.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
