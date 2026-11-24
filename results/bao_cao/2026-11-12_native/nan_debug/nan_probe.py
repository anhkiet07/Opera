import sys, runpy, math, torch, torch.nn.functional as F
sys.path.insert(0, '/home/vmo/Opera')
import src.models.irregular_baselines as ib
step = {'n': 0, 'hist': []}
orig_fwd = ib.WarpformerEncoder.forward
def fwd(self, batch, return_repr=False, obs_descriptor=None):
    x, m, t, obs = ib._flatten(batch)
    t = ib._time_scale(batch, t)
    v = self.value_enc((x * m).unsqueeze(-1)); v = (v + self.var_emb) * m.unsqueeze(-1)
    tok = v.sum(2) / m.sum(2, keepdim=True).clamp(min=1.0) + self.time_proj(self.temb(t))
    tok = tok * obs.unsqueeze(-1)
    s = self.score(tok).squeeze(-1)
    dg = F.softplus(s) * obs
    dg = dg.clone(); dg[:, 0] = 0.0
    gmax = (torch.cumsum(dg, -1) * obs).amax(1)
    pos = gmax[gmax > 0]
    nobs = obs.sum(1)
    if self.training:
        step['last'] = dict(score_min=s[obs > 0].min().item() if (obs > 0).any() else float('nan'),
                            score_max=s[obs > 0].max().item() if (obs > 0).any() else float('nan'),
                            gmax_min=pos.min().item() if pos.numel() else float('nan'),
                            n_gmax0=int((gmax == 0).sum()), nobs_min=int(nobs.min()),
                            tok_absmax=tok.abs().max().item())
    out = orig_fwd(self, batch, return_repr, obs_descriptor)
    p = out[0] if isinstance(out, tuple) else out
    if self.training and not torch.isfinite(p).all():
        print('[probe] NONFINITE OUTPUT at step', step['n'], step['last'], flush=True)
    return out
ib.WarpformerEncoder.forward = fwd
orig_step = torch.optim.Adam.step
def ostep(self, *a, **k):
    step['n'] += 1
    gn = 0.0; bad = []
    for g in self.param_groups:
        for p in g['params']:
            if p.grad is not None:
                gnp = p.grad.norm().item(); gn += gnp ** 2
                if not math.isfinite(gnp): bad.append(tuple(p.shape))
    gn = gn ** .5
    step['hist'].append(gn)
    if step['n'] % 500 == 0:
        h = step['hist'][-500:]
        print(f"[probe] step {step['n']} gradnorm median {sorted(h)[250]:.3g} max {max(h):.3g} | {step.get('last')}", flush=True)
    if bad or not math.isfinite(gn):
        print(f"[probe] NONFINITE GRAD at step {step['n']} gn={gn} bad_params={bad[:6]} prev_gn={step['hist'][-6:-1]} fwd={step.get('last')}", flush=True)
        sys.exit(3)
    r = orig_step(self, *a, **k)
    for g in self.param_groups:
        for p in g['params']:
            if not torch.isfinite(p).all():
                print(f"[probe] NONFINITE PARAM after step {step['n']} shape={tuple(p.shape)} gn={gn} fwd={step.get('last')}", flush=True); sys.exit(4)
    return r
torch.optim.Adam.step = ostep
sys.argv = ['scripts/train.py'] + sys.argv[1:]
runpy.run_path('/home/vmo/Opera/scripts/train.py', run_name='__main__')
