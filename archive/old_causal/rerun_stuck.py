"""Find Phase 3 forecasting runs that early-stopped on the initial constant-predictor plateau and rerun them
with a minimum-epoch guard, rebuilding each command line from the args stored in its results JSON.

A run is "stuck" when its best epoch is <= STUCK_BEST and it ran fewer than STUCK_EPOCHS epochs
(i.e. the validation loss never improved after the first few epochs and patience ended the run).

Usage: python rerun_stuck.py [--glob "results/p3*.json"] [--min_epochs 40] [--dry]
"""
import argparse
import glob
import json
import os
import subprocess
import sys

STUCK_BEST, STUCK_EPOCHS = 5, 30
PASS = ['dataset', 'history', 'patch_size', 'stride', 'hid_dim', 'nhead', 'nlayer', 'batch_size', 'lr', 'gpu',
        'epoch', 'patience', 'seed', 'n', 'mode', 'model', 'eval_views', 'run_tag', 'wandb_project',
        'causal_mode', 'causal_graph', 'causal_lambda', 'adaptive_query', 'train_view', 'n_views',
        'consistency_lambda', 'consistency_target', 'consistency_dist']
FLAGS = ['irr_emb']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='results/p3*.json')
    ap.add_argument('--min_epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=None, help='Override the stored patience')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    py = sys.executable
    for f in sorted(glob.glob(a.glob)):
        r = json.load(open(f))
        already_stuck = f.endswith('.stuck.json')
        early = r.get('best_epoch', 99) <= STUCK_BEST and r.get('epochs_run', 99) < STUCK_EPOCHS
        # constant predictor: the error is identical under every observation view
        mses = [v['mse'] for v in r.get('robustness', {}).values()]
        constant = len(mses) > 1 and (max(mses) - min(mses)) < 1e-4
        if not (early or constant or already_stuck):
            continue
        live = f.replace('.stuck.json', '.json')
        if already_stuck and os.path.exists(live):
            continue                      # a successful rerun already replaced it
        args = dict(r['args'])
        if a.patience is not None:
            args['patience'] = a.patience
        cmd = [py, 'train_forecasting.py', '--min_epochs', str(a.min_epochs)]
        for k in PASS:
            if k in args and args[k] is not None:
                cmd += ['-n' if k == 'n' else f'--{k}', str(args[k])]
        for k in FLAGS:
            if args.get(k):
                cmd.append(f'--{k}')
        print(f"[stuck] {r['run_tag']}: best_epoch={r['best_epoch']} epochs_run={r['epochs_run']} -> rerun")
        if a.dry:
            print(' '.join(cmd)); continue
        if not already_stuck:
            os.rename(f, f.replace('.json', '.stuck.json'))
        with open(f'logs/rerun_{r["run_tag"]}.log', 'w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        print(f"[done] {r['run_tag']}")


if __name__ == '__main__':
    main()
