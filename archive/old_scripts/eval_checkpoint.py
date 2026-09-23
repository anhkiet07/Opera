"""Re-evaluate saved forecasting checkpoints on additional sampling views and merge into results/<tag>.json.

Usage: python eval_checkpoint.py --tags p0_quite_itr_s1 p1_aug_only_s1 --views variable:0.5,stride:2,stride:3
Rebuilds the model from the args stored in the results file, loads models/<tag>.pt, evaluates each view on
the test set with the same fixed seed the training scripts use, and adds the numbers under "robustness".
"""
import os
import sys
import json
import math
import argparse

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch

import utils
import sampling_views as sv
from data.parse import parse_datasets
from evaluation import evaluation
from models.quite import Model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--views', type=str, default=','.join(sv.UNSEEN_EVAL_VIEWS))
    ap.add_argument('--force', action='store_true', help='Re-evaluate views that are already present')
    a = ap.parse_args()
    views = [v for v in a.views.split(',') if v]

    for tag in a.tags:
        path = f'results/{tag}.json'
        res = json.load(open(path))
        todo = [v for v in views if a.force or v not in res.get('robustness', {})]
        if not todo:
            print(f'[skip] {tag}: views present'); continue
        args = argparse.Namespace(**res['args'])
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(getattr(args, 'gpu', '0'))
        utils.setup_seed(args.seed)
        if args.irr_emb:
            d = parse_datasets(args, patch_ts=True, max_ts=False)
        elif args.model in ('itransformer', 's_mamba'):
            d = parse_datasets(args, patch_ts=False, max_ts=True)
        else:
            d = parse_datasets(args, patch_ts=True, max_ts=True)
        args.ndim = d['input_dim']
        args.npatch = int(math.ceil((args.history - args.patch_size) / args.stride)) + 1
        args.scale_patch_size = args.patch_size / (args.history + args.pred_window)
        args.task = 'forecasting'
        model = Model(args).to(args.device)
        model.load_state_dict(torch.load(f'models/{tag}.pt', map_location=args.device))
        model.eval()
        with torch.no_grad():
            for view in todo:
                g = sv.make_generator(12345, args.device)
                r, var_mse = evaluation(model, d['test_dataloader'], d['n_test_batches'], view=view, gen=g)
                res.setdefault('robustness', {})[view] = {k: r[k] for k in ('mse', 'rmse', 'mae', 'mape')}
                res['robustness'][view]['mse_per_var'] = [float(x) for x in var_mse.detach().cpu().tolist()]
                print(f'{tag} [{view}] MSE {r["mse"]:.5f} MAE {r["mae"]:.5f}')
        json.dump(res, open(path, 'w'), indent=2)
        del model, d
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
