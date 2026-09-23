"""Re-evaluate saved P12/P19 classification checkpoints on additional sampling views (fold 0 only)
and merge into results/<tag>.json under robustness_folds[0] / robustness_mean.

Usage: python eval_checkpoint_cls.py --tags p0cls_quite_patchtst_P12 --views variable:0.5,stride:2,stride:3
"""
import os
import sys
import json
import math
import argparse

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

import sampling_views as sv
from data import DATA_ROOT
from models.quite import Model
from utils import (get_data_split, getStats, getStats_static,
                   tensorize_normalize_extract_feature_patch, evaluate_model_patch)

CFG = {'P12': dict(d_static=9, variables_num=36, n_class=2, history=48, split='/splits/phy12_split{k}.npy'),
       'P19': dict(d_static=6, variables_num=34, n_class=2, history=60, split='/splits/phy19_split{k}_new.npy')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--views', type=str, default=','.join(sv.UNSEEN_EVAL_VIEWS))
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    views = [v for v in a.views.split(',') if v]
    cache = {}

    for tag in a.tags:
        path = f'results/{tag}.json'
        res = json.load(open(path))
        rf = res['robustness_folds'][0]
        todo = [v for v in views if a.force or v not in rf]
        if not todo:
            print(f'[skip] {tag}: views present'); continue
        args = argparse.Namespace(**res['args'])
        args.device = torch.device('cuda:0')
        cfg = CFG[args.dataset]
        args.d_static, args.n_class, args.history = cfg['d_static'], cfg['n_class'], cfg['history']
        args.ndim = cfg['variables_num']
        args.npatch = int(math.ceil((args.history - args.patch_size) / args.stride)) + 1
        args.scale_patch_size = args.patch_size / args.history
        args.task = 'classification'
        key = (args.dataset, args.patch_size, args.stride)
        if key not in cache:
            torch.manual_seed(0); np.random.seed(0)
            base = os.path.join(DATA_ROOT, args.dataset)
            Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base, cfg['split'].format(k=1), dataset=args.dataset)
            T, F = Ptrain[0]['arr'].shape
            D = len(Ptrain[0]['extended_static'])
            tr = np.zeros((len(Ptrain), T, F)); st = np.zeros((len(Ptrain), D))
            for i in range(len(Ptrain)):
                tr[i] = Ptrain[i]['arr']; st[i] = Ptrain[i]['extended_static']
            mf, stdf = getStats(tr); ms, ss = getStats_static(st, dataset=args.dataset)
            Pt, Pm, Ps, Ptime, yt, maxlen = tensorize_normalize_extract_feature_patch(Ptest, ytest, mf, stdf, ms, ss, args)
            cache[key] = (Pt, Pm, Ps, Ptime, ytest.ravel())
        Pt, Pm, Ps, Ptime, y = cache[key]
        args.maxlen = 0
        model = Model(args).to(args.device)
        model.load_state_dict(torch.load(f'models_classification/{tag}_fold0.pt', map_location=args.device))
        model.eval()
        with torch.no_grad():
            for view in todo:
                g = sv.make_generator(12345, Pt.device)
                Pv, Mv = sv.apply_view(Pt, Ptime, Pm, view, g)
                out = evaluate_model_patch(model, Pv, Mv, Ps, Ptime, n_classes=args.n_class, batch_size=args.batch_size)
                probs = torch.softmax(out, dim=1).cpu().numpy(); pred = probs.argmax(1)
                r = {'acc': float((y == pred).mean() * 100),
                     'auroc': float(roc_auc_score(y, probs[:, 1]) * 100),
                     'auprc': float(average_precision_score(y, probs[:, 1]) * 100)}
                rf[view] = r
                res['robustness_mean'][view] = dict(r, **{m + '_std': 0.0 for m in r})
                print(f'{tag} [{view}] auroc={r["auroc"]:.2f} auprc={r["auprc"]:.2f}')
        json.dump(res, open(path, 'w'), indent=2)
        del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
