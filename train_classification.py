"""Entrypoint: train a QuITE-equipped MTS backbone for IMTS classification.

Examples:
    # QuITE-PatchTST on P19 (binary classification → AUROC/AUPRC reported)
    python train_classification.py --dataset P19 --gpu 0 --epoch 100 \
        --batch_size 64 --lr 1e-3 --nhead 1 --nlayer 3 --hid_dim 32 \
        --patch_size 3.75 --stride 3.75 --model patchtst --irr_emb --mode quite

    # QuITE-PatchTST on PAM (8-class activity → Precision/Recall/F1 reported)
    python train_classification.py --dataset PAM --gpu 0 --epoch 100 \
        --batch_size 64 --lr 1e-3 --nhead 2 --nlayer 3 --hid_dim 128 \
        --patch_size 10 --stride 10 --model patchtst --irr_emb --mode quite
"""
import os
import math
import time
import argparse
import warnings
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser('QuITE Classification')

# ----- Training -----
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--patience', type=int, default=10,
                    help='Early-stopping patience (no val-loss improvement)')
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--logmode', type=str, default='a')

# ----- Dataset -----
parser.add_argument('--dataset', type=str, default='P12', choices=['P12', 'P19', 'PAM'])

# ----- Model & Embedding -----
parser.add_argument('--model', type=str, required=True,
                    choices=['patchtst', 'patchmixer', 'tmix', 'itransformer', 's_mamba', 'timexer'],
                    help='MTS backbone')
parser.add_argument('--mode', type=str, default='False',
                    choices=['quite', 'mean', 'mtand', 'add', 'concat', 'False'],
                    help="Embedding mode: 'quite'=QuITE (paper main), 'mean'/'mtand'=baselines (Table 5), "
                         "'add'/'concat'=non-irregular baselines, 'False'=vanilla backbone")
parser.add_argument('--irr_emb', action='store_true',
                    help='Use QuITE-style query-based embedding (paper main method)')
parser.add_argument('-hd', '--hid_dim', type=int, default=64)
parser.add_argument('--nhead', type=int, default=4,
                    help='Paper §6.1 standardizes QuITE-equipped backbones to 4 heads')
parser.add_argument('--nlayer', type=int, default=1)
parser.add_argument('-ps', '--patch_size', type=float, default=6)
parser.add_argument('--stride', type=float, default=6)
parser.add_argument('--dropout', type=float, default=0.1)

import wandb_utils
from sampling_views import DEFAULT_EVAL_VIEWS
wandb_utils.add_args(parser)

# ----- Sampling-invariant training (Phase 1) & robustness evaluation -----
parser.add_argument('--train_view', type=str, default='none',
                    help="Sampling view applied to training inputs: none | random:p | bursty:p | variable:p | mixed | a|b")
parser.add_argument('--n_views', type=int, default=1, help='Views per batch when train_view != none')
parser.add_argument('--consistency_lambda', type=float, default=0.0,
                    help='Weight of the cross-view consistency loss (0 = off)')
parser.add_argument('--consistency_target', type=str, default='repr', choices=['repr', 'pred'])
parser.add_argument('--consistency_dist', type=str, default='cos', choices=['cos', 'mse'])
parser.add_argument('--eval_views', type=str, default=','.join(DEFAULT_EVAL_VIEWS),
                    help='Comma-separated views for the final cross-sampling robustness table')
parser.add_argument('--n_folds', type=int, default=5, help='Number of Raindrop splits to run (max 5)')
parser.add_argument('--run_tag', type=str, default=None, help='Checkpoint / results file tag (auto if omitted)')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import json
import sampling_views as sv
from models.quite import Model
from data import DATA_ROOT
from utils import (
    get_data_split, getStats, getStats_static,
    tensorize_normalize_extract_feature_patch,
    tensorize_normalize_extract_feature_pam_patch,
    evaluate_model_patch, get_logger, makedirs,
)

args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

model_path = './models_classification/'
makedirs(model_path)


def layer_of_patches(n_patch):
    if n_patch == 1:
        return 1
    if n_patch % 2 == 0:
        return 1 + layer_of_patches(n_patch / 2)
    return layer_of_patches(n_patch + 1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Dataset-specific config (no split_idx — we use k+1 per fold)
DATASET_CFG = {
    'P12': dict(base_path=os.path.join(DATA_ROOT, 'P12'), d_static=9, variables_num=36, n_class=2, history=48),
    'P19': dict(base_path=os.path.join(DATA_ROOT, 'P19'), d_static=6, variables_num=34, n_class=2, history=60),
    'PAM': dict(base_path=os.path.join(DATA_ROOT, 'PAM'), d_static=0, variables_num=17, n_class=8, history=60),
}
cfg = DATASET_CFG[args.dataset]
base_path = cfg['base_path']
d_static = args.d_static = cfg['d_static']
variables_num = cfg['variables_num']
n_class = args.n_class = cfg['n_class']
args.history = cfg['history']

# Metric arrays
acc_arr, auprc_arr, auroc_arr = [], [], []
precision_arr, recall_arr, f1_arr = [], [], []

makedirs("logs/")
log_path = f"logs/{args.dataset}_{args.model}_{args.mode}_{args.hid_dim}hdims_{args.nlayer}nlayers_{args.nhead}nheads.log"
logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode)
logger.info(args)
_cons = f'_view-{args.train_view}_lam{args.consistency_lambda}' if args.train_view != 'none' else ''
args.run_tag = args.run_tag or args.wandb_name or f'{args.dataset}_{args.model}_{args.mode}{_cons}'
wandb_utils.init(args, name=args.run_tag,
                 tags=[args.dataset, 'classification', args.model, args.mode], group=args.dataset)
view_gen = sv.make_generator(0, torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
train_times, robust_folds, infer_ms = [], [], []

# Run 5 seeds × 5 splits
for k in range(args.n_folds):
    torch.manual_seed(k); torch.cuda.manual_seed(k); np.random.seed(k)
    split_idx = k + 1
    logger.info(f"Using split: {split_idx}")

    if args.dataset == 'P12':
        split_path = f'/splits/phy12_split{split_idx}.npy'
    elif args.dataset == 'P19':
        split_path = f'/splits/phy19_split{split_idx}_new.npy'
    elif args.dataset == 'PAM':
        split_path = f'/splits/PAM_split_{split_idx}.npy'

    Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path, dataset=args.dataset)
    print(len(Ptrain), len(Pval), len(Ptest), len(ytrain), len(yval), len(ytest))

    args.ndim = variables_num
    args.npatch = int(math.ceil((args.history - args.patch_size) / args.stride)) + 1
    args.patch_layer = layer_of_patches(args.npatch)
    args.scale_patch_size = args.patch_size / args.history
    args.task = 'classification'

    # Normalize & tensorize
    if args.dataset in ('P12', 'P19'):
        T, F = Ptrain[0]['arr'].shape
        D = len(Ptrain[0]['extended_static'])
        Ptrain_tensor = np.zeros((len(Ptrain), T, F))
        Ptrain_static_tensor = np.zeros((len(Ptrain), D))
        for i in range(len(Ptrain)):
            Ptrain_tensor[i] = Ptrain[i]['arr']
            Ptrain_static_tensor[i] = Ptrain[i]['extended_static']

        mf, stdf = getStats(Ptrain_tensor)
        ms, ss = getStats_static(Ptrain_static_tensor, dataset=args.dataset)

        Ptrain_tensor, Ptrain_mask_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor, maxlen = \
            tensorize_normalize_extract_feature_patch(Ptrain, ytrain, mf, stdf, ms, ss, args)
        Pval_tensor, Pval_mask_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor, maxlen = \
            tensorize_normalize_extract_feature_patch(Pval, yval, mf, stdf, ms, ss, args)
        Ptest_tensor, Ptest_mask_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor, maxlen = \
            tensorize_normalize_extract_feature_patch(Ptest, ytest, mf, stdf, ms, ss, args)

    elif args.dataset == 'PAM':
        mf, stdf = getStats(Ptrain)

        Ptrain_tensor, Ptrain_mask_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor, maxlen = \
            tensorize_normalize_extract_feature_pam_patch(Ptrain, ytrain, mf, stdf, args)
        Pval_tensor, Pval_mask_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor, maxlen = \
            tensorize_normalize_extract_feature_pam_patch(Pval, yval, mf, stdf, args)
        Ptest_tensor, Ptest_mask_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor, maxlen = \
            tensorize_normalize_extract_feature_pam_patch(Ptest, ytest, mf, stdf, args)

    args.maxlen = 0 if args.irr_emb else maxlen
    model = Model(args).to(args.device)
    logger.info(model)
    logger.info(f'parameters: {count_parameters(model)}')

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Batching strategy
    if args.dataset != 'PAM':
        # Upsample minority class for binary tasks
        idx_0 = np.where(ytrain == 0)[0]
        idx_1 = np.where(ytrain == 1)[0]
        expanded_idx_1 = np.concatenate([idx_1, idx_1, idx_1], axis=0)
        K0 = len(idx_0) // int(args.batch_size / 2)
        K1 = len(expanded_idx_1) // int(args.batch_size / 2)
        n_batches = np.min([K0, K1])
    else:
        all_indices = np.arange(len(ytrain))
        n_batches = len(all_indices) // args.batch_size

    best_val_epoch = 0
    best_loss_val = float('inf')
    save_time = None

    # Training loop
    for epoch in range(args.epoch):
        if epoch - best_val_epoch > args.patience:
            break

        model.train()
        if args.dataset != 'PAM':
            np.random.shuffle(expanded_idx_1); I1 = expanded_idx_1
            np.random.shuffle(idx_0); I0 = idx_0
        else:
            np.random.shuffle(all_indices)

        st = time.time()
        for n in range(n_batches):
            if args.dataset != 'PAM':
                idx0_b = I0[n * int(args.batch_size / 2):(n + 1) * int(args.batch_size / 2)]
                idx1_b = I1[n * int(args.batch_size / 2):(n + 1) * int(args.batch_size / 2)]
                idx = np.concatenate([idx0_b, idx1_b], axis=0)
            else:
                idx = all_indices[n * args.batch_size:(n + 1) * args.batch_size]

            P = Ptrain_tensor[idx].cuda()
            P_mask = Ptrain_mask_tensor[idx].cuda()
            P_static = Ptrain_static_tensor[idx].cuda() if d_static != 0 else None
            P_time = Ptrain_time_tensor[idx].cuda()
            y = ytrain_tensor[idx].cuda()

            optimizer.zero_grad()
            extra = {}
            if args.train_view == 'none' and args.consistency_lambda <= 0:
                outputs = model.classification(P, P_time, P_mask, P_static)
                loss = criterion(outputs, y)
            else:
                nv = max(args.n_views, 2 if args.consistency_lambda > 0 else 1)
                logits_l, z_l, task_l = [], [], []
                for _v in range(nv):
                    Pv, Mv = sv.apply_view(P, P_time, P_mask, args.train_view, view_gen)
                    lg, z = model.classification(Pv, P_time, Mv, P_static, return_repr=True)
                    logits_l.append(lg); z_l.append(z); task_l.append(criterion(lg, y))
                loss = torch.stack(task_l).mean()
                if args.consistency_lambda > 0:
                    if args.consistency_target == 'repr':
                        cons = sv.consistency_loss(z_l[0], z_l[1], args.consistency_dist)
                    else:
                        cons = sv.consistency_loss(torch.log_softmax(logits_l[0], -1), torch.log_softmax(logits_l[1], -1), 'mse')
                    loss = loss + args.consistency_lambda * cons
                    extra['train/consistency'] = cons.item()
                if nv >= 2:
                    extra.update(sv.repr_diagnostics(z_l[0], z_l[1]))
            loss.backward()
            optimizer.step()
        train_time = time.time() - st
        train_times.append(train_time)
        logger.info(f'training time per epoch :{train_time:.3f}s')

        # Validation
        model.eval()
        with torch.no_grad():
            st = time.time()
            out_val = evaluate_model_patch(
                model, Pval_tensor, Pval_mask_tensor, Pval_static_tensor, Pval_time_tensor,
                n_classes=n_class, batch_size=args.batch_size,
            )
            logger.info(f"val time per epoch :{time.time() - st:.2f}s")

            # Val loss on raw logits
            val_loss = criterion(out_val.cuda(), torch.from_numpy(yval.squeeze(1)).long().cuda())

            probs_val = torch.softmax(out_val, dim=1).detach().cpu().numpy()
            y_val_pred = np.argmax(probs_val, axis=1)
            acc_val = np.sum(yval.ravel() == y_val_pred.ravel()) / yval.shape[0]

            if n_class == 2:
                auc_val = roc_auc_score(yval, probs_val[:, 1])
                aupr_val = average_precision_score(yval, probs_val[:, 1])
                logger.info(
                    f"Validation: Epoch {epoch}, train_loss:{loss.item():.4f}, "
                    f"val_loss:{val_loss.item():.4f}, acc_val:{acc_val * 100:.2f}, "
                    f"aupr_val:{aupr_val * 100:.2f}, auc_val:{auc_val * 100:.2f}"
                )
            else:
                prec_val = precision_score(yval.ravel(), y_val_pred, average='macro')
                rec_val = recall_score(yval.ravel(), y_val_pred, average='macro')
                f1_val = f1_score(yval.ravel(), y_val_pred, average='macro')
                logger.info(
                    f"Validation: Epoch {epoch}, train_loss:{loss.item():.4f}, "
                    f"val_loss:{val_loss.item():.4f}, acc_val:{acc_val * 100:.2f}, "
                    f"prec_val:{prec_val * 100:.2f}, rec_val:{rec_val * 100:.2f}, "
                    f"f1_val:{f1_val * 100:.2f}"
                )

            metrics = {f'fold{k}/train_loss': loss.item(), f'fold{k}/val_loss': val_loss.item(),
                       f'fold{k}/val_acc': acc_val * 100, f'fold{k}/epoch_time_s': train_time, f'fold{k}/epoch': epoch}
            if n_class == 2:
                metrics.update({f'fold{k}/val_auroc': auc_val * 100, f'fold{k}/val_auprc': aupr_val * 100})
            else:
                metrics.update({f'fold{k}/val_f1': f1_val * 100})
            metrics.update({f'fold{k}/{kk}' if not kk.startswith('repr') else kk: vv for kk, vv in extra.items()})
            if torch.cuda.is_available():
                metrics['gpu/peak_mem_mb'] = torch.cuda.max_memory_allocated() / 2**20
            wandb_utils.log(metrics)

            if val_loss.item() < best_loss_val:
                best_val_epoch = epoch
                best_loss_val = val_loss.item()
                save_time = str(int(time.time()))
                torch.save(model.state_dict(), f"{model_path}{args.run_tag}_fold{k}.pt")

    # Testing
    if save_time is None:
        logger.warning(f"No model saved for fold {k}; skipping testing.")
        continue

    model_filename = f"{model_path}{args.run_tag}_fold{k}.pt"
    if not os.path.exists(model_filename):
        logger.error(f"Model file missing: {model_filename}")
        continue

    model.eval()
    model.load_state_dict(torch.load(model_filename))
    logger.info(f"Loaded model for testing: {model_filename}")

    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        st = time.time()
        out_test = evaluate_model_patch(
            model, Ptest_tensor, Ptest_mask_tensor, Ptest_static_tensor, Ptest_time_tensor,
            n_classes=n_class, batch_size=args.batch_size,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_ms.append(1000.0 * (time.time() - st) / max(Ptest_tensor.shape[0], 1))
        logger.info(f"test time per epoch :{time.time() - st:.2f}s")

        probs = torch.softmax(out_test, dim=1).detach().cpu().numpy()
        ypred = np.argmax(probs, axis=1)
        y_test_real = ytest.ravel()
        acc = np.sum(y_test_real == ypred) / y_test_real.shape[0]

        if n_class == 2:
            auc = roc_auc_score(y_test_real, probs[:, 1])
            aupr = average_precision_score(y_test_real, probs[:, 1])
            logger.info(f'Testing: AUROC = {auc * 100:.2f} | AUPRC = {aupr * 100:.2f} | Accuracy = {acc * 100:.2f}')
            auprc_arr.append(aupr * 100)
            auroc_arr.append(auc * 100)
        else:
            prec = precision_score(y_test_real, ypred, average='macro')
            rec = recall_score(y_test_real, ypred, average='macro')
            f1 = f1_score(y_test_real, ypred, average='macro')
            logger.info(f'Testing: PRECISION = {prec * 100:.2f} | RECALL = {rec * 100:.2f} | '
                        f'F1 = {f1 * 100:.2f} | Accuracy = {acc * 100:.2f}')
            precision_arr.append(prec * 100)
            recall_arr.append(rec * 100)
            f1_arr.append(f1 * 100)
        acc_arr.append(acc * 100)
        fold_test = {f'test/fold{k}/acc': acc * 100, f'test/fold{k}/best_epoch': best_val_epoch}
        if n_class == 2:
            fold_test.update({f'test/fold{k}/auroc': auc * 100, f'test/fold{k}/auprc': aupr * 100})
        else:
            fold_test.update({f'test/fold{k}/precision': prec * 100, f'test/fold{k}/recall': rec * 100,
                              f'test/fold{k}/f1': f1 * 100})
        wandb_utils.summary(fold_test)

        # Cross-sampling robustness of this fold's best model (test inputs viewed, labels untouched)
        rob = {}
        for view in [v.strip() for v in args.eval_views.split(',') if v.strip()]:
            g = sv.make_generator(12345, Ptest_tensor.device)
            Pv, Mv = sv.apply_view(Ptest_tensor, Ptest_time_tensor, Ptest_mask_tensor, view, g)
            out_v = evaluate_model_patch(model, Pv, Mv, Ptest_static_tensor, Ptest_time_tensor,
                                         n_classes=n_class, batch_size=args.batch_size)
            probs_v = torch.softmax(out_v, dim=1).detach().cpu().numpy()
            pred_v = np.argmax(probs_v, axis=1)
            r = {'acc': float(np.mean(y_test_real == pred_v) * 100)}
            if n_class == 2:
                r['auroc'] = float(roc_auc_score(y_test_real, probs_v[:, 1]) * 100)
                r['auprc'] = float(average_precision_score(y_test_real, probs_v[:, 1]) * 100)
            else:
                r['f1'] = float(f1_score(y_test_real, pred_v, average='macro') * 100)
            rob[view] = r
            logger.info(f'Robustness fold{k} [{view}]: ' + ', '.join(f'{a}={b:.2f}' for a, b in r.items()))
        robust_folds.append(rob)

# Aggregate
logger.info('------------------------------------------')
logger.info(f'args.dataset: {args.dataset}')
if n_class == 2:
    logger.info(f'AUPRC    = {np.mean(auprc_arr):.1f}±{np.std(auprc_arr):.1f}')
    logger.info(f'AUROC    = {np.mean(auroc_arr):.1f}±{np.std(auroc_arr):.1f}')
else:
    logger.info(f'PRECISION = {np.mean(precision_arr):.1f}±{np.std(precision_arr):.1f}')
    logger.info(f'RECALL    = {np.mean(recall_arr):.1f}±{np.std(recall_arr):.1f}')
    logger.info(f'F1        = {np.mean(f1_arr):.1f}±{np.std(f1_arr):.1f}')
logger.info(f'Accuracy  = {np.mean(acc_arr):.1f}±{np.std(acc_arr):.1f}')

final = {'test/acc_mean': np.mean(acc_arr), 'test/acc_std': np.std(acc_arr)}
if n_class == 2:
    final.update({'test/auroc_mean': np.mean(auroc_arr), 'test/auroc_std': np.std(auroc_arr),
                  'test/auprc_mean': np.mean(auprc_arr), 'test/auprc_std': np.std(auprc_arr)})
else:
    final.update({'test/f1_mean': np.mean(f1_arr), 'test/f1_std': np.std(f1_arr),
                  'test/precision_mean': np.mean(precision_arr), 'test/recall_mean': np.mean(recall_arr)})
if torch.cuda.is_available():
    final['peak_gpu_mem_mb'] = torch.cuda.max_memory_allocated() / 2**20
final['train_time_per_epoch_s'] = float(np.mean(train_times)) if train_times else None
final['infer_ms_per_sample'] = float(np.mean(infer_ms)) if infer_ms else None
final['n_params'] = count_parameters(model)
robust_mean = {}
if robust_folds:
    for view in robust_folds[0]:
        robust_mean[view] = {m: float(np.mean([rf[view][m] for rf in robust_folds])) for m in robust_folds[0][view]}
        robust_mean[view].update({m + '_std': float(np.std([rf[view][m] for rf in robust_folds])) for m in robust_folds[0][view]})
        for m, v in robust_mean[view].items():
            final[f'robust/{view}/{m}'] = v
        logger.info(f'Robustness mean [{view}]: ' + ', '.join(f'{a}={b:.2f}' for a, b in robust_mean[view].items()))
wandb_utils.summary(final)
try:
    import wandb
    if wandb.run is not None and robust_mean:
        cols = ['view'] + list(next(iter(robust_mean.values())).keys())
        tbl = wandb.Table(columns=cols, data=[[v] + [r[c] for c in cols[1:]] for v, r in robust_mean.items()])
        plots = {}
        for m in [c for c in cols[1:] if not c.endswith('_std')]:
            plots[f'robustness/{m}_by_view'] = wandb.plot.bar(tbl, 'view', m, title=f'Test {m.upper()} by observation view')
        wandb.log(plots)
except Exception as e:
    logger.info(f'wandb table skipped: {e}')
makedirs('results/')
with open(f'results/{args.run_tag}.json', 'w') as f:
    json.dump({'run_tag': args.run_tag,
               'args': {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))},
               'final': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in final.items()},
               'robustness_folds': robust_folds, 'robustness_mean': robust_mean}, f, indent=2)
logger.info(f'Saved results/{args.run_tag}.json')
wandb_utils.finish()
