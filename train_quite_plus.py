"""Entrypoint: train QuITE++ for IMTS forecasting.

Example:
    python train_quite_plus.py --dataset physionet --history 24 \
        --patch_size 6 --stride 6 --hid_dim 64 --nhead 4 --nlayer 2 \
        --seed 1 --gpu 0
"""
import os
import sys
import time
import math
import datetime
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from random import SystemRandom

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser('QuITE++ Forecasting')

# ----- Training -----
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--logmode', type=str, default='a')
parser.add_argument('--load', type=str, default=None,
                    help='Experiment ID to resume; if None, a fresh one is generated')

# ----- Dataset -----
parser.add_argument('--dataset', type=str, default='physionet',
                    choices=['physionet', 'mimic', 'ushcn', 'activity'])
parser.add_argument('--history', type=int, default=24,
                    help='Historical window length (hours / months / ms by dataset)')
parser.add_argument('--pred_window', type=int, default=1)
parser.add_argument('-n', type=int, default=int(1e8),
                    help='Cap on dataset size (debug); when < 12000 a debug log file is used')
parser.add_argument('--quantization', type=float, default=0.0,
                    help='Time quantization (PhysioNet only)')

# ----- Model (QuITE++) -----
parser.add_argument('-hd', '--hid_dim', type=int, default=64)
parser.add_argument('--nhead', type=int, default=4)
parser.add_argument('--nlayer', type=int, default=2)
parser.add_argument('-ps', '--patch_size', type=float, default=24)
parser.add_argument('--stride', type=float, default=24)
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
parser.add_argument('--run_tag', type=str, default=None, help='Checkpoint / results file tag (auto if omitted)')

args = parser.parse_args()
args.npatch = int(np.ceil((args.history - args.patch_size) / args.stride)) + 1
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch.optim as optim
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

import json
import utils
import sampling_views as sv
from data.parse import parse_datasets
from evaluation import compute_all_losses, evaluation
from models.quite_plus import Model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def layer_of_patches(n_patch):
    if n_patch == 1:
        return 1
    if n_patch % 2 == 0:
        return 1 + layer_of_patches(n_patch / 2)
    return layer_of_patches(n_patch + 1)


if __name__ == '__main__':
    utils.setup_seed(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.PID = os.getpid()
    print("PID, device:", args.PID, args.device)

    experimentID = args.load or int(SystemRandom().random() * 100000)

    input_command = sys.argv
    ind = [i for i, a in enumerate(input_command) if a == "--load"]
    if len(ind) == 1:
        i = ind[0]
        input_command = input_command[:i] + input_command[(i + 2):]
    input_command = " ".join(input_command)

    data_obj = parse_datasets(args, patch_ts=True, max_ts=True)
    args.ndim = data_obj["input_dim"]
    args.npatch = int(math.ceil((args.history - args.patch_size) / args.stride)) + 1
    args.patch_layer = layer_of_patches(args.npatch)
    args.scale_patch_size = args.patch_size / (args.history + args.pred_window)
    args.task = 'forecasting'

    model = Model(args).to(args.device)

    if args.n < 12000:
        log_path = f"logs/{args.dataset}_quite_plus_debug.log"
    else:
        log_path = (
            f"logs/{args.dataset}_quite_plus_{args.hid_dim}hdims_"
            f"{args.nlayer}nlayers_{args.npatch}npatch_{args.patch_size}patchsize_"
            f"{args.history}history_{args.pred_window}pred_{args.nhead}nhead_"
            f"{args.lr}lr_{args.seed}seed.log"
        )
    utils.makedirs("logs/")
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode)
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(input_command)
    logger.info(args)
    logger.info(model)
    logger.info(f'parameters:{count_parameters(model)}')
    args.run_tag = args.run_tag or args.wandb_name or f'{args.dataset}_h{args.history}_quite_plus_s{args.seed}'
    wandb_utils.init(args, name=args.run_tag, tags=[args.dataset, 'forecasting', 'quite_plus'],
                     group=f'{args.dataset}_h{args.history}')
    wandb_utils.summary({'n_params': count_parameters(model)})
    view_gen = sv.make_generator(args.seed, args.device)
    eval_gen_seed = 12345
    ckpt_path = f'./models/{args.run_tag}.pt'
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_times = []

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_batches = data_obj["n_train_batches"]
    best_val_mse = np.inf
    test_res = None

    for itr in range(args.epoch):
        model.train()
        st = time.time()
        epoch_loss, epoch_n = 0.0, 0
        for _ in tqdm(range(num_batches)):
            optimizer.zero_grad()
            batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
            if batch_dict is None:
                continue
            train_res = compute_all_losses(model, batch_dict)
            train_res["loss"].backward()
            optimizer.step()
            epoch_loss += train_res['mse']; epoch_n += 1
        train_time = time.time() - st
        train_times.append(train_time)
        logger.info(f'training time per epoch :{train_time:.3f}s')

        model.eval()
        with torch.no_grad():
            st = time.time()
            val_res, _ = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"])
            logger.info(f"val time per epoch :{time.time() - st:.2f}s")

            if val_res["mse"] < best_val_mse:
                best_val_mse = val_res["mse"]
                best_iter = itr
                st = time.time()
                test_res, _ = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])
                logger.info(f"test time per epoch :{time.time() - st:.2f}s")
                torch.save(model.state_dict(), ckpt_path)

            logger.info(f'- Epoch {itr:03d}, ExpID {experimentID}')
            logger.info(f"Train - Loss (one batch): {train_res['loss'].item():.5f}")
            logger.info(
                f"Val - Loss, MSE, RMSE, MAE, MAPE: {val_res['loss']:.5f}, "
                f"{val_res['mse']:.5f}, {val_res['rmse']:.5f}, {val_res['mae']:.5f}, "
                f"{val_res['mape'] * 100:.2f}%"
            )
            if test_res is not None:
                logger.info(
                    f"Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: {best_iter}, "
                    f"{test_res['loss']:.5f}, {test_res['mse']:.5f}, "
                    f"{test_res['rmse']:.5f}, {test_res['mae']:.5f}, "
                    f"{test_res['mape'] * 100:.2f}%"
                )

            metrics = {
                'epoch': itr,
                'train/loss': epoch_loss / max(epoch_n, 1),
                'train/loss_last_batch': train_res['loss'].item(),
                'train/epoch_time_s': train_time,
                'val/mse': val_res['mse'], 'val/rmse': val_res['rmse'],
                'val/mae': val_res['mae'], 'val/mape': val_res['mape'],
                'best_epoch': best_iter,
            }
            if test_res is not None:
                metrics.update({'test/mse': test_res['mse'], 'test/rmse': test_res['rmse'],
                                'test/mae': test_res['mae'], 'test/mape': test_res['mape']})
            for k in ('consistency', 'repr/pos_cos', 'repr/neg_cos', 'repr/std'):
                if k in train_res:
                    metrics['train/' + k if k == 'consistency' else k] = train_res[k]
            if torch.cuda.is_available():
                metrics['gpu/peak_mem_mb'] = torch.cuda.max_memory_allocated() / 2**20
            wandb_utils.log(metrics, step=itr)

        if itr - best_iter >= args.patience:
            logger.info("Exp has been early stopped!")
            break

    # ---------------- Final report: cost + cross-sampling robustness ----------------
    results = {'run_tag': args.run_tag,
               'args': {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))},
               'n_params': count_parameters(model), 'best_epoch': best_iter, 'best_val_mse': best_val_mse,
               'epochs_run': itr + 1, 'train_time_per_epoch_s': float(np.mean(train_times)),
               'total_train_time_s': float(np.sum(train_times))}
    if torch.cuda.is_available():
        results['peak_gpu_mem_mb'] = torch.cuda.max_memory_allocated() / 2**20
    if test_res is not None:
        results['test'] = {k: test_res[k] for k in ('mse', 'rmse', 'mae', 'mape', 'infer_ms_per_series')}
        wandb_utils.summary({'best_epoch': best_iter, 'best_val_mse': best_val_mse,
                             'test_mse': test_res['mse'], 'test_rmse': test_res['rmse'],
                             'test_mae': test_res['mae'], 'test_mape': test_res['mape'],
                             'train_time_per_epoch_s': results['train_time_per_epoch_s'],
                             'peak_gpu_mem_mb': results.get('peak_gpu_mem_mb'),
                             'infer_ms_per_series': test_res['infer_ms_per_series']})
        # Robustness: best checkpoint evaluated on the test set under each sampling view
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
        model.eval()
        robust = {}
        with torch.no_grad():
            for view in [v.strip() for v in args.eval_views.split(',') if v.strip()]:
                g = sv.make_generator(eval_gen_seed, args.device)
                r, _ = evaluation(model, data_obj['test_dataloader'], data_obj['n_test_batches'], view=view, gen=g)
                robust[view] = {k: r[k] for k in ('mse', 'rmse', 'mae', 'mape')}
                logger.info(f"Robustness [{view}] MSE {r['mse']:.5f} MAE {r['mae']:.5f}")
                wandb_utils.summary({f'robust/{view}/mse': r['mse'], f'robust/{view}/mae': r['mae']})
        results['robustness'] = robust
        try:
            import wandb
            if wandb.run is not None:
                tbl = wandb.Table(columns=['view', 'mse', 'rmse', 'mae', 'mape'],
                                  data=[[v, r['mse'], r['rmse'], r['mae'], r['mape']] for v, r in robust.items()])
                wandb.log({'robustness/mse_by_view': wandb.plot.bar(tbl, 'view', 'mse', title='Test MSE by observation view'),
                           'robustness/mae_by_view': wandb.plot.bar(tbl, 'view', 'mae', title='Test MAE by observation view')})
        except Exception as e:
            logger.info(f'wandb table skipped: {e}')
    utils.makedirs('results/')
    with open(f'results/{args.run_tag}.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Saved results/{args.run_tag}.json')
    wandb_utils.finish()
