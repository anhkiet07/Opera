#!/bin/bash
# Phase 0 (reproduce QuITE) + Phase 1 (Sampling-Invariant QuITE) on PhysioNet, history 24 -> 24.
#
# Every run shares: iTransformer backbone, hid=64, 4 heads, 3 layers, patch 6, bs 32, lr 1e-3,
# epoch cap EPOCH, early-stop patience PATIENCE. All runs log to W&B (project "quite") and write
# results/<run_tag>.json (test metrics, cost, cross-sampling robustness table).
#
# Usage:  bash jobs/phase01_physionet.sh            (from the repo root, venv active)
#         EPOCH=40 PATIENCE=8 SEEDS="1 2 3" bash jobs/phase01_physionet.sh
set -u
PY=${PY:-.venv/Scripts/python}
GPU=${GPU:-0}
BS=${BS:-32}
EPOCH=${EPOCH:-40}
PATIENCE=${PATIENCE:-8}
SEEDS=${SEEDS:-"1 2 3"}
GROUP=${GROUP:-phase01}

COMMON="--dataset physionet --history 24 --patch_size 6 --stride 6 --hid_dim 64 --nhead 4 --nlayer 3 \
        --batch_size $BS --lr 1e-3 --gpu $GPU --epoch $EPOCH --patience $PATIENCE --wandb_project quite"

run() {  # run <tag> <extra args...>
    local tag="$1"; shift
    if [ -f "results/${tag}.json" ]; then echo "[skip] $tag (results exist)"; return; fi
    echo "===== $tag ====="
    $PY train_forecasting.py $COMMON --run_tag "$tag" "$@" 2>&1 | grep -v "it/s\]\|s/it\]" | grep -E "Epoch|Val -|Test -|Robustness|Saved|Error|error"
}

for seed in $SEEDS; do
    # ---- Phase 0: reproduce QuITE + iTransformer, and the vanilla backbone ----
    run "p0_quite_itr_s${seed}"        --irr_emb --model itransformer --mode quite --seed $seed
    # ---- Phase 1 main comparison ----
    run "p1_aug_only_s${seed}"         --irr_emb --model itransformer --mode quite --seed $seed \
                                       --train_view mixed --n_views 2 --consistency_lambda 0
    run "p1_cons_lam0.1_s${seed}"      --irr_emb --model itransformer --mode quite --seed $seed \
                                       --train_view mixed --n_views 2 --consistency_lambda 0.1
done

# ---- Phase 0: original backbone (no QuITE; vanilla time-agnostic embedding) ----
run "p0_vanilla_itr_s1"                --model itransformer --mode False --seed 1

# ---- Phase 1 ablations (seed 1) ----
run "p1_cons_lam0.01_s1"               --irr_emb --model itransformer --mode quite --seed 1 \
                                       --train_view mixed --n_views 2 --consistency_lambda 0.01
run "p1_cons_lam1.0_s1"                --irr_emb --model itransformer --mode quite --seed 1 \
                                       --train_view mixed --n_views 2 --consistency_lambda 1.0
run "p1_cons_pred_lam0.1_s1"           --irr_emb --model itransformer --mode quite --seed 1 \
                                       --train_view mixed --n_views 2 --consistency_lambda 0.1 --consistency_target pred
run "p1_cons_randomview_lam0.1_s1"     --irr_emb --model itransformer --mode quite --seed 1 \
                                       --train_view random:0.5 --n_views 2 --consistency_lambda 0.1
run "p1_cons_burstyview_lam0.1_s1"     --irr_emb --model itransformer --mode quite --seed 1 \
                                       --train_view bursty:0.5 --n_views 2 --consistency_lambda 0.1

echo "ALL FORECASTING RUNS DONE"
