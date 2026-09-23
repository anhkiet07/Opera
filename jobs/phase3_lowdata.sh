#!/bin/bash
# Phase 3, low-data regime: same synthetic task with only 1500 samples (900 train). An inductive bias should
# matter most here. Configs: plain QuITE, true / shuffled / random / identity graph (soft, lambda 2), 3 seeds.
set -u
PY=${PY:-.venv/Scripts/python}
GPU=${GPU:-0}
N=${N:-1500}
EPOCH=${EPOCH:-150}
PATIENCE=${PATIENCE:-15}
SEEDS=${SEEDS:-"1 2 3"}
VIEWS="none,random:0.5,random:0.7,variable:0.3,dropvars:5-6-7-8-9,dropvars:1-4,dropvars:0-3,dropvars:9"
COMMON="--dataset synthetic -n $N --history 24 --patch_size 6 --stride 6 --hid_dim 64 --nhead 4 --nlayer 3 \
        --batch_size 32 --lr 1e-3 --gpu $GPU --epoch $EPOCH --patience $PATIENCE --wandb_project quite \
        --irr_emb --model itransformer --mode quite --eval_views $VIEWS"
run() {
    local tag="$1"; shift
    if [ -f "results/${tag}.json" ]; then echo "[skip] $tag (results exist)"; return; fi
    echo "===== $tag ====="
    $PY train_forecasting.py $COMMON --run_tag "$tag" "$@" 2>&1 | grep -E "Saved|Error|Traceback"
}
for seed in $SEEDS; do
    run "p3low_quite_s${seed}"              --seed $seed
    run "p3low_soft_true_lam2_s${seed}"     --seed $seed --causal_mode soft --causal_graph true     --causal_lambda 2
    run "p3low_soft_shuffled_lam2_s${seed}" --seed $seed --causal_mode soft --causal_graph shuffled --causal_lambda 2
    run "p3low_soft_random_lam2_s${seed}"   --seed $seed --causal_mode soft --causal_graph random   --causal_lambda 2
    run "p3low_soft_identity_lam2_s${seed}" --seed $seed --causal_mode soft --causal_graph identity --causal_lambda 2
    run "p3low_hard_true_s${seed}"          --seed $seed --causal_mode hard --causal_graph true
done
echo "PHASE3-LOW DONE"
