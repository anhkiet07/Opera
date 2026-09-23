#!/bin/bash
# Follow-up: the soft estimated-graph bias in low-data PhysioNet looked more robust on seed 1. Seeds 2-3 plus
# shuffled / random soft controls (same edge count as the estimated graph, lambda 2).
set -u
PY=${PY:-.venv/Scripts/python}
GPU=${GPU:-0}
VIEWS="none,random:0.3,random:0.5,random:0.7,bursty:0.3,bursty:0.5,variable:0.3,variable:0.5,stride:2,stride:3"
LOW="--dataset physionet -n 1500 --history 24 --patch_size 6 --stride 6 --hid_dim 64 --nhead 4 --nlayer 3 \
     --batch_size 32 --lr 1e-3 --gpu $GPU --wandb_project quite --irr_emb --model itransformer --mode quite \
     --eval_views $VIEWS --epoch 150 --patience 20 --min_epochs 40"
run() {
    local tag="$1"; shift
    if [ -f "results/${tag}.json" ]; then echo "[skip] $tag (results exist)"; return; fi
    echo "===== $tag ====="
    $PY train_forecasting.py $LOW --run_tag "$tag" "$@" 2>&1 | grep -E "Saved|Error|Traceback"
}
for seed in 2 3; do
    run "p3b_low_soft_est_lam2_s${seed}"      --seed $seed --causal_mode soft --causal_graph true     --causal_lambda 2 --prior_source estimated
done
for seed in 1 2 3; do
    run "p3b_low_soft_shuffled_est_s${seed}"  --seed $seed --causal_mode soft --causal_graph shuffled --causal_lambda 2 --prior_source estimated
    run "p3b_low_soft_random_est_s${seed}"    --seed $seed --causal_mode soft --causal_graph random   --causal_lambda 2 --prior_source estimated
done
echo "PHASE3B-SOFT DONE"
