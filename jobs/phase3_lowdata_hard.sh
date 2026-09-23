#!/bin/bash
# Phase 3 low-data regime, hard-mask controls: the hard mask forces each variable token to attend only to the
# variables listed in M (plus itself). With the oracle graph this is a correct structural constraint; with a
# shuffled / random graph it forces attention to the wrong variables; with identity it makes the backbone
# channel-independent. 3 seeds each.
set -u
PY=${PY:-.venv/Scripts/python}
GPU=${GPU:-0}
N=${N:-1500}
EPOCH=${EPOCH:-150}
PATIENCE=${PATIENCE:-15}
SEEDS=${SEEDS:-"1 2 3"}
VIEWS="none,random:0.5,random:0.7,variable:0.3,dropvars:5-6-7-8-9,dropvars:1-4,dropvars:0-3,dropvars:9"
COMMON="--dataset synthetic -n $N --history 24 --patch_size 6 --stride 6 --hid_dim 64 --nhead 4 --nlayer 3 \
        --batch_size 32 --lr 1e-3 --gpu $GPU --epoch $EPOCH --patience $PATIENCE --min_epochs 40 --wandb_project quite \
        --irr_emb --model itransformer --mode quite --eval_views $VIEWS"
run() {
    local tag="$1"; shift
    if [ -f "results/${tag}.json" ]; then echo "[skip] $tag (results exist)"; return; fi
    echo "===== $tag ====="
    $PY train_forecasting.py $COMMON --run_tag "$tag" "$@" 2>&1 | grep -E "Saved|Error|Traceback"
}
for seed in $SEEDS; do
    run "p3low_hard_shuffled_s${seed}" --seed $seed --causal_mode hard --causal_graph shuffled
    run "p3low_hard_identity_s${seed}" --seed $seed --causal_mode hard --causal_graph identity
    run "p3low_hard_random_s${seed}"   --seed $seed --causal_mode hard --causal_graph random
done
echo "PHASE3-LOWHARD DONE"
