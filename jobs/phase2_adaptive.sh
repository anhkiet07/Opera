#!/bin/bash
# Phase 2: Adaptive (observation-aware) query embedding.
#   q(X) = q_base + MLP(stats(X)), stats in {full, density, gaps, avail}, 'random' = noise control.
# Forecasting: PhysioNet 24h->24h, iTransformer, same budget as Phase 0/1 (epoch cap 30, patience 8, bs 32).
# Evaluation uses the extended view list (adds unseen regimes: variable:0.5, stride:2, stride:3).
set -u
PY=${PY:-.venv/Scripts/python}
GPU=${GPU:-0}
BS=${BS:-32}
EPOCH=${EPOCH:-30}
PATIENCE=${PATIENCE:-8}
SEEDS=${SEEDS:-"1 2"}
VIEWS="none,random:0.3,random:0.5,random:0.7,bursty:0.3,bursty:0.5,variable:0.3,variable:0.5,stride:2,stride:3"

COMMON="--dataset physionet --history 24 --patch_size 6 --stride 6 --hid_dim 64 --nhead 4 --nlayer 3 \
        --batch_size $BS --lr 1e-3 --gpu $GPU --epoch $EPOCH --patience $PATIENCE --wandb_project quite \
        --irr_emb --model itransformer --mode quite --eval_views $VIEWS"

run() {
    local tag="$1"; shift
    if [ -f "results/${tag}.json" ]; then echo "[skip] $tag (results exist)"; return; fi
    echo "===== $tag ====="
    $PY train_forecasting.py $COMMON --run_tag "$tag" "$@" 2>&1 | grep -E "Robustness|Saved|Error|Traceback"
}

for seed in $SEEDS; do
    # standard training (no views): plain QuITE already exists as p0_quite_itr_s<seed>
    run "p2_adaptive_full_s${seed}"        --seed $seed --adaptive_query full
    # mixed-regime training (train on mixture, test on unseen regimes): compare with p1_aug_only_s<seed>
    run "p2_adaptive_full_mixed_s${seed}"  --seed $seed --adaptive_query full --train_view mixed --n_views 2
done

# ---- ablations (seed 1, standard training): which statistics matter, and a noise control ----
run "p2_adaptive_density_s1"  --seed 1 --adaptive_query density
run "p2_adaptive_gaps_s1"     --seed 1 --adaptive_query gaps
run "p2_adaptive_avail_s1"    --seed 1 --adaptive_query avail
run "p2_adaptive_random_s1"   --seed 1 --adaptive_query random
run "p2_adaptive_random_mixed_s1" --seed 1 --adaptive_query random --train_view mixed --n_views 2

# ---- classification (P12, fold 1) ----
CLS="--dataset P12 --patch_size 6 --stride 6 --hid_dim 64 --nhead 4 --nlayer 3 --batch_size $BS --lr 1e-3 \
     --gpu $GPU --epoch $EPOCH --patience $PATIENCE --n_folds 1 --wandb_project quite --irr_emb --model patchtst --mode quite \
     --eval_views $VIEWS"
runc() {
    local tag="$1"; shift
    if [ -f "results/${tag}.json" ]; then echo "[skip] $tag (results exist)"; return; fi
    echo "===== $tag ====="
    $PY train_classification.py $CLS --run_tag "$tag" "$@" 2>&1 | grep -E "Testing|Robustness mean|Saved|Error|Traceback"
}
runc "p2cls_adaptive_full_patchtst_P12"        --adaptive_query full
runc "p2cls_adaptive_full_mixed_patchtst_P12"  --adaptive_query full --train_view mixed --n_views 2

$PY summarize_results.py > /dev/null
echo "PHASE2 DONE"
