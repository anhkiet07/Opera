#!/bin/bash
# Phase 3: causal structure as an attention bias, on the synthetic dataset with a KNOWN graph.
#   Graph (cause -> effect): 0->1, 1->2, 0->3, 4->2, 5->6, 7->8 ; 9 isolated. Target Y = variable 2.
# Views for the critical experiment (inputs re-observed, targets untouched):
#   dropvars:5-6-7-8-9  remove all variables that are NON-causal for Y
#   dropvars:1-4        remove Y's direct causes
#   dropvars:0-3        remove Y's grand-cause and a sibling
#   random:p / variable:0.3   generic sparsity
set -u
PY=${PY:-.venv/Scripts/python}
GPU=${GPU:-0}
EPOCH=${EPOCH:-100}
PATIENCE=${PATIENCE:-10}
SEEDS=${SEEDS:-"1 2 3"}
VIEWS="none,random:0.5,random:0.7,variable:0.3,dropvars:5-6-7-8-9,dropvars:1-4,dropvars:0-3,dropvars:9"

COMMON="--dataset synthetic --history 24 --patch_size 6 --stride 6 --hid_dim 64 --nhead 4 --nlayer 3 \
        --batch_size 32 --lr 1e-3 --gpu $GPU --epoch $EPOCH --patience $PATIENCE --wandb_project quite \
        --model itransformer --eval_views $VIEWS"

run() {
    local tag="$1"; shift
    if [ -f "results/${tag}.json" ]; then echo "[skip] $tag (results exist)"; return; fi
    echo "===== $tag ====="
    $PY train_forecasting.py $COMMON --run_tag "$tag" "$@" 2>&1 | grep -E "Robustness|Saved|Error|Traceback"
}

Q="--irr_emb --mode quite"
for seed in $SEEDS; do
    run "p3_vanilla_s${seed}"            --mode False --seed $seed
    run "p3_quite_s${seed}"              $Q --seed $seed
    run "p3_soft_true_lam2_s${seed}"     $Q --seed $seed --causal_mode soft --causal_graph true     --causal_lambda 2
    run "p3_soft_random_lam2_s${seed}"   $Q --seed $seed --causal_mode soft --causal_graph random   --causal_lambda 2
    run "p3_soft_shuffled_lam2_s${seed}" $Q --seed $seed --causal_mode soft --causal_graph shuffled --causal_lambda 2
done

# ---- single-seed variants ----
run "p3_soft_identity_lam2_s1"   $Q --seed 1 --causal_mode soft --causal_graph identity  --causal_lambda 2
run "p3_soft_noisy0.3_lam2_s1"   $Q --seed 1 --causal_mode soft --causal_graph noisy:0.3 --causal_lambda 2
run "p3_soft_true_lam0.5_s1"     $Q --seed 1 --causal_mode soft --causal_graph true      --causal_lambda 0.5
run "p3_soft_true_lam5_s1"       $Q --seed 1 --causal_mode soft --causal_graph true      --causal_lambda 5
run "p3_hard_true_s1"            $Q --seed 1 --causal_mode hard --causal_graph true
run "p3_learned_s1"              $Q --seed 1 --causal_mode learned
# causal bias on the vanilla backbone (does the graph help without QuITE?)
run "p3_vanilla_soft_true_lam2_s1" --mode False --seed 1 --causal_mode soft --causal_graph true --causal_lambda 2

$PY summarize_results.py > /dev/null
echo "PHASE3 DONE"
