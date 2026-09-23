#!/bin/bash
# Phase 3b: structural priors on PhysioNet in the regime where the synthetic study found them to work:
# HARD attention constraints, LOW-data training. Priors: LLM-elicited physiological graph, a graph estimated
# from the same training split (estimate_graph.py), and their union. Controls: shuffled / identity / random hard
# masks, and the unconstrained encoder. Low data = first 1500 records (900 train / 300 val / 300 test).
set -u
PY=${PY:-.venv/Scripts/python}
GPU=${GPU:-0}
SEEDS=${SEEDS:-"1 2 3"}
VIEWS="none,random:0.3,random:0.5,random:0.7,bursty:0.3,bursty:0.5,variable:0.3,variable:0.5,stride:2,stride:3"
BASE="--dataset physionet --history 24 --patch_size 6 --stride 6 --hid_dim 64 --nhead 4 --nlayer 3 \
      --batch_size 32 --lr 1e-3 --gpu $GPU --wandb_project quite --irr_emb --model itransformer --mode quite \
      --eval_views $VIEWS"
LOW="$BASE -n 1500 --epoch 150 --patience 20 --min_epochs 40"
FULL="$BASE --epoch 30 --patience 8 --min_epochs 10"

run() {  # run <tag> <common> <extra...>
    local tag="$1"; local common="$2"; shift 2
    if [ -f "results/${tag}.json" ]; then echo "[skip] $tag (results exist)"; return; fi
    echo "===== $tag ====="
    $PY train_forecasting.py $common --run_tag "$tag" "$@" 2>&1 | grep -E "prior source|Saved|Error|Traceback"
}

# ---------------- low data, hard masks ----------------
for seed in $SEEDS; do
    run "p3b_low_plain_s${seed}"           "$LOW" --seed $seed
    run "p3b_low_hard_llm_s${seed}"        "$LOW" --seed $seed --causal_mode hard --causal_graph true     --prior_source llm
    run "p3b_low_hard_est_s${seed}"        "$LOW" --seed $seed --causal_mode hard --causal_graph true     --prior_source estimated
    run "p3b_low_hard_hybrid_s${seed}"     "$LOW" --seed $seed --causal_mode hard --causal_graph true     --prior_source hybrid
    run "p3b_low_hard_shuffled_s${seed}"   "$LOW" --seed $seed --causal_mode hard --causal_graph shuffled --prior_source hybrid
    run "p3b_low_hard_identity_s${seed}"   "$LOW" --seed $seed --causal_mode hard --causal_graph identity --prior_source llm
    run "p3b_low_hard_random_s${seed}"     "$LOW" --seed $seed --causal_mode hard --causal_graph random   --prior_source hybrid
done
# low data, soft estimated prior (does the estimated graph help as a soft bias?)
run "p3b_low_soft_est_lam2_s1"             "$LOW" --seed 1 --causal_mode soft --causal_graph true --causal_lambda 2 --prior_source estimated
run "p3b_low_soft_hybrid_lam2_s1"          "$LOW" --seed 1 --causal_mode soft --causal_graph true --causal_lambda 2 --prior_source hybrid

# ---------------- full data, estimated / hybrid graphs (reference: p0_quite_itr_s*, p3phys_hard_llm_s1) ----------------
for seed in 1 2; do
    run "p3b_full_hard_est_s${seed}"       "$FULL" --seed $seed --causal_mode hard --causal_graph true --prior_source estimated
    run "p3b_full_hard_hybrid_s${seed}"    "$FULL" --seed $seed --causal_mode hard --causal_graph true --prior_source hybrid
done
run "p3b_full_soft_est_lam2_s1"            "$FULL" --seed 1 --causal_mode soft --causal_graph true --causal_lambda 2 --prior_source estimated
run "p3b_full_hard_shuffled_hybrid_s1"     "$FULL" --seed 1 --causal_mode hard --causal_graph shuffled --prior_source hybrid

$PY summarize_results.py > /dev/null
echo "PHASE3B DONE"
