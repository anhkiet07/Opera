#!/bin/bash
# Phase 3 on real data: LLM-elicited physiological prior graph (data/physionet_prior.py) as an attention
# bias for QuITE + iTransformer on PhysioNet 24h->24h. Same budget as Phase 0/1 (epoch cap 30, patience 8,
# bs 32), so p0_quite_itr_s<seed> is the no-bias reference. Controls: shuffled / random / identity prior.
set -u
PY=${PY:-.venv/Scripts/python}
GPU=${GPU:-0}
EPOCH=${EPOCH:-30}
PATIENCE=${PATIENCE:-8}
SEEDS=${SEEDS:-"1 2"}
VIEWS="none,random:0.3,random:0.5,random:0.7,bursty:0.3,bursty:0.5,variable:0.3,variable:0.5,stride:2,stride:3"
COMMON="--dataset physionet --history 24 --patch_size 6 --stride 6 --hid_dim 64 --nhead 4 --nlayer 3 \
        --batch_size 32 --lr 1e-3 --gpu $GPU --epoch $EPOCH --patience $PATIENCE --wandb_project quite \
        --irr_emb --model itransformer --mode quite --eval_views $VIEWS"
run() {
    local tag="$1"; shift
    if [ -f "results/${tag}.json" ]; then echo "[skip] $tag (results exist)"; return; fi
    echo "===== $tag ====="
    $PY train_forecasting.py $COMMON --run_tag "$tag" "$@" 2>&1 | grep -E "Robustness|Saved|Error|Traceback"
}
for seed in $SEEDS; do
    run "p3phys_soft_llm_lam2_s${seed}"      --seed $seed --causal_mode soft --causal_graph true     --causal_lambda 2
    run "p3phys_soft_shuffled_lam2_s${seed}" --seed $seed --causal_mode soft --causal_graph shuffled --causal_lambda 2
done
run "p3phys_soft_llm_lam1_s1"       --seed 1 --causal_mode soft --causal_graph true     --causal_lambda 1
run "p3phys_soft_llm_lam5_s1"       --seed 1 --causal_mode soft --causal_graph true     --causal_lambda 5
run "p3phys_soft_random_lam2_s1"    --seed 1 --causal_mode soft --causal_graph random   --causal_lambda 2
run "p3phys_hard_llm_s1"            --seed 1 --causal_mode hard --causal_graph true
run "p3phys_learned_s1"             --seed 1 --causal_mode learned
$PY summarize_results.py > /dev/null
echo "PHASE3-PHYS DONE"
