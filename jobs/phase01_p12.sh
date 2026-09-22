#!/bin/bash
# Phase 0 (reproduce QuITE + PatchTST on P12) + Phase 1 (sampling consistency) for classification.
# Uses the first N_FOLDS Raindrop splits. Results -> results/<run_tag>.json, W&B project "quite".
set -u
PY=${PY:-.venv/Scripts/python}
GPU=${GPU:-0}
BS=${BS:-32}
EPOCH=${EPOCH:-40}
PATIENCE=${PATIENCE:-8}
N_FOLDS=${N_FOLDS:-2}

COMMON="--dataset P12 --patch_size 6 --stride 6 --hid_dim 64 --nhead 4 --nlayer 3 \
        --batch_size $BS --lr 1e-3 --gpu $GPU --epoch $EPOCH --patience $PATIENCE --n_folds $N_FOLDS --wandb_project quite"

run() {
    local tag="$1"; shift
    if [ -f "results/${tag}.json" ]; then echo "[skip] $tag (results exist)"; return; fi
    echo "===== $tag ====="
    $PY train_classification.py $COMMON --run_tag "$tag" "$@" 2>&1 | grep -v "it/s\]\|s/it\]" | grep -E "Validation|Testing|Robustness|Saved|AUROC|Error|error"
}

run "p0cls_quite_patchtst_P12"          --irr_emb --model patchtst --mode quite
run "p1cls_cons_lam0.1_patchtst_P12"    --irr_emb --model patchtst --mode quite --train_view mixed --n_views 2 --consistency_lambda 0.1
run "p1cls_aug_only_patchtst_P12"       --irr_emb --model patchtst --mode quite --train_view mixed --n_views 2 --consistency_lambda 0

echo "ALL CLASSIFICATION RUNS DONE"
