#!/usr/bin/env bash
# Ablation: top-k of the DSL target (paper appendix "Target Sparsity").
# generative / ERNIE-4.5-0.3B-PT, stackoverflow, K=50, topk in {10,20,50,100,500,1000}.
cd "$(dirname "$0")/.." && source scripts/common.sh

lm="ERNIE-4.5-0.3B-PT"
ds="stackoverflow"
TOPKS=(10 20 50 100 500 1000)

for topk in "${TOPKS[@]}"; do
  dsl_train --model generative --data_path "$(data_path "$ds" "$lm")" \
    --num_topics 50 --topk "$topk" \
    --num_seeds "$SEEDS" --num_epochs "$EPOCHS" $EVAL_FLAGS
done
