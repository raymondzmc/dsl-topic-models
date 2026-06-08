#!/usr/bin/env bash
# Ablation: target sparsity ratio (paper appendix "Target Sparsity and Topic Diversity").
# dsl / Llama-3.2-1B-Instruct, all 3 datasets, all K, sparsity_ratio in {0.1..0.9}.
cd "$(dirname "$0")/.." && source scripts/common.sh

lm="Llama-3.2-1B-Instruct"
SPARSITY=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)

for ds in "${DATASETS[@]}"; do
  for K in "${KS[@]}"; do
    for s in "${SPARSITY[@]}"; do
      dsl_train --model dsl --data_path "$(data_path "$ds" "$lm")" \
        --num_topics "$K" --sparsity_ratio "$s" \
        --num_seeds "$SEEDS" --num_epochs "$EPOCHS" $EVAL_FLAGS
    done
  done
done
