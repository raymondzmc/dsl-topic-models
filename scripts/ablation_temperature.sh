#!/usr/bin/env bash
# Ablation: softmax temperature tau (paper "Temperature Values" + appendix).
# generative / ERNIE-4.5-0.3B-PT, all 3 datasets, all K, tau in {0.5,1,...,9}.
cd "$(dirname "$0")/.." && source scripts/common.sh

lm="ERNIE-4.5-0.3B-PT"
TEMPERATURES=(0.5 1 2 3 4 5 6 7 8 9)

for ds in "${DATASETS[@]}"; do
  for K in "${KS[@]}"; do
    for tau in "${TEMPERATURES[@]}"; do
      dsl_train --model generative --data_path "$(data_path "$ds" "$lm")" \
        --num_topics "$K" --temperature "$tau" \
        --num_seeds "$SEEDS" --num_epochs "$EPOCHS" $EVAL_FLAGS
    done
  done
done
