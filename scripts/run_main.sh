#!/usr/bin/env bash
# Main DSL results (paper Table "Automatic Topic Evaluation"): ProdLDA + DSL.
# dsl model x 5 LMs x 3 datasets x K in {25,50,75,100}, 5 seeds.
cd "$(dirname "$0")/.." && source scripts/common.sh

for lm in "${LMS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for K in "${KS[@]}"; do
      dsl_train --model dsl --data_path "$(data_path "$ds" "$lm")" \
        --num_topics "$K" --num_seeds "$SEEDS" --num_epochs "$(model_epochs dsl)" $EVAL_FLAGS
    done
  done
done
