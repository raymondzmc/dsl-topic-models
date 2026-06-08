#!/usr/bin/env bash
# DSL applied to other topic-model backbones (paper: ECRTM+DSL, FASTopic+DSL, ETM+DSL).
# {dsl_ecrtm, dsl_fastopic, dsl_etm} x 5 LMs x 3 datasets x K, 5 seeds.
cd "$(dirname "$0")/.." && source scripts/common.sh

BACKBONES=(dsl_ecrtm dsl_fastopic dsl_etm)

for model in "${BACKBONES[@]}"; do
  for lm in "${LMS[@]}"; do
    for ds in "${DATASETS[@]}"; do
      for K in "${KS[@]}"; do
        dsl_train --model "$model" --data_path "$(data_path "$ds" "$lm")" \
          --num_topics "$K" --num_seeds "$SEEDS" --num_epochs "$(model_epochs "$model")" $EVAL_FLAGS
      done
    done
  done
done
