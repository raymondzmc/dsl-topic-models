#!/usr/bin/env bash
# DSL applied to other topic-model backbones (paper: ECRTM+DSL, FASTopic+DSL, ETM+DSL).
# {generative_ecrtm, generative_fastopic, generative_etm} x 5 LMs x 3 datasets x K, 5 seeds.
cd "$(dirname "$0")/.." && source scripts/common.sh

BACKBONES=(generative_ecrtm generative_fastopic generative_etm)

for model in "${BACKBONES[@]}"; do
  for lm in "${LMS[@]}"; do
    for ds in "${DATASETS[@]}"; do
      for K in "${KS[@]}"; do
        dsl_train --model "$model" --data_path "$(data_path "$ds" "$lm")" \
          --num_topics "$K" --num_seeds "$SEEDS" --num_epochs "$EPOCHS" $EVAL_FLAGS
      done
    done
  done
done
