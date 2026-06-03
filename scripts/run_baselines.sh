#!/usr/bin/env bash
# Baseline topic models (paper main table, "Baselines" block).
# 8 baselines x 3 datasets x K in {25,50,75,100}, 5 seeds.
# All baselines use the Llama-3.1-8B-processed corpus (BoW + contextual embeddings).
cd "$(dirname "$0")/.." && source scripts/common.sh

BASELINES=(lda prodlda zeroshot combined etm bertopic fastopic ecrtm)

for model in "${BASELINES[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for K in "${KS[@]}"; do
      dsl_train --model "$model" --data_path "$(data_path "$ds" "$BASELINE_EMB_LM")" \
        --num_topics "$K" --num_seeds "$SEEDS" --num_epochs "$EPOCHS" $EVAL_FLAGS
    done
  done
done
