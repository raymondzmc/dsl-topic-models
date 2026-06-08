#!/usr/bin/env bash
# Ablation: use the BoW target instead of the DSL soft-label target (isolates the DSL signal).
# dsl --ablation_use_bow_target --loss_type CE,
# {ERNIE, Llama-3.1-8B, Llama-3.2-1B} x 3 datasets x all K.
cd "$(dirname "$0")/.." && source scripts/common.sh

MODELS=(ERNIE-4.5-0.3B-PT Llama-3.1-8B-Instruct Llama-3.2-1B-Instruct)

for lm in "${MODELS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for K in "${KS[@]}"; do
      dsl_train --model dsl --data_path "$(data_path "$ds" "$lm")" \
        --num_topics "$K" --ablation_use_bow_target --loss_type CE \
        --num_seeds "$SEEDS" --num_epochs "$EPOCHS" $EVAL_FLAGS
    done
  done
done
