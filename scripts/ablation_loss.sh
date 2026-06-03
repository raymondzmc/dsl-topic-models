#!/usr/bin/env bash
# Ablation: reconstruction loss type (KL default vs. cross-entropy).
# generative with --loss_type CE, {ERNIE, Llama-3.1-8B, Llama-3.2-1B} x 3 datasets, K=100.
cd "$(dirname "$0")/.." && source scripts/common.sh

MODELS=(ERNIE-4.5-0.3B-PT Llama-3.1-8B-Instruct Llama-3.2-1B-Instruct)

for lm in "${MODELS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    dsl_train --model generative --data_path "$(data_path "$ds" "$lm")" \
      --num_topics 100 --loss_type CE \
      --num_seeds "$SEEDS" --num_epochs "$EPOCHS" $EVAL_FLAGS
  done
done
