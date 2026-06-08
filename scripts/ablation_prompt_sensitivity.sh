#!/usr/bin/env bash
# Ablation: instruction-prompt sensitivity (paper "Instruction Prompt Sensitivity" + appendix).
# Re-process 20Newsgroups/ERNIE with 5 rephrased instruction templates, then train
# {dsl, zeroshot, prodlda, fastopic} at K=50. Results go to results/prompt_<variant>/.
cd "$(dirname "$0")/.." && source scripts/common.sh

lm="ERNIE-4.5-0.3B-PT"
VARIANTS=(variant_1 variant_2 variant_3 variant_4 variant_5)
MODELS=(dsl zeroshot prodlda fastopic)

for v in "${VARIANTS[@]}"; do
  save="20_newsgroups_${lm}_vocab_2000_last_${v}"
  if [ ! -d "data/processed_data/${save}" ]; then
    echo "+ process 20_newsgroups / ERNIE with instructions/${v}.jinja"
    "$PY" -m dsl_topic.cli.process_dataset --dataset SetFit/20_newsgroups \
      --content_key text --label_key label --split all \
      --model_name baidu/ERNIE-4.5-0.3B-PT --vocab_size 2000 --embedding_method last \
      --instruction_template "instructions/${v}.jinja" \
      --batch_size "$BATCH_SIZE" --save_name "$save" --no_upload \
      || { echo "  [warn] processing failed for ${v} (skipping)"; continue; }
  fi
  for model in "${MODELS[@]}"; do
    dsl_train --model "$model" --data_path "${HF_NAMESPACE}/${save}" \
      --num_topics 50 --num_seeds "$SEEDS" --num_epochs "$EPOCHS" \
      --output_dir "results/prompt_${v}" $EVAL_FLAGS
  done
done
