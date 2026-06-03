#!/usr/bin/env bash
# Ablation: vocabulary size |V| (paper appendix "Experiments with Different Vocabulary Sizes").
# Re-process 20Newsgroups/ERNIE at |V| in {500,1000,2000,4000}, then train generative + 8 baselines.
# Results go to results/vocab_<V>/ (the source dataset name is identical across |V|, so we
# disambiguate via --output_dir).
cd "$(dirname "$0")/.." && source scripts/common.sh

lm="ERNIE-4.5-0.3B-PT"
VOCABS=(500 1000 2000 4000)
MODELS=(generative lda prodlda zeroshot combined etm bertopic fastopic ecrtm)

for V in "${VOCABS[@]}"; do
  save="20_newsgroups_${lm}_vocab_${V}_last"
  if [ ! -d "data/processed_data/${save}" ]; then
    echo "+ process 20_newsgroups / ERNIE at |V|=${V}"
    "$PY" -m dsl_topic.cli.process_dataset --dataset SetFit/20_newsgroups \
      --content_key text --label_key label --split all \
      --model_name baidu/ERNIE-4.5-0.3B-PT --vocab_size "$V" --embedding_method last \
      --batch_size "$BATCH_SIZE" --save_name "$save" --no_upload \
      || { echo "  [warn] processing failed for |V|=${V} (skipping)"; continue; }
  fi
  for model in "${MODELS[@]}"; do
    for K in "${KS[@]}"; do
      dsl_train --model "$model" --data_path "${HF_NAMESPACE}/${save}" \
        --num_topics "$K" --num_seeds "$SEEDS" --num_epochs "$EPOCHS" \
        --output_dir "results/vocab_${V}" $EVAL_FLAGS
    done
  done
done
