#!/usr/bin/env bash
# Regenerate all pre-processed LM soft-label datasets locally (3 datasets x 5 LMs).
#
# This is the FALLBACK to downloading the authors' processed datasets from the HF
# Hub (which the training scripts do automatically). Regenerating runs each LM in
# bf16 over the corpus, so numbers may drift slightly from the published ones;
# download the HF datasets for bit-exact reproduction (see the README).
#
# Resumable: skips a dataset that already exists under data/processed_data/.
cd "$(dirname "$0")/.." && source scripts/common.sh

declare -A LM_HF=(
  [ERNIE-4.5-0.3B-PT]="baidu/ERNIE-4.5-0.3B-PT"
  [Qwen3.5-0.8B]="Qwen/Qwen3.5-0.8B"
  [Llama-3.2-1B-Instruct]="meta-llama/Llama-3.2-1B-Instruct"
  [Llama-3.1-8B-Instruct]="meta-llama/Llama-3.1-8B-Instruct"
  [Phi-3-mini-128k-instruct]="microsoft/Phi-3-mini-128k-instruct"
)

ds_args() {  # source-dataset args per dataset (matches the paper's preprocessing)
  case "$1" in
    20_newsgroups) echo "--dataset SetFit/20_newsgroups --content_key text --label_key label --split all" ;;
    tweet_topic)   echo "--dataset data/raw_data/tweet_topic.tsv --content_key text --label_key label" ;;
    stackoverflow) echo "--dataset data/raw_data/stackoverflow.tsv --content_key text --label_key label" ;;
  esac
}

for lm in "${LMS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    save="${ds}_${lm}_vocab_2000_last"
    if [ -d "data/processed_data/${save}" ]; then
      echo "[skip] data/processed_data/${save} already exists"
      continue
    fi
    echo "+ process ${ds} with ${LM_HF[$lm]} (batch_size=${BATCH_SIZE})"
    "$PY" -m dsl_topic.cli.process_dataset $(ds_args "$ds") \
      --model_name "${LM_HF[$lm]}" \
      --vocab_size 2000 --embedding_method last \
      --batch_size "$BATCH_SIZE" --save_name "$save" --no_upload \
      || echo "  [warn] processing failed: ${ds} / ${lm} (continuing)"
  done
done
