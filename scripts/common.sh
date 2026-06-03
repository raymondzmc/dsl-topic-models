#!/usr/bin/env bash
# Shared configuration and helpers for the DSL-Topic experiment scripts.
#
# All scripts run SEQUENTIALLY on a single GPU (CUDA_VISIBLE_DEVICES=0 by default)
# and are RESUMABLE: a run whose results/<dataset>/<run>/averaged_results.json
# already exists is skipped, so you can safely re-run any script after an
# interruption.
#
# Environment overrides:
#   PYTHON=...        python interpreter to use            (default: python)
#   CUDA_VISIBLE_DEVICES=N  GPU to use                     (default: 0)
#   NUM_SEEDS=N       random seeds per run                 (default: 5)
#   NUM_EPOCHS=N      training epochs                      (default: 100)
#   SKIP_LLM=1        skip the OpenAI gpt-4o rating metric (default: compute it)
#   USE_WANDB=1       also log to Weights & Biases         (default: local only)
#   HF_NAMESPACE=...  HF Hub namespace for auto-download    (default: raymondzmc)
#   BATCH_SIZE=N      process_dataset batch size           (default: 1, as in paper)
set -o pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PYTHON:-python}"
SEEDS="${NUM_SEEDS:-5}"
EPOCHS="${NUM_EPOCHS:-100}"
HF_NAMESPACE="${HF_NAMESPACE:-raymondzmc}"
BATCH_SIZE="${BATCH_SIZE:-1}"

EVAL_FLAGS=""
[ "${SKIP_LLM:-0}" = "1" ] && EVAL_FLAGS="$EVAL_FLAGS --skip_llm_rating"
[ "${USE_WANDB:-0}" = "1" ] && EVAL_FLAGS="$EVAL_FLAGS --wandb"

# Experiment grid (matches the paper).
DATASETS=(20_newsgroups tweet_topic stackoverflow)
KS=(25 50 75 100)
LMS=(ERNIE-4.5-0.3B-PT Qwen3.5-0.8B Llama-3.2-1B-Instruct Llama-3.1-8B-Instruct Phi-3-mini-128k-instruct)
# Baselines were trained on the Llama-3.1-8B-processed corpus (BoW + embeddings).
BASELINE_EMB_LM="Llama-3.1-8B-Instruct"

# Resolve a data path: "<HF_NAMESPACE>/<dataset>_<lm>_vocab_2000_last".
# loaders.py resolves this local-first (data/processed_data/<name>) and falls
# back to downloading the HF dataset of the same name.
data_path() {  # args: <dataset> <lm>
  echo "${HF_NAMESPACE}/${1}_${2}_vocab_2000_last"
}

# Train one configuration; never aborts the whole sweep on a single failure.
dsl_train() {
  echo "+ run_topic_model $*"
  "$PY" -m dsl_topic.cli.train "$@" \
    || echo "  [warn] run failed: $* (continuing)"
}
