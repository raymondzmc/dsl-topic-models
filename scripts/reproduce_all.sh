#!/usr/bin/env bash
# Master driver: run EVERY experiment in the paper, sequentially, on one GPU.
#
# Datasets are auto-downloaded from the HF Hub on first use; if download is
# unavailable, run `scripts/process_all.sh` first to regenerate them locally.
# Every step is resumable (completed runs are skipped), so this script can be
# safely re-run after an interruption.
#
# Tips:
#   SKIP_LLM=1 bash scripts/reproduce_all.sh   # skip the OpenAI gpt-4o rating metric
#   NUM_SEEDS=1 NUM_EPOCHS=10 ...              # a fast (non-paper) dry run
cd "$(dirname "$0")/.." && source scripts/common.sh

echo "================ DSL-Topic: full reproduction ================"
echo "Python=$PY  GPU=$CUDA_VISIBLE_DEVICES  seeds=$SEEDS  epochs=$EPOCHS  eval_flags='$EVAL_FLAGS'"
echo "Datasets auto-download from HF (namespace=$HF_NAMESPACE); run process_all.sh to regenerate."
echo "=============================================================="

# Uncomment to regenerate all processed datasets locally instead of downloading:
# bash scripts/process_all.sh

bash scripts/run_main.sh
bash scripts/run_baselines.sh
bash scripts/run_backbones.sh

bash scripts/ablation_temperature.sh
bash scripts/ablation_sparsity.sh
bash scripts/ablation_topk.sh
bash scripts/ablation_loss.sh
bash scripts/ablation_embedding.sh
bash scripts/ablation_bow_target.sh
bash scripts/ablation_vocab_size.sh
bash scripts/ablation_prompt_sensitivity.sh

bash scripts/run_retrieval.sh

echo
echo "All experiments complete. Summarize with:  $PY -m dsl_topic.cli.summarize results"
