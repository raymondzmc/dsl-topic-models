#!/usr/bin/env bash
# Convenience wrapper to train a single configuration.
# Usage: scripts/run_dsl.sh <model> <lm> <dataset> <K> [extra dsl-train flags...]
# Example: scripts/run_dsl.sh dsl ERNIE-4.5-0.3B-PT 20_newsgroups 50 --temperature 2
cd "$(dirname "$0")/.." && source scripts/common.sh

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <model> <lm> <dataset> <K> [extra flags...]"
  echo "  model:   dsl | dsl_ecrtm | dsl_fastopic | dsl_etm | <baseline>"
  echo "  lm:      ${LMS[*]}"
  echo "  dataset: ${DATASETS[*]}"
  exit 1
fi

model="$1"; lm="$2"; ds="$3"; K="$4"; shift 4
dsl_train --model "$model" --data_path "$(data_path "$ds" "$lm")" \
  --num_topics "$K" --num_seeds "$SEEDS" --num_epochs "$(model_epochs "$model")" $EVAL_FLAGS "$@"
