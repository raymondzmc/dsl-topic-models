#!/usr/bin/env bash
# Retrieval evaluation (paper "Retrieval Evaluation"): topic-guided document retrieval.
# Computes Precision@{1,5,10} for every trained run under results/ and writes retrieval.json
# next to each averaged_results.json. Run this AFTER the training scripts.
cd "$(dirname "$0")/.." && source scripts/common.sh

echo "Computing topic-guided retrieval Precision@k over all runs in results/ ..."
"$PY" -m dsl_topic.cli.retrieval --results_dir results "$@"
