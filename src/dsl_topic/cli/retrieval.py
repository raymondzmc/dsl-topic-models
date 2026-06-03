"""Compute topic-guided retrieval Precision@k from locally-saved runs.

For each run produced by ``dsl-train`` (``results/<dataset>/<run_name>/`` with
``seed_*/model_output.pt`` and ``labels.json``), rank documents by the KL
divergence between their topic distributions, compute Precision@{1,5,10},
average across seeds, and write ``retrieval.json`` next to ``averaged_results.json``.

Reuses the core math in :mod:`dsl_topic.evaluation.retrieval`.
"""
import os
import json
import glob
import argparse
from collections import defaultdict

import numpy as np
import torch

from dsl_topic.evaluation.retrieval import evaluate_single_seed


def _load_labels(run_dir: str):
    labels_path = os.path.join(run_dir, "labels.json")
    if not os.path.exists(labels_path):
        return None
    with open(labels_path, encoding="utf-8") as f:
        return np.array(json.load(f))


def evaluate_run_dir(run_dir: str, subset_size: int = -1, device=None, force: bool = False):
    """Compute and persist Precision@k for one run directory."""
    out_path = os.path.join(run_dir, "retrieval.json")
    if os.path.exists(out_path) and not force:
        print(f"[skip] {out_path} already exists (use --force to recompute)")
        return None

    labels = _load_labels(run_dir)
    if labels is None:
        print(f"[skip] {run_dir}: no labels.json (retrieval needs ground-truth labels)")
        return None

    seed_dirs = [
        d for d in sorted(glob.glob(os.path.join(run_dir, "seed_*")))
        if os.path.exists(os.path.join(d, "model_output.pt"))
    ]
    if not seed_dirs:
        print(f"[skip] {run_dir}: no seed_*/model_output.pt found")
        return None

    per_seed = defaultdict(list)
    for sd in seed_dirs:
        res = evaluate_single_seed(
            os.path.join(sd, "model_output.pt"), labels,
            subset_size=subset_size, device=device,
        )
        for k, v in res.items():
            per_seed[k].append(v)
        print(f"  {os.path.basename(sd)}: " + ", ".join(f"{k}={v:.4f}" for k, v in res.items()))

    out = {k: float(np.mean(v)) for k, v in per_seed.items()}
    out.update({f"{k}_std": float(np.std(v)) for k, v in per_seed.items()})
    out["num_seeds"] = len(seed_dirs)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[done] {out_path}")
    return out


def iter_run_dirs(results_dir: str, dataset: str = None):
    pattern = os.path.join(results_dir, dataset if dataset else "*", "*")
    for d in sorted(glob.glob(pattern)):
        if os.path.isdir(d) and glob.glob(os.path.join(d, "seed_*")):
            yield d


def main():
    parser = argparse.ArgumentParser(
        description="Topic-guided retrieval Precision@k from local dsl-train results")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Root results directory (default: results/)")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Evaluate a single run directory instead of scanning results_dir")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Restrict scanning to results_dir/<dataset>/")
    parser.add_argument("--subset_size", type=int, default=-1,
                        help="Subsample this many documents for retrieval (-1 = all)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if retrieval.json already exists")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.run_dir:
        evaluate_run_dir(args.run_dir, subset_size=args.subset_size, device=device, force=args.force)
        return

    found = False
    for run_dir in iter_run_dirs(args.results_dir, args.dataset):
        found = True
        print(f"\n=== {run_dir} ===")
        evaluate_run_dir(run_dir, subset_size=args.subset_size, device=device, force=args.force)
    if not found:
        print(f"No runs found under {args.results_dir}. Train models first with dsl-train.")


if __name__ == "__main__":
    main()
