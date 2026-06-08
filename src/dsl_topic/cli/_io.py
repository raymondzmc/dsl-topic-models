"""Shared result-persistence helpers for the training CLI.

Centralizes the exact ``torch.save`` / ``json.dump`` patterns used by both the
training loop (``run``) and the W&B re-evaluation path (``run_reevaluate``) in
``dsl_topic.cli.train``. JSON is written with the same options every call site
used before this was extracted: default ``ensure_ascii`` (no ``indent``,
no ``sort_keys``) and a ``utf-8`` file handle. The intentional asymmetry
between the two paths (which files each writes, and the conditional guards) is
preserved by the call sites, not by these primitives.
"""
import os
import json

import torch


def _dump_json(obj, path: str) -> None:
    """Write ``obj`` as JSON to ``path`` (matches every original call site)."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f)


def save_model_output(model_output: dict, seed_dir: str) -> None:
    """Save ``model_output.pt`` and ``topics.json`` for one seed."""
    torch.save(model_output, os.path.join(seed_dir, 'model_output.pt'))
    _dump_json(model_output['topics'], os.path.join(seed_dir, 'topics.json'))


def save_evaluation(evaluation_results: dict, seed_dir: str) -> None:
    """Save ``evaluation_results.json`` for one seed."""
    _dump_json(evaluation_results, os.path.join(seed_dir, 'evaluation_results.json'))


def save_aggregate(averaged_results: dict, results_dir: str) -> None:
    """Save ``averaged_results.json`` for a run."""
    _dump_json(averaged_results, os.path.join(results_dir, 'averaged_results.json'))


def save_labels(labels, results_dir: str) -> None:
    """Save ``labels.json`` for a run."""
    _dump_json(labels, os.path.join(results_dir, 'labels.json'))
