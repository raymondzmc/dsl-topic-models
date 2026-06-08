"""Byte-identity tests for the cli/_io.py result-writer helpers (PR5).

Writes via the helpers and via the original inline json.dump / torch.save calls
into two temp dirs and asserts the files are byte-for-byte identical, confirming
ensure_ascii / no-indent / utf-8 parity. No GPU/network needed.
"""
import json
import filecmp

import torch

from dsl_topic.cli import _io


def _inline_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f)


def test_dump_json_byte_identical(tmp_path):
    obj = {"b": 2, "a": 1, "unicode": "café — ünïcôde", "nested": [1, {"x": 0.5}]}
    p_helper = tmp_path / "helper.json"
    p_inline = tmp_path / "inline.json"
    _io._dump_json(obj, str(p_helper))
    _inline_json(obj, str(p_inline))
    assert filecmp.cmp(p_helper, p_inline, shallow=False)


def test_save_evaluation_and_aggregate_and_labels(tmp_path):
    seed_dir = tmp_path / "seed_0"
    seed_dir.mkdir()
    results_dir = tmp_path
    evaluation = {"topic_diversity": 0.5, "inverted_rbo": 0.9, "training_time": 1.23}
    averaged = {"topic_diversity": 0.5}
    labels = [0, 1, 1, 2]

    _io.save_evaluation(evaluation, str(seed_dir))
    _io.save_aggregate(averaged, str(results_dir))
    _io.save_labels(labels, str(results_dir))

    # Compare against inline equivalents.
    ref = tmp_path / "ref"
    ref.mkdir()
    _inline_json(evaluation, str(ref / "evaluation_results.json"))
    _inline_json(averaged, str(ref / "averaged_results.json"))
    _inline_json(labels, str(ref / "labels.json"))

    assert filecmp.cmp(seed_dir / "evaluation_results.json", ref / "evaluation_results.json", shallow=False)
    assert filecmp.cmp(results_dir / "averaged_results.json", ref / "averaged_results.json", shallow=False)
    assert filecmp.cmp(results_dir / "labels.json", ref / "labels.json", shallow=False)


def test_save_model_output_writes_pt_and_topics(tmp_path):
    seed_dir = tmp_path / "seed_0"
    seed_dir.mkdir()
    model_output = {"topics": [["a", "b"], ["c", "d"]], "training_time": 2.0}
    _io.save_model_output(model_output, str(seed_dir))

    assert (seed_dir / "model_output.pt").exists()
    loaded = torch.load(seed_dir / "model_output.pt", weights_only=False)
    assert loaded == model_output

    ref = tmp_path / "topics_ref.json"
    _inline_json(model_output["topics"], str(ref))
    assert filecmp.cmp(seed_dir / "topics.json", ref, shallow=False)
