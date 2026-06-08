"""Structural tests for the model-construction registry (PR7).

The registry must cover exactly the CLI's ``--model`` choices, and the thin
``train_model`` dispatcher must preserve the original unknown-model error.
No GPU/data needed (no model is constructed).
"""
import argparse

import pytest

from dsl_topic.cli.train import ALL_MODELS, train_model
from dsl_topic.cli._model_builders import MODEL_BUILDERS


def test_registry_covers_all_models():
    assert set(MODEL_BUILDERS) == ALL_MODELS


def test_unknown_model_raises():
    args = argparse.Namespace()
    with pytest.raises(ValueError, match="Unknown model: nope"):
        train_model(
            model_name="nope", args=args, seed=0, checkpoint_dir="/tmp",
            local_data_path="/tmp", vocab=[], bow_corpus=[],
        )


def test_shared_ctm_builder_for_zeroshot_and_combined():
    # zeroshot and combined historically share one branch.
    assert MODEL_BUILDERS["zeroshot"] is MODEL_BUILDERS["combined"]
