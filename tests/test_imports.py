"""Smoke test: the package and all CLI entry points import cleanly.

This is the fast guard for the src/ layout + import wiring; it needs no data,
GPU, or API keys.
"""
import importlib

import pytest

CORE_MODULES = [
    "dsl_topic",
    "dsl_topic.paths",
    "dsl_topic.settings",
    "dsl_topic.models.dsl.prodlda",
    "dsl_topic.models.dsl.etm",
    "dsl_topic.models.dsl.ecrtm",
    "dsl_topic.models.dsl.fastopic",
    "dsl_topic.models.dsl._objective",
    "dsl_topic.models.baselines.fastopic",
    "dsl_topic.models.baselines.octis.CTM",
    "dsl_topic.models.baselines.octis.ETM",
    "dsl_topic.models.baselines.octis.LDA",
    "dsl_topic.models.baselines.octis.ProdLDA",
    "dsl_topic.models.baselines.topmost.ECRTM.ECRTM",
    "dsl_topic.data.loaders",
    "dsl_topic.data.ctm_dataset",
    "dsl_topic.data.octis_dataset",
    "dsl_topic.evaluation.metrics",
    "dsl_topic.evaluation.coherence",
    "dsl_topic.evaluation.diversity",
    "dsl_topic.evaluation.retrieval",
    "dsl_topic.prompts.renderer",
]

CLI_MODULES = [
    "dsl_topic.cli.process_dataset",
    "dsl_topic.cli.train",
    "dsl_topic.cli.retrieval",
    "dsl_topic.cli.summarize",
]

HELPER_MODULES = [
    "dsl_topic.cli._io",
    "dsl_topic.cli._model_builders",
]


@pytest.mark.parametrize("module", CORE_MODULES + CLI_MODULES + HELPER_MODULES)
def test_module_imports(module):
    importlib.import_module(module)


@pytest.mark.parametrize("module", CLI_MODULES)
def test_cli_has_main(module):
    mod = importlib.import_module(module)
    assert callable(getattr(mod, "main", None)), f"{module} is missing a main() entry point"


def test_settings_optional_without_env():
    """settings must construct even with no credentials present."""
    from dsl_topic.settings import settings
    # All credential fields are optional (None by default); accessing them is fine.
    _ = settings.openai_api_key, settings.hf_token, settings.wandb_api_key
