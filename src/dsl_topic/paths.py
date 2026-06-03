"""Centralized filesystem paths for data and heavy evaluation assets.

Everything resolves under ``DATA_DIR``, which defaults to ``./data`` relative to
the current working directory (so run the CLIs / scripts from the repository
root) and can be overridden with the ``DSL_TOPIC_DATA_DIR`` environment variable.

Layout::

    <DATA_DIR>/
        raw_data/        # committed source corpora (.tsv)
        processed_data/  # LM soft-label datasets (downloaded from HF or regenerated)
        wikipedia/       # Palmetto JAR + Lucene index for the C_V (cv_wiki) metric
"""
import os

DATA_DIR = os.path.abspath(os.environ.get("DSL_TOPIC_DATA_DIR", "data"))

PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed_data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
WIKIPEDIA_DIR = os.path.join(DATA_DIR, "wikipedia")

# Palmetto Wikipedia-based C_V coherence (the paper's reported "C_V"); optional.
PALMETTO_JAR = os.path.join(WIKIPEDIA_DIR, "palmetto-0.1.0-jar-with-dependencies.jar")
WIKIPEDIA_INDEX = os.path.join(WIKIPEDIA_DIR, "wikipedia_bd")
