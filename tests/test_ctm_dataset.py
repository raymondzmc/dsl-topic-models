"""Bit-identity tests for the ctm_dataset.py function split (PR4).

Feeds a tiny in-memory HuggingFace Dataset through
get_ctm_dataset_from_processed_data and asserts the produced x_embeddings / y
arrays exactly equal the original inline computation, for both target modes and
both embedding layouts (1D and per-layer 2D). No GPU/network needed.
"""
import numpy as np
import pytest
from collections import Counter

from datasets import Dataset

from dsl_topic.data.ctm_dataset import get_ctm_dataset_from_processed_data

VOCAB = ["alpha", "beta", "gamma", "delta", "epsilon"]
LOGITS_DIM = len(VOCAB)
EMB_DIM = 4
N_LAYERS = 3


def _make_dataset(two_d_embeddings):
    rng = np.random.RandomState(0)
    rows = []
    bows = ["alpha beta beta", "gamma", "delta delta epsilon alpha", "beta gamma gamma gamma"]
    for bow in bows:
        if two_d_embeddings:
            emb = rng.randn(N_LAYERS, EMB_DIM).astype(np.float32).tolist()
        else:
            emb = rng.randn(EMB_DIM).astype(np.float32).tolist()
        logits = rng.randn(LOGITS_DIM).astype(np.float32).tolist()
        rows.append({"input_embeddings": emb, "next_word_logits": logits, "bow": bow})
    return Dataset.from_list(rows)


def _expected(dataset, use_bow_target, layer_idx=-1):
    """The exact pre-refactor inline computation."""
    n_samples = len(dataset)
    first_emb = np.array(dataset[0]["input_embeddings"])
    first_logits = np.array(dataset[0]["next_word_logits"])
    emb_dim = first_emb.shape[1] if first_emb.ndim == 2 else first_emb.shape[0]
    logits_dim = first_logits.shape[0]
    token2idx = {token: i for i, token in enumerate(VOCAB)}

    x_embeddings = np.zeros((n_samples, emb_dim), dtype=np.float32)
    for i, item in enumerate(dataset):
        emb = np.array(item["input_embeddings"])
        x_embeddings[i] = emb[layer_idx] if emb.ndim == 2 else emb

    if use_bow_target:
        y = np.zeros((n_samples, len(VOCAB)), dtype=np.float32)
        for i, item in enumerate(dataset):
            for token, count in Counter(item["bow"].split()).items():
                if token in token2idx:
                    y[i, token2idx[token]] = count
    else:
        y = np.zeros((n_samples, logits_dim), dtype=np.float32)
        for i, item in enumerate(dataset):
            y[i] = np.array(item["next_word_logits"])
    return x_embeddings, y


@pytest.mark.parametrize("two_d", [False, True])
@pytest.mark.parametrize("use_bow_target", [False, True])
def test_matches_inline(two_d, use_bow_target):
    dataset = _make_dataset(two_d)
    out = get_ctm_dataset_from_processed_data(dataset, VOCAB, use_bow_target=use_bow_target)
    exp_x, exp_y = _expected(dataset, use_bow_target)
    assert np.array_equal(out.x_embeddings, exp_x)
    assert np.array_equal(out.y, exp_y)
    assert out.idx2token == {i: t for i, t in enumerate(VOCAB)}
