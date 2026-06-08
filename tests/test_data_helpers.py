"""Unit tests for the data helpers extracted during refactoring.

These prove the extracted helpers are byte-for-byte equivalent to the original
inline logic they replaced, so behavior is preserved. No GPU/data/keys needed.
"""
import random

from dsl_topic.data.loaders import filter_empty_documents


def _inline_filter(bow_corpus, labels=None):
    """The exact pre-refactor inline expression (3 copies in the codebase)."""
    non_empty_indices = [i for i, doc in enumerate(bow_corpus) if len(doc) > 0]
    filtered_corpus = [bow_corpus[i] for i in non_empty_indices]
    filtered_labels = [labels[i] for i in non_empty_indices] if labels is not None else None
    return filtered_corpus, filtered_labels


def _random_corpus(rng):
    n = rng.randint(0, 30)
    corpus = []
    for _ in range(n):
        length = rng.choice([0, 0, 1, 2, 5])  # bias toward some empties
        corpus.append([f"w{rng.randint(0, 9)}" for _ in range(length)])
    return corpus


def test_matches_inline_with_labels():
    rng = random.Random(0)
    for _ in range(200):
        corpus = _random_corpus(rng)
        labels = [rng.randint(0, 4) for _ in corpus]
        assert filter_empty_documents(corpus, labels) == _inline_filter(corpus, labels)


def test_matches_inline_without_labels():
    rng = random.Random(1)
    for _ in range(200):
        corpus = _random_corpus(rng)
        got_corpus, got_labels = filter_empty_documents(corpus)
        ref_corpus, ref_labels = _inline_filter(corpus)
        assert got_corpus == ref_corpus
        assert got_labels is None and ref_labels is None


def test_order_and_alignment_preserved():
    corpus = [["a"], [], ["b", "c"], [], [], ["d"]]
    labels = [10, 11, 12, 13, 14, 15]
    fc, fl = filter_empty_documents(corpus, labels)
    assert fc == [["a"], ["b", "c"], ["d"]]
    assert fl == [10, 12, 15]  # labels of the surviving indices 0, 2, 5
