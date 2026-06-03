"""Smoke test for the evaluation metrics on tiny synthetic data.

Verifies that evaluate_topic_model() produces the diversity / purity metrics in
sane ranges, and that the OpenAI LLM rating is cleanly skipped when requested
(so the test needs no API key, GPU, downloaded data, or Palmetto/Java).
"""
import numpy as np

from dsl_topic.evaluation.metrics import evaluate_topic_model


def _synthetic_output():
    topics = [["cat", "dog", "fish"], ["tree", "bird", "leaf"]]
    # 2 topics x 4 documents (topic-document matrix); argmax → hard assignment.
    tdm = np.array([[0.9, 0.8, 0.1, 0.2],
                    [0.1, 0.2, 0.9, 0.8]], dtype=float)
    return {"topics": topics, "topic-document-matrix": tdm}


def test_evaluate_topic_model_skips_llm_rating():
    out = _synthetic_output()
    labels = [0, 0, 1, 1]

    results = evaluate_topic_model(
        out, top_words=3, labels=labels, skip_llm_rating=True,
    )

    # Core metrics are present and in range.
    assert 0.0 <= results["topic_diversity"] <= 1.0
    assert 0.0 <= results["inverted_rbo"] <= 1.0
    assert 0.0 <= results["purity"] <= 1.0
    # The gensim cv/npmi coherence were removed; C_V is Palmetto-only (cv_wiki),
    # which is skipped here since the Wikipedia index isn't available.
    assert "cv" not in results and "npmi" not in results
    # LLM rating must be skipped (no OpenAI call).
    assert "llm_rating" not in results


def test_perfect_purity():
    out = _synthetic_output()
    # Documents 0,1 -> topic 0; documents 2,3 -> topic 1; labels match exactly.
    results = evaluate_topic_model(out, top_words=3, labels=[0, 0, 1, 1], skip_llm_rating=True)
    assert results["purity"] == 1.0
