"""CPU end-to-end + determinism test for the DSL ProdLDA backbone.

Builds a tiny synthetic CTM dataset (no data download, no GPU, no API keys) and
trains ``DSLProdLDA`` for a couple of epochs. Asserts that the pipeline runs,
topics are extracted, the always-on metrics are finite, and that a fixed seed
reproduces identical topics. This is the only test that exercises the full
train→evaluate path, so it is the guard against the model/objective/eval wiring
silently breaking.
"""
import numpy as np

from dsl_topic.models.baselines.octis.contextualized_topic_models.datasets import CTMDataset
from dsl_topic.models.dsl.prodlda import DSLProdLDA
from dsl_topic.evaluation.metrics import evaluate_topic_model
from dsl_topic.cli.train import set_seed

VOCAB_SIZE, EMB_DIM, N_DOCS, NUM_TOPICS, TOP_WORDS = 30, 16, 40, 5, 5


def _synthetic_dataset():
    """A deterministic, tiny stand-in for a processed DSL dataset."""
    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(VOCAB_SIZE)]
    x_embeddings = rng.standard_normal((N_DOCS, EMB_DIM)).astype(np.float32)
    teacher_logits = rng.standard_normal((N_DOCS, VOCAB_SIZE)).astype(np.float32)  # the DSL target y
    idx2token = {i: w for i, w in enumerate(vocab)}
    ds = CTMDataset(x_bow=None, x_embeddings=x_embeddings, idx2token=idx2token, y=teacher_logits)
    return ds, vocab


def _train_once(ds, seed=0):
    set_seed(seed)
    model = DSLProdLDA(
        vocab_size=VOCAB_SIZE, embedding_size=EMB_DIM, num_topics=NUM_TOPICS,
        hidden_sizes=(32,), activation="softplus", solver="adam",
        num_epochs=2, batch_size=16, lr=1e-3,
        loss_weight=1.0, sparsity_ratio=1.0, topk=None,
        loss_type="KL", temperature=3.0, top_words=TOP_WORDS,
    )
    model.fit(ds)
    return model.get_info()


def test_dsl_prodlda_end_to_end_cpu():
    ds, vocab = _synthetic_dataset()
    info = _train_once(ds)

    assert info.get("topics") is not None, "training produced NaN topics"
    assert len(info["topics"]) == NUM_TOPICS
    vocab_set = set(vocab)
    for topic in info["topics"]:
        assert 1 <= len(topic) <= TOP_WORDS
        assert all(word in vocab_set for word in topic)

    metrics = evaluate_topic_model(info, top_words=TOP_WORDS, labels=None, skip_llm_rating=True)
    for key in ("topic_diversity", "inverted_rbo"):
        assert key in metrics, f"missing metric {key}"
        assert np.isfinite(metrics[key]) and 0.0 <= metrics[key] <= 1.0


def test_dsl_prodlda_seed_determinism():
    ds, _ = _synthetic_dataset()
    first = _train_once(ds, seed=0)
    second = _train_once(ds, seed=0)
    assert first["topics"] == second["topics"], "same seed should reproduce identical topics"
