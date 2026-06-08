"""Faithfulness tests for the shared DSL objective helpers (PR6).

Each helper is checked against an inlined copy of the exact pre-refactor block
it replaced, on fixed-seed tensors. Version-independent and CPU-only: if these
pass, the four DSL models (which only call these helpers) reproduce their
original masking / loss / topic-extraction bit-for-bit.
"""
import math

import numpy as np
import torch

from dsl_topic.models.dsl._objective import (
    topk_target, distillation_loss, extract_topics,
)


def _logits(seed=0, n=5, v=12):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, v, generator=g)


# --- topk_target -----------------------------------------------------------

def test_topk_multiplicative_matches_inline():
    y = _logits()
    for sparsity in (1.0, 0.5, 0.25):
        k = math.ceil(sparsity * y.size(1))
        idx = torch.topk(y, k=k, dim=1)[1]
        mask = torch.zeros_like(y)
        mask.scatter_(1, idx, 1.0)
        ref = y * mask
        got = topk_target(y, sparsity_ratio=sparsity, mask_mode="multiplicative")
        assert torch.equal(got, ref)


def test_topk_neg_inf_matches_inline():
    y = _logits()
    for sparsity in (1.0, 0.5):
        k = math.ceil(sparsity * y.size(1))
        vals, idx = torch.topk(y, k=k, dim=1)
        ref = torch.full_like(y, float('-inf'))
        ref.scatter_(1, idx, vals)
        got = topk_target(y, sparsity_ratio=sparsity, mask_mode="neg_inf")
        assert torch.equal(got, ref)


def test_topk_k_precedence_over_sparsity():
    # prodlda passes both k and sparsity_ratio; k must win when not None.
    y = _logits()
    vals, idx = torch.topk(y, k=4, dim=1)
    ref = torch.full_like(y, float('-inf'))
    ref.scatter_(1, idx, vals)
    got = topk_target(y, k=4, sparsity_ratio=0.9, mask_mode="neg_inf")
    assert torch.equal(got, ref)


# --- distillation_loss -----------------------------------------------------

def _student(seed=1, n=5, v=12):
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(n, v, generator=g), dim=-1)


def test_kl_matches_inline():
    masked = topk_target(_logits(), sparsity_ratio=0.5, mask_mode="multiplicative")
    student = _student()
    T = 3.0
    teacher_probs = torch.softmax(masked / T, dim=-1).clamp_min(1e-9)
    s = student.clamp_min(1e-9)
    ref = torch.sum(teacher_probs * torch.log(teacher_probs / s), dim=1)
    got = distillation_loss(masked, student, loss_type="KL", temperature=T,
                            ce_softmax_teacher=False)
    assert torch.equal(got, ref)


def test_ce_raw_teacher_matches_inline():
    # etm/ecrtm CE: weight by the raw masked logits (no teacher softmax).
    masked = topk_target(_logits(), sparsity_ratio=0.5, mask_mode="multiplicative")
    student = _student()
    ref = -torch.sum(masked * torch.log(student + 1e-10), dim=1)
    got = distillation_loss(masked, student, loss_type="CE", temperature=3.0,
                            ce_softmax_teacher=False)
    assert torch.equal(got, ref)


def test_ce_softmax_teacher_matches_inline():
    # prodlda CE: softmax the teacher first.
    masked = topk_target(_logits(), sparsity_ratio=0.5, mask_mode="neg_inf")
    student = _student()
    T = 3.0
    teacher_probs = torch.softmax(masked / T, dim=-1)
    ref = -torch.sum(teacher_probs * torch.log(student + 1e-10), dim=1)
    got = distillation_loss(masked, student, loss_type="CE", temperature=T,
                            ce_softmax_teacher=True)
    assert torch.equal(got, ref)


# --- extract_topics --------------------------------------------------------

def _inline_extract(beta, top_words, idx2token=None, vocab=None):
    topics = []
    for k in range(beta.shape[0]):
        if np.isnan(beta[k]).any():
            return None
        top_indices = beta[k].argsort()[-top_words:][::-1]
        if idx2token is not None:
            topics.append([idx2token[i] for i in top_indices])
        elif vocab is not None:
            topics.append([vocab[i] for i in top_indices])
        else:
            topics.append(list(top_indices))
    return topics


def test_extract_topics_matches_inline():
    rng = np.random.RandomState(0)
    beta = rng.randn(3, 7).astype("float32")
    idx2token = {i: f"w{i}" for i in range(7)}
    vocab = [f"v{i}" for i in range(7)]
    for kw in ({}, {"idx2token": idx2token}, {"vocab": vocab},
               {"idx2token": idx2token, "vocab": vocab}):
        assert extract_topics(beta, 4, **kw) == _inline_extract(beta, 4, **kw)


def test_extract_topics_nan_guard():
    beta = np.ones((3, 5), dtype="float32")
    beta[1, 0] = np.nan
    assert extract_topics(beta, 3) is None
