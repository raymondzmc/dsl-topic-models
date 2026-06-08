"""Shared building blocks for the DSL training objective.

The four DSL trainers (etm/ecrtm/fastopic/prodlda) repeat the same top-k target
masking, CE/KL distillation loss, topic extraction, and theta-inference loops.
These helpers factor out that duplication **without unifying the load-bearing
differences** between models — those differences are expressed as parameters so
each call site reproduces its original computation bit-for-bit:

- mask_mode: etm/ecrtm zero out non-top-k logits (``multiplicative``); prodlda/
  fastopic fill them with ``-inf`` (``neg_inf``). These differ once sparsity<1.
- ce_softmax_teacher: etm/ecrtm weight CE by the raw masked logits; prodlda
  softmaxes the teacher first.

Each loss helper returns the **un-reduced (batch,) vector** so every caller
keeps its own mean/sum reduction at the call site.
"""
import math
from typing import Literal, Optional

import numpy as np
import torch


def topk_target(
    teacher_logits: torch.Tensor,
    *,
    k: Optional[int] = None,
    sparsity_ratio: Optional[float] = None,
    mask_mode: Literal["multiplicative", "neg_inf"],
) -> torch.Tensor:
    """Keep only the top-k logits per row; mask the rest.

    ``k`` takes precedence when not None; otherwise
    ``k = ceil(sparsity_ratio * V)`` where ``V = teacher_logits.size(1)`` — the
    exact precedence prodlda used.

    mask_mode='multiplicative' (etm/ecrtm): non-top-k entries become 0.0.
    mask_mode='neg_inf' (prodlda/fastopic): non-top-k entries become -inf.
    """
    if k is None:
        k = math.ceil(sparsity_ratio * teacher_logits.size(1))

    if mask_mode == "multiplicative":
        topk_indices = torch.topk(teacher_logits, k=k, dim=1)[1]
        mask = torch.zeros_like(teacher_logits)
        mask.scatter_(1, topk_indices, 1.0)
        return teacher_logits * mask
    elif mask_mode == "neg_inf":
        topk_vals, topk_idx = torch.topk(teacher_logits, k=k, dim=1)
        masked_logits = torch.full_like(teacher_logits, float('-inf'))
        masked_logits.scatter_(1, topk_idx, topk_vals)
        return masked_logits
    else:
        raise ValueError(f"Invalid mask_mode: {mask_mode}")


def distillation_loss(
    masked_logits: torch.Tensor,
    student_probs: torch.Tensor,
    *,
    loss_type: Literal["CE", "KL"],
    temperature: float,
    ce_softmax_teacher: bool,
) -> torch.Tensor:
    """Per-row distillation loss, shape (batch,). No reduction is applied.

    KL (identical across all callers):
        t = softmax(masked_logits / T).clamp_min(1e-9)
        s = student_probs.clamp_min(1e-9)
        return sum(t * log(t / s), dim=1)

    CE with ce_softmax_teacher=False (etm/ecrtm):
        return -sum(masked_logits * log(student_probs + 1e-10), dim=1)
    CE with ce_softmax_teacher=True (prodlda):
        t = softmax(masked_logits / T)
        return -sum(t * log(student_probs + 1e-10), dim=1)
    """
    if loss_type == 'CE':
        if ce_softmax_teacher:
            teacher_probs = torch.softmax(masked_logits / temperature, dim=-1)
            return -torch.sum(teacher_probs * torch.log(student_probs + 1e-10), dim=1)
        return -torch.sum(masked_logits * torch.log(student_probs + 1e-10), dim=1)
    elif loss_type == 'KL':
        teacher_probs = torch.softmax(masked_logits / temperature, dim=-1).clamp_min(1e-9)
        student_probs = student_probs.clamp_min(1e-9)
        return torch.sum(teacher_probs * torch.log(teacher_probs / student_probs), dim=1)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")


def extract_topics(
    beta: np.ndarray,
    top_words: int,
    *,
    idx2token: Optional[dict] = None,
    vocab: Optional[list] = None,
) -> Optional[list]:
    """Extract the top-``top_words`` words per topic row of ``beta``.

    Returns ``None`` for the whole result if any topic row contains NaN (the
    original NaN guard). Word resolution ladder: ``idx2token`` if given, else
    ``vocab`` if given, else the raw integer indices. ``beta`` has shape
    ``(num_topics, vocab_size)``.
    """
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


def theta_inference(
    model: torch.nn.Module,
    ctm_dataset,
    *,
    device: torch.device,
    batch_size: int,
    theta_fn,
) -> np.ndarray:
    """Eval-mode DataLoader inference loop shared by etm/ecrtm/fastopic.

    Iterates ``ctm_dataset`` (shuffle=False), moves ``x_embeddings`` to
    ``device``, applies the model-specific ``theta_fn(x)`` and concatenates the
    per-batch results into an ``(N, K)`` array.
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(ctm_dataset, batch_size=batch_size, shuffle=False)
    all_theta = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch['x_embeddings'].to(device)
            all_theta.append(theta_fn(x).cpu().numpy())
    return np.concatenate(all_theta, axis=0)
