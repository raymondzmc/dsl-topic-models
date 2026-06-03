"""Topic-guided retrieval evaluation.

Ranks documents by the KL divergence between their topic distributions and reports
Precision@k. The CLI in ``dsl_topic.cli.retrieval`` calls ``evaluate_single_seed``
over the per-seed ``model_output.pt`` files written by ``dsl-train``.
"""
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm


def compute_pairwise_kl_divergence_torch(P, device):
    """
    Compute pairwise KL divergence between rows of a matrix P using PyTorch.
    KL(P_i || P_j) = sum_k P_i[k] * (log P_i[k] - log P_j[k])
    This version computes the matrix row-by-row to be more memory-efficient.
    Args:
        P (torch.Tensor): Input tensor of shape (n_docs, n_features) where rows are distributions.
        device (torch.device): The device (CPU or CUDA) to perform calculations on.
    Returns:
        torch.Tensor: Pairwise KL divergence matrix of shape (n_docs, n_docs).
    """
    P = P.to(device)
    n_docs, n_features = P.shape
    P_norm = P / P.sum(dim=1, keepdims=True)
    P_stable = P_norm + 1e-12
    logP_all = P_stable.log()
    kl_matrix = torch.zeros((n_docs, n_docs))

    print(f"  Calculating KL divergence row-by-row for {n_docs} documents...")
    for i in tqdm(range(n_docs), desc="  KL Div Row", unit="doc", leave=False, dynamic_ncols=True):
        log_diff = logP_all[i, :].unsqueeze(0) - logP_all
        product = P_norm[i, :].unsqueeze(0) * log_diff
        kl_matrix[i, :] = product.sum(dim=-1).cpu()
    kl_matrix.fill_diagonal_(0)
    return kl_matrix


def compute_precision_at_k(retrieved_indices, query_label, all_labels, k_values=[1, 5, 10]):
    """Compute precision@k for the retrieved documents."""
    results = {}

    for k in k_values:
        if k > len(retrieved_indices):
            continue
        top_k_indices = retrieved_indices[:k]
        top_k_labels = all_labels[top_k_indices]
        precision = np.mean(top_k_labels == query_label)
        results[f'precision@{k}'] = precision

    return results


def apply_subsetting(labels, retrieval_representation, subset_size):
    """Apply subsetting for an even number of documents per label."""
    label_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        label_to_indices[label].append(i)

    min_count = min(len(indices) for indices in label_to_indices.values())
    if subset_size < len(label_to_indices) * min_count:
        docs_per_label = subset_size // len(label_to_indices)
        min_count = min(min_count, docs_per_label)

    subset_indices = sum([indices[:min_count] for indices in label_to_indices.values()], [])
    labels = labels[subset_indices]
    retrieval_representation = retrieval_representation[subset_indices]
    return labels, retrieval_representation


def evaluate_single_seed(
    model_output_path: str,
    labels: np.ndarray,
    subset_size: int = -1,
    device: torch.device = None,
) -> dict:
    """
    Evaluate retrieval metrics for a single seed's model output.

    Args:
        model_output_path: Path to model_output.pt file
        labels: Document labels (already filtered for empty docs)
        subset_size: Number of documents to subset (-1 for all)
        device: Torch device to use

    Returns:
        Dictionary with precision@k results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model output
    output = torch.load(model_output_path, weights_only=False)
    if 'topic-document-matrix' not in output.keys():
        raise ValueError(f"topic-document-matrix not found in output keys at {model_output_path}")

    topic_distribution_np: np.ndarray = output['topic-document-matrix'].transpose()

    # Validate alignment with labels
    if topic_distribution_np.shape[0] != len(labels):
        raise ValueError(
            f"Shape mismatch between topic distributions ({topic_distribution_np.shape[0]}) "
            f"and labels ({len(labels)}). The model was likely trained with filtered data."
        )

    # Apply subsetting if requested
    eval_labels = labels.copy()
    if subset_size > 0:
        eval_labels, topic_distribution_np = apply_subsetting(
            eval_labels, topic_distribution_np, subset_size
        )

    print(f"  Evaluating {topic_distribution_np.shape[0]} documents...")

    # Compute pairwise KL divergence
    topic_distribution_tensor = torch.from_numpy(topic_distribution_np).float()
    kl_matrix = compute_pairwise_kl_divergence_torch(topic_distribution_tensor, device)
    similarity_matrix = kl_matrix.cpu().numpy()

    # Calculate precision@k for each document
    n_docs = similarity_matrix.shape[0]
    k_values = [1, 5, 10]
    max_k = max(k_values)

    precision_results = {k: [] for k in k_values}

    for i in tqdm(range(n_docs), desc="  Precision@k", leave=False, dynamic_ncols=True):
        # Get indices sorted by KL divergence (ascending, smaller is more similar)
        retrieved_indices = np.argsort(similarity_matrix[i])
        # Remove self
        retrieved_indices = retrieved_indices[retrieved_indices != i][:max_k]

        precisions = compute_precision_at_k(retrieved_indices, eval_labels[i], eval_labels, k_values)

        for k_val, precision in precisions.items():
            precision_results[int(k_val.split('@')[1])].append(precision)

    # Compute mean precision for each k
    results = {}
    for k in k_values:
        if precision_results[k]:
            results[f'precision@{k}'] = float(np.mean(precision_results[k]))

    return results
