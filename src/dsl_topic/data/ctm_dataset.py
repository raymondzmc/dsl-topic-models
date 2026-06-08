import torch
import numpy as np
from dsl_topic.models.baselines.octis.contextualized_topic_models.datasets import CTMDataset
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from collections import Counter


def _resolve_dataset(data: Union[Dict[str, Any], Dataset]) -> Dataset:
    """Unwrap the input into a HuggingFace ``Dataset`` (or raise)."""
    if isinstance(data, Dataset):
        return data
    elif isinstance(data, dict) and 'hf_dataset' in data:
        return data['hf_dataset']
    else:
        raise ValueError("Invalid data type")


def _build_x_embeddings(
    dataset: Dataset,
    embedding_model: Optional[SentenceTransformer],
    layer_idx: int,
    n_samples: int,
    emb_dim: int,
) -> np.ndarray:
    """Build the (n_samples, emb_dim) topic-model input embeddings.

    With ``embedding_model`` (ablation), encode the ``bow`` texts with that
    SentenceTransformer; otherwise read the precomputed ``input_embeddings``
    (selecting ``layer_idx`` when per-layer states were saved).
    """
    # If embedding_model is provided, generate embeddings from texts
    # Use the 'bow' field (space-separated tokens) as the document text
    if embedding_model is not None:
        print("Generating embeddings using provided SentenceTransformer model...")
        if torch.cuda.is_available():
            embedding_model = embedding_model.to('cuda')
        texts = dataset['bow']
        x_embeddings = np.array(embedding_model.encode(texts, show_progress_bar=True, batch_size=32))
    else:
        # Pre-allocate arrays for original embeddings
        x_embeddings = np.zeros((n_samples, emb_dim), dtype=np.float32)
        print(f"Loading {n_samples} embeddings...")
        for i, item in enumerate(tqdm(dataset, desc="Extracting embeddings")):
            emb = np.array(item['input_embeddings'])
            if emb.ndim == 2:
                x_embeddings[i] = emb[layer_idx]
            else:
                x_embeddings[i] = emb
    return x_embeddings


def _build_target(
    dataset: Dataset,
    vocab: List[str],
    token2idx: Dict[str, int],
    use_bow_target: bool,
    n_samples: int,
    logits_dim: int,
) -> np.ndarray:
    """Build the reconstruction target: BoW counts (ablation) or LLM logits."""
    # If use_bow_target is True, compute BoW representation as target
    if use_bow_target:
        print("Computing BoW target...")
        y = np.zeros((n_samples, len(vocab)), dtype=np.float32)
        for i, item in enumerate(tqdm(dataset, desc="Computing BoW")):
            tokens = item['bow'].split()
            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                if token in token2idx:
                    y[i, token2idx[token]] = count
    else:
        # Use LLM predicted logits as target
        y = np.zeros((n_samples, logits_dim), dtype=np.float32)
        print(f"Loading {n_samples} logits...")
        for i, item in enumerate(tqdm(dataset, desc="Extracting logits")):
            y[i] = np.array(item['next_word_logits'])
    return y


def get_ctm_dataset_from_processed_data(
    data: Union[Dict[str, Any], Dataset],
    vocab: List[str],
    layer_idx: int = -1,
    embedding_model: Optional[SentenceTransformer] = None,
    use_bow_target: bool = False,
) -> CTMDataset:
    """Create CTM dataset from processed dsl model data.

    Args:
        data: Dictionary with 'input_embeddings' and 'next_word_logits' OR HuggingFace Dataset
        vocab: Vocabulary list
        layer_idx: Which layer's embeddings to use (-1 for last layer)

        Ablation Experiments:
            embedding_model: Use another embedding model to generate x_embeddings
            use_bow_target: Use BoW rather than LLM predicted targets

    Returns:
        CTMDataset ready for training and inference
    """
    # Check if data is a HuggingFace Dataset or a dict containing one
    dataset = _resolve_dataset(data)

    # Optimized loading from HuggingFace dataset with progress bar
    n_samples = len(dataset)

    # Peek at first element to determine shapes
    first_item = dataset[0]
    first_emb = np.array(first_item['input_embeddings'])
    first_logits = np.array(first_item['next_word_logits'])

    # Determine embedding shape from original embeddings (for fallback)
    if first_emb.ndim == 2:
        emb_dim = first_emb.shape[1]
    else:
        emb_dim = first_emb.shape[0]

    logits_dim = first_logits.shape[0]

    # Build vocab lookup for BoW target
    token2idx = {token: i for i, token in enumerate(vocab)}

    x_embeddings = _build_x_embeddings(
        dataset, embedding_model, layer_idx, n_samples, emb_dim
    )
    y = _build_target(
        dataset, vocab, token2idx, use_bow_target, n_samples, logits_dim
    )

    # Create idx2token mapping
    idx2token = {i: token for i, token in enumerate(vocab)}
    dataset = CTMDataset(
        x_bow=None,
        x_embeddings=x_embeddings,
        idx2token=idx2token,
        y=y,
    )
    return dataset