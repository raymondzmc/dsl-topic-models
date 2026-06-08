"""Dataset loading utilities for topic modeling."""

import os
import csv
import json
import shutil
from typing import Optional
from dataclasses import dataclass
from datasets import (
    load_dataset,
    load_from_disk,
    get_dataset_config_names,
    concatenate_datasets,
    Dataset,
)
from huggingface_hub import hf_hub_download

# Directory for storing processed datasets locally (resolves under DATA_DIR,
# i.e. ./data/processed_data by default; override with DSL_TOPIC_DATA_DIR).
from dsl_topic.paths import PROCESSED_DATA_DIR


@dataclass
class TrainingData:
    """Container for training data."""
    processed_dataset: Optional[dict]  # For dsl models (embeddings + logits)
    vocab: list[str]
    bow_corpus: list[list[str]]
    labels: Optional[list]
    metadata: Optional[dict]
    local_path: str


def get_hf_dataset(dataset_name: str, split: Optional[str] = None) -> Dataset:
    """Load a dataset from HuggingFace Hub.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to load (None or 'all' loads all splits)
        
    Returns:
        Concatenated dataset from all specified splits
    """
    configs = get_dataset_config_names(dataset_name, trust_remote_code=True)
    if 'default' in configs:
        configs = ['default']

    if len(configs) == 0:
        raise ValueError(f"No dataset configs found for {dataset_name}")

    datasets = []
    for cfg in configs:
        dataset = load_dataset(dataset_name, cfg, trust_remote_code=True)
        if split is None or split == 'all':
            all_splits = list(dataset.keys())
        else:
            all_splits = [split]

        for s in all_splits:
            datasets.append(dataset[s])

    dataset = concatenate_datasets(datasets)
    return dataset


def get_local_dataset(dataset_path: str) -> Dataset:
    """Load a dataset from a local TSV file.
    
    Args:
        dataset_path: Path to the TSV file
        
    Returns:
        Dataset loaded from the TSV file
    """
    import pandas as pd
    
    if not os.path.basename(dataset_path).endswith('.tsv'):
        raise ValueError(f"Dataset {dataset_path} is not a TSV file")
    
    # Use pandas with on_bad_lines to handle rows with extra tabs in text content
    df = pd.read_csv(
        dataset_path, 
        sep='\t', 
        on_bad_lines='warn',  # Skip malformed rows and warn
        engine='python',  # Python engine is more forgiving
    )
    
    # Convert pandas DataFrame to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    return dataset


def load_or_download_dataset(
    repo_id: str,
    force_download: bool = False,
    revision: Optional[str] = None,
) -> tuple[Dataset, list[str], dict]:
    """Load processed dataset from local cache or download from HuggingFace Hub.

    First tries to load from local processed_data/ directory. If not found,
    downloads from HuggingFace Hub and caches locally.

    Args:
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name') or dataset name
        force_download: Force re-download even if local exists
        revision: Pin the HF Hub revision (commit hash/tag/branch) for an
            immutable, reproducible download. Defaults to the
            ``DSL_TOPIC_HF_REVISION`` env var if set, else the latest revision.
            Note: only affects the *download* path — a pre-existing local cache is
            used as-is, so clear it (or pass ``force_download=True``) to switch
            revisions.

    Returns:
        Tuple of (dataset, vocab, metadata)
    """
    if revision is None:
        revision = os.environ.get("DSL_TOPIC_HF_REVISION")
    # Use basename of repo_id as local directory name
    local_name = repo_id.split("/")[-1]
    local_path = os.path.join(PROCESSED_DATA_DIR, local_name)
    
    # Try to load from local first
    if os.path.exists(local_path) and not force_download:
        print(f"Loading dataset from local cache: {local_path}")
        dataset = load_from_disk(local_path)
        
        vocab_file = os.path.join(local_path, "vocab.json")
        metadata_file = os.path.join(local_path, "metadata.json")
        
        if os.path.exists(vocab_file) and os.path.exists(metadata_file):
            with open(vocab_file, 'r') as f:
                vocab = json.load(f)
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            return dataset, vocab, metadata
        else:
            print(f"Warning: vocab.json or metadata.json not found in {local_path}")
    
    # Download from HuggingFace Hub
    print(f"Downloading dataset from HuggingFace Hub: {repo_id}"
          + (f" @ {revision}" if revision else ""))
    dataset = load_dataset(repo_id, split='train', revision=revision)

    # Download vocab and metadata
    vocab_hf_path = hf_hub_download(repo_id=repo_id, filename='vocab.json', repo_type='dataset', revision=revision)
    metadata_hf_path = hf_hub_download(repo_id=repo_id, filename='metadata.json', repo_type='dataset', revision=revision)
    
    with open(vocab_hf_path, 'r') as f:
        vocab = json.load(f)
    with open(metadata_hf_path, 'r') as f:
        metadata = json.load(f)
    
    # Save to local cache
    os.makedirs(local_path, exist_ok=True)
    dataset.save_to_disk(local_path)
    
    # Copy vocab and metadata to local path
    shutil.copy(vocab_hf_path, os.path.join(local_path, "vocab.json"))
    shutil.copy(metadata_hf_path, os.path.join(local_path, "metadata.json"))
    
    print(f"Dataset cached locally at: {local_path}")
    
    return dataset, vocab, metadata


def load_bow(dataset: Dataset) -> list[str]:
    """Load bag-of-words representations from a dataset.
    
    Args:
        dataset: HuggingFace Dataset with 'bow' column
        
    Returns:
        List of bow strings
    """
    if 'bow' not in dataset.column_names:
        raise ValueError("Dataset does not have 'bow' column")
    return dataset['bow']


def load_labels(dataset: Dataset) -> list:
    """Load labels from a dataset.
    
    Args:
        dataset: HuggingFace Dataset with 'label' column
        
    Returns:
        List of labels
    """
    if 'label' not in dataset.column_names:
        raise ValueError("Dataset does not have 'label' column")
    return dataset['label']


def load_vocab_from_hub(repo_id: str) -> list[str]:
    """Load vocabulary from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID
        
    Returns:
        List of vocabulary words
    """
    vocab_path = hf_hub_download(repo_id=repo_id, filename='vocab.json', repo_type='dataset')
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    return vocab


def load_metadata_from_hub(repo_id: str) -> dict:
    """Load metadata from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID
        
    Returns:
        Metadata dictionary
    """
    metadata_path = hf_hub_download(repo_id=repo_id, filename='metadata.json', repo_type='dataset')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata


def filter_empty_documents(
    bow_corpus: list[list[str]],
    labels: Optional[list] = None,
) -> tuple[list[list[str]], Optional[list]]:
    """Drop zero-token documents, preserving order and label alignment.

    OCTIS cannot handle empty documents, so they are removed before building
    OCTIS files / datasets. Iterates ``enumerate(bow_corpus)`` in ascending
    order and selects with that same index list, so the surviving documents
    (and their labels) keep their original relative order and stay aligned.

    Returns ``(filtered_corpus, filtered_labels)``; ``filtered_labels`` is
    ``None`` iff ``labels`` is ``None``.
    """
    non_empty_indices = [i for i, doc in enumerate(bow_corpus) if len(doc) > 0]
    filtered_corpus = [bow_corpus[i] for i in non_empty_indices]
    filtered_labels = [labels[i] for i in non_empty_indices] if labels is not None else None
    return filtered_corpus, filtered_labels


def prepare_octis_files(
    local_path: str,
    bow_corpus: list[list[str]],
    vocab: list[str],
    labels: Optional[list] = None
) -> None:
    """Prepare OCTIS-compatible files from processed dataset.
    
    Creates bow_dataset.txt, vocabulary.txt, corpus.tsv, and optionally numeric_labels.txt.
    Empty documents are skipped as OCTIS cannot handle them.
    
    Args:
        local_path: Directory to save files
        bow_corpus: List of tokenized documents
        vocab: List of vocabulary words
        labels: Optional list of labels
    """
    os.makedirs(local_path, exist_ok=True)

    # Filter out empty documents and corresponding labels
    filtered_corpus, filtered_labels = filter_empty_documents(bow_corpus, labels)

    # Save bow_dataset.txt (always overwrite to handle limit_dataset)
    bow_path = os.path.join(local_path, 'bow_dataset.txt')
    with open(bow_path, 'w', encoding='utf-8') as f:
        for doc in filtered_corpus:
            f.write(' '.join(doc) + '\n')
    
    # Save vocabulary.txt (OCTIS format)
    vocab_txt_path = os.path.join(local_path, 'vocabulary.txt')
    with open(vocab_txt_path, 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(f"{word}\n")
    
    # Save corpus.tsv (OCTIS format)
    corpus_path = os.path.join(local_path, 'corpus.tsv')
    with open(corpus_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for doc in filtered_corpus:
            writer.writerow([' '.join(doc), 'train', ''])
    
    # Save labels if available
    if filtered_labels is not None:
        labels_path = os.path.join(local_path, 'numeric_labels.txt')
        with open(labels_path, 'w', encoding='utf-8') as f:
            for label in filtered_labels:
                f.write(f"{label}\n")


def load_training_data(
    data_path: str,
    for_dsl: bool = True,
    revision: Optional[str] = None,
) -> TrainingData:
    """Load training data from local cache or HuggingFace Hub.
    
    First checks if a local copy exists in processed_data/. If not, downloads
    from HuggingFace Hub and caches locally.
    
    Handles both dsl models (which need embeddings + logits) and
    baseline models (which only need BOW data).
    
    Args:
        data_path: HuggingFace repo ID (e.g., 'username/dataset-name') or dataset name
        for_dsl: Whether loading for dsl model (needs embeddings)
        
    Returns:
        TrainingData containing all necessary data for training
    """
    # Determine local path for caching (use basename of data_path)
    local_name = data_path.split("/")[-1]
    local_data_path = os.path.join(PROCESSED_DATA_DIR, local_name)
    
    # Load from local cache or download from HuggingFace Hub
    dataset, vocab, metadata = load_or_download_dataset(data_path, revision=revision)
    
    # Get labels if available
    labels = list(dataset['label']) if 'label' in dataset.column_names else None
    
    # Create BOW corpus from dataset
    from tqdm import tqdm
    print("Extracting BOW corpus...")
    bow_corpus = [bow.split() for bow in tqdm(dataset['bow'], desc="Loading BOW")]
    
    # For baseline models, filter out empty documents to avoid OCTIS issues
    if not for_dsl:
        bow_corpus, labels = filter_empty_documents(bow_corpus, labels)

        # Prepare OCTIS-compatible files
        prepare_octis_files(local_data_path, bow_corpus, vocab, labels)
    
    # For dsl models, also extract embeddings and logits
    processed_dataset = None
    if for_dsl:
        processed_dataset = dataset
    
    return TrainingData(
        processed_dataset=processed_dataset,
        vocab=vocab,
        bow_corpus=bow_corpus,
        labels=labels,
        metadata=metadata,
        local_path=local_data_path,
    )

