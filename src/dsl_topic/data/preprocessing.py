"""Processing utilities for topic modeling dataset creation."""

import os
import json
import time
import platform
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Optional
from tqdm import tqdm
from dsl_topic.prompts import jinja_template_manager


def get_device_info(device: torch.device) -> dict:
    """Get device information including CPU and GPU details.
    
    Args:
        device: PyTorch device
        
    Returns:
        Dictionary with device information
    """
    device_info = {
        "device_type": str(device),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    
    # CPU info
    device_info["cpu_count"] = os.cpu_count()
    
    # GPU info if available
    if torch.cuda.is_available():
        device_info["cuda_available"] = True
        device_info["cuda_version"] = torch.version.cuda
        device_info["gpu_count"] = torch.cuda.device_count()
        if device.type == "cuda":
            gpu_idx = device.index if device.index is not None else 0
            device_info["gpu_name"] = torch.cuda.get_device_name(gpu_idx)
            device_info["gpu_memory_total"] = torch.cuda.get_device_properties(gpu_idx).total_memory
            device_info["gpu_memory_allocated"] = torch.cuda.memory_allocated(gpu_idx)
    else:
        device_info["cuda_available"] = False
    
    return device_info


def collate_fn(
    batch: list[dict],
    tokenizer,
    content_key: str,
    label_key: Optional[str],
    vocab_set: set[str],
    instruction_template: str,
    prompt_template: str
) -> dict:
    """Collate function for DataLoader that prepares batched inputs with left padding.
    
    Args:
        batch: List of examples from dataset
        tokenizer: HuggingFace tokenizer
        content_key: Key for text content
        label_key: Key for labels (optional)
        vocab_set: Set of vocabulary words
        instruction_template: Jinja template for instruction
        prompt_template: Jinja template for prompt
        
    Returns:
        Dictionary with batched tensors and metadata
    """
    instruction = jinja_template_manager.render(instruction_template)
    contexts = []
    example_ids = []
    labels = []
    bow_lines = []
    
    for i, example in enumerate(batch):
        context = jinja_template_manager.render(
            prompt_template, 
            instruction=instruction, 
            document=example[content_key]
        )
        contexts.append(context.rstrip())
        example_ids.append(example['id'] if 'id' in example else example.get('idx', i))
        
        # Include label if label_key is specified
        if label_key is not None and label_key in example:
            labels.append(example[label_key])
        else:
            labels.append(None)
        
        # Create bow_line from words
        if 'words' in example:
            filtered_words = [w for w in example['words'] if w in vocab_set]
            bow_lines.append(" ".join(filtered_words))
        else:
            bow_lines.append("")
    
    # Tokenize with left padding
    encoded = tokenizer(
        contexts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        return_attention_mask=True
    )
    
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'contexts': contexts,
        'ids': example_ids,
        'labels': labels,
        'bow_lines': bow_lines
    }


def extract_embeddings(
    hidden_states: tuple,
    attention_mask: torch.Tensor,
    batch_idx: int,
    hidden_state_layer: Optional[int],
    embedding_method: str
) -> list:
    """Extract embeddings from model hidden states.
    
    Args:
        hidden_states: Tuple of hidden states from model
        attention_mask: Attention mask tensor
        batch_idx: Index in batch
        hidden_state_layer: Layer to extract from (None for all layers)
        embedding_method: 'mean' or 'last'
        
    Returns:
        Embeddings as list (or list of lists for all layers)
    """
    # Use the last-token hidden state (the position aligned with the next-token DSL target).
    if hidden_state_layer is not None:
        return hidden_states[hidden_state_layer][batch_idx, -1].float().cpu().tolist()
    # Save the last-token hidden state from every layer.
    return [h[batch_idx, -1].float().cpu().tolist() for h in hidden_states]


def write_batch_to_parquet(
    examples: list[dict],
    batch_num: int,
    output_dir: str,
    compression: str = 'snappy'
) -> Optional[str]:
    """Write a batch of processed examples to a Parquet file.
    
    This function is used for incremental writing during dataset processing
    to avoid holding all examples in memory at once.
    
    Args:
        examples: List of processed example dictionaries. Each should have keys:
            - id: Example identifier
            - context: Text context
            - next_word: Predicted next word
            - next_word_logits: List of logits for vocab words
            - input_embeddings: Embeddings (list or list of lists)
            - bow: Bag of words string
            - label (optional): Example label
        batch_num: Batch number for file naming
        output_dir: Directory to write Parquet files to
        compression: Parquet compression algorithm (default: 'snappy')
        
    Returns:
        Path to written Parquet file, or None if examples is empty
    """
    if not examples:
        return None
    
    # Convert to columnar format
    data = {
        'id': [ex['id'] for ex in examples],
        'context': [ex['context'] for ex in examples],
        'next_word': [ex['next_word'] for ex in examples],
        'next_word_logits': [ex['next_word_logits'] for ex in examples],
        'input_embeddings': [ex['input_embeddings'] for ex in examples],
        'bow': [ex['bow'] for ex in examples],
    }
    if 'label' in examples[0]:
        data['label'] = [ex['label'] for ex in examples]
    
    # Create PyArrow table
    table = pa.table(data)
    
    # Write to Parquet file
    parquet_path = os.path.join(output_dir, f'batch_{batch_num:06d}.parquet')
    pq.write_table(table, parquet_path, compression=compression)
    
    return parquet_path


def convert_parquet_to_arrow(
    parquet_files: list[str],
    arrow_path: str,
    show_progress: bool = True
) -> int:
    """Convert multiple Parquet files to a single Arrow IPC stream file.
    
    This is memory-efficient: reads and writes one batch at a time without
    loading all data into memory.
    
    Args:
        parquet_files: List of paths to Parquet files to convert
        arrow_path: Output path for the Arrow IPC stream file
        show_progress: Whether to show a progress bar
        
    Returns:
        Number of rows written
    """
    writer = None
    sink = None
    rows_written = 0
    
    iterator = tqdm(parquet_files, desc="Converting to Arrow format") if show_progress else parquet_files
    
    for pf in iterator:
        # Read parquet file as a table
        table = pq.read_table(pf)
        
        if writer is None:
            # Initialize IPC stream writer with schema from first file
            sink = pa.OSFile(arrow_path, 'wb')
            writer = pa.ipc.new_stream(sink, table.schema)
        
        # Write record batches
        for batch in table.to_batches():
            writer.write_batch(batch)
            rows_written += batch.num_rows
    
    if writer is not None:
        writer.close()
    if sink is not None:
        sink.close()
    
    return rows_written


def save_hf_dataset_from_parquet(
    parquet_files: list[str],
    output_dir: str,
    vocab: list[str],
    metadata: dict,
    dataset_name: str,
    description: str = ""
) -> str:
    """Save Parquet files as a HuggingFace-compatible dataset directory.
    
    Creates the Arrow data file and all necessary metadata files for
    compatibility with `datasets.load_from_disk()`.
    
    Args:
        parquet_files: List of paths to Parquet files
        output_dir: Directory to save the dataset
        vocab: Vocabulary list to save
        metadata: Metadata dictionary to save
        dataset_name: Name of the dataset (used for fingerprint)
        description: Dataset description
        
    Returns:
        Path to the saved dataset directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert Parquet files to Arrow format
    arrow_path = os.path.join(output_dir, 'data-00000-of-00001.arrow')
    rows_written = convert_parquet_to_arrow(parquet_files, arrow_path)
    
    print(f"  Written {rows_written} rows to Arrow format")
    
    # Create dataset_info.json (minimal - let HuggingFace infer features from Arrow)
    arrow_size = os.path.getsize(arrow_path)
    dataset_info = {
        "description": description,
        "citation": "",
        "homepage": "",
        "license": "",
        "splits": {
            "train": {
                "name": "train",
                "num_bytes": arrow_size,
                "num_examples": rows_written,
            }
        },
        "download_size": arrow_size,
        "dataset_size": arrow_size,
    }
    
    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Create state.json (HuggingFace datasets format)
    state = {
        "_data_files": [{"filename": "data-00000-of-00001.arrow"}],
        "_fingerprint": f"{dataset_name}_{int(time.time())}",
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": "train",
    }
    
    with open(os.path.join(output_dir, 'state.json'), 'w') as f:
        json.dump(state, f, indent=2)
    
    # Save vocab
    with open(os.path.join(output_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab, f, indent=2)
    
    # Save metadata
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return output_dir


def upload_dataset_to_hub(
    local_path: str,
    hf_repo_name: str,
    private: bool = False
) -> bool:
    """Upload a locally saved dataset to HuggingFace Hub.
    
    Uploads the dataset (Arrow file) and additional files (vocab.json, metadata.json)
    to a HuggingFace dataset repository.
    
    Args:
        local_path: Path to the local dataset directory
        hf_repo_name: HuggingFace repository ID (e.g., 'username/dataset-name')
        private: Whether to make the HF dataset private (default: False for public)
        
    Returns:
        True if upload succeeded, False otherwise
    """
    from datasets import load_from_disk
    from huggingface_hub import HfApi
    
    print(f"\nUploading dataset to HuggingFace Hub: {hf_repo_name}")
    
    try:
        # Load the dataset for pushing
        dataset = load_from_disk(local_path)
        
        # Push dataset to hub
        dataset.push_to_hub(
            hf_repo_name,
            private=private,
        )
        print(f"  ✓ Dataset pushed to hub ({len(dataset)} examples)")
        
        # Upload additional files (vocab.json, metadata.json)
        api = HfApi()
        
        vocab_path = os.path.join(local_path, "vocab.json")
        metadata_path = os.path.join(local_path, "metadata.json")
        
        if os.path.exists(vocab_path):
            api.upload_file(
                path_or_fileobj=vocab_path,
                path_in_repo="vocab.json",
                repo_id=hf_repo_name,
                repo_type="dataset",
            )
            with open(vocab_path, 'r') as f:
                vocab = json.load(f)
            print(f"  ✓ vocab.json uploaded ({len(vocab)} words)")
        
        if os.path.exists(metadata_path):
            api.upload_file(
                path_or_fileobj=metadata_path,
                path_in_repo="metadata.json",
                repo_id=hf_repo_name,
                repo_type="dataset",
            )
            print(f"  ✓ metadata.json uploaded")
        
        print(f"\n✓ Dataset successfully uploaded to: https://huggingface.co/datasets/{hf_repo_name}")
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to upload to HuggingFace Hub: {e}")
        print(f"  Dataset is still saved locally at: {local_path}")
        return False

