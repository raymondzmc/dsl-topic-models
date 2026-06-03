"""Validate a processed soft-label dataset (downloaded or regenerated).

Checks that the on-disk dataset has the fields DSL training needs, with the right
shapes. Useful after `scripts/process_all.sh` or a manual download.

Usage:
    python tests/validate_processed_data.py 20_newsgroups_ERNIE-4.5-0.3B-PT_vocab_2000_last
    python tests/validate_processed_data.py /abs/path/to/processed_dataset_dir
"""
import os
import sys
import json

import numpy as np
from datasets import load_from_disk

from dsl_topic.paths import PROCESSED_DATA_DIR

REQUIRED_FIELDS = ["next_word_logits", "input_embeddings", "bow"]


def validate(name_or_path: str) -> bool:
    path = name_or_path
    if not os.path.isdir(path):
        path = os.path.join(PROCESSED_DATA_DIR, name_or_path)
    if not os.path.isdir(path):
        print(f"FAIL: dataset directory not found: {path}")
        return False

    print(f"Validating {path}")
    ds = load_from_disk(path)
    if hasattr(ds, "keys"):  # DatasetDict
        ds = ds[list(ds.keys())[0]]

    ok = True
    cols = set(ds.column_names)
    for field in REQUIRED_FIELDS:
        if field not in cols:
            print(f"  FAIL: missing required field '{field}'")
            ok = False
    print(f"  rows: {len(ds)}")
    print(f"  columns: {sorted(cols)}")

    vocab_path = os.path.join(path, "vocab.json")
    if os.path.exists(vocab_path):
        vocab = json.load(open(vocab_path, encoding="utf-8"))
        print(f"  vocab size: {len(vocab)}")
        if ok and len(ds) > 0:
            logits = ds[0]["next_word_logits"]
            if len(logits) != len(vocab):
                print(f"  FAIL: next_word_logits length {len(logits)} != vocab size {len(vocab)}")
                ok = False
            emb = np.array(ds[0]["input_embeddings"])
            # input_embeddings is (num_layers, hidden) when all layers are stored,
            # or (hidden,) when a single layer was selected at processing time.
            print(f"  input_embeddings shape: {emb.shape} (hidden dim = {emb.shape[-1]})")
            print(f"  logits dim: {len(logits)}")
    else:
        print("  WARN: vocab.json not found next to the dataset")

    if "label" not in cols:
        print("  WARN: no 'label' field — Purity and retrieval metrics will be unavailable")

    print("  OK" if ok else "  INVALID")
    return ok


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(0 if validate(sys.argv[1]) else 1)


if __name__ == "__main__":
    main()
