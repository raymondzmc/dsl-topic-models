"""Tokenization and text processing utilities for topic modeling."""

import spacy
from typing import Optional

_SPACY_MODEL = "en_core_web_lg"
_nlp = None


def _get_nlp():
    """Lazily load (and cache) the spaCy model used for tokenization.

    Loaded on first use — only ``dsl-process`` tokenizes raw text — so importing
    this module (and therefore the rest of the package and the test suite) does
    NOT require the large ``en_core_web_lg`` model or any network access.
    """
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(_SPACY_MODEL)
        except OSError as exc:
            raise OSError(
                f"spaCy model '{_SPACY_MODEL}' is required for dataset processing "
                f"(dsl-process). Install it once with:\n"
                f"    python -m spacy download {_SPACY_MODEL}"
            ) from exc
    return _nlp


def tokenize_document(
    text: str,
    tokenizer,
    vocab: Optional[list[str]] = None,
) -> tuple[list[str], list[list[int]], list[tuple[int, int]]]:
    """Tokenize a single document into words and token IDs.

    Args:
        text: Document text
        tokenizer: HuggingFace tokenizer
        vocab: Optional vocabulary to filter words

    Returns:
        Tuple of (words, token_ids, offsets)
    """
    doc = _get_nlp()(text)
    word_list, token_list, word_offsets = [], [], []
    
    for word in doc:
        if (
            (vocab is not None and word.text in vocab)
            or (
                vocab is None
                and word.is_alpha
                and not word.is_stop
                and not word.is_sent_start
                and len(word.text) > 2
                and word.is_lower
            )
        ):
            start, end = word.idx, word.idx + len(word)
            word_token_ids = tokenizer.encode(f" {word.text}", add_special_tokens=False)
            if len(word_token_ids) == 0:
                raise ValueError(f"Word {word.text} not found in tokenizer")
            
            word_list.append(word.text)
            token_list.append(word_token_ids)
            word_offsets.append((start, end))
    
    return word_list, token_list, word_offsets


def tokenize_dataset_batch(
    batch: dict,
    tokenizer,
    content_key: str,
    vocab: Optional[list[str]] = None
) -> dict:
    """Tokenize a batch of documents.

    Args:
        batch: Batch dictionary with content_key column
        tokenizer: HuggingFace tokenizer
        content_key: Key for the text content in batch
        vocab: Optional vocabulary to filter words

    Returns:
        Dictionary with 'words', 'token_ids', 'offsets', 'content' keys
    """
    words = []
    token_ids = []
    offsets = []
    
    for text in batch[content_key]:
        word_list, token_list, word_offsets = tokenize_document(
            text, tokenizer, vocab
        )
        words.append(word_list)
        token_ids.append(token_list)
        offsets.append(word_offsets)
    
    return {
        'words': words,
        'token_ids': token_ids,
        'offsets': offsets,
        'content': batch[content_key]
    }


def create_bow(words: list[str], vocab_set: set[str]) -> str:
    """Create bag-of-words string from word list.
    
    Args:
        words: List of words from document
        vocab_set: Set of vocabulary words
        
    Returns:
        Space-separated string of filtered words
    """
    filtered_words = [w for w in words if w in vocab_set]
    return " ".join(filtered_words)


def create_bow_batch(batch: dict, vocab_set: set[str]) -> dict:
    """Create bag-of-words for a batch of documents.
    
    Args:
        batch: Batch dictionary with 'words' column
        vocab_set: Set of vocabulary words
        
    Returns:
        Dictionary with 'bow' key
    """
    bow_lines = []
    for words in batch["words"]:
        bow_lines.append(create_bow(words, vocab_set))
    return {"bow": bow_lines}

