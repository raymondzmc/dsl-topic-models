"""Model-construction registry for the training CLI.

Each ``_build_<name>`` function contains the exact body of the corresponding
branch of the original ``train_model`` if/elif dispatch, and returns the same
model-output dict (``topics`` / ``topic-document-matrix`` / ...). The registry
``MODEL_BUILDERS`` maps each ``--model`` key to its builder; ``train.py`` keeps
the ``ALL_MODELS`` choice set unchanged and only delegates dispatch here.

The per-branch model-specific imports (BERTopic, gensim, OCTIS/TopMost/FASTopic
trainers, the DSL classes) live here rather than in ``train.py`` so they are
loaded in the same places as before, and ``scipy.sparse`` / the GloVe helper
stay as local imports inside their branches to preserve import timing.
"""
import os
from dataclasses import dataclass

import argparse
import torch
from sentence_transformers import SentenceTransformer

from bertopic import BERTopic
from gensim.downloader import load as gensim_load
from sklearn.feature_extraction.text import CountVectorizer
from dsl_topic.models.baselines.octis import LDA, ProdLDA, CTM, ETM
from dsl_topic.models.baselines.fastopic import FASTopicTrainer
from dsl_topic.models.baselines.topmost.ECRTM import ECRTMTrainer
from dsl_topic.models.baselines.topmost.data import RawDataset
from dsl_topic.models.dsl.prodlda import DSLProdLDA
from dsl_topic.models.dsl.etm import DSLETM
from dsl_topic.models.dsl.ecrtm import DSLECRTM
from dsl_topic.models.dsl.fastopic import DSLFASTopic


@dataclass(frozen=True)
class BuildContext:
    """Everything a model builder needs (mirrors train_model's parameters)."""
    args: argparse.Namespace
    seed: int
    checkpoint_dir: str
    local_data_path: str
    vocab: list
    bow_corpus: list
    ctm_dataset: object = None
    octis_dataset: object = None


def _build_dsl(ctx: BuildContext) -> dict:
    args, vocab, ctm_dataset = ctx.args, ctx.vocab, ctx.ctm_dataset
    if ctm_dataset is None:
        raise ValueError("DSL model requires ctm_dataset")

    model = DSLProdLDA(
        vocab_size=len(vocab),
        embedding_size=ctm_dataset.x_embeddings.shape[1],
        num_topics=args.num_topics,
        activation=args.activation,
        hidden_sizes=tuple([args.hidden_size] * args.num_hidden_layers),
        solver=args.solver,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_weight=args.loss_weight,
        sparsity_ratio=args.sparsity_ratio,
        topk=args.topk,
        loss_type=args.loss_type,
        temperature=args.temperature,
        top_words=args.top_words,
    )
    model.fit(ctm_dataset)
    return model.get_info()


def _build_dsl_etm(ctx: BuildContext) -> dict:
    args, vocab, ctm_dataset = ctx.args, ctx.vocab, ctx.ctm_dataset
    if ctm_dataset is None:
        raise ValueError("dsl_etm requires ctm_dataset")
    idx2token = {i: w for i, w in enumerate(vocab)}
    model = DSLETM(
        vocab_size=len(vocab),
        embedding_size=ctm_dataset.x_embeddings.shape[1],
        num_topics=args.num_topics,
        t_hidden_size=args.hidden_size,
        activation=args.activation,
        dropout=0.5,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        temperature=args.temperature,
        loss_weight=args.loss_weight,
        sparsity_ratio=args.sparsity_ratio,
        loss_type=args.loss_type,
        top_words=args.top_words,
    )
    model.fit(ctm_dataset)
    info = model.get_info(idx2token=idx2token)
    theta = model.get_theta(ctm_dataset)
    info['topic-document-matrix'] = theta.T
    return info


def _build_dsl_ecrtm(ctx: BuildContext) -> dict:
    args, vocab, ctm_dataset = ctx.args, ctx.vocab, ctx.ctm_dataset
    if ctm_dataset is None:
        raise ValueError("dsl_ecrtm requires ctm_dataset")

    import scipy.sparse
    glove_path = os.path.join(ctx.local_data_path, 'glove_word_embeddings.npz')
    if os.path.exists(glove_path):
        pretrained_WE = scipy.sparse.load_npz(glove_path).toarray().astype('float32')
    else:
        from dsl_topic.models.baselines.topmost.ECRTM.preprocess import get_word_embeddings
        pretrained_WE_sparse = get_word_embeddings(vocab, embedding_model='glove-wiki-gigaword-200')
        scipy.sparse.save_npz(glove_path, pretrained_WE_sparse)
        pretrained_WE = pretrained_WE_sparse.toarray().astype('float32')

    model = DSLECRTM(
        vocab_size=len(vocab),
        embedding_size=ctm_dataset.x_embeddings.shape[1],
        num_topics=args.num_topics,
        vocab=vocab,
        pretrained_WE=pretrained_WE,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        loss_weight=args.loss_weight,
        sparsity_ratio=args.sparsity_ratio,
        loss_type=args.loss_type,
        top_words=args.top_words,
    )
    model.fit(ctm_dataset)
    info = model.get_info()
    theta = model.get_theta(ctm_dataset)
    info['topic-document-matrix'] = theta.T
    return info


def _build_dsl_fastopic(ctx: BuildContext) -> dict:
    args, vocab, ctm_dataset = ctx.args, ctx.vocab, ctx.ctm_dataset
    if ctm_dataset is None:
        raise ValueError("dsl_fastopic requires ctm_dataset")

    model = DSLFASTopic(
        vocab_size=len(vocab),
        embedding_size=ctm_dataset.x_embeddings.shape[1],
        num_topics=args.num_topics,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        top_words=args.top_words,
        vocab=vocab,
    )
    model.fit(ctm_dataset)
    info = model.get_info()
    theta = model.get_theta(ctm_dataset)
    info['topic-document-matrix'] = theta.T
    return info


def _build_lda(ctx: BuildContext) -> dict:
    args = ctx.args
    model = LDA(num_topics=args.num_topics, random_state=ctx.seed)
    return model.train_model(dataset=ctx.octis_dataset, top_words=args.top_words)


def _build_prodlda(ctx: BuildContext) -> dict:
    args = ctx.args
    model = ProdLDA(
        num_topics=args.num_topics,
        batch_size=args.batch_size,
        lr=args.lr,
        activation=args.activation,
        solver=args.solver,
        num_layers=args.num_hidden_layers,
        num_neurons=args.hidden_size,
        num_epochs=args.num_epochs,
        use_partitions=False,
    )
    return model.train_model(dataset=ctx.octis_dataset, top_words=args.top_words)


def _build_ctm(ctx: BuildContext) -> dict:
    args = ctx.args
    model = CTM(
        num_topics=args.num_topics,
        num_layers=args.num_hidden_layers,
        num_neurons=args.hidden_size,
        batch_size=args.batch_size,
        lr=args.lr,
        activation=args.activation,
        solver=args.solver,
        num_epochs=args.num_epochs,
        inference_type=args.model,
        bert_path=os.path.join(ctx.local_data_path, 'gte-large-en-v1.5'),
        bert_model='Alibaba-NLP/gte-large-en-v1.5',
        use_partitions=False,
    )
    model.set_seed(ctx.seed)
    return model.train_model(dataset=ctx.octis_dataset, top_words=args.top_words)


def _build_etm(ctx: BuildContext) -> dict:
    args = ctx.args
    word2vec_path = 'word2vec-google-news-300.kv'
    if not os.path.exists(word2vec_path):
        word2vec = gensim_load('word2vec-google-news-300')
        word2vec.save_word2vec_format(word2vec_path, binary=True)

    model = ETM(
        num_topics=args.num_topics,
        use_partitions=False,
        train_embeddings=False,
        embeddings_path=word2vec_path,
        embeddings_type='word2vec',
        binary_embeddings=True,
    )
    return model.train_model(
        dataset=ctx.octis_dataset,
        top_words=args.top_words,
        op_path=os.path.join(ctx.checkpoint_dir, 'checkpoint.pt'),
    )


def _build_bertopic(ctx: BuildContext) -> dict:
    args, vocab, bow_corpus = ctx.args, ctx.vocab, ctx.bow_corpus
    embedding_model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
    text_corpus = [' '.join(word_list) for word_list in bow_corpus]
    embeddings = embedding_model.encode(
        text_corpus,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Constrain BERTopic to use the preprocessed vocabulary for fair comparison
    # This ensures topics are distributions over the same vocab as ProdLDA/ETM/ZeroshotTM
    vectorizer = CountVectorizer(vocabulary={w: i for i, w in enumerate(vocab)})

    model = BERTopic(
        vectorizer_model=vectorizer,
        language='english',
        top_n_words=args.top_words,
        nr_topics=args.num_topics + 1,
        calculate_probabilities=True,
        verbose=True,
        low_memory=False,
    )
    output = model.fit_transform(text_corpus, embeddings=embeddings)
    all_topics = model.get_topics()
    topics = [
        # Filter out empty strings that BERTopic adds when topic has fewer words than top_n_words
        [word_prob[0] for word_prob in topic if word_prob[0].strip()]
        for topic_id, topic in all_topics.items() if topic_id != -1
    ]
    return {
        'topics': topics,
        'topic-document-matrix': output[1].transpose(),
    }


def _build_fastopic(ctx: BuildContext) -> dict:
    args, bow_corpus = ctx.args, ctx.bow_corpus
    text_corpus = [' '.join(word_list) for word_list in bow_corpus]
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dataset = RawDataset(text_corpus, device=device)
    embedding_model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
    trainer = FASTopicTrainer(
        dataset=dataset,
        num_topics=args.num_topics,
        num_top_words=args.top_words,
        doc_embed_model=embedding_model,
        low_memory=True,
        low_memory_batch_size=262144,
    )
    top_words, doc_topic_dist = trainer.train()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'topics': [topic_string.split(' ') for topic_string in top_words],
        'topic-document-matrix': doc_topic_dist.transpose(),
    }


def _build_ecrtm(ctx: BuildContext) -> dict:
    args, vocab, bow_corpus = ctx.args, ctx.vocab, ctx.bow_corpus
    # Convert bow_corpus to BoW matrix using CountVectorizer
    import scipy.sparse

    text_corpus = [' '.join(word_list) for word_list in bow_corpus]
    vocab2id = {word: idx for idx, word in enumerate(vocab)}

    vectorizer = CountVectorizer(vocabulary=vocab2id, token_pattern=r'(?u)\b\w+\b')
    bow_matrix = vectorizer.fit_transform(text_corpus).toarray().astype('float32')

    # Load or compute GloVe word embeddings
    glove_path = os.path.join(ctx.local_data_path, 'glove_word_embeddings.npz')
    if os.path.exists(glove_path):
        print(f"Loading cached GloVe embeddings from {glove_path}")
        pretrained_WE = scipy.sparse.load_npz(glove_path).toarray().astype('float32')
    else:
        print(f"Computing GloVe embeddings for vocabulary...")
        from dsl_topic.models.baselines.topmost.ECRTM.preprocess import get_word_embeddings
        pretrained_WE_sparse = get_word_embeddings(vocab, embedding_model='glove-wiki-gigaword-200')
        scipy.sparse.save_npz(glove_path, pretrained_WE_sparse)
        pretrained_WE = pretrained_WE_sparse.toarray().astype('float32')
        print(f"Cached GloVe embeddings to {glove_path}")

    # Initialize and train ECRTM
    trainer = ECRTMTrainer(
        vocab_size=len(vocab),
        num_topics=args.num_topics,
        vocab=vocab,
        pretrained_WE=pretrained_WE,
    )

    beta = trainer.train(bow_matrix, verbose=True)
    topics = trainer.get_topics(beta, top_words=args.top_words)
    theta = trainer.get_theta(bow_matrix)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'topics': topics,
        'topic-document-matrix': theta.T,
    }


MODEL_BUILDERS = {
    'dsl': _build_dsl,
    'dsl_etm': _build_dsl_etm,
    'dsl_ecrtm': _build_dsl_ecrtm,
    'dsl_fastopic': _build_dsl_fastopic,
    'lda': _build_lda,
    'prodlda': _build_prodlda,
    'zeroshot': _build_ctm,
    'combined': _build_ctm,
    'etm': _build_etm,
    'bertopic': _build_bertopic,
    'fastopic': _build_fastopic,
    'ecrtm': _build_ecrtm,
}
