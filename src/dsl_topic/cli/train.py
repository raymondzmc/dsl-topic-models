"""Unified script for training and evaluating topic models."""

import os
import json
import time
import argparse
import random
import tempfile
import numpy as np
import torch
import wandb
from sentence_transformers import SentenceTransformer
from dsl_topic.data.loaders import load_training_data
from dsl_topic.data.octis_dataset import prepare_octis_dataset
from dsl_topic.data.ctm_dataset import get_ctm_dataset_from_processed_data
from dsl_topic.evaluation.metrics import compute_aggregate_results, evaluate_topic_model
from dsl_topic.cli._io import (
    save_model_output, save_evaluation, save_aggregate, save_labels, _dump_json,
)
from dsl_topic.settings import settings
from dsl_topic.cli._model_builders import BuildContext, MODEL_BUILDERS


DSL_MODELS = {'dsl', 'dsl_etm', 'dsl_ecrtm', 'dsl_fastopic'}
BASELINE_MODELS = {'lda', 'prodlda', 'zeroshot', 'combined', 'etm', 'bertopic', 'fastopic', 'ecrtm'}
ALL_MODELS = DSL_MODELS | BASELINE_MODELS


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def train_model(
    model_name: str,
    args: argparse.Namespace,
    seed: int,
    checkpoint_dir: str,
    local_data_path: str,
    vocab: list[str],
    bow_corpus: list[list[str]],
    ctm_dataset = None,
    octis_dataset = None,
) -> dict:
    """Train a topic model and return output dictionary."""
    builder = MODEL_BUILDERS.get(model_name)
    if builder is None:
        raise ValueError(f"Unknown model: {model_name}")
    ctx = BuildContext(
        args=args,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        local_data_path=local_data_path,
        vocab=vocab,
        bow_corpus=bow_corpus,
        ctm_dataset=ctm_dataset,
        octis_dataset=octis_dataset,
    )
    return builder(ctx)


def run_reevaluate(args: argparse.Namespace):
    """Re-evaluate a model from a previous W&B run."""
    if args.wandb_project is None:
        raise ValueError("--wandb_project is required when using --load_run_id_or_name")
    
    run_id_or_name = args.load_run_id_or_name
    wandb_project = args.wandb_project
    
    print(f"\n{'='*60}")
    print(f"Re-evaluating from W&B run: {run_id_or_name}")
    print(f"Project: {settings.wandb_entity}/{wandb_project}")
    print(f"{'='*60}\n")
    
    api = wandb.Api()
    
    # Find run by ID first, then by name
    source_run = None
    try:
        source_run = api.run(f"{settings.wandb_entity}/{wandb_project}/{run_id_or_name}")
        print(f"Found run by ID: {source_run.name} ({source_run.id})")
    except wandb.errors.CommError:
        print(f"Run ID '{run_id_or_name}' not found, searching by name...")
        runs = api.runs(
            f"{settings.wandb_entity}/{wandb_project}",
            filters={"display_name": run_id_or_name},
            order="-created_at",
        )
        runs_list = list(runs)
        
        if len(runs_list) == 0:
            raise ValueError(f"No run found with ID or name: {run_id_or_name}")
        
        if len(runs_list) > 1:
            print(f"⚠️  WARNING: Found {len(runs_list)} runs with name '{run_id_or_name}', using most recent")
            for i, r in enumerate(runs_list[:5]):
                print(f"   {i+1}. ID: {r.id}, Created: {r.created_at}")
        
        source_run = runs_list[0]
        print(f"Using run: {source_run.name} ({source_run.id})")
    
    # Find model artifact
    print("\nSearching for model artifact...")
    artifacts = list(source_run.logged_artifacts())
    model_artifacts = [a for a in artifacts if a.type == "model"]
    
    if len(model_artifacts) == 0:
        raise ValueError(f"No model artifacts found for run {source_run.id}")
    
    artifact = model_artifacts[-1]
    print(f"Found artifact: {artifact.name} (v{artifact.version})")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_dir = artifact.download(root=temp_dir)
        print(f"Downloaded to: {artifact_dir}")
        
        # Load labels
        labels_path = os.path.join(artifact_dir, 'labels.json')
        labels = None
        if os.path.exists(labels_path):
            with open(labels_path, encoding='utf-8') as f:
                labels = json.load(f)
            print(f"Loaded {len(labels)} labels")
        
        # Get metadata
        metadata = artifact.metadata or {}
        num_seeds = metadata.get('num_seeds', 1)
        top_words = args.top_words
        model_name = metadata.get('model', 'unknown')
        dataset_name = metadata.get('dataset', 'unknown')
        num_topics = metadata.get('num_topics', 0)
        
        print(f"\nMetadata: model={model_name}, dataset={dataset_name}, K={num_topics}, seeds={num_seeds}")
        
        # Initialize new run with same name as source (to replace in visualization)
        new_run = wandb.init(
            project=wandb_project,
            entity=settings.wandb_entity,
            name=source_run.name,
            config={
                "source_run_id": source_run.id,
                "source_run_name": source_run.name,
                "model": model_name,
                "dataset": dataset_name,
                "num_topics": num_topics,
                "num_seeds": num_seeds,
                "top_words": top_words,
                "reevaluation": True,
            },
            mode='online' if not args.wandb_offline else 'offline',
        )
        
        results_dir = os.path.join(temp_dir, 'reevaluated')
        os.makedirs(results_dir, exist_ok=True)
        
        for seed in range(num_seeds):
            seed_dir = os.path.join(artifact_dir, f"seed_{seed}")
            new_seed_dir = os.path.join(results_dir, f"seed_{seed}")
            os.makedirs(new_seed_dir, exist_ok=True)
            
            model_output_path = os.path.join(seed_dir, 'model_output.pt')
            if not os.path.exists(model_output_path):
                print(f"[Seed {seed}] model_output.pt not found, skipping")
                continue
            
            print(f"[Seed {seed}] Re-evaluating...")
            model_output = torch.load(model_output_path, weights_only=False)
            training_time = model_output.get('training_time', 0)
            
            # Copy model output
            torch.save(model_output, os.path.join(new_seed_dir, 'model_output.pt'))

            # Copy topics
            if 'topics' in model_output:
                _dump_json(model_output['topics'], os.path.join(new_seed_dir, 'topics.json'))

            # Re-evaluate
            evaluation_results = evaluate_topic_model(
                model_output,
                top_words=top_words,
                labels=labels,
            )
            evaluation_results['training_time'] = training_time

            save_evaluation(evaluation_results, new_seed_dir)

            print(f"[Seed {seed}] {evaluation_results}")
            new_run.log({f"seed_{seed}/{k}": v for k, v in evaluation_results.items()})
        
        # Aggregated results
        averaged_results = compute_aggregate_results(results_dir)
        save_aggregate(averaged_results, results_dir)

        # Copy labels and vocab embeddings
        if labels is not None:
            save_labels(labels, results_dir)
        
        new_run.log({f"avg/{k}": v for k, v in averaged_results.items()})
        print(f"\nAveraged: {averaged_results}")
        
        # Upload artifact
        has_labels = labels is not None
        new_artifact = wandb.Artifact(
            name=f"{model_name}-K{num_topics}-{dataset_name}",
            type="model",
            description=f"Re-evaluated from {source_run.name} ({source_run.id})",
            metadata={
                "model": model_name,
                "dataset": dataset_name,
                "num_topics": num_topics,
                "num_seeds": num_seeds,
                "top_words": top_words,
                "has_labels": has_labels,
                "source_run_id": source_run.id,
                "reevaluation": True,
            }
        )
        new_artifact.add_dir(results_dir)
        new_run.log_artifact(new_artifact)
        new_run.finish()
        
        print(f"\n{'='*60}")
        print("Re-evaluation complete!")
        print(f"View: https://wandb.ai/{settings.wandb_entity}/{wandb_project}")
        print(f"{'='*60}")


def run(args: argparse.Namespace):
    """Main training and evaluation loop."""
    is_dsl = args.model in DSL_MODELS
    training_data = load_training_data(args.data_path, for_dsl=is_dsl)

    octis_data_path = training_data.local_path
    metadata = training_data.metadata or {}
    original_dataset = metadata.get('args', {}).get('dataset', '')
    if original_dataset:
        # Extract basename and remove extension (e.g. 'SetFit/20_newsgroups' -> '20_newsgroups',
        # 'data/raw_data/stackoverflow.tsv' -> 'stackoverflow')
        dataset_name = os.path.basename(original_dataset).split('.')[0]
    else:
        # Fallback: use folder name (shouldn't happen with properly processed data)
        dataset_name = os.path.basename(training_data.local_path)
    
    # Extract model name for dsl models (e.g., 'baidu/ERNIE-4.5-0.3B-PT' -> 'ERNIE-4.5-0.3B-PT')
    model_name_suffix = ""
    if args.model in DSL_MODELS:
        original_model_name = metadata.get('args', {}).get('model_name', '')
        if original_model_name:
            model_name_suffix = f"_{os.path.basename(original_model_name)}"
    
    # Prepare OCTIS dataset for baseline models
    # This also filters empty documents and returns the filtered corpus/labels for evaluation
    octis_dataset = None
    eval_corpus = training_data.bow_corpus
    eval_labels = training_data.labels
    if args.model in BASELINE_MODELS:
        octis_dataset, eval_corpus, eval_labels = prepare_octis_dataset(
            octis_data_path,
            training_data.bow_corpus,
            training_data.vocab,
            training_data.labels,
        )
    
    # Pre-compute CTM dataset for dsl models (only once, not per seed)
    ctm_dataset = None
    if args.model in DSL_MODELS:
        # Load embedding model for ablation if specified
        ablation_embedding_model = None
        if args.ablation_embedding_model:
            print(f"Loading ablation embedding model: {args.ablation_embedding_model}")
            ablation_embedding_model = SentenceTransformer(
                args.ablation_embedding_model, trust_remote_code=True
            )
        
        ctm_dataset = get_ctm_dataset_from_processed_data(
            training_data.processed_dataset,
            training_data.vocab,
            embedding_model=ablation_embedding_model,
            use_bow_target=args.ablation_use_bow_target,
        )

        # Report BoW sparsity statistics
        bow_corpus = training_data.bow_corpus
        vocab_size = len(training_data.vocab)
        nnz_per_doc = [len(set(doc)) for doc in bow_corpus]
        avg_nnz = sum(nnz_per_doc) / len(nnz_per_doc)
        print(f"\n--- BoW Sparsity ---")
        print(f"Vocab size: {vocab_size}")
        print(f"Avg unique tokens per doc: {avg_nnz:.1f} / {vocab_size} ({100*avg_nnz/vocab_size:.1f}%)")
        print(f"Min: {min(nnz_per_doc)}, Max: {max(nnz_per_doc)}")
        if args.topk is not None:
            print(f"Top-k target: {args.topk} / {vocab_size} ({100*args.topk/vocab_size:.1f}%)")
        print(f"-------------------\n")

    # Build config
    wandb_project = args.wandb_project if args.wandb_project else dataset_name
    wandb_config = {
        'model': args.model,
        'dataset': dataset_name,
        'num_topics': args.num_topics,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'hidden_size': args.hidden_size,
        'num_hidden_layers': args.num_hidden_layers,
        'activation': args.activation,
        'solver': args.solver,
        'top_words': args.top_words,
        'num_seeds': args.num_seeds,
    }
    if args.model in DSL_MODELS:
        wandb_config.update({
            'loss_weight': args.loss_weight,
            'sparsity_ratio': args.sparsity_ratio,
            'topk': args.topk,
            'loss_type': args.loss_type,
            'temperature': args.temperature,
            'ablation_embedding_model': args.ablation_embedding_model,
            'ablation_use_bow_target': args.ablation_use_bow_target,
        })

    # Build run name with ablation suffixes
    run_name = f"{args.model}{model_name_suffix}_K{args.num_topics}"
    if args.model in DSL_MODELS:
        if args.ablation_use_bow_target:
            run_name += "_bow-target"
        if args.ablation_embedding_model:
            # Extract short model name (e.g., "gte-large-en-v1.5" from "Alibaba-NLP/gte-large-en-v1.5")
            ablation_emb_name = os.path.basename(args.ablation_embedding_model)
            run_name += f"_{ablation_emb_name}"
        if args.loss_type == 'CE':
            run_name += "_CE"
        if args.topk is not None:
            run_name += f"_topk{args.topk}"
        elif args.sparsity_ratio != 1.0:
            run_name += f"_sparsity{args.sparsity_ratio}"
        if args.temperature != 3.0:
            run_name += f"_temp{args.temperature}"

    # Persistent local results directory = source of truth. Resumable across runs.
    results_dir = os.path.join(args.output_dir, dataset_name, run_name)
    if os.path.exists(os.path.join(results_dir, 'averaged_results.json')):
        print(f"\n[skip] Results already exist at {results_dir} (delete dir to re-run).")
        return
    os.makedirs(results_dir, exist_ok=True)

    wb_run = None
    if args.wandb:
        wb_run = wandb.init(
            project=wandb_project,
            entity=settings.require('wandb_entity', 'W&B logging (--wandb)'),
            name=run_name,
            config=wandb_config,
            mode='online' if not args.wandb_offline else 'offline',
        )

    all_results = []

    for seed in range(args.num_seeds):
        set_seed(seed)
        seed_dir = os.path.join(results_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        print(f"\n[Seed {seed}] Training {args.model}...")
        start_time = time.time()

        model_output = train_model(
            model_name=args.model,
            args=args,
            seed=seed,
            checkpoint_dir=seed_dir,
            local_data_path=octis_data_path,
            vocab=training_data.vocab,
            bow_corpus=eval_corpus,  # Use filtered corpus for consistent doc alignment
            ctm_dataset=ctm_dataset,
            octis_dataset=octis_dataset,
        )

        training_time = time.time() - start_time
        model_output['training_time'] = training_time
        print(f"[Seed {seed}] Trained in {training_time:.2f}s")

        # Save model output and topics
        save_model_output(model_output, seed_dir)

        # Evaluate
        print(f"[Seed {seed}] Evaluating...")
        evaluation_results = evaluate_topic_model(
            model_output,
            top_words=args.top_words,
            labels=eval_labels,
            skip_llm_rating=args.skip_llm_rating,
        )
        evaluation_results['training_time'] = training_time

        save_evaluation(evaluation_results, seed_dir)

        if args.wandb:
            wb_run.log({f"seed_{seed}/{k}": v for k, v in evaluation_results.items()})
        all_results.append(evaluation_results)

    # Aggregated results
    averaged_results = compute_aggregate_results(results_dir)
    save_aggregate(averaged_results, results_dir)

    # Save labels for re-evaluation (use filtered labels to match model output)
    has_labels = eval_labels is not None
    if has_labels:
        labels_list = eval_labels
        if hasattr(labels_list, 'tolist'):
            labels_list = labels_list.tolist()
        save_labels(labels_list, results_dir)

    # Save vocab for reproducibility
    if training_data.vocab is not None:
        _dump_json(training_data.vocab, os.path.join(results_dir, 'vocab.json'))

    # Save config for reproducibility
    _dump_json(wandb_config, os.path.join(results_dir, 'config.json'))

    if args.wandb:
        wb_run.log({f"avg/{k}": v for k, v in averaged_results.items()})
        print("\nUploading artifact to wandb...")
        artifact = wandb.Artifact(
            name=f"{args.model}-K{args.num_topics}-{dataset_name}",
            type="model",
            description=f"Topic model: {args.model} on {dataset_name} (K={args.num_topics}, seeds={args.num_seeds})",
            metadata={
                "model": args.model,
                "dataset": dataset_name,
                "num_topics": args.num_topics,
                "num_seeds": args.num_seeds,
                "top_words": args.top_words,
                "has_labels": has_labels,
            }
        )
        artifact.add_dir(results_dir)
        wb_run.log_artifact(artifact)
        wb_run.finish()
        print(f"\nView run: https://wandb.ai/{settings.wandb_entity}/{wandb_project}")

    print(f"\n[done] Results written to {results_dir}")
    print(f"       Aggregated metrics: {os.path.join(results_dir, 'averaged_results.json')}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate topic models")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default=None, help='Path to data directory or HF repo ID')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='dsl',
                        choices=list(ALL_MODELS),
                        help="Model to train. DSL (paper) methods: 'dsl'=DSL-ProdLDA "
                             "(the main method), 'dsl_ecrtm'/'dsl_fastopic'/"
                             "'dsl_etm'=DSL on that backbone. The rest are baselines.")
    parser.add_argument('--num_topics', type=int, default=25, help='Number of topics')
    parser.add_argument('--top_words', type=int, default=15, help='Top words per topic')
    
    # Training arguments
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of random seeds')
    parser.add_argument('--num_epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=200, help='Hidden layer size')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='Hidden layers')
    parser.add_argument('--activation', type=str, default='softplus', help='Activation')
    parser.add_argument('--solver', type=str, default='adam', help='Optimizer')
    
    # DSL model arguments
    parser.add_argument('--loss_weight', type=float, default=1e3, help='Reconstruction loss weight')
    parser.add_argument('--sparsity_ratio', type=float, default=1.0, help='Sparsity ratio')
    parser.add_argument('--topk', type=int, default=None, help='Top-k words to keep in LLM target (overrides sparsity_ratio)')
    parser.add_argument('--loss_type', type=str, default='KL', choices=['KL', 'CE'], help='Loss type')
    parser.add_argument('--temperature', type=float, default=3.0, help='Softmax temperature')
    
    # Ablation arguments (dsl model only)
    parser.add_argument('--ablation_embedding_model', type=str, default=None,
                        help='Use a different SentenceTransformer model for embeddings (ablation)')
    parser.add_argument('--ablation_use_bow_target', action='store_true',
                        help='Use BoW as target instead of LLM predictions (ablation)')
    
    # Output / evaluation
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory for local results, the source of truth (default: results/)')
    parser.add_argument('--skip_llm_rating', action='store_true',
                        help='Skip the OpenAI gpt-4o LLM topic-coherence rating metric')

    # Wandb arguments (optional; off by default)
    parser.add_argument('--wandb', action='store_true',
                        help='Also log metrics and a model artifact to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default=None, help='W&B project name')
    parser.add_argument('--wandb_offline', action='store_true', help='Offline mode')
    parser.add_argument('--load_run_id_or_name', type=str, default=None,
                        help='Load from previous W&B run for re-evaluation (requires --wandb credentials)')

    args = parser.parse_args()
    
    if args.load_run_id_or_name:
        run_reevaluate(args)
    elif args.data_path is None:
        parser.error("the following arguments are required: --data_path")
    else:
        run(args)


if __name__ == '__main__':
    main()
