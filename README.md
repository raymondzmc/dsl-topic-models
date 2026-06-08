# Improving Topic Modeling by Distilling Soft Labels from Language Models

Official implementation of **"Improving Topic Modeling by Distilling Soft Labels from Language Models"** (ICML 2026).

Neural topic models are usually trained to reconstruct a document's **Bag-of-Words
(BoW)**, which ignores context and struggles with short or sparse text. **DSL**
instead trains the topic model to reconstruct a **dense, semantically grounded soft
label** obtained from a small language model: we prompt the LM for the document's
theme, project the next-token distribution onto the topic-model vocabulary, and
distill it into the topic model with a KL objective — using the LM's last hidden
state as the document representation. DSL is a drop-in training objective that works
across topic-model families (ProdLDA, ECRTM, FASTopic, ETM).

```
                prompt π + document x
                        │
                 ┌──────▼───────┐         next-token logits → softmax(·/τ)
                 │  small LM     │ ───────────────────►  y_DSL  (soft target over V)
                 └──────┬───────┘
                  last hidden state x_emb
                        │
                 ┌──────▼───────┐   KL( y_DSL ‖ ŷ )   interpretable topics θ, β
                 │ topic model M │ ──────────────────►  (document-topic + topic-word)
                 └──────────────┘
```

---

## 1. Installation

```bash
git clone --recurse-submodules https://github.com/raymondzmc/dsl-topic-models.git
cd dsl-topic-models

# Create an environment (Python 3.10–3.12) and install the package.
conda create -n dsl-topic python=3.12 -y && conda activate dsl-topic
pip install -e .                 # installs the `dsl_topic` package + CLIs
# For the EXACT tested versions (paper reproduction):
#   pip install -r requirements.lock && pip install -e . --no-deps

# (optional) copy the credentials template — none are required for training
cp .env.example .env
```

This installs four console commands: **`dsl-process`** (build soft-label datasets),
**`dsl-train`** (train + evaluate), **`dsl-retrieval`** (retrieval metric), and
**`dsl-summarize`** (print results). A CUDA GPU is recommended; the paper used a
single H100.

**API keys are optional** (set them in `.env`) and only enable specific features:

| Key | Needed for |
|---|---|
| `OPENAI_API_KEY` | the `LLM` topic-rating metric (`gpt-4o`); omit it or pass `--skip_llm_rating` |
| `HF_TOKEN` | downloading/uploading processed datasets on the HF Hub (only if private) |
| `WANDB_API_KEY`, `WANDB_ENTITY` | logging to Weights & Biases (only with `--wandb`) |

## 2. Data and evaluation assets

Training reads pre-processed datasets named `<dataset>_<lm>_vocab_2000_last`, each a
HuggingFace dataset where every document carries `next_word_logits` (the LM next-token
logits over the 2,000-word vocabulary — the **DSL target**), `input_embeddings` (the LM
last-layer hidden state — the topic-model input), `bow` (Bag-of-Words), and `label`
(ground-truth class, for Purity / retrieval). `data/loaders.py` resolves a data path
**local-first** (`data/processed_data/<name>/`) then falls back to the HF Hub, so:

- **Download (recommended, bit-exact):** the launch scripts pass
  `<HF_NAMESPACE>/<name>` (default namespace `raymondzmc`), so datasets auto-download on
  first use. Set `HF_NAMESPACE=you` to use your own copies. For an immutable,
  reproducible download pin a Hub revision with `DSL_TOPIC_HF_REVISION=<commit>`.
- **Regenerate locally:** `bash scripts/process_all.sh` rebuilds all 3 datasets × 5 LMs
  from `data/raw_data/` (StackOverflow, TweetTopic — committed) and `SetFit/20_newsgroups`
  (from HF). Because LM inference in bf16 is not bit-deterministic, regenerated targets can
  differ marginally from the published ones — prefer download for exact numbers.

Override the data root with the `DSL_TOPIC_DATA_DIR` env var (default `./data`).

**Optional assets:**
- **`C_V` (`cv_wiki`)** — the paper's coherence metric uses the Palmetto JAR over a
  Wikipedia index. Install Java and place `data/wikipedia/palmetto-0.1.0-jar-with-dependencies.jar`
  + `data/wikipedia/wikipedia_bd/` (Lucene index) from the
  [Palmetto project](https://github.com/dice-group/Palmetto). If absent, `cv_wiki` is
  simply omitted (all other metrics still run).
- **word2vec** — the **ETM** baseline initializes from `word2vec-google-news-300`, which
  `dsl-train --model etm` downloads automatically via gensim (cached, gitignored).

## 3. Quickstart (no API keys)

```bash
# Train a baseline and write local JSON results. The processed dataset is
# auto-downloaded from the HF Hub on first use; or run scripts/process_all.sh first.
dsl-train --model prodlda \
  --data_path raymondzmc/20_newsgroups_Llama-3.1-8B-Instruct_vocab_2000_last \
  --num_topics 25 --num_seeds 1 --num_epochs 20 --skip_llm_rating

dsl-summarize results          # tabulate metrics from results/
```

Results are written to `results/<dataset>/<run_name>/` as JSON (the source of truth).
`--skip_llm_rating` avoids the only paid metric; Weights & Biases is **off by default**
(add `--wandb` to also log there).

## 4. Reproducing the paper

Everything runs **sequentially on one GPU** (`CUDA_VISIBLE_DEVICES=0`) and is
**resumable** — a run whose `results/<dataset>/<run>/averaged_results.json` already exists
is skipped, so you can stop and restart freely.

```bash
bash scripts/reproduce_all.sh                                  # the full matrix
SKIP_LLM=1 bash scripts/reproduce_all.sh                       # skip the paid OpenAI metric
NUM_SEEDS=1 NUM_EPOCHS=10 SKIP_LLM=1 bash scripts/reproduce_all.sh   # fast smoke
```

**Conventions.** Datasets `{20_newsgroups, tweet_topic, stackoverflow}`; LMs
`{ERNIE-4.5-0.3B-PT, Qwen3.5-0.8B, Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct,
Phi-3-mini-128k-instruct}`; `K ∈ {25,50,75,100}`; 5 seeds; 100 epochs (the ECRTM/FASTopic DSL backbones use 200 to match their
baselines). Per-script
overrides: `PYTHON`, `CUDA_VISIBLE_DEVICES`, `NUM_SEEDS`, `NUM_EPOCHS`, `SKIP_LLM=1`,
`USE_WANDB=1`, `HF_NAMESPACE`, `BATCH_SIZE` (see `scripts/common.sh`).

**Core results**

| Paper result | Script | Grid (× 5 seeds, K∈{25,50,75,100}) |
|---|---|---|
| Main table — ProdLDA + DSL | `scripts/run_main.sh` | `dsl` × 5 LMs × 3 datasets |
| Main table — baselines | `scripts/run_baselines.sh` | {lda, prodlda, zeroshot, combined, etm, bertopic, fastopic, ecrtm} × 3 datasets (Llama-3.1-8B corpus) |
| Other backbones + DSL | `scripts/run_backbones.sh` | {dsl_ecrtm, dsl_fastopic, dsl_etm} × 5 LMs × 3 datasets |
| Retrieval (Precision@k) | `scripts/run_retrieval.sh` | P@{1,5,10} for every run in `results/` (run after training) |

**Ablations**

| Ablation | Script | Grid |
|---|---|---|
| Temperature τ | `scripts/ablation_temperature.sh` | ERNIE × 3 datasets × 4 K × τ∈{0.5,1,2,3,4,5,6,7,8,9} |
| Target sparsity | `scripts/ablation_sparsity.sh` | Llama-3.2-1B × 3 datasets × 4 K × ratio∈{0.1…0.9} |
| Target top-k | `scripts/ablation_topk.sh` | ERNIE × stackoverflow × K=50 × topk∈{10,20,50,100,500,1000} |
| Loss type (CE vs KL) | `scripts/ablation_loss.sh` | {ERNIE, Llama-8B, Llama-1B} × 3 datasets × K=100 |
| Embedding source (GTE) | `scripts/ablation_embedding.sh` | {ERNIE, Llama-8B, Llama-1B} × 3 datasets × 4 K |
| BoW target (no DSL) | `scripts/ablation_bow_target.sh` | {ERNIE, Llama-8B, Llama-1B} × 3 datasets × 4 K |
| Vocabulary size \|V\| | `scripts/ablation_vocab_size.sh` | reprocess 20news/ERNIE at \|V\|∈{500,1000,2000,4000}, then dsl + 8 baselines → `results/vocab_<V>/` |
| Prompt sensitivity | `scripts/ablation_prompt_sensitivity.sh` | reprocess 20news/ERNIE with 5 instruction variants, then {dsl, zeroshot, prodlda, fastopic} at K=50 → `results/prompt_<variant>/` |

The vocab-size and prompt ablations vary the *data* (not a CLI flag), so they write under
`results/vocab_<V>/` and `results/prompt_<variant>/` to avoid colliding with main runs.

**A single configuration:**
```bash
scripts/run_dsl.sh <model> <lm> <dataset> <K> [extra dsl-train flags...]
scripts/run_dsl.sh dsl ERNIE-4.5-0.3B-PT 20_newsgroups 50     # one main-table cell
```

**Inspect results:** `dsl-summarize results [--dataset 20_newsgroups]`.

**Runtime (single H100).** Topic-model training consumes **pre-computed** LM features (no
LM forward pass at train time), so each run (5 seeds × 100 epochs) is minutes; LDA/BERTopic
and the Palmetto coherence dominate wall-clock. The full matrix is hundreds of runs —
roughly a day or two end-to-end. Use `SKIP_LLM=1` to avoid OpenAI calls.

**Determinism.** Each seed is fixed via `set_seed()` (`torch`/`cuda`/`numpy`/`random` plus
`cudnn.deterministic=True`, `cudnn.enabled=False`); baselines such as `lda`/`zeroshot` also
take `random_state=seed`. Because training reads pre-computed LM features, a given run is
reproducible on the **same** GPU/driver. Two caveats: (1) *regenerating* the datasets
(`process_all.sh`) runs the LM in bf16, which is not bit-deterministic — download the
published HF datasets for exact numbers (see §2), or pass `dsl-process --deterministic`
(float32 + deterministic kernels) to regenerate reproducible targets at some speed cost;
(2) the I-RBO metric re-encodes topic
words with a per-call index map (`evaluation/diversity.py:get_word2index`) — this is
deterministic for the metric (RBO is invariant under a consistent relabeling), so it is not
a bug and should not be "fixed" with `sorted()`.

## 5. Repository structure

```
src/dsl_topic/
  cli/            process_dataset.py · train.py · retrieval.py · summarize.py   (entry points)
  models/
    dsl/          prodlda.py · ecrtm.py · fastopic.py · etm.py   ← the authors' DSL models
    baselines/    octis/ · topmost/ · fastopic/                  ← third-party baselines
  data/           loaders.py · preprocessing.py · tokenization.py · {ctm,octis}_dataset.py
  evaluation/     metrics.py · coherence.py (C_V) · diversity.py (I-RBO) · retrieval.py · rbo.py
  prompts/        renderer.py + templates/   (Jinja prompts: DSL targets + LLM rating)
  paths.py        data-directory resolution (override with DSL_TOPIC_DATA_DIR)
  settings.py     optional API keys (.env)
scripts/          experiment launch scripts (see §4)
tests/            smoke + numeric-equivalence tests
```

The model registry keys passed to `--model` (`dsl`, `dsl_ecrtm`,
`dsl_fastopic`, `dsl_etm`, and the baselines) match the paper; only the
internal module/class layout is reorganized. They are dispatched through
`dsl_topic/cli/_model_builders.py:MODEL_BUILDERS`, and the authoritative default
hyperparameters live in the `dsl-train` argparse (`dsl_topic/cli/train.py`).

## 6. Citation

```bibtex
@inproceedings{li2026dsltopic,
  title = 	 {Improving Topic Modeling by Distilling Soft Labels from Language Models},
  author =       {Li, Raymond and Abaskohi, Amirhossein and Li, Chuyuan and Murray, Gabriel and Carenini, Giuseppe},
  booktitle = 	 {Proceedings of the 43rd International Conference on Machine Learning},
  pages = 	 {to appear},
  year = 	 {2026},
  editor = 	 {to be announced},
  volume = 	 {to be announced},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {Jul},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/},
}
```

_Accepted at ICML 2026. The PMLR volume, page range, editors, and final PDF/URL will be
filled in once the proceedings are published._

## 7. Acknowledgements

The baselines under `src/dsl_topic/models/baselines/` adapt
[OCTIS](https://github.com/MIND-Lab/OCTIS) (ProdLDA, CombinedTM, ZeroShotTM, ETM, LDA),
[TopMost](https://github.com/BobXWu/TopMost) (ECRTM), and
[FASTopic](https://github.com/BobXWu/FASTopic) — adapted to the local namespace and
trimmed to what the paper uses. Please cite those works if you use the baselines.
