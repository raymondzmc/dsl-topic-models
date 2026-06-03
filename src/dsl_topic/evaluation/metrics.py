import os
import json
import time
import numpy as np
from typing import Optional
from openai import OpenAI, RateLimitError, APIError
from collections import defaultdict
from dsl_topic.settings import settings
from dsl_topic.prompts import jinja_template_manager
from dsl_topic.evaluation.abc import AbstractMetric
from dsl_topic.evaluation.diversity import TopicDiversity, InvertedRBO
from dsl_topic.evaluation.coherence import PalmettoCoherence
from sklearn.metrics.cluster import contingency_matrix
from sklearn import metrics
from concurrent.futures import ThreadPoolExecutor, as_completed


def compute_llm_rating(topics: list[list[str]], model: str = "gpt-4o"):
    system_prompt = jinja_template_manager.render("topic_ratings_system.jinja")
    
    def render_messages(topic: list[str]):
        user_prompt = jinja_template_manager.render(
            "topic_ratings_user.jinja",
            topic=topic,
        )
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]
        return messages

    def get_single_topic_rating(topic):
        messages = render_messages(topic)
        client = OpenAI(api_key=settings.openai_api_key)
        
        rating: Optional[int] = None
        temperature: float = 0.0
        num_attempts: int = 0
        max_attempts: int = 10
        base_delay: float = 1.0
        
        while rating is None and num_attempts < max_attempts:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=1,
                )
                _rating = int(response.choices[0].message.content)
                
                if _rating in [1, 2, 3]:
                    rating = _rating
                else:
                    temperature += 0.1
                    num_attempts += 1
            except RateLimitError as e:
                # Exponential backoff for rate limits
                delay = base_delay * (2 ** num_attempts)
                print(f"Rate limit hit for topic, waiting {delay:.1f}s before retry...")
                time.sleep(delay)
                num_attempts += 1
            except APIError as e:
                # API errors - retry with backoff
                delay = base_delay * (2 ** num_attempts)
                print(f"API error for topic \"{topic[:3]}...\": {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                num_attempts += 1
            except ValueError as e:
                # Invalid response format
                print(f"Invalid response for topic \"{topic[:3]}...\": {e}")
                temperature += 0.1
                num_attempts += 1
            except Exception as e:
                print(f"Error for topic \"{topic[:3]}...\": {e}")
                temperature += 0.1
                num_attempts += 1
                
        if rating is None:
            raise RuntimeError(f"Could not get LLM rating for topic {topic[:3]}... after {max_attempts} attempts.")
            
        return rating

    # Use ThreadPoolExecutor with reduced parallelism to avoid rate limits
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        future_to_topic = {executor.submit(get_single_topic_rating, topic): i for i, topic in enumerate(topics)}
        
        # Collect results ensuring order matches input topics
        topic_ratings = [None] * len(topics)
        for future in as_completed(future_to_topic):
            index = future_to_topic[future]
            rating = future.result()
            topic_ratings[index] = rating

    return topic_ratings


def compute_purity_score(topic_document_matrix, labels ):
    """
    Compute cluster purity (and optionally inverse & harmonic purity)
    for a topic model given ground‑truth document labels.

    Parameters
    ----------
    labels : array‑like, shape (n_documents,)
        Ground‑truth class label for each document.

    topic_document_matrix : ndarray, shape (K, n_documents)
        Topic weights/probabilities per document.  Each column j is the
        distribution θ_{·,j} over K topics for document j.

    Returns
    -------
    purity : float
        Standard cluster purity in [0, 1].

    inverse_purity : float
        Inverse cluster purity in [0, 1].

    harmonic_purity : float
        Harmonic mean of purity and inverse purity in [0, 1].
    """
    labels = np.asarray(labels)
    # sanity check
    if topic_document_matrix.shape[1] != labels.shape[0]:
        raise ValueError(
            "topic_document_matrix must have the same number "
            "of columns as the length of `labels`."
        )

    # 1. Hard‐assign every document to its most probable topic
    y_pred = topic_document_matrix.argmax(axis=0)  # shape (n_documents,)

    # 2. Build contingency matrix: rows=gold classes, cols=predicted topics
    cmat = contingency_matrix(labels, y_pred)
    n_samples = cmat.sum()

    # 3. Purity: for each predicted cluster, count how many docs come
    #    from its dominant class, then normalise.
    purity = cmat.max(axis=0).sum() / n_samples

    # Inverse purity (a.k.a. completeness)
    inverse_purity = cmat.max(axis=1).sum() / n_samples

    # Harmonic purity (F1 between purity & inverse purity)
    harmonic_purity = (
        2 * purity * inverse_purity / (purity + inverse_purity)
        if (purity + inverse_purity) > 0
        else 0.0
    )

    return purity, inverse_purity, harmonic_purity


def evaluate_topic_model(model_output, top_words=10, labels=None,
                         skip_llm_rating=False):
    assert 'topics' in model_output, "model_output must contain 'topics'"

    evaluation_results = {}

    td = TopicDiversity(topk=top_words)
    td_score = td.score(model_output)
    print("Topic Diversity:", td_score)
    evaluation_results['topic_diversity'] = float(td_score)
    
    irbo = InvertedRBO(topk=top_words)
    irbo_score = irbo.score(model_output)
    print("Inverted RBO:", irbo_score)
    evaluation_results['inverted_rbo'] = float(irbo_score)

    if labels is not None and model_output.get('topic-document-matrix') is not None:
        purity_score, inverse_purity, harmonic_purity = compute_purity_score(model_output['topic-document-matrix'], labels)
        print("Purity:", purity_score)
        evaluation_results['purity'] = float(purity_score)
        print("Inverse Purity:", inverse_purity)
        evaluation_results['inverse_purity'] = float(inverse_purity)
        print("Harmonic Purity:", harmonic_purity)
        evaluation_results['harmonic_purity'] = float(harmonic_purity)
        
        ari_score = metrics.adjusted_rand_score(labels, model_output['topic-document-matrix'].argmax(axis=0))
        print("ARI:", ari_score)
        evaluation_results['ari'] = float(ari_score)
        mis_score = metrics.normalized_mutual_info_score(labels, model_output['topic-document-matrix'].argmax(axis=0))
        print("MIS:", mis_score)
        evaluation_results['mis'] = float(mis_score)

    # Wikipedia-based C_V using Palmetto (paths resolve under DATA_DIR)
    from dsl_topic.paths import PALMETTO_JAR as palmetto_jar, WIKIPEDIA_INDEX as wikipedia_index
    
    jar_exists = os.path.exists(palmetto_jar)
    index_exists = os.path.exists(wikipedia_index)
    
    if jar_exists and index_exists:
        try:
            palmetto_cv = PalmettoCoherence(
                palmetto_jar=palmetto_jar,
                wikipedia_index=wikipedia_index,
                measure="C_V",
                topk=top_words,
            )
            cv_wiki_score = palmetto_cv.score(model_output)
            if cv_wiki_score != -1.0:
                print("CV (Wikipedia):", cv_wiki_score)
                evaluation_results['cv_wiki'] = float(cv_wiki_score)
            else:
                print("CV (Wikipedia): skipped (Palmetto returned -1.0)")
        except Exception as e:
            print(f"Palmetto coherence skipped: {e}")
    else:
        missing = []
        if not jar_exists:
            missing.append(f"JAR not found: {palmetto_jar}")
        if not index_exists:
            missing.append(f"Index not found: {wikipedia_index}")
        print(f"CV (Wikipedia): skipped ({'; '.join(missing)})")

    # LLM topic-coherence rating (gpt-4o). Requires an OpenAI key; skipped otherwise.
    if skip_llm_rating:
        print("LLM Rating: skipped (--skip_llm_rating)")
    elif not settings.openai_api_key:
        print("LLM Rating: skipped (no OPENAI_API_KEY set)")
    else:
        llm_ratings = compute_llm_rating(model_output['topics'])
        llm_average_rating = float(np.mean(llm_ratings))
        print("LLM Rating:", llm_average_rating)
        evaluation_results['llm_rating'] = llm_average_rating
    return evaluation_results


def compute_aggregate_results(results_path):
    aggregated_results = defaultdict(float)
    counts = defaultdict(int)
    for seed_dir in os.listdir(results_path):
        results_file = os.path.join(results_path, seed_dir, 'evaluation_results.json')
        if os.path.exists(results_file):
            results = json.load(open(results_file, encoding='utf-8'))

            # Backwards compatibility (used to save topics and results)
            if isinstance(results, list):
                results = results[1]

            assert isinstance(results, dict)
            for k, v in results.items():
                aggregated_results[k] += v
                counts[k] += 1

    metrics = aggregated_results.keys()
    for k in metrics:
        aggregated_results[k] /= counts[k]
        print(f"[{k}] {aggregated_results[k]} (from {counts[k]} runs)")
    return aggregated_results
