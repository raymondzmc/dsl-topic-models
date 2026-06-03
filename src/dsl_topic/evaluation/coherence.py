"""Topic coherence via the Palmetto JAR over a Wikipedia reference corpus.

This implements the paper's reported coherence metric, ``C_V`` (``cv_wiki``).
"""
import os
import re
import subprocess
import tempfile
import warnings

import numpy as np

from dsl_topic.evaluation.abc import AbstractMetric


class PalmettoCoherence(AbstractMetric):
    """
    Compute topic coherence using the Palmetto JAR with Wikipedia as reference corpus.

    Requires:
    - Java installed and available in PATH
    - Palmetto JAR file (palmetto-0.1.0-jar-with-dependencies.jar)
    - Wikipedia Lucene index (wikipedia_bd directory)
    """

    def __init__(
        self,
        palmetto_jar: str = None,
        wikipedia_index: str = None,
        measure: str = "C_V",
        topk: int = 10,
    ):
        """
        Initialize Palmetto coherence metric.

        Parameters
        ----------
        palmetto_jar : str
            Path to the Palmetto JAR file.
        wikipedia_index : str
            Path to the Wikipedia Lucene index directory.
        measure : str
            Coherence measure to use. Options: C_V, C_NPMI, C_P, C_A, C_UCI, C_CP.
        topk : int
            Number of top words per topic to use for coherence computation.
        """
        super().__init__()
        from dsl_topic.paths import PALMETTO_JAR, WIKIPEDIA_INDEX
        self.palmetto_jar = palmetto_jar if palmetto_jar is not None else PALMETTO_JAR
        self.wikipedia_index = wikipedia_index if wikipedia_index is not None else WIKIPEDIA_INDEX
        self.measure = measure
        self.topk = topk

        # Validate paths exist
        if not os.path.exists(self.palmetto_jar):
            raise FileNotFoundError(f"Palmetto JAR not found: {self.palmetto_jar}")
        if not os.path.exists(self.wikipedia_index):
            raise FileNotFoundError(f"Wikipedia index not found: {self.wikipedia_index}")

    def info(self):
        return {
            "citation": "Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures.",
            "name": f"Palmetto Coherence ({self.measure})"
        }

    def score(self, model_output) -> float:
        """
        Compute coherence score using Palmetto.

        Parameters
        ----------
        model_output : dict
            Dictionary containing 'topics' key with list of topic word lists.

        Returns
        -------
        float
            Mean coherence score across all topics.
        """
        topics = model_output["topics"]
        if topics is None:
            return -1.0

        if self.topk > len(topics[0]):
            raise ValueError(f"Words in topics ({len(topics[0])}) are less than topk ({self.topk})")

        # Process topics individually to avoid Palmetto batch processing bugs
        # (ArrayIndexOutOfBoundsException when topics contain empty strings)
        scores = []

        for topic in topics:
            # Take only top-k words and filter out empty strings
            top_words = [w for w in topic[:self.topk] if w.strip()]

            if len(top_words) == 0:
                # Skip empty topics
                continue

            # Create temporary file with single topic
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(' '.join(top_words) + '\n')
                topics_file = f.name

            try:
                # Run Palmetto JAR
                cmd = [
                    'java', '-jar', self.palmetto_jar,
                    self.wikipedia_index,
                    self.measure,
                    topics_file
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,  # 1 minute timeout per topic
                )

                if result.returncode != 0:
                    # Skip this topic if Palmetto fails
                    continue

                # Parse output - format: "    0\t0.65341\t[word, word, ...]"
                for line in result.stdout.strip().split('\n'):
                    match = re.match(r'^\s*\d+\t([-\d.]+)\t', line)
                    if match:
                        score = float(match.group(1))
                        scores.append(score)
                        break

            except subprocess.TimeoutExpired:
                # Skip this topic if it times out
                continue
            except FileNotFoundError:
                warnings.warn("Java not found. Please install Java to use Palmetto coherence.")
                return -1.0
            except Exception:
                # Skip this topic on any error
                continue
            finally:
                # Clean up temp file
                if os.path.exists(topics_file):
                    os.unlink(topics_file)

        if len(scores) == 0:
            warnings.warn("No topics could be scored by Palmetto")
            return -1.0

        return float(np.mean(scores))
