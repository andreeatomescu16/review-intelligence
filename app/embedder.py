"""
embedder.py

Wraps the SentenceTransformer model used to produce dense vector embeddings
for review highlights and pain points.

Model choice — multi-qa-MiniLM-L6-cos-v1:
  This model was explicitly fine-tuned on semantic similarity / question-answer
  retrieval tasks (hence "multi-qa"), making its embedding space better
  calibrated for "same meaning, different words" comparisons than the generic
  all-MiniLM-L6-v2. Both models have the same memory footprint (~22M parameters,
  ~80 MB on disk).
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"


class ReviewEmbedder:
    """
    Singleton-style wrapper around a SentenceTransformer model.

    The model is loaded once at instantiation time. Callers should create a
    single instance and reuse it for all embed() calls to avoid repeated
    model loading overhead.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME) -> None:
        """
        Load the SentenceTransformer model.

        Args:
            model_name: HuggingFace model identifier. Defaults to
                        multi-qa-MiniLM-L6-cos-v1.
        """
        logger.info("Loading embedding model '%s'.", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded.")

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of strings into L2-normalised embedding vectors.

        L2 normalisation is applied so that the dot product between any two
        vectors equals their cosine similarity. This allows the deduplicator
        to compute the full pairwise cosine similarity matrix with a single
        matrix multiply (embeddings @ embeddings.T), which is both numerically
        correct and computationally efficient.

        Args:
            texts: List of strings to embed. Must be non-empty.

        Returns:
            2D numpy array of shape (len(texts), embedding_dim) with each row
            being a unit-length vector.
        """
        if not texts:
            # Return an empty array with the correct number of dimensions so
            # downstream code can always assume a 2D array.
            return np.empty((0, self._model.get_sentence_embedding_dimension()))

        raw_embeddings: np.ndarray = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # L2-normalise each row: divide by its Euclidean norm.
        # After this step: ||v|| = 1 for every row, so v_i · v_j = cos(v_i, v_j).
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        # Guard against zero-norm vectors (e.g. empty strings) to avoid NaN
        norms = np.where(norms == 0, 1.0, norms)
        normalised_embeddings = raw_embeddings / norms

        return normalised_embeddings
