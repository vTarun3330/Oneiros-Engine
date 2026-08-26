"""
Text embedding utilities for Oneiros Engine.
"""
from typing import List, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from config import memory_config


class EmbeddingModel:
    """
    Wrapper for sentence-transformers embedding model.
    """

    def __init__(self, model_name: str = None):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name or memory_config.embedding_model
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings.

        Args:
            texts: Single text or list of texts to encode
            normalize: Whether to L2-normalize embeddings

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )

        return np.array(embeddings).astype('float32')

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0-1)
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        return float(np.dot(emb1[0], emb2[0]))


# Lazy-loaded global instance
_embedding_model = None


def get_embedding_model() -> EmbeddingModel:
    """Get or create the global embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model


def get_embeddings(texts: Union[str, List[str]]) -> np.ndarray:
    """
    Convenience function to get embeddings.

    Args:
        texts: Text(s) to embed

    Returns:
        Embedding array
    """
    return get_embedding_model().encode(texts)
