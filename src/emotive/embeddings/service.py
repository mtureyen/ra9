"""Embedding service wrapping mxbai-embed-large for local inference."""

from __future__ import annotations

from emotive.logging import get_logger

logger = get_logger("embeddings")

EMBEDDING_DIM = 1024


class EmbeddingService:
    """Generates embeddings using a sentence-transformers model.

    Model is loaded lazily on first call to keep startup fast.
    """

    def __init__(self, model_name: str = "mixedbread-ai/mxbai-embed-large-v1") -> None:
        self._model_name = model_name
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {self._model_name}")
        self._model = SentenceTransformer(self._model_name)
        logger.info("Embedding model loaded")

    def embed_text(self, text: str) -> list[float]:
        """Generate a 1024-dim embedding for a single text."""
        self._load_model()
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        self._load_model()
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM
