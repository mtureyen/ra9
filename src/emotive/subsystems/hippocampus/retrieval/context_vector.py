"""Temporal Context Model — drifting context vector.

A unit-length vector in high-dimensional space that drifts gradually
with each exchange. Memories encoded with a context snapshot. Retrieval
matches current context against stored contexts.

Brain analog: Temporal Context Model (Howard & Kahana, 2002).
  - β=0.92 drift rate (higher = slower change)
  - Context half-life: ~5-10 exchanges
  - After 5 exchanges on a new topic, similarity to old topic drops ~50%

Sources: Howard & Kahana (2002), PMC2630591.
"""

from __future__ import annotations

import math

import numpy as np


class ContextVector:
    """Drifting context vector implementing the Temporal Context Model.

    The vector updates toward each new input, creating a gradually
    shifting representation of "what we've been talking about."
    Memories encoded during similar contexts surface more easily
    when that context returns.
    """

    def __init__(self, dim: int = 256, beta: float = 0.92) -> None:
        self.dim = dim
        self.beta = beta
        # Initialize with small random vector, normalized
        rng = np.random.default_rng()
        vec = rng.standard_normal(dim).astype(np.float32)
        self._vector = vec / np.linalg.norm(vec)

    @property
    def vector(self) -> np.ndarray:
        """Current context vector (read-only view)."""
        return self._vector.copy()

    def drift(self, input_embedding: list[float] | np.ndarray) -> None:
        """Update context toward current input.

        The context blends toward the new input at rate (1 - β).
        Result is re-normalized to unit length.
        """
        inp = np.asarray(input_embedding, dtype=np.float32)

        # Truncate or pad to match dim
        if len(inp) > self.dim:
            inp = inp[: self.dim]
        elif len(inp) < self.dim:
            inp = np.pad(inp, (0, self.dim - len(inp)))

        # Normalize input
        norm = np.linalg.norm(inp)
        if norm < 1e-10:
            return  # skip zero vectors
        inp = inp / norm

        # TCM update: context_t = β * context_{t-1} + (1-β) * input_t
        self._vector = self.beta * self._vector + (1 - self.beta) * inp

        # Re-normalize to unit sphere
        vec_norm = np.linalg.norm(self._vector)
        if vec_norm > 1e-10:
            self._vector = self._vector / vec_norm

    def similarity_to(self, stored_context: list[float] | np.ndarray) -> float:
        """Cosine similarity between current context and a stored context.

        Returns 0.0 if stored context is None/empty.
        """
        if stored_context is None:
            return 0.0

        stored = np.asarray(stored_context, dtype=np.float32)
        if len(stored) == 0:
            return 0.0

        # Truncate or pad
        if len(stored) > self.dim:
            stored = stored[: self.dim]
        elif len(stored) < self.dim:
            stored = np.pad(stored, (0, self.dim - len(stored)))

        norm = np.linalg.norm(stored)
        if norm < 1e-10:
            return 0.0
        stored = stored / norm

        return float(np.dot(self._vector, stored))

    def snapshot(self) -> list[float]:
        """Return current context as a plain list for DB storage."""
        return self._vector.tolist()

    def reset(self) -> None:
        """Reset to random context (new session)."""
        rng = np.random.default_rng()
        vec = rng.standard_normal(self.dim).astype(np.float32)
        self._vector = vec / np.linalg.norm(vec)
