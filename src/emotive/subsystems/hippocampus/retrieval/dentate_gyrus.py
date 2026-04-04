"""Dentate Gyrus — pattern separation.

Transforms similar inputs into highly dissimilar, non-overlapping
representations. Prevents catastrophic interference — without DG,
similar memories would blur together.

When "do you remember floret?" and "do you remember lena?" arrive,
create DISTINCT retrieval paths despite similar sentence structure.

Mechanism: if a new query overlaps heavily with a recent query but
targets a different person/topic, subtract the shared component to
create an orthogonal retrieval vector.

Brain analog: dentate gyrus granule cells.
Sources: PMC2976779, Science 1152882.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class PatternSeparator:
    """Orthogonalize similar queries to prevent retrieval blurring.

    Maintains a short history of recent query vectors (last 3 exchanges).
    When a new query has high overlap with a recent one, subtracts the
    shared component to create a distinct retrieval vector.
    """

    def __init__(self, history_size: int = 3) -> None:
        self._recent: deque[tuple[np.ndarray, str | None]] = deque(
            maxlen=history_size,
        )

    def separate(
        self,
        query_embedding: list[float] | np.ndarray,
        detected_person: str | None = None,
    ) -> np.ndarray:
        """Apply pattern separation to a query embedding.

        If the query overlaps > 0.7 with a recent query but targets
        a DIFFERENT person/topic, orthogonalize by subtracting the
        shared component.

        Returns the (possibly modified) query embedding as numpy array.
        """
        query = np.asarray(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-10:
            return query

        query_unit = query / query_norm

        for recent_vec, recent_person in self._recent:
            overlap = float(np.dot(query_unit, recent_vec))

            if overlap > 0.7:
                # High overlap with a recent query
                if detected_person != recent_person:
                    # Different person/topic — orthogonalize
                    # Subtract the component of query along the recent vector
                    projection = overlap * recent_vec
                    separated = query_unit - 0.5 * projection

                    # Re-normalize
                    sep_norm = np.linalg.norm(separated)
                    if sep_norm > 1e-10:
                        separated = separated / sep_norm
                        # Scale back to original magnitude
                        query = separated * query_norm

        # Store this query in history
        self._recent.append((query_unit.copy(), detected_person))

        return query

    def reset(self) -> None:
        """Clear history (new session)."""
        self._recent.clear()
