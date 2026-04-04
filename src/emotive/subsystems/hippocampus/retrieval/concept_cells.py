"""Person-node cache — the Jennifer Aniston neuron.

When a person's name appears in input, directly activate ALL memories
tagged with that person. Bypasses embedding similarity entirely.
The name IS the key.

Built at session boot from DMN self-schema person_context.
Refreshed when new people are encountered.

Brain analog: concept cells in medial temporal lobe.
Sources: Quiroga et al. (2005), "Jennifer Aniston neuron."
"""

from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.orm import Session

from emotive.db.models.memory import Memory
from emotive.logging import get_logger

if TYPE_CHECKING:
    from emotive.subsystems.dmn.schema import SelfSchema

logger = get_logger("retrieval.person_node")


class PersonNodeCache:
    """Maps known person names to their tagged memory IDs.

    Built at session boot. Updated when new people appear.
    """

    def __init__(self) -> None:
        # name (lowercase) → set of memory UUIDs
        self._nodes: dict[str, set[uuid.UUID]] = {}
        # All known names for quick detection
        self._known_names: set[str] = set()

    def build_from_schema(
        self,
        schema: SelfSchema | None,
        db_session: Session,
    ) -> None:
        """Build cache from DMN self-schema person_context.

        Queries DB for all active memories tagged with each known person.
        """
        self._nodes.clear()
        self._known_names.clear()

        if schema is None or not schema.person_context:
            return

        # Extract person names from schema
        for name in schema.person_context:
            name_lower = name.lower().strip()
            if not name_lower:
                continue
            self._known_names.add(name_lower)

            # Query all memories tagged with this person
            stmt = (
                select(Memory.id)
                .where(Memory.is_archived.is_(False))
                .where(Memory.tags.any(name_lower))
            )
            rows = db_session.execute(stmt).scalars().all()
            self._nodes[name_lower] = set(rows)

        logger.info(
            "Person-node cache built: %d people, %d total memories",
            len(self._nodes),
            sum(len(ids) for ids in self._nodes.values()),
        )

    def detect_person(self, text: str) -> str | None:
        """Detect a known person name in the input text.

        Returns the first matching person name (lowercase), or None.
        Case-insensitive word boundary matching.
        """
        text_lower = text.lower()
        for name in self._known_names:
            # Word boundary match to avoid partial matches
            pattern = r"\b" + re.escape(name) + r"\b"
            if re.search(pattern, text_lower):
                return name
        return None

    def get_memory_ids(self, person: str) -> set[uuid.UUID]:
        """Get all memory IDs associated with a person.

        Returns empty set if person not found.
        """
        return self._nodes.get(person.lower().strip(), set())

    def add_memory(self, person: str, memory_id: uuid.UUID) -> None:
        """Register a new memory for a person (called at encoding time)."""
        name_lower = person.lower().strip()
        if name_lower not in self._nodes:
            self._nodes[name_lower] = set()
            self._known_names.add(name_lower)
        self._nodes[name_lower].add(memory_id)

    @property
    def known_people(self) -> set[str]:
        """All known person names."""
        return self._known_names.copy()
