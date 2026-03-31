"""Shared application context for dependency injection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from emotive.config import ConfigManager
from emotive.embeddings.service import EmbeddingService
from emotive.runtime.event_bus import EventBus

if TYPE_CHECKING:
    from emotive.thalamus.dispatcher import Thalamus


@dataclass
class AppContext:
    """Shared services available to all tools via ctx.lifespan_context."""

    session_factory: type
    embedding_service: EmbeddingService
    config_manager: ConfigManager
    event_bus: EventBus
    # Phase 1.5: thalamus orchestrator (set after initialization)
    thalamus: Thalamus | None = field(default=None, repr=False)
