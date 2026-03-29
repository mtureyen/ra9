"""Shared application context for dependency injection."""

from __future__ import annotations

from dataclasses import dataclass

from emotive.config import ConfigManager
from emotive.embeddings.service import EmbeddingService
from emotive.runtime.event_bus import EventBus


@dataclass
class AppContext:
    """Shared services available to all tools via ctx.lifespan_context."""

    session_factory: type
    embedding_service: EmbeddingService
    config_manager: ConfigManager
    event_bus: EventBus
