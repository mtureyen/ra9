"""FastMCP server entrypoint for Emotive AI."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastmcp import FastMCP

from emotive.app_context import AppContext
from emotive.config import ConfigManager
from emotive.db.engine import SessionFactory
from emotive.embeddings.service import EmbeddingService
from emotive.runtime.event_bus import EventBus, create_db_handler
from emotive.tools.atomic.appraise import appraise_tool
from emotive.tools.atomic.create_episode import create_episode_tool
from emotive.tools.atomic.decay_memories import decay_memories_tool
from emotive.tools.atomic.link_memories import link_memories_tool
from emotive.tools.atomic.store_memory import store_memory_tool
from emotive.tools.composite.begin_session import begin_session_tool
from emotive.tools.composite.consolidate import consolidate_tool
from emotive.tools.composite.end_session import end_session_tool
from emotive.tools.composite.experience_event import experience_event_tool
from emotive.tools.composite.recall import recall_tool
from emotive.tools.observability.export_timeline import export_timeline_tool
from emotive.tools.observability.get_history import get_history_tool
from emotive.tools.observability.get_state import get_state_tool
from emotive.tools.observability.reset_layer import reset_layer_tool
from emotive.tools.observability.search_memories import search_memories_tool
from emotive.tools.observability.set_config import set_config_tool


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Initialize shared services on startup."""
    config_manager = ConfigManager()
    config = config_manager.get()
    embedding_service = EmbeddingService(model_name=config.embedding_model)
    event_bus = EventBus()
    event_bus.subscribe_all(create_db_handler(SessionFactory))

    yield AppContext(
        session_factory=SessionFactory,
        embedding_service=embedding_service,
        config_manager=config_manager,
        event_bus=event_bus,
    )


mcp = FastMCP(
    "Emotive AI",
    instructions=(
        "Emotive AI emotional memory system. Use begin_session at conversation start, "
        "experience_event when something emotionally significant happens, "
        "store_memory for general information, recall to retrieve memories, "
        "and end_session when done."
    ),
    lifespan=lifespan,
)

# Composite tools (normal conversation flow)
mcp.add_tool(begin_session_tool)
mcp.add_tool(end_session_tool)
mcp.add_tool(recall_tool)
mcp.add_tool(consolidate_tool)
mcp.add_tool(experience_event_tool)

# Atomic tools (single operations)
mcp.add_tool(store_memory_tool)
mcp.add_tool(appraise_tool)
mcp.add_tool(create_episode_tool)
mcp.add_tool(link_memories_tool)
mcp.add_tool(decay_memories_tool)

# Observability tools
mcp.add_tool(get_state_tool)
mcp.add_tool(get_history_tool)
mcp.add_tool(search_memories_tool)
mcp.add_tool(set_config_tool)
mcp.add_tool(reset_layer_tool)
mcp.add_tool(export_timeline_tool)


if __name__ == "__main__":
    mcp.run()
