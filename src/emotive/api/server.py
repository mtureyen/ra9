"""FastAPI server wrapping the thalamus.

Same brain as the terminal chat — different interface. The thalamus,
all subsystems, and the LLM adapter are identical. This server
exposes them as HTTP/SSE endpoints for the Tauri desktop app.

Binds to 127.0.0.1 only. No authentication (single-user desktop app).
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from emotive.app_context import AppContext
from emotive.config import ConfigManager
from emotive.db.engine import SessionFactory
from emotive.db.queries.memory_queries import search_by_embedding
from emotive.embeddings.service import EmbeddingService
from emotive.logging import get_logger
from emotive.runtime.event_bus import EventBus, create_db_handler
from emotive.thalamus.dispatcher import Thalamus
from emotive.thalamus.session import boot_session, end_session

logger = get_logger("api")


# ── State ──────────────────────────────────────────────────────────────

class AppState:
    """Singleton state for the API server. One brain per process."""

    def __init__(self) -> None:
        self.app_context: AppContext | None = None
        self.thalamus: Thalamus | None = None
        self.session_active: bool = False

    def require_thalamus(self) -> Thalamus:
        if self.thalamus is None:
            raise HTTPException(503, "Brain not initialized")
        return self.thalamus

    def require_session(self) -> Thalamus:
        t = self.require_thalamus()
        if not self.session_active:
            raise HTTPException(400, "No active session — call POST /session/boot first")
        return t


state = AppState()


# ── Lifespan ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Boot the brain on startup, clean up on shutdown."""
    logger.info("Initializing brain...")

    config_manager = ConfigManager()
    config = config_manager.get()
    embedding_service = EmbeddingService(model_name=config.embedding_model)
    event_bus = EventBus()
    event_bus.subscribe_all(create_db_handler(SessionFactory))

    app_context = AppContext(
        session_factory=SessionFactory,
        embedding_service=embedding_service,
        config_manager=config_manager,
        event_bus=event_bus,
    )

    thalamus = Thalamus(app_context)
    app_context.thalamus = thalamus

    state.app_context = app_context
    state.thalamus = thalamus

    logger.info("Brain ready. API server running.")
    yield

    # Shutdown: end session if active
    if state.session_active:
        try:
            end_session(thalamus)
            logger.info("Session ended on shutdown")
        except Exception:
            logger.exception("Failed to end session on shutdown")

    logger.info("API server stopped.")


# ── App ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ra9 — Emotive AI",
    description="HTTP API for the ra9 cognitive architecture",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1420", "http://127.0.0.1:1420", "tauri://localhost"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


class MemorySearchRequest(BaseModel):
    q: str
    limit: int = 10


# ── Endpoints ──────────────────────────────────────────────────────────

@app.post("/session/boot")
async def api_boot_session():
    """Boot a new conversation session. Returns session_id."""
    thalamus = state.require_thalamus()

    if state.session_active:
        raise HTTPException(400, "Session already active — end it first")

    conv_id = boot_session(thalamus)
    state.session_active = True

    schema = thalamus.dmn.current
    schema_info = f"{len(schema.traits)} traits, {len(schema.core_facts)} core facts" if schema else "none"

    logger.info("Session booted: %s (schema: %s)", conv_id, schema_info)

    return {
        "session_id": str(conv_id),
        "self_schema": schema_info,
    }


@app.post("/session/end")
async def api_end_session():
    """End the active session. Triggers consolidation + export. May take 5-15 seconds."""
    thalamus = state.require_session()

    result = end_session(thalamus)
    state.session_active = False

    logger.info("Session ended")
    return result


@app.post("/chat")
async def api_chat(req: ChatRequest):
    """Send a message. Returns SSE stream of text chunks + final debug dict."""
    thalamus = state.require_session()

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            async for chunk in thalamus.process_input(req.message):
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"

            # Send debug dict as final event
            debug = thalamus.last_debug or {}
            # Convert any non-serializable values
            safe_debug = _safe_json(debug)
            yield f"data: {json.dumps({'type': 'done', 'debug': safe_debug})}\n\n"
        except Exception as e:
            logger.exception("Chat stream error")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/mood")
async def api_mood():
    """Current mood dimensions."""
    thalamus = state.require_session()
    return thalamus.mood.current


@app.get("/embodied")
async def api_embodied():
    """Current energy, cognitive load, comfort."""
    thalamus = state.require_session()
    return thalamus.embodied.to_dict()


@app.get("/schema")
async def api_schema():
    """Current self-schema from DMN."""
    thalamus = state.require_thalamus()
    schema = thalamus.dmn.current
    if schema:
        return asdict(schema)
    return {}


@app.get("/memories")
async def api_memories(q: str, limit: int = 10):
    """Search memories by text query."""
    ctx = state.app_context
    if not ctx:
        raise HTTPException(503, "Brain not initialized")

    embedding = ctx.embedding_service.embed_text(q)
    session = ctx.session_factory()
    try:
        results = search_by_embedding(session, embedding, limit=limit)
        return [_safe_json(r) for r in results]
    finally:
        session.close()


@app.get("/history")
async def api_history():
    """Conversation turns for current session."""
    thalamus = state.require_session()
    turns = thalamus.prefrontal.buffer.turns
    return [{"role": t.role, "content": t.content} for t in turns]


@app.get("/health")
async def api_health():
    """Health check."""
    return {
        "status": "ok",
        "brain": state.thalamus is not None,
        "session": state.session_active,
    }


# ── Helpers ────────────────────────────────────────────────────────────

def _safe_json(obj: dict) -> dict:
    """Make a dict JSON-serializable by converting non-standard types."""
    result = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            result[k] = _safe_json(v)
        elif isinstance(v, (str, int, float, bool, type(None))):
            result[k] = v
        elif isinstance(v, (list, tuple)):
            result[k] = [_safe_item(item) for item in v]
        else:
            result[k] = str(v)
    return result


def _safe_item(item):
    if isinstance(item, dict):
        return _safe_json(item)
    elif isinstance(item, (str, int, float, bool, type(None))):
        return item
    return str(item)
