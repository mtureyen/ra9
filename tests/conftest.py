"""Shared fixtures for the Emotive AI test suite."""

from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from emotive.config import ConfigManager
from emotive.config.schema import EmotiveConfig
from emotive.db.models import Base
from emotive.db.models.temperament import Temperament
from emotive.embeddings.service import EmbeddingService
from emotive.runtime.event_bus import EventBus
from emotive.runtime.working_memory import WorkingMemory


@pytest.fixture(scope="session")
def db_engine():
    import os

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture(scope="session")
def seed_temperament(db_engine):
    session = sessionmaker(bind=db_engine)()
    if not session.get(Temperament, 1):
        session.add(Temperament(id=1))
        session.commit()
    session.close()


@pytest.fixture()
def db_session(db_engine, seed_temperament):
    """Rolls back after each test via nested transactions."""
    conn = db_engine.connect()
    trans = conn.begin()
    session = Session(bind=conn)

    nested = conn.begin_nested()

    @event.listens_for(session, "after_transaction_end")
    def restart_savepoint(session, transaction):
        nonlocal nested
        if transaction.nested and not transaction._parent.nested:
            nested = conn.begin_nested()

    yield session

    session.close()
    trans.rollback()
    conn.close()


@pytest.fixture()
def session_factory(db_session):
    def _factory():
        return db_session

    return _factory


@pytest.fixture(scope="session")
def embedding_service():
    svc = EmbeddingService()
    svc.embed_text("warmup")
    return svc


@pytest.fixture()
def event_bus():
    return EventBus()


@pytest.fixture()
def config():
    return EmotiveConfig()


@pytest.fixture()
def config_manager(tmp_path):
    import json

    # Write Phase 2 config so emotional + mood tools are enabled in tests
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "phase": 2,
        "layers": {
            "temperament": True,
            "episodes": True,
            "mood": True,
            "personality": False,
            "identity": False,
        },
    }))
    return ConfigManager(config_path)


@pytest.fixture()
def working_memory(event_bus):
    return WorkingMemory(capacity=5, event_bus=event_bus)


@pytest.fixture()
def app_context(session_factory, embedding_service, config_manager, event_bus):
    from emotive.app_context import AppContext

    return AppContext(
        session_factory=session_factory,
        embedding_service=embedding_service,
        config_manager=config_manager,
        event_bus=event_bus,
    )


@pytest.fixture()
def mcp_client(app_context):
    from fastmcp import Client

    from emotive.server import mcp

    original_lifespan = mcp._lifespan

    @asynccontextmanager
    async def test_lifespan(server):
        yield app_context

    mcp._lifespan = test_lifespan
    client = Client(mcp)
    yield client
    mcp._lifespan = original_lifespan
