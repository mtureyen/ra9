"""Close all open sessions. Used as a Claude Code exit hook."""

from __future__ import annotations

from emotive.config import ConfigManager
from emotive.db.engine import SessionFactory
from emotive.embeddings.service import EmbeddingService
from emotive.memory.session_cleanup import close_orphaned_sessions


def close_open_sessions() -> None:
    """Find and close any unclosed conversation sessions."""
    config_manager = ConfigManager()
    config = config_manager.get()
    embedding_service = EmbeddingService(model_name=config.embedding_model)

    session = SessionFactory()
    try:
        cleaned = close_orphaned_sessions(
            session,
            embedding_service,
            config,
            limit=50,
        )
        session.commit()
        if cleaned:
            print(f"Closed {cleaned} orphaned session(s)")
    except Exception as e:
        session.rollback()
        print(f"Error closing sessions: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    close_open_sessions()
