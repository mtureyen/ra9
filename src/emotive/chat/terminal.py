"""Terminal chat loop: stdin/stdout through the cognitive pipeline.

Usage: python -m emotive.chat
   or: emotive-chat

The brain boots, Ryo wakes up, and you talk in the terminal.
Type 'exit' or 'quit' to end the session gracefully.
"""

from __future__ import annotations

import asyncio
import sys

from emotive.app_context import AppContext
from emotive.config import ConfigManager
from emotive.db.engine import SessionFactory
from emotive.embeddings.service import EmbeddingService
from emotive.logging import get_logger
from emotive.runtime.event_bus import EventBus, create_db_handler
from emotive.thalamus.dispatcher import Thalamus
from emotive.thalamus.session import boot_session, end_session

logger = get_logger("chat")


def _create_app_context() -> AppContext:
    """Initialize all shared services (same as server.py lifespan)."""
    config_manager = ConfigManager()
    config = config_manager.get()
    embedding_service = EmbeddingService(model_name=config.embedding_model)
    event_bus = EventBus()
    event_bus.subscribe_all(create_db_handler(SessionFactory))

    return AppContext(
        session_factory=SessionFactory,
        embedding_service=embedding_service,
        config_manager=config_manager,
        event_bus=event_bus,
    )


async def run_terminal() -> None:
    """Main terminal chat loop."""
    print("Initializing brain...", flush=True)

    # 1. Initialize services
    app = _create_app_context()

    # 2. Create thalamus (initializes all subsystems)
    thalamus = Thalamus(app)
    app.thalamus = thalamus

    # 3. Boot session (staged: orphans → conversation → self-schema → ready)
    print("Booting session...", flush=True)
    conv_id = boot_session(thalamus)
    print(f"Session: {conv_id}\n")
    print("Ryo is ready. Type 'exit' to end.\n", flush=True)

    # 4. Chat loop
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
                break

            # Stream response through cognitive pipeline
            print("Ryo: ", end="", flush=True)
            try:
                async for chunk in thalamus.process_input(user_input):
                    print(chunk, end="", flush=True)
                print()  # newline after response
            except Exception:
                print("\n[Error generating response]", flush=True)
                logger.exception("Error during process_input")

    except KeyboardInterrupt:
        print()  # clean line after ^C

    # 5. Graceful shutdown
    print("\nEnding session...", flush=True)
    try:
        result = end_session(thalamus)
        archived = result.get("episodes_archived", 0)
        schema = result.get("self_schema_regenerated", False)
        print(f"Episodes archived: {archived}")
        print(f"Self-schema regenerated: {schema}")
    except Exception:
        logger.exception("Error during session end")
    print("Goodbye.")


def main() -> None:
    """Entry point for emotive-chat command."""
    asyncio.run(run_terminal())


if __name__ == "__main__":
    main()
