"""Terminal chat loop: stdin/stdout through the cognitive pipeline.

Usage: python -m emotive.chat
   or: emotive-chat

The brain boots, you talk in the terminal.
Type 'exit' or 'quit' to end the session gracefully.

Brain activity is written to logs/brain.log — watch it with:
    python -m emotive.debug
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from emotive.app_context import AppContext
from emotive.config import ConfigManager
from emotive.db.engine import SessionFactory
from emotive.embeddings.service import EmbeddingService
from emotive.logging import get_logger
from emotive.runtime.event_bus import EventBus, create_db_handler
from emotive.thalamus.dispatcher import Thalamus
from emotive.thalamus.session import boot_session, end_session

logger = get_logger("chat")


def _setup_logging() -> tuple[Path, Path]:
    """Set up file logging. Returns (session_log_path, brain_log_path)."""
    log_dir = Path(os.environ.get("RA9_LOG_DIR", "logs"))
    log_dir.mkdir(exist_ok=True)

    session_log = log_dir / "session.log"
    brain_log = log_dir / "brain.log"

    # Structured JSON logs → session.log
    from emotive.logging import StructuredFormatter

    file_handler = logging.FileHandler(session_log, mode="a", encoding="utf-8")
    file_handler.setFormatter(StructuredFormatter())

    root = logging.getLogger("emotive")
    root.addHandler(file_handler)
    root.setLevel(logging.INFO)

    # Suppress stderr output — all logs go to files only
    for handler in list(root.handlers):
        if isinstance(handler, logging.StreamHandler) and handler.stream is not None:
            if hasattr(handler.stream, "name") and handler.stream.name == "<stderr>":
                root.removeHandler(handler)
    # Also suppress child loggers' stderr handlers
    for name in logging.Logger.manager.loggerDict:
        if name.startswith("emotive."):
            child = logging.getLogger(name)
            for handler in list(child.handlers):
                if isinstance(handler, logging.StreamHandler):
                    child.removeHandler(handler)

    return session_log, brain_log


def _write_brain_status(brain_log: Path, debug: dict) -> None:
    """Write formatted brain activity to brain.log for the debug monitor."""
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    lines = []

    lines.append(f"─── {now} ───")

    # Appraisal
    emotion = debug["final_emotion"]
    intensity = debug["final_intensity"]
    if debug["reappraised"]:
        lines.append(
            f"  amygdala: {debug['fast_emotion']} ({debug['fast_intensity']:.2f}) "
            f"→ reappraised: {emotion} ({intensity:.2f})"
        )
    else:
        lines.append(f"  amygdala: {emotion} ({intensity:.2f})")

    # Recall
    count = debug.get("recalled_count", 0)
    if count > 0:
        top = debug.get("recalled_top", "")
        lines.append(f'  recalled: {count} memories (top: "{top}")')
    else:
        lines.append("  recalled: 0 memories")

    # Encoding
    if debug.get("encoded"):
        lines.append(f"  encoded: episode + memory ({emotion}, {intensity:.2f})")
    else:
        lines.append("  encoded: none (below threshold)")

    # Intent
    if debug.get("intent_detected"):
        lines.append("  intent: detected → enhanced encoding")

    # Loop detection
    if debug.get("loop_detected"):
        lines.append("  loop: detected → novelty nudge queued for next exchange")

    # Gist
    gist_count = debug.get("gist_compressed", 0)
    if gist_count > 0:
        lines.append(f"  gist: {gist_count} turns compressed")

    lines.append("")

    with open(brain_log, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


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
    session_log, brain_log = _setup_logging()

    print("Initializing brain...", flush=True)

    # 1. Initialize services
    app = _create_app_context()

    # 2. Create thalamus (initializes all subsystems)
    thalamus = Thalamus(app)
    app.thalamus = thalamus

    # 3. Boot session
    print("Booting session...", flush=True)
    conv_id = boot_session(thalamus)

    schema = thalamus.dmn.current
    traits_count = len(schema.traits) if schema else 0
    facts_count = len(schema.core_facts) if schema else 0

    print(f"Session: {conv_id}")
    print(f"Self-schema: {traits_count} traits, {facts_count} core facts")
    print(f"Debug: python -m emotive.debug")
    print()
    print("Ready. Type 'exit' to end.", flush=True)
    print()

    # Write boot info to brain log
    _write_brain_status(brain_log, {
        "fast_emotion": "boot", "fast_intensity": 0,
        "reappraised": False, "final_emotion": "boot", "final_intensity": 0,
        "recalled_count": 0, "recalled_top": None,
        "encoded": False, "intent_detected": False, "gist_compressed": 0,
    })

    # 4. Chat loop
    interrupted = False
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

            # Stream response
            print("Ryo: ", end="", flush=True)
            try:
                async for chunk in thalamus.process_input(user_input):
                    print(chunk, end="", flush=True)
                print()  # newline after response
                print()  # blank line before next "You: " prompt

                # Write brain activity to debug log (not to chat terminal)
                if thalamus.last_debug:
                    _write_brain_status(brain_log, thalamus.last_debug)

            except KeyboardInterrupt:
                print("\n[interrupted]")
                interrupted = True
                break
            except asyncio.CancelledError:
                print("\n[interrupted]")
                interrupted = True
                break
            except Exception:
                print("\n[Error generating response]", flush=True)
                logger.exception("Error during process_input")

    except KeyboardInterrupt:
        interrupted = True
        print()

    # 5. Graceful shutdown — always runs, even on Ctrl+C
    print("\nEnding session...", flush=True)
    try:
        result = end_session(thalamus)
        archived = result.get("episodes_archived", 0)
        schema_regen = result.get("self_schema_regenerated", False)
        print(f"Episodes archived: {archived}")
        print(f"Self-schema regenerated: {schema_regen}")
    except Exception:
        logger.exception("Error during session end")
    print("Goodbye.")


def main() -> None:
    """Entry point for emotive-chat command."""
    asyncio.run(run_terminal())


if __name__ == "__main__":
    main()
