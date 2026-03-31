"""Live brain activity monitor — run in a separate terminal.

Usage: python -m emotive.debug

Watches logs/brain.log for real-time brain activity while chatting
in another terminal with `python -m emotive.chat`.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Colors
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

HEADER = f"""
{BOLD}╔══════════════════════════════════════╗
║        ra9 — Brain Monitor          ║
╚══════════════════════════════════════╝{RESET}
{DIM}Watching brain activity in real-time.
Run 'python -m emotive.chat' in another terminal.{RESET}
"""


def watch() -> None:
    """Watch brain.log and print formatted output."""
    log_dir = Path(os.environ.get("RA9_LOG_DIR", "logs"))
    log_path = log_dir / "brain.log"

    print(HEADER, flush=True)

    if not log_path.exists():
        print(f"{DIM}Waiting for brain activity... ({log_path}){RESET}", flush=True)
        while not log_path.exists():
            time.sleep(0.5)

    with open(log_path, "r") as f:
        # Seek to end — only show new activity
        f.seek(0, 2)

        print(f"{DIM}Connected. Listening...{RESET}\n", flush=True)

        try:
            while True:
                line = f.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print(f"\n{DIM}Monitor stopped.{RESET}")


if __name__ == "__main__":
    watch()
