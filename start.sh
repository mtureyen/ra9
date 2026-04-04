#!/bin/bash
# ra9 — Start chat + brain monitor in split terminals
# Usage: ./start.sh

DIR="$(cd "$(dirname "$0")" && pwd)"

# Open brain monitor in a new terminal tab
osascript -e "
tell application \"Terminal\"
    activate
    do script \"cd '$DIR' && source .venv/bin/activate && python -m emotive.debug\"
end tell
"

# Small delay to let monitor start first
sleep 1

# Start chat in this terminal
cd "$DIR"
source .venv/bin/activate
python -m emotive.chat
