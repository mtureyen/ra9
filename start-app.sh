#!/bin/bash
# Start ra9 desktop app: API server + Tauri frontend
# Usage: ./start-app.sh

DIR="$(cd "$(dirname "$0")" && pwd)"

# Kill any existing API server on port 8000
lsof -ti:8000 | xargs kill 2>/dev/null

# Start API server in background
cd "$DIR"
source .venv/bin/activate
python -m emotive.api &
API_PID=$!

# Wait for API to be ready
sleep 2

# Start Tauri dev mode
cd "$DIR/frontend"
npx tauri dev

# When Tauri closes, kill API server
kill $API_PID 2>/dev/null
