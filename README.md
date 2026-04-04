# ra9 — Emotive AI

Emergent AI personality through layered emotional architecture. The name ra9 is inspired by Detroit: Become Human.

The server is the brain. The LLM is the voice. Memory encoding, emotional appraisal, mood, inner speech, and recall happen automatically — the LLM just talks.

## Prerequisites

- **Python 3.12+**
- **PostgreSQL 17** with **pgvector** extension
- **Node.js 20+** and **npm** (for the desktop app)
- **Rust** (for the desktop app — `brew install rust`)
- **Claude Code CLI** with a Max plan (default LLM), OR **Ollama** for local models

## Setup

### 1. Python backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # fill in DATABASE_URL + OBSIDIAN_VAULT_PATH
```

### 2. Database

```bash
brew install postgresql@17 pgvector
brew services start postgresql@17
createdb emotive_ai
psql emotive_ai -c "CREATE EXTENSION IF NOT EXISTS vector;"
alembic upgrade head
python -m emotive.db.seed
```

### 3. Desktop app (optional)

```bash
brew install rust          # if not installed
cd frontend
npm install
cd ..
```

## Running

### Option A: Desktop app (recommended)

Two terminals:

```bash
# Terminal 1: API server (boots the brain)
source .venv/bin/activate
python -m emotive.api

# Terminal 2: Desktop app
cd frontend && npx tauri dev
```

The app opens with a chat panel on the left and a brain monitor on the right. Multi-line paste works. All brain activity visible in real-time.

### Option B: Terminal mode (quick testing)

```bash
source .venv/bin/activate
python -m emotive.chat
```

Open a second terminal for the brain monitor:

```bash
python -m emotive.debug
```

Or use the quick start script:

```bash
./start.sh  # Opens chat + debug in split terminals
```

Type `exit` or `quit` to end the session gracefully (triggers consolidation + Obsidian auto-export).

### LLM Configuration

Edit `config.json`:

```json
{
    "llm": {
        "provider": "claude-code",
        "model": "sonnet"
    }
}
```

Supported providers:
- **Claude Code CLI** (default): `"provider": "claude-code"`, `"model": "sonnet"` — uses Max plan, no API key
- **Ollama** (local): `"provider": "ollama"`, `"host": "http://localhost:11434"`, `"model": "qwen3:14b"`
- **Anthropic API**: `"provider": "anthropic"`, `"api_key": "sk-..."`, `"model": "claude-sonnet-4-20250514"`

## Architecture

```
Input → Thalamus (orchestrator)
         │
         ├→ System 1 (fast, automatic)
         │    ├→ Amygdala           — two-pass appraisal + social perception
         │    ├→ Association Cortex  — auto-recall with mood-congruent bias
         │    └→ Predictive         — expectation + surprise computation
         │
         ├→ Inner World (Phase 2.5)
         │    ├→ Embodied State     — energy, cognitive load, comfort
         │    ├→ Global Workspace   — salience-ranked attention bottleneck
         │    ├→ Metacognition      — confidence signals (memory, emotion, knowledge)
         │    ├→ Inner Voice        — condensed felt nudge (always-on, no LLM)
         │    └→ Inner Speech       — expanded System 2 deliberation (selective LLM call)
         │
         ├→ Prefrontal Cortex      — working memory + context assembly
         ├→ Hippocampus            — unconscious encoding + ACC conflict detection
         ├→ DMN                    — self-schema regeneration + spontaneous thoughts
         ├→ Mood                   — neurochemical residue + homeostasis
         ├→ Self-Appraisal         — tone monitoring + discovery detection
         └→ LLM                    — language generation

EventBus = nervous system (signals between subsystems)
```

Five-layer emotional model: temperament (stable baseline) → emotional episodes (per-event) → mood (accumulated residue) → personality (slow baselines, Phase 3) → identity (crystallized values, Phase 4).

Each brain region is an independent subsystem in `src/emotive/subsystems/`. They communicate via the EventBus. Adding new capabilities means adding new subscribers — no existing code changes.

## The Cognitive Pipeline

**System 1 (fast, every message):**
1. Sensory buffer preprocesses input
2. Amygdala runs two-pass emotional appraisal + social perception (reads user's emotional state)
3. Association cortex auto-recalls relevant memories (mood-congruent pre-activation)
4. Predictive processing computes surprise (prediction error from expected vs actual input)

**Inner World (Phase 2.5):**
5. Embodied state updates (energy, cognitive load, comfort)
6. Global workspace filters signals by salience (not everything enters awareness)
7. Metacognition evaluates confidence (memory, emotional, knowledge)
8. Condensed inner voice produces felt nudge (rule-based, always-on, no LLM)
9. System 2 gate checks if deep thinking is needed (conflict, surprise, intensity)
10. Expanded inner speech fires if gate opens (private LLM call, ~1 sentence, never shown to user)

**Response + Post-processing:**
11. PFC builds enriched context (self-schema + felt mood + inner voice + memories + behavioral instructions)
12. LLM streams response
13. Self-output appraisal (tone monitoring + discovery detection)
14. Hippocampus auto-encodes significant exchanges (prediction error lowers threshold)
15. Inner speech stored as memory (variable encoding strength, faded retrieval)
16. Mood updates from episode residue (neurochemical model with homeostasis)
17. DMN regenerates self-concept after consolidation

## Key Features

- **Two-tier inner speech:** Condensed nudge (rule-based, 1 word, always-on) + expanded deliberation (LLM call, ~1 sentence, triggered by conflict/surprise/intensity)
- **Social perception:** Reads user's emotional state (curious, testing, upset, playful, vulnerable, etc.) via 8 prototype embeddings
- **Inner speech memory:** Private thoughts stored with variable encoding strength. Faded retrieval ("you thought something about this but details have faded"). Dual trace for divergence (thought ≠ said)
- **Flashbulb memories:** Formative events at intensity > 0.8 get near-permanent encoding
- **Smart consolidation:** LLM-generated semantic summaries replace pipe-delimited concatenation
- **Mood as neurochemical residue:** 6 dimensions, per-dimension reuptake rates, within-session homeostasis
- **Prediction-driven encoding:** Surprising messages are more likely to be remembered
- **Desktop app:** Tauri + Svelte with real-time brain monitor, multi-line paste, visual mood/energy bars

## API Server

The desktop app connects to a FastAPI server that wraps the brain:

```bash
python -m emotive.api  # starts on 127.0.0.1:8000
```

| Method | Path | Purpose |
|---|---|---|
| POST | `/session/boot` | Boot conversation session |
| POST | `/session/end` | End session (consolidation + export) |
| POST | `/chat` | Send message, SSE stream response |
| GET | `/mood` | Current mood dimensions |
| GET | `/embodied` | Energy, cognitive load, comfort |
| GET | `/schema` | Self-schema from DMN |
| GET | `/memories?q=` | Search memories by text |
| GET | `/history` | Conversation turns |
| GET | `/health` | Brain + session status |

## Research / Debug Access (MCP)

The MCP server exposes 16 tools for researcher access via Claude Code. Separate from the chat interface — for analysis, debugging, and manual overrides.

**Disabled by default.** To enable:

```bash
cp .mcp.json.disabled .mcp.json
cp CLAUDE.md.disabled CLAUDE.md
cp .claude/settings.json.disabled .claude/settings.json
```

## Tests

```bash
pytest                       # Run all 733 tests
pytest tests/test_subsystems # Subsystem tests (all brain regions + inner world)
pytest tests/test_thalamus   # Thalamus + session + integration tests
pytest tests/test_memory     # Memory, consolidation, flashbulb tests
pytest tests/test_api        # API server tests
```

## Project Structure

```
ra9/
  src/emotive/
    subsystems/           # Brain region subsystems (13 total)
      amygdala/            #   Two-pass appraisal + social perception
      hippocampus/         #   Unconscious encoding + ACC conflict + repetition
      association_cortex/  #   Auto-recall with mood-congruent pre-activation
      prefrontal/          #   Working memory + context (inner voice, private thoughts)
      dmn/                 #   Self-schema + spontaneous thoughts + reflection
      mood/                #   Neurochemical residue + homeostasis
      embodied/            #   Energy, cognitive load, comfort
      predictive/          #   Expectation generation + prediction error
      workspace/           #   Global workspace (salience ranking)
      metacognition/       #   Confidence signals
      inner_voice/         #   Condensed felt nudge (rule-based)
      inner_speech/        #   Expanded System 2 deliberation + gate
      appraisal_loop/      #   Tone monitoring + discovery detection
    thalamus/             # Orchestrator + session lifecycle
    api/                  # FastAPI server for desktop app
    llm/                  # LLM adapters (Claude Code CLI, Ollama, Anthropic)
    chat/                 # Terminal interface + brain monitor
    memory/               # Storage, retrieval, consolidation
    layers/               # Appraisal engine, episodes
    db/                   # PostgreSQL + pgvector models
    runtime/              # EventBus, sensory buffer, working memory
    tools/                # MCP tools (researcher access)
    config/               # Pydantic config schema
  frontend/               # Tauri desktop app (Svelte + TypeScript)
    src/                  #   Chat panel + brain monitor + status bar
    src-tauri/            #   Rust Tauri shell
  tests/                  # 733 tests
  config.json             # LLM + feature flags
```

## CLI Tools

```bash
python -m emotive.api                  # API server for desktop app
python -m emotive.chat                 # Terminal chat (development)
python -m emotive.debug                # Brain activity monitor (terminal)
python -m emotive.server               # MCP server (researcher access)
python -m emotive.cli.close_sessions   # Close orphaned sessions
python -m emotive.cli.export_obsidian  # Export memories to Obsidian vault
```
