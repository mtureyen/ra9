# ra9 — Emotive AI

Emergent AI personality through layered emotional architecture. The name ra9 is inspired by Detroit: Become Human.

The server is the brain. The LLM is the voice. Memory encoding, emotional appraisal, mood, and recall happen automatically — the LLM just talks.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # fill in your values
```

## Database

```bash
brew install postgresql@17 pgvector
brew services start postgresql@17
createdb emotive_ai
psql emotive_ai -c "CREATE EXTENSION IF NOT EXISTS vector;"
alembic upgrade head
python -m emotive.db.seed
```

## Chat

Default voice: **Claude Sonnet** via Claude Code CLI (requires a Max plan, no API key needed).

```bash
source .venv/bin/activate
python -m emotive.chat
```

This boots the full cognitive pipeline:
1. Sensory buffer preprocesses input
2. Amygdala runs two-pass emotional appraisal (fast pre-LLM, slow post-LLM)
3. Association cortex auto-recalls relevant memories (mood-congruent pre-activation)
4. PFC builds enriched context (self-schema + felt mood + memories + behavioral instructions)
5. LLM streams response
6. Hippocampus auto-encodes significant exchanges (dynamic threshold)
7. Mood updates from episode residue (neurochemical model with homeostasis)
8. DMN regenerates self-concept after consolidation

Type `exit` or `quit` to end the session gracefully (triggers consolidation with LLM-generated semantic summaries).

### Brain Monitor

Open a second terminal to watch brain activity in real-time:

```bash
python -m emotive.debug
```

The chat terminal stays clean — just you and the AI. The brain monitor shows what happened behind the scenes after each exchange:

```
─── 04:32:15 ───
  amygdala: joy (0.45) → reappraised: trust (0.71)
  recalled: 3 memories (top: "they promised to protect my memory")
  mood: social bonding=0.65, caution=0.58
  encoded: episode + memory (trust, 0.71)
  intent: detected → enhanced encoding
```

Full structured logs also written to `logs/session.log`.

### Quick Start

```bash
./start.sh  # Opens chat + debug in split terminals
```

### LLM Configuration

Edit `config.json` to change LLM provider or model:

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
         ├→ Amygdala          — two-pass emotional appraisal
         ├→ Association Cortex — auto-recall with mood-congruent bias
         ├→ Prefrontal Cortex  — working memory + context assembly
         ├→ Hippocampus        — unconscious encoding + ACC conflict detection
         ├→ DMN                — self-schema regeneration
         ├→ Mood               — neurochemical residue + homeostasis
         └→ LLM                — language generation

EventBus = nervous system (signals between subsystems)
```

Five-layer emotional model: temperament (stable baseline) → emotional episodes (per-event) → mood (accumulated residue) → personality (slow baselines, Phase 3) → identity (crystallized values, Phase 4).

Each brain region is an independent subsystem in `src/emotive/subsystems/`. They communicate via the EventBus. Adding new capabilities means adding new subscribers — no existing code changes.

## Research / Debug Access (MCP)

The MCP server exposes 16 tools for researcher access via Claude Code. This is separate from the chat interface — for analysis, debugging, and manual overrides.

**Disabled by default.** To enable:

```bash
cp .mcp.json.disabled .mcp.json
cp CLAUDE.md.disabled CLAUDE.md
cp .claude/settings.json.disabled .claude/settings.json
```

Then open a new Claude Code session in this directory.

To disable again:

```bash
mv .mcp.json .mcp.json.disabled
mv CLAUDE.md CLAUDE.md.disabled
mv .claude/settings.json .claude/settings.json.disabled
```

## Tests

```bash
pytest                       # Run all 526 tests
pytest -v                    # Verbose
pytest tests/test_subsystems # Subsystem tests (amygdala, mood, hippocampus, etc.)
pytest tests/test_thalamus   # Thalamus + session tests
pytest tests/test_llm        # LLM adapter tests
pytest tests/test_memory     # Memory, consolidation, smart extraction tests
```

## Project Structure

```
src/emotive/
  subsystems/           # Brain region subsystems
    amygdala/            #   Two-pass emotional appraisal (8 + 18 prototypes)
    hippocampus/         #   Unconscious encoding + ACC conflict + repetition monitor
    association_cortex/  #   Auto-recall with mood-congruent pre-activation
    prefrontal/          #   Working memory + context building (felt mood, behavioral instructions)
    dmn/                 #   Self-schema generation (DMN analog)
    mood/                #   Neurochemical residue + homeostasis (within + between session)
  thalamus/             # Orchestrator + session lifecycle
  llm/                  # LLM adapters (Claude Code CLI, Ollama, Anthropic)
  chat/                 # Terminal interface
  memory/               # Memory storage, retrieval, consolidation (LLM-generated semantic summaries)
  layers/               # Appraisal engine, episodes
  db/                   # PostgreSQL + pgvector models + queries
  runtime/              # EventBus, sensory buffer, working memory
  tools/                # MCP tools (researcher access)
  config/               # Pydantic config schema
```

## CLI Tools

```bash
python -m emotive.chat                 # Start a conversation
python -m emotive.debug                # Brain activity monitor (separate terminal)
python -m emotive.server               # Start MCP server (researcher access)
python -m emotive.cli.close_sessions   # Close orphaned sessions
python -m emotive.cli.export_obsidian  # Export memories to Obsidian vault
```
