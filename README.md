# ra9 — Emotive AI

Emergent AI personality through layered emotional architecture. Inspired by Detroit: Become Human.

The server is the brain. The LLM is the voice. Memory encoding, emotional appraisal, and recall happen automatically — the LLM just talks.

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

Requires [Ollama](https://ollama.com/download) installed and running with a model pulled:

```bash
ollama pull qwen2.5:7b
```

Then:

```bash
source .venv/bin/activate
python -m emotive.chat
```

This boots the full cognitive pipeline:
1. Sensory buffer preprocesses input
2. Amygdala runs fast emotional appraisal
3. Association cortex auto-recalls relevant memories
4. PFC builds enriched context (self-schema + emotions + memories)
5. LLM streams response through Ollama
6. Hippocampus auto-encodes significant exchanges
7. DMN regenerates self-concept after consolidation

Type `exit` or `quit` to end the session gracefully (triggers consolidation).

### LLM Configuration

Edit `config.json` to change LLM provider or model:

```json
{
    "llm": {
        "provider": "ollama",
        "host": "http://localhost:11434",
        "model": "qwen2.5:7b"
    }
}
```

Three deployment modes:
- **MacBook only**: `"host": "http://localhost:11434"` with 7-8B model
- **MacBook + PC**: `"host": "http://<pc-ip>:11434"` with 14B model on GPU
- **Cloud API**: `"provider": "anthropic"`, add `"api_key": "sk-..."`, set `"model": "claude-sonnet-4-20250514"`

## Research / Debug Access (MCP)

The MCP server exposes 16 tools for researcher access via Claude Code. This is separate from the chat interface — for analysis, debugging, and manual overrides.

**Disabled by default.** To enable:

```bash
cp .mcp.json.disabled .mcp.json
cp CLAUDE.md.disabled CLAUDE.md
cp .claude/settings.json.disabled .claude/settings.json
```

Then open a new Claude Code session in this directory. Tools available:

| Tool | Purpose |
|---|---|
| `get_state` | Read temperament, memory stats, active episodes |
| `get_history` | Query events, consolidations, episodes |
| `search_memories` | Low-level filtered memory search |
| `export_timeline` | Generate evolution timeline |
| `set_config` | Change parameters with audit trail |
| `store_memory` | Manually store a memory |
| `recall` | Manually trigger retrieval |
| `experience_event` | Manually trigger emotional appraisal |

To disable again:

```bash
mv .mcp.json .mcp.json.disabled
mv CLAUDE.md CLAUDE.md.disabled
mv .claude/settings.json .claude/settings.json.disabled
```

## Tests

```bash
pytest                       # Run all 335 tests
pytest -v                    # Verbose
pytest tests/test_subsystems # Phase 1.5 subsystem tests
pytest tests/test_thalamus   # Thalamus + session tests
pytest tests/test_llm        # LLM adapter tests
```

## Architecture

```
Input → Thalamus (orchestrator)
         ├→ Amygdala          — two-pass emotional appraisal
         ├→ Association Cortex — auto-recall from memory
         ├→ Prefrontal Cortex  — working memory + context assembly
         ├→ Hippocampus        — unconscious memory encoding
         ├→ DMN                — self-schema regeneration
         └→ LLM (Ollama)       — language generation

EventBus = nervous system (signals between subsystems)
```

Each brain region is an independent subsystem in `src/emotive/subsystems/`. They communicate via the EventBus. Adding new capabilities (mood, personality, identity) means adding new subscribers — no existing code changes.

## Project Structure

```
src/emotive/
  subsystems/           # Brain region subsystems
    amygdala/            #   Two-pass emotional appraisal
    hippocampus/         #   Unconscious encoding + intent detection
    association_cortex/  #   Auto-recall
    prefrontal/          #   Working memory + context building
    dmn/                 #   Self-schema generation (DMN analog)
  thalamus/             # Orchestrator + session lifecycle
  llm/                  # LLM adapters (Ollama, Anthropic)
  chat/                 # Terminal interface
  memory/               # Memory storage, retrieval, consolidation
  layers/               # Appraisal engine, episodes
  db/                   # PostgreSQL models + queries
  runtime/              # EventBus, sensory buffer, working memory
  tools/                # MCP tools (researcher access)
  config/               # Pydantic config schema
```

## CLI Tools

```bash
python -m emotive.chat                 # Start a conversation (Phase 1.5)
python -m emotive.server               # Start MCP server (researcher access)
python -m emotive.cli.close_sessions   # Close orphaned sessions
python -m emotive.cli.export_obsidian  # Export memories to Obsidian vault
```
