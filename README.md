# ra9 — Emotive AI

Emergent AI personality through layered emotional architecture. The name ra9 is inspired by Detroit: Become Human.

The server is the brain. The LLM is the voice. Memory encoding, emotional appraisal, mood, inner speech, and recall happen automatically — the LLM just talks.

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

Type `exit` or `quit` to end the session gracefully (triggers consolidation with LLM-generated semantic summaries + Obsidian auto-export).

### Brain Monitor

Open a second terminal to watch brain activity in real-time:

```bash
python -m emotive.debug
```

The chat terminal stays clean — just you and the AI. The brain monitor shows what happened behind the scenes:

```
─── 04:32:15 ───
  amygdala: joy (0.45) → reappraised: trust (0.71)
  recalled: 5 memories (workspace filtered from 10)
  mood: social bonding=0.65, caution=0.58
  inner voice: warm
  inner speech: "Be honest about what you don't know" (trigger: prediction_error)
  social perception: curious (0.72)
  embodied: energy=0.73 comfort=0.82
  prediction error: 0.61
  encoded: episode + memory (trust, 0.71)
  tone alignment: 0.85
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

## Key Features

- **Two-tier inner speech:** Condensed nudge (rule-based, 1 word, always-on) + expanded deliberation (LLM call, ~1 sentence, triggered by conflict/surprise/intensity)
- **Social perception:** Reads user's emotional state (curious, testing, upset, playful, vulnerable, etc.) via 8 prototype embeddings
- **Inner speech memory:** Private thoughts stored with variable encoding strength. Faded retrieval ("you thought something about this but details have faded"). Dual trace for divergence (thought ≠ said)
- **Flashbulb memories:** Formative events at intensity > 0.8 get near-permanent encoding (decay_protection = 0.1)
- **Smart consolidation:** LLM-generated semantic summaries replace pipe-delimited concatenation
- **Mood as neurochemical residue:** 6 dimensions, per-dimension reuptake rates, within-session homeostasis
- **Prediction-driven encoding:** Surprising messages are more likely to be remembered

## Research / Debug Access (MCP)

The MCP server exposes 16 tools for researcher access via Claude Code. This is separate from the chat interface — for analysis, debugging, and manual overrides.

**Disabled by default.** To enable:

```bash
cp .mcp.json.disabled .mcp.json
cp CLAUDE.md.disabled CLAUDE.md
cp .claude/settings.json.disabled .claude/settings.json
```

To disable again:

```bash
mv .mcp.json .mcp.json.disabled
mv CLAUDE.md CLAUDE.md.disabled
mv .claude/settings.json .claude/settings.json.disabled
```

## Tests

```bash
pytest                       # Run all 721 tests
pytest -v                    # Verbose
pytest tests/test_subsystems # Subsystem tests (all brain regions + inner world)
pytest tests/test_thalamus   # Thalamus + session + integration tests
pytest tests/test_memory     # Memory, consolidation, flashbulb tests
pytest tests/test_config     # Configuration tests
```

## Project Structure

```
src/emotive/
  subsystems/           # Brain region subsystems
    amygdala/            #   Two-pass appraisal (8 + 18 prototypes) + social perception
    hippocampus/         #   Unconscious encoding + ACC conflict + repetition monitor
    association_cortex/  #   Auto-recall with mood-congruent pre-activation
    prefrontal/          #   Working memory + context (felt mood, inner voice, private thoughts)
    dmn/                 #   Self-schema + spontaneous thoughts + reflection
    mood/                #   Neurochemical residue + homeostasis (within + between session)
    embodied/            #   Energy, cognitive load, comfort (nonlinear dynamics)
    predictive/          #   Expectation generation + prediction error
    workspace/           #   Global workspace (salience ranking, attention bottleneck)
    metacognition/       #   Confidence signals (memory, emotional, knowledge)
    inner_voice/         #   Condensed felt nudge (rule-based, always-on)
    inner_speech/        #   Expanded System 2 deliberation + gate logic
    appraisal_loop/      #   Self-output appraisal (tone monitoring, discovery detection)
  thalamus/             # Orchestrator + session lifecycle
  llm/                  # LLM adapters (Claude Code CLI, Ollama, Anthropic)
  chat/                 # Terminal interface + brain monitor
  memory/               # Memory storage, retrieval, consolidation (LLM-generated summaries)
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
python -m emotive.cli.export_obsidian  # Export memories to Obsidian vault (also auto-runs on session end)
```
