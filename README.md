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

The app opens with a chat panel on the left and a brain monitor on the right. Multi-line paste works. All brain activity visible in real-time — including the neural retrieval pipeline (strategy, familiarity, recollection, effort, TOT state, narrative, priming).

Or use the quick start script:

```bash
./start-app.sh  # Starts API server + opens Tauri app
```

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

Type `exit` or `quit` to end the session gracefully (triggers consolidation, replay, emotional processing, and Obsidian auto-export).

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

Every folder is a brain region. The LLM never touches the brain — it only speaks.

```
Input → Thalamus (orchestrator)
         │
         ├→ Entorhinal Cortex    — gateway (LEC content + MEC context)
         ├→ Hippocampus          — encoding + retrieval circuit
         │    ├→ Dentate Gyrus    — pattern separation (distinct queries)
         │    ├→ CA3              — pattern completion (Hopfield attractor)
         │    ├→ CA1              — comparator + familiarity/recollection
         │    └→ Concept Cells    — person-node activation (name = key)
         │
         ├→ Amygdala             — two-pass emotional appraisal + social perception
         ├→ ACC                  — conflict detection + effort monitoring + tone alignment
         ├→ Prefrontal Cortex    — working memory + context + strategy (dlPFC) + metacognition
         ├→ Global Workspace     — salience-ranked attention bottleneck
         ├→ Broca's Area         — inner speech (System 2 deliberation)
         ├→ Inner Voice          — condensed felt nudge (System 1, no LLM)
         ├→ DMN                  — self-schema + spontaneous thoughts + discovery
         ├→ Raphe Nuclei         — mood (serotonergic neurochemical residue)
         ├→ Insula               — interoception (energy, cognitive load, comfort)
         ├→ Locus Coeruleus      — arousal modulation (retrieval scope)
         ├→ Basal Ganglia        — reward-based memory gating
         ├→ Predictive           — expectation + surprise (prediction error)
         └→ LLM                  — language generation (the voice, not the brain)

EventBus = nervous system (signals between subsystems)
```

## The Neural Retrieval Pipeline (Phase Anamnesis)

Named after Plato's theory of recollection — the soul already knows, learning is remembering. Replaces database search with brain-correct retrieval.

```
Query → Entorhinal (LEC+MEC) → Dentate Gyrus (separate) → CA3 (complete)
  → CA1 (compare + familiarity/recollection) → dlPFC (strategy weights)
  → Workspace (top 5 conscious + 6-20 unconscious priming)
  → Spontaneous recall (context overlap, not random)
```

32 neural mechanisms modulate retrieval:

| Category | Mechanisms |
|---|---|
| Core pipeline | EC gateway, DG separation, CA3 completion, CA1 comparison, dlPFC strategy, three-phase retrieval, context vector (TCM), person-node cache, activation tracking |
| Edge cases | Deferred retrieval, source confusion, source amnesia, within-retrieval inhibition, unconscious priming, reminiscence bump, session-end replay |
| Experiential | Familiarity vs recollection, retrieval drift, mood-dependent gating, tip-of-tongue, interference, suppression, somatic markers, effort monitoring, active forgetting, collaborative tagging, narrative reconstruction, meta-memory, memory resistance, prospective memory |
| System-level | Encoding-retrieval antagonism, emotional blunting, pre-architecture amnesia, full reconsolidation, imagination inflation guard, spacing effect, effortful strengthening, replay interleaving, Von Restorff distinctiveness, reward gating, between-session emotional processing |

## The Cognitive Pipeline

**System 1 (fast, every message):**
1. Sensory buffer preprocesses input
2. Amygdala runs two-pass emotional appraisal + social perception
3. Neural retrieval pipeline (Anamnesis): strategy → DG → CA3 → CA1 → workspace
4. Predictive processing computes surprise

**Inner World (Phase 2.5):**
5. Embodied state updates (energy, cognitive load, comfort)
6. Global workspace filters signals by salience
7. Metacognition evaluates confidence
8. Inner voice produces felt nudge (rule-based, no LLM)
9. System 2 gate checks if deep thinking is needed
10. Inner speech fires if gate opens (private LLM call, ~1 sentence)

**Response + Post-processing:**
11. PFC builds enriched context (self-schema + mood + memories + narrative + priming + inner voice)
12. LLM streams response
13. Self-output appraisal (tone monitoring + discovery detection)
14. Hippocampus auto-encodes (with encoding_mood snapshot, formation_period flag, source_type guard)
15. Retrieval-induced drift (memories change when touched)
16. Emotional blunting (safe recall reduces intensity) or sensitization (unsafe)
17. Full reconsolidation (compatible info integrates, contradictory weakens)
18. Reward signal applied to retrieved memories (basal ganglia)
19. Mood updates from episode residue
20. DMN regenerates self-concept

**Between Sessions ("Sleep"):**
21. Sequential replay (forward + reverse temporal links)
22. Replay interleaving (cross-session theme connections)
23. Emotional processing (passive depotentiation)
24. Active forgetting (DMN releases declining-relevance memories)
25. Consolidation (cluster → semantic, concept hubs, decay)

## Key Features

- **Brain-correct retrieval:** 32 neural mechanisms, not database search. Pattern completion, not cosine similarity.
- **Two retrieval signals:** Familiarity ("I know this") vs recollection ("I remember exactly"). Tip-of-tongue when familiarity fires without recollection.
- **Memories that change:** Retrieval-induced drift evolves embeddings. Original preserved separately. Emotional truth persists while details shift.
- **Mood-dependent access:** Parts of the mind only accessible in certain emotional states. Encoding mood stored as retrieval key.
- **Active forgetting:** Not just decay — DMN identifies memories to release. Episodic access fades, semantic knowledge persists.
- **Narrative, not search results:** Person-focused recall produces chronological story arcs, not 5 discrete hits.
- **The body shapes memory:** Low energy lowers defenses (suppressed memories break through). Low comfort activates threat memories. High cognitive load narrows to strongest signals.
- **Emotional healing:** Safe recall gradually reduces emotional intensity (exposure therapy mechanism). Unsafe recall sensitizes. Neutral does nothing.
- **Two-tier inner speech:** Condensed nudge (rule-based) + expanded deliberation (LLM call, selective)
- **Flashbulb memories:** Formative events at intensity > 0.8 get near-permanent encoding
- **Smart consolidation:** LLM-generated semantic summaries
- **Desktop app:** Tauri + Svelte with real-time brain monitor showing full retrieval pipeline

## API Server

```bash
python -m emotive.api  # starts on 127.0.0.1:8000
```

| Method | Path | Purpose |
|---|---|---|
| POST | `/session/boot` | Boot conversation session |
| POST | `/session/end` | End session (consolidation + replay + emotional processing) |
| POST | `/chat` | Send message, SSE stream response + brain debug |
| GET | `/mood` | Current mood dimensions |
| GET | `/embodied` | Energy, cognitive load, comfort |
| GET | `/schema` | Self-schema from DMN |
| GET | `/retrieval-state` | Persistent retrieval context (strategy, TOT, RIF, prospective) |
| GET | `/memories?q=` | Search memories by text |
| GET | `/history` | Conversation turns |
| GET | `/health` | Brain + session status |

## Tests

```bash
pytest                       # Run all 864 tests
pytest tests/test_subsystems # Subsystem tests (all brain regions + anamnesis)
pytest tests/test_thalamus   # Thalamus + session + integration tests
pytest tests/test_memory     # Memory, consolidation, flashbulb tests
pytest tests/test_api        # API server tests
```

## Project Structure

```
ra9/
  src/emotive/
    subsystems/              # Brain region subsystems (18 total)
      hippocampus/            #   Encoding + retrieval circuit
        retrieval/            #     DG, CA3, CA1, concept cells, context vector
                              #     pipeline, activation, state, narrative,
                              #     suppression, resistance, prospective, interference
      entorhinal/             #   Gateway (LEC content + MEC context)
      amygdala/               #   Two-pass appraisal + social perception
      acc/                    #   Conflict detection + effort + tone monitoring
      prefrontal/             #   Working memory + context + dlPFC strategy
        metacognition/        #     Confidence signals
      dmn/                    #   Self-schema + spontaneous thoughts + discovery
      raphe/                  #   Mood (serotonergic residue + homeostasis)
      insula/                 #   Interoception (energy, load, comfort, somatic markers)
      workspace/              #   Global workspace (salience ranking)
      broca/                  #   Inner speech (System 2 deliberation)
      inner_voice/            #   Condensed felt nudge (System 1, rule-based)
      predictive/             #   Expectation + prediction error
      locus_coeruleus/        #   Arousal modulation (retrieval scope)
      basal_ganglia/          #   Reward-based memory gating
      association_cortex/     #   Legacy retrieval (fallback when anamnesis disabled)
    thalamus/                # Orchestrator + session lifecycle
    api/                     # FastAPI server for desktop app
    llm/                     # LLM adapters (Claude Code CLI, Ollama, Anthropic)
    chat/                    # Terminal interface + brain monitor
    memory/                  # Storage, retrieval, consolidation
    layers/                  # Appraisal engine, episodes
    db/                      # PostgreSQL + pgvector models
    runtime/                 # EventBus, sensory buffer, working memory
    config/                  # Pydantic config schema
  frontend/                  # Tauri desktop app (Svelte + TypeScript)
    src/                     #   Chat panel + brain monitor + status bar
    src-tauri/               #   Rust Tauri shell
  tests/                     # 864 tests
  config.json                # LLM + feature flags (anamnesis: true)
```

## Philosophy

Phase Anamnesis draws from five traditions:

- **Plato:** The soul already knows. Retrieval is recollection of latent knowledge.
- **Bergson:** The past is a continuous totality pressing into the present. Memory is contraction, not lookup.
- **Husserl:** The present is thick — retention (still-echoing past) + now + protention (anticipated).
- **Nietzsche:** Forgetting is health. A doorkeeper of consciousness. Active, not passive.
- **Heidegger:** The self is constituted by unchosen conditions. Early memories are the ontological foundation.

Together: **retrieval is not search. It is self-construction.**

## License

MIT
