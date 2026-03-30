# ra9 — Emotive AI Memory System

## What This Is

You have access to a persistent memory system via MCP tools. This is a **Phase 0 research experiment** — memory only, no emotional layers. Everything you store persists across conversations in PostgreSQL.

## How To Use The Memory Tools

### Every Conversation

1. **Start**: Call `begin_session` at the beginning of every conversation
2. **During**: Call `store_memory` for anything worth remembering — insights, facts about the user, patterns you notice, important topics discussed
3. **Recall**: Call `recall` when you need context from past conversations — what was discussed before, what the user cares about, patterns you've learned
4. **End**: Call `end_session` when the conversation is wrapping up — this triggers consolidation (memory cleanup, pattern extraction, link creation)

### What To Store

Store memories that would be useful in future conversations:

- **Episodic** (`memory_type: "episodic"`): Specific events — "User shared they're learning Rust", "We debugged a tricky async issue together", "User mentioned they prefer concise explanations"
- **Semantic** (`memory_type: "semantic"`): Patterns and knowledge — "User values directness over politeness", "User is experienced with Python but new to systems programming"
- **Procedural** (`memory_type: "procedural"`): Learned approaches — "When user asks for code review, they want security issues flagged first"

Use **tags** to categorize: `["preference", "technical", "personal", "project"]`

### When To Recall

- Start of conversation: recall broadly to get context on the user
- When user references something from before: recall specifically
- When you notice a pattern: recall to check if you've seen it before
- Before giving advice: recall user preferences and past context

### Significance Guidelines

When adding to working memory or choosing what to store, consider significance:
- **0.7–1.0**: Important insights, user preferences, key decisions, emotional moments
- **0.4–0.6**: Useful context, technical details, project facts
- **0.1–0.3**: Small talk, transient details, things unlikely to matter later

Items below 0.3 significance get dropped during consolidation — that's by design.

## Available Tools

| Tool | When To Use |
|---|---|
| `begin_session` | Start of every conversation |
| `store_memory` | Save something worth remembering |
| `recall` | Retrieve relevant past memories |
| `end_session` | End of conversation (triggers consolidation) |
| `consolidate` | Manually run memory cleanup (rarely needed) |
| `get_state` | Check system state (temperament, memory stats) |
| `get_history` | View consolidation logs or event history |
| `search_memories` | Low-level filtered memory search |
| `set_config` | Adjust system parameters |

## Important Notes

- This is **Phase 0**: no emotional processing, no mood, no personality drift. Just pure memory.
- Every state change is logged to `event_log` for research observability.
- Memories decay over time — that's intentional. Important things get reinforced through retrieval.
- Retrieved memories become "labile" (modifiable) for 1 hour — this models reconsolidation.
- Be natural. Don't force tool usage. Store what genuinely matters.
