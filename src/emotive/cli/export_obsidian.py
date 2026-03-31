"""Export memories and links to Obsidian vault as markdown files."""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select

from emotive.db.engine import SessionFactory
from emotive.db.models.conversation import Conversation
from emotive.db.models.memory import Memory, MemoryLink


def sanitize_filename(text: str, max_len: int = 60) -> str:
    """Create a safe filename from memory content."""
    clean = re.sub(r"[^\w\s-]", "", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:max_len] if clean else "untitled"


def memory_to_markdown(
    mem: Memory,
    links_out: list[tuple[str, str, float]],
    links_in: list[tuple[str, str, float]],
) -> str:
    """Convert a memory to an Obsidian markdown note."""
    significance = ""
    if mem.metadata_ and "significance" in mem.metadata_:
        significance = f"\nsignificance: {mem.metadata_['significance']}"

    # YAML list format for Obsidian tag support
    if mem.tags:
        tags_yaml = "\ntags:\n" + "\n".join(f"  - {t}" for t in mem.tags)
    else:
        tags_yaml = "\ntags: []"

    # Hashtags in body for clickability
    hashtags = " ".join(f"#{t}" for t in mem.tags) if mem.tags else ""

    md = f"""---
id: {mem.id}
type: {mem.memory_type}{tags_yaml}
confidence: {mem.confidence}
detail_retention: {mem.detail_retention}
decay_rate: {mem.decay_rate}
is_formative: {mem.is_formative}
retrieval_count: {mem.retrieval_count}{significance}
created_at: {mem.created_at}
---

# {mem.content[:80]}

{hashtags}

{mem.content}
"""

    if mem.metadata_ and len(mem.metadata_) > 0:
        # Filter out significance since it's in frontmatter
        extra = {k: v for k, v in mem.metadata_.items() if k != "significance"}
        if extra:
            md += "\n## Metadata\n\n"
            for k, v in extra.items():
                md += f"- **{k}**: {v}\n"

    if links_out or links_in:
        md += "\n## Links\n\n"
        for target_name, link_type, strength in links_out:
            md += f"- → [[{target_name}]] ({link_type}, strength: {strength})\n"
        for source_name, link_type, strength in links_in:
            md += f"- ← [[{source_name}]] ({link_type}, strength: {strength})\n"

    return md


def generate_index(
    memories: list[Memory],
    conversations: list[Conversation],
    link_count: int,
) -> str:
    """Generate the index note with stats and links to all memories."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    type_counts: dict[str, int] = {}
    for m in memories:
        type_counts[m.memory_type] = type_counts.get(m.memory_type, 0) + 1

    md = f"""---
generated: {now}
---

# Ryo's Memory Graph

**Exported**: {now}
**Total memories**: {len(memories)}
**Links**: {link_count}
**Conversations**: {len(conversations)}

## By Type

"""
    for t, c in sorted(type_counts.items()):
        md += f"- **{t}**: {c}\n"

    for mem_type in ["semantic", "episodic", "procedural"]:
        typed = [m for m in memories if m.memory_type == mem_type]
        if not typed:
            continue
        md += f"\n## {mem_type.title()} Memories\n\n"
        for m in typed:
            name = sanitize_filename(m.content)
            md += f"- [[{name}]]\n"

    return md


def export(vault_path: str | None = None) -> None:
    """Export all memories to Obsidian vault."""
    if vault_path is None:
        vault_path = os.environ.get("OBSIDIAN_VAULT_PATH")
    if not vault_path:
        print("Error: OBSIDIAN_VAULT_PATH not set in .env")
        return

    vault = Path(vault_path)
    vault.mkdir(parents=True, exist_ok=True)

    session = SessionFactory()
    try:
        # Load all active memories
        memories = list(
            session.execute(
                select(Memory)
                .where(Memory.is_archived.is_(False))
                .order_by(Memory.created_at)
            ).scalars().all()
        )

        # Load all links
        links = list(session.execute(select(MemoryLink)).scalars().all())

        # Load conversations
        conversations = list(
            session.execute(
                select(Conversation).order_by(Conversation.started_at)
            ).scalars().all()
        )

        # Build name lookup: memory_id -> filename
        names: dict[str, str] = {}
        used_names: set[str] = set()
        for m in memories:
            name = sanitize_filename(m.content)
            # Deduplicate
            if name in used_names:
                name = f"{name} {str(m.id)[:8]}"
            used_names.add(name)
            names[str(m.id)] = name

        # Build link maps
        links_out: dict[str, list[tuple[str, str, float]]] = {}
        links_in: dict[str, list[tuple[str, str, float]]] = {}
        for link in links:
            src = str(link.source_memory_id)
            tgt = str(link.target_memory_id)
            if src in names and tgt in names:
                links_out.setdefault(src, []).append(
                    (names[tgt], link.link_type, link.strength)
                )
                links_in.setdefault(tgt, []).append(
                    (names[src], link.link_type, link.strength)
                )

        # Clean export folder (memories subfolder only)
        memories_dir = vault / "memories"
        if memories_dir.exists():
            shutil.rmtree(memories_dir)
        memories_dir.mkdir()

        # Write memory files
        for m in memories:
            mid = str(m.id)
            name = names[mid]
            md = memory_to_markdown(
                m,
                links_out.get(mid, []),
                links_in.get(mid, []),
            )
            (memories_dir / f"{name}.md").write_text(md)

        # Write index
        index_md = generate_index(memories, conversations, len(links))
        (vault / "INDEX.md").write_text(index_md)

        print(f"Exported {len(memories)} memories, {len(links)} links to {vault}")
        print(f"  Semantic: {sum(1 for m in memories if m.memory_type == 'semantic')}")
        print(f"  Episodic: {sum(1 for m in memories if m.memory_type == 'episodic')}")
        print(f"  Procedural: {sum(1 for m in memories if m.memory_type == 'procedural')}")

    finally:
        session.close()


if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    export()
