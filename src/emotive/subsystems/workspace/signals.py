"""Workspace signal dataclasses for the global workspace broadcast.

Signals compete for attention in the workspace. Only the most salient
signals enter the LLM context (broadcast); the rest stay unconscious.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkspaceSignal:
    """A single signal competing for workspace attention."""

    source: str  # "amygdala", "memory", "conflict", "prediction", "mood"
    content: Any  # the data (memory dict, emotion string, etc.)
    salience: float  # 0-1, how urgently it demands attention
    signal_type: str  # "emotion", "memory", "conflict", "prediction", "mood_shift"


@dataclass
class WorkspaceOutput:
    """Result of workspace competition: what gets broadcast vs suppressed."""

    broadcast: list[WorkspaceSignal] = field(default_factory=list)
    unconscious: list[WorkspaceSignal] = field(default_factory=list)
    broadcast_memories: list[dict] = field(default_factory=list)
