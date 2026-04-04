"""Backwards-compatible re-export.

PatternSeparator lives in hippocampus/retrieval/dentate_gyrus.py
(anatomically correct — DG is part of hippocampus, not entorhinal).
This module re-exports for any code that imports from here.
"""

from emotive.subsystems.hippocampus.retrieval.dentate_gyrus import PatternSeparator

__all__ = ["PatternSeparator"]
