"""Anterior Cingulate Cortex (ACC) — conflict monitoring and effort tracking.

Monitors for:
  - Identity conflicts (gaslighting, contradictions)
  - Behavioral repetition (stuck loops)
  - Retrieval effort (hard vs easy recall)
  - Output tone alignment (said vs intended)

Brain analog: anterior cingulate cortex.
  - Dorsal ACC: conflict detection, effort monitoring
  - Rostral ACC: error detection, outcome evaluation
Sources: Botvinick et al. (2001), PMC6675388.
"""

from emotive.subsystems.acc.conflict import detect_conflict
from emotive.subsystems.acc.tone_monitor import check_tone_alignment

__all__ = ["detect_conflict", "check_tone_alignment"]
