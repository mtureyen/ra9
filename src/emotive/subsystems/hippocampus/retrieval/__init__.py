"""Hippocampal retrieval circuit — DG → CA3 → CA1 → output.

The core neural retrieval pipeline. Pattern separation (DG) ensures
distinct queries don't blur. Pattern completion (CA3) converges on
stored patterns from partial cues. Context comparison (CA1) verifies
match and produces familiarity/recollection dual signal.

Brain analog: hippocampal formation retrieval pathway.
Sources: PMC4792674, Nature s41467-017-02752-1.
"""

from emotive.subsystems.hippocampus.retrieval.pipeline import (
    RetrievalResult,
    run_retrieval,
)

__all__ = ["run_retrieval", "RetrievalResult"]
