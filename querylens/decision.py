"""
decision.py — Layer 2 for QueryLens.

Phase 0: the contract only. No decision logic yet — just the shape of
the data Layer 2 promises to produce, plus a stub class whose decide()
returns a "refuse everything" default.

Subsequent phases (see docs/layer2_plan.md §6) fill this in:
  Phase 1: confidence classifier
  Phase 2: LLM gate (Streamlit integration)
  Phase 3: fallback
  Phase 4: input validation
  Phase 5: safety patterns
  Phase 6: telemetry
  Phase 7: diversity
  Phase 8: score-gap demotion
  Phase 9: tests covering every branch
  Phase 10: calibration
  Phase 11: centralised prompts
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── Public output contract ───────────────────────────────────────────────────
#
# Every field is set explicitly at construction. Fields with defaults are
# the ones added in later phases — keeping them defaulted means Phase 0
# code can build a SearchDecision without specifying them.


@dataclass
class SearchDecision:
    """The single object Layer 2 hands to Layer 3.

    Layer 3's contract: read these fields, render UI, call the LLM iff
    `should_generate_answer` is True. Layer 3 does NOT look at raw
    scores, the corpus, or make any policy decisions itself.
    """

    # — confidence band (filled in Phase 1) —
    confidence: str                          # "high" | "medium" | "low" | "none"

    # — two-axis decision (Phase 2 wires them; Phase 1 sets them based
    #   on confidence). Invariant we maintain: should_generate_answer
    #   implies should_show_results. The reverse is not required.
    should_show_results: bool                # softer: render top passages
    should_generate_answer: bool             # stricter: actually call the LLM

    # — what to display / pass to LLM —
    final_results: List[Tuple[int, float]]   # (passage_idx, score) pairs

    # — user-facing notes (Phase 1+ fills) —
    warning: Optional[str]                   # banner text, or None

    # — fallback bookkeeping (Phase 3 sets) —
    fallback_triggered: bool                 # did we swap to dense-only?
    fallback_method: Optional[str]           # which method took over

    # — diagnostics (kept for telemetry and debugging) —
    top_score: float                         # primary path's #1 score
    score_gap: float                         # #1 minus #2 score

    # — optional flags added by later phases —
    ambiguous: bool = False                  # set by Phase 8 (score-gap demote)
    rejected: bool = False                   # set by Phase 4/5 (validation/safety)
    rejected_reason: Optional[str] = None    # "validation" | "safety" | None


# ── The decision layer ───────────────────────────────────────────────────────


class DecisionLayer:
    """Layer 2 of QueryLens. See module docstring for the phase plan.

    Architectural rules (do not violate as we add phases):
      1. NO ML model dependencies. Pure logic on numpy + dicts.
      2. Immutable inputs: never mutate upstream score lists.
      3. The output is a SearchDecision — never raw scores, never side effects
         beyond the optional telemetry write that Phase 6 adds.
    """

    def __init__(self) -> None:
        """Phase 0: no parameters yet. Future phases add config knobs here
        (diversifier choice, telemetry path, etc.)."""
        pass

    def decide(self, query: str) -> SearchDecision:
        """Phase 0 stub: return a default 'refuse everything' SearchDecision.

        The default refuses to show results AND refuses to generate. This
        is the safe baseline — until later phases add real classification,
        the system never acts on uncertain evidence.

        Args:
            query: the user's query string (unused in Phase 0; future
                   phases use it for validation and safety checks).

        Returns:
            SearchDecision with everything in the most-conservative state.
        """
        return self._default_decision()

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _default_decision() -> SearchDecision:
        """The 'refuse everything' baseline. Used by Phase 0; later phases
        will return enriched SearchDecisions, but this stays the safe
        fallback for any path that doesn't build a custom one."""
        return SearchDecision(
            confidence="none",
            should_show_results=False,
            should_generate_answer=False,
            final_results=[],
            warning=None,
            fallback_triggered=False,
            fallback_method=None,
            top_score=float("-inf"),
            score_gap=0.0,
        )
