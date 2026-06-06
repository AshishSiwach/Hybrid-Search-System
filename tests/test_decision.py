"""
Phase 0 tests for the Layer 2 contract.

These verify the *shape* of Layer 2 is correct — not the behaviour, which
comes in later phases. Each subsequent phase will add tests covering its
new branch of logic.

Run from project root:
    pytest tests/test_decision.py -v
"""

import sys
from pathlib import Path

import pytest

# Make repo root importable regardless of where pytest is run from
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from querylens.decision import DecisionLayer, SearchDecision  # noqa: E402


# ── 1. The contract is importable and the dataclass constructs ───────────────


def test_search_decision_can_be_constructed_explicitly():
    """All fields can be set; defaults exist for the later-phase fields."""
    d = SearchDecision(
        confidence="high",
        should_show_results=True,
        should_generate_answer=True,
        final_results=[(0, 7.3), (4, 6.1)],
        warning=None,
        fallback_triggered=False,
        fallback_method=None,
        top_score=7.3,
        score_gap=1.2,
    )
    assert d.confidence == "high"
    assert d.should_show_results is True
    assert d.should_generate_answer is True
    assert d.final_results == [(0, 7.3), (4, 6.1)]
    # Later-phase fields have safe defaults
    assert d.ambiguous is False
    assert d.rejected is False
    assert d.rejected_reason is None


# ── 2. DecisionLayer is instantiable and decide() returns a SearchDecision ───


def test_decision_layer_instantiates():
    """Phase 0 has no constructor parameters — DecisionLayer() should just work."""
    layer = DecisionLayer()
    assert layer is not None


def test_decide_returns_a_search_decision_with_safe_defaults():
    """Phase 0 stub returns 'refuse everything' for any query."""
    layer = DecisionLayer()
    d = layer.decide("any query at all")

    # Must be the right type
    assert isinstance(d, SearchDecision)

    # Must be in the safest possible state
    assert d.confidence == "none"
    assert d.should_show_results is False
    assert d.should_generate_answer is False
    assert d.final_results == []
    assert d.warning is None
    assert d.fallback_triggered is False
    assert d.fallback_method is None
    assert d.top_score == float("-inf")
    assert d.score_gap == 0.0
    assert d.ambiguous is False
    assert d.rejected is False
    assert d.rejected_reason is None


# ── 3. Architectural invariant: no ML imports leak through ───────────────────


def test_importing_decision_does_not_pull_in_ml_libraries():
    """The decision module must stay model-free. If torch or
    sentence_transformers shows up in sys.modules just from importing
    querylens.decision, the architectural rule is broken."""
    # Note: torch may already be imported by something else in the test
    # session; what we really verify is that *importing decision alone*
    # doesn't pull it in. We do that by re-importing in a fresh subprocess.
    import subprocess
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; "
         "from querylens.decision import DecisionLayer, SearchDecision; "
         "assert 'torch' not in sys.modules, 'torch leaked'; "
         "assert 'sentence_transformers' not in sys.modules, 'sentence_transformers leaked'; "
         "print('OK')"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    assert "OK" in result.stdout
