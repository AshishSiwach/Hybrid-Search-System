"""
Tests for decision.py — the QueryLens Layer 2 decision logic.

Run from repo root:
    pytest tests/test_decision.py -v

These tests deliberately do NOT load any models. The decision layer is pure
logic on numpy arrays and dicts — the entire suite finishes in milliseconds.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make repo root importable when running pytest from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decision import DecisionLayer, SearchDecision   # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def layer():
    """Telemetry off so tests don't write JSONL to disk."""
    return DecisionLayer(diversifier="jaccard", enable_telemetry=False)


@pytest.fixture
def layer_mmr():
    return DecisionLayer(diversifier="mmr", enable_telemetry=False)


@pytest.fixture
def corpus():
    """
    Small fake corpus designed to trip every Layer 2 check:

      0, 1, 2 — share 8 of 9 tokens each → Jaccard ≈ 0.80 (above 0.75 cutoff)
               and live on en.wikipedia.org / www.wikipedia.org (same normalised domain)
      3       — unrelated tokens, distinct domain
      4       — unrelated tokens, distinct domain

    Token-level Jaccard math:
      0 vs 1: |{a..h} ∩ {a..h,k}| / |{a..h,i} ∪ {a..h,k}| = 8/10 = 0.80
    """
    passages = [
        "alpha beta gamma delta epsilon zeta eta theta iota",     # 0 — near-dup base
        "alpha beta gamma delta epsilon zeta eta theta kappa",    # 1 — Jaccard 0.80 vs 0
        "alpha beta gamma delta epsilon zeta eta theta lambda",   # 2 — Jaccard 0.80 vs 0
        "mitochondrion powerhouse cell organelle structure",      # 3 — unrelated
        "gradient descent optimisation neural training loss",     # 4 — unrelated
    ]
    metadata = [
        {"url": "https://en.wikipedia.org/page0"},
        {"url": "https://en.wikipedia.org/page1"},
        {"url": "https://www.wikipedia.org/page2"},  # 'www.' stripped → wikipedia.org
        {"url": "https://example.com/cell-biology"},
        {"url": "https://other.org/ml-basics"},
    ]
    return passages, metadata


@pytest.fixture
def embeddings():
    """
    L2-normalised 3-D vectors. Passages 0, 1, 2 are near-identical in
    embedding space (intended duplicates). 3 and 4 are orthogonal.
    """
    raw = np.array([
        [1.0, 0.0, 0.0],     # passage 0
        [0.99, 0.10, 0.0],   # passage 1 — cos(0,1) ≈ 0.99
        [0.98, 0.15, 0.05],  # passage 2 — cos(0,2) ≈ 0.98
        [0.0, 1.0, 0.0],     # passage 3 — orthogonal to 0
        [0.0, 0.0, 1.0],     # passage 4 — orthogonal to 0
    ], dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


# ── 1. Empty / degenerate inputs ─────────────────────────────────────────────

def test_empty_results_returns_none_confidence(layer, corpus):
    passages, metadata = corpus
    d = layer.decide("query", [], {}, passages, metadata)
    assert d.confidence == "none"
    assert d.should_generate_answer is False
    assert d.final_results == []
    assert d.warning is not None


def test_single_result_has_infinite_score_gap(layer, corpus):
    passages, metadata = corpus
    d = layer.decide("query", [(0, 5.0)], {}, passages, metadata)
    assert d.score_gap == float("inf")
    assert d.confidence == "high"


# ── 2. Confidence classification ─────────────────────────────────────────────

def test_classify_high(layer, corpus):
    passages, metadata = corpus
    d = layer.decide("q", [(0, 5.0), (4, 4.0)], {}, passages, metadata)
    assert d.confidence == "high"
    assert d.should_generate_answer is True


def test_classify_medium(layer, corpus):
    passages, metadata = corpus
    d = layer.decide("q", [(0, 1.0), (4, 0.1)], {}, passages, metadata)
    assert d.confidence == "medium"
    assert d.should_generate_answer is True


def test_classify_low(layer, corpus):
    passages, metadata = corpus
    d = layer.decide("q", [(0, -1.0), (4, -1.5)], {}, passages, metadata)
    assert d.confidence == "low"
    assert d.should_generate_answer is True   # low still generates, just with warning


def test_classify_none_gates_llm(layer, corpus):
    passages, metadata = corpus
    d = layer.decide("q", [(0, -5.0), (4, -6.0)], {}, passages, metadata)
    assert d.confidence == "none"
    assert d.should_generate_answer is False   # the critical LLM gate


# ── 3. Score-gap demotion ────────────────────────────────────────────────────

def test_score_gap_demotes_high_to_medium(layer, corpus):
    passages, metadata = corpus
    # Both well into "high" range but only 0.1 apart → ambiguous
    d = layer.decide("q", [(0, 5.0), (4, 4.95)], {}, passages, metadata)
    assert d.ambiguous is True
    assert d.confidence == "medium"
    assert "ambiguous" in d.warning.lower()


def test_score_gap_demotes_medium_to_low(layer, corpus):
    passages, metadata = corpus
    d = layer.decide("q", [(0, 1.0), (4, 0.9)], {}, passages, metadata)
    assert d.ambiguous is True
    assert d.confidence == "low"


def test_score_gap_does_not_demote_low(layer, corpus):
    """Low confidence with a tied #2 stays low — nothing lower to demote to."""
    passages, metadata = corpus
    d = layer.decide("q", [(0, -1.0), (4, -1.1)], {}, passages, metadata)
    assert d.confidence == "low"   # not demoted


def test_score_gap_no_demotion_when_clearly_best(layer, corpus):
    passages, metadata = corpus
    d = layer.decide("q", [(0, 5.0), (4, 1.0)], {}, passages, metadata)
    assert d.ambiguous is False
    assert d.confidence == "high"


# ── 4. Diversity — Jaccard ───────────────────────────────────────────────────

def test_jaccard_drops_lexical_duplicates(layer, corpus):
    """Passages 0/1/2 share most tokens — only one should survive."""
    passages, metadata = corpus
    results = [(0, 5.0), (1, 4.8), (2, 4.6), (3, 4.0), (4, 3.5)]
    d = layer.decide("q", results, {}, passages, metadata)
    idxs = [i for i, _ in d.final_results]
    # At most one of {0, 1, 2} should remain (the highest-scoring one)
    assert sum(i in (0, 1, 2) for i in idxs) <= 1
    assert 3 in idxs and 4 in idxs


def test_jaccard_caps_per_domain(layer, corpus):
    """All-wikipedia.org results should be capped at MAX_PER_DOMAIN = 2."""
    passages = ["alpha alpha unique words one", "beta beta unique words two",
                "gamma gamma unique words three", "delta delta unique words four"]
    metadata = [{"url": f"https://en.wikipedia.org/page{i}"} for i in range(4)]
    results = [(0, 5.0), (1, 4.5), (2, 4.0), (3, 3.5)]
    d = layer.decide("q", results, {}, passages, metadata)
    assert len(d.final_results) == 2   # capped


# ── 5. Diversity — MMR ───────────────────────────────────────────────────────

def test_mmr_promotes_semantically_diverse(layer_mmr, corpus, embeddings):
    """
    Input: top-scored is the wiki/photosynthesis cluster (0/1/2).
    MMR should pick #0 first, then skip semantically-identical 1 and 2
    even though they're next in rerank order, and reach for 3 and 4.
    """
    passages, metadata = corpus
    results = [(0, 5.0), (1, 4.8), (2, 4.6), (3, 4.0), (4, 3.5)]
    d = layer_mmr.decide("q", results, {}, passages, metadata, embeddings=embeddings)
    idxs = [i for i, _ in d.final_results]
    assert idxs[0] == 0                          # top-scored stays #1
    assert 3 in idxs and 4 in idxs               # diverse picks surfaced
    assert sum(i in (1, 2) for i in idxs) == 0   # near-dupes filtered


def test_mmr_falls_back_to_jaccard_when_no_embeddings(layer_mmr, corpus):
    """MMR layer with embeddings=None should silently degrade to Jaccard."""
    passages, metadata = corpus
    results = [(0, 5.0), (1, 4.8), (3, 4.0)]
    d = layer_mmr.decide("q", results, {}, passages, metadata, embeddings=None)
    # Should still get a valid decision; near-dup 1 dropped
    idxs = [i for i, _ in d.final_results]
    assert 0 in idxs
    assert 3 in idxs
    assert 1 not in idxs


# ── 6. Dense-aware fallback ──────────────────────────────────────────────────

def test_fallback_fires_when_primary_none_and_dense_good(layer, corpus):
    passages, metadata = corpus
    # Primary scores all below LOW_THRESHOLD → confidence "none"
    primary = [(0, -5.0), (1, -5.5)]
    # Dense fallback has a strong cosine (well above DENSE_MEDIUM)
    fallback = {"Dense only": [(3, 0.80), (4, 0.55)]}
    d = layer.decide("q", primary, fallback, passages, metadata)
    assert d.fallback_triggered is True
    assert d.fallback_method == "Dense only"
    assert d.confidence == "high"   # dense top score 0.80 ≥ DENSE_HIGH
    assert 3 in [i for i, _ in d.final_results]


def test_fallback_does_not_fire_when_dense_also_bad(layer, corpus):
    """Dense fallback below DENSE_LOW → no swap, decision stays 'none'."""
    passages, metadata = corpus
    primary  = [(0, -5.0), (1, -5.5)]
    fallback = {"Dense only": [(3, 0.15), (4, 0.10)]}   # all below DENSE_LOW
    d = layer.decide("q", primary, fallback, passages, metadata)
    assert d.fallback_triggered is False
    assert d.confidence == "none"
    assert d.should_generate_answer is False


def test_fallback_does_not_fire_when_primary_ok(layer, corpus):
    passages, metadata = corpus
    primary  = [(0, 2.0), (4, 1.0)]
    fallback = {"Dense only": [(3, 0.95)]}   # would be high, but never consulted
    d = layer.decide("q", primary, fallback, passages, metadata)
    assert d.fallback_triggered is False


# ── 7. Query validation ──────────────────────────────────────────────────────

def test_validate_empty_query(layer):
    d = layer.validate_query("")
    assert d is not None
    assert d.rejected is True
    assert d.should_generate_answer is False


def test_validate_whitespace_only(layer):
    d = layer.validate_query("   \t\n  ")
    assert d is not None and d.rejected is True


def test_validate_too_short(layer):
    d = layer.validate_query("a")
    assert d is not None and d.rejected is True


def test_validate_pure_punctuation(layer):
    d = layer.validate_query("???!!!")
    assert d is not None and d.rejected is True


def test_validate_too_long(layer):
    d = layer.validate_query("x" * 1000)
    assert d is not None and d.rejected is True


def test_validate_normal_query_passes(layer):
    d = layer.validate_query("what is photosynthesis")
    assert d is None   # no rejection


# ── 8. Diversifier configuration ─────────────────────────────────────────────

def test_invalid_diversifier_raises():
    with pytest.raises(ValueError):
        DecisionLayer(diversifier="foo", enable_telemetry=False)


def test_telemetry_disabled_does_not_write(tmp_path, corpus):
    layer = DecisionLayer(
        diversifier="jaccard",
        enable_telemetry=False,
        telemetry_path=str(tmp_path / "should_not_exist.jsonl"),
    )
    passages, metadata = corpus
    layer.decide("q", [(0, 5.0)], {}, passages, metadata)
    assert not (tmp_path / "should_not_exist.jsonl").exists()


def test_telemetry_enabled_writes_jsonl(tmp_path, corpus):
    log_path = tmp_path / "decisions.jsonl"
    layer = DecisionLayer(
        diversifier="jaccard",
        enable_telemetry=True,
        telemetry_path=str(log_path),
    )
    passages, metadata = corpus
    layer.decide("test query", [(0, 5.0), (4, 4.0)], {}, passages, metadata)
    assert log_path.exists()
    import json
    entry = json.loads(log_path.read_text().strip())
    assert entry["confidence"] == "high"
    assert entry["query"] == "test query"
    assert entry["diversifier"] == "jaccard"
