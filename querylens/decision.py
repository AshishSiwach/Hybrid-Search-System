"""
decision.py — Layer 2: decision logic for QueryLens.

Sits between retrieval (Layer 1) and LLM generation (Layer 3) and answers:
  - Is the query worth running at all?              → validate_query()
  - How confident are we in these results?          → confidence threshold
  - Should we generate an answer at all?            → should_generate_answer
  - Are the top results diverse enough?             → MMR or Jaccard filter
  - Should we fall back to a different method?      → dense-aware fallback
  - What did we decide, and why?                    → JSONL telemetry

Architectural contract:
  - No ML model dependencies. Pure logic on numpy + dicts.
  - Immutable inputs: never mutates upstream score lists.
  - Output is a single dataclass (SearchDecision) — Layer 3 only reads it.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np


# ── Public output contract ───────────────────────────────────────────────────

@dataclass
class SearchDecision:
    confidence: str                          # "high" | "medium" | "low" | "none"
    should_generate_answer: bool
    final_results: List[Tuple[int, float]]   # diversity-filtered (idx, score) pairs
    warning: Optional[str]
    fallback_triggered: bool
    fallback_method: Optional[str]
    top_score: float
    score_gap: float
    ambiguous: bool = False                  # True when top results are tied
    rejected: bool = False                   # True when query validation failed


# ── The decision layer ───────────────────────────────────────────────────────

class DecisionLayer:
    """
    Layer 2 of the QueryLens predict → decide → act pipeline.

    Threshold notes
    ---------------
    Cross-encoder (Hybrid + Reranker) scores are raw logits, roughly [-6, +10]
    on MS MARCO. Dense fallback scores are cosine similarities in [0, 1].
    Each path has its own threshold set — never compare logits to cosines.
    All thresholds should be re-derived via calibrate_thresholds.py against
    your actual corpus, not trusted as seeded values.
    """

    # ─ Cross-encoder logit thresholds (Hybrid + Reranker, primary path) ─
    HIGH_THRESHOLD   =  9.93
    MEDIUM_THRESHOLD =  8.50
    LOW_THRESHOLD    =  5.92

    # ─ Dense cosine thresholds (fallback path, [0, 1] for L2-normalised vecs) ─
    # SEEDED VALUES — not yet data-calibrated. Affect only ~2–5 % of queries
    # (those where the primary path returns "none" AND Dense Only has results).
    # Revisit via calibrate_thresholds.py once results/decisions.jsonl shows
    # fallback_triggered firing on > 5 % of queries — until then, the seeded
    # values cover the long tail acceptably.
    DENSE_HIGH   = 0.65
    DENSE_MEDIUM = 0.45
    DENSE_LOW    = 0.30

    # ─ Score-gap demotion: top result barely beats #2 → confidence drops one tier ─
    SCORE_GAP_AMBIGUOUS_THRESHOLD = 0.5

    # ─ Diversity filter knobs ─
    MAX_PER_DOMAIN       = 2      # cap results from any single normalised domain
    MAX_RESULTS          = 10     # cap on diversified output (caller still slices [:5])
    DUPLICATE_THRESHOLD  = 0.75   # Jaccard token-overlap cutoff (lexical dedup)
    MMR_LAMBDA           = 0.7    # λ ↔ relevance vs diversity. 1.0 = pure relevance.
    MMR_HARD_DUP_COSINE  = 0.95   # cosine above this = definitely a duplicate, drop

    # ─ Query validation knobs ─
    MIN_QUERY_LENGTH = 2
    MAX_QUERY_LENGTH = 500

    def __init__(
        self,
        diversifier: str = "mmr",
        enable_telemetry: bool = True,
        telemetry_path: str = "results/decisions.jsonl",
    ):
        """
        Args:
            diversifier:      "mmr" (semantic, needs embeddings) or "jaccard" (lexical).
                              Falls back to Jaccard automatically when embeddings are not
                              provided to decide().
            enable_telemetry: When True, append every decision to telemetry_path as JSONL.
                              Off in tests.
            telemetry_path:   Where to append decision logs.
        """
        if diversifier not in ("mmr", "jaccard"):
            raise ValueError(f"diversifier must be 'mmr' or 'jaccard', got {diversifier!r}")
        self.diversifier      = diversifier
        self.enable_telemetry = enable_telemetry
        self.telemetry_path   = Path(telemetry_path)

    # ── 1. Query validation ──────────────────────────────────────────────────

    def validate_query(self, query: str) -> Optional[SearchDecision]:
        """
        Cheap pre-flight check. Returns a rejected SearchDecision if the query
        is unusable; returns None when the query is OK to pass to Layer 1.

        Caller pattern:
            decision = layer.validate_query(q)
            if decision is None:
                decision = layer.decide(q, ...)
        """
        q = (query or "").strip()

        reason: Optional[str] = None
        if not q:
            reason = "Please enter a query."
        elif len(q) < self.MIN_QUERY_LENGTH:
            reason = "Query is too short — try a few more words."
        elif len(q) > self.MAX_QUERY_LENGTH:
            reason = f"Query is too long (over {self.MAX_QUERY_LENGTH} characters)."
        elif not any(c.isalnum() for c in q):
            reason = "Query needs at least some letters or numbers."

        if reason is None:
            return None

        decision = SearchDecision(
            confidence="none",
            should_generate_answer=False,
            final_results=[],
            warning=reason,
            fallback_triggered=False,
            fallback_method=None,
            top_score=float("-inf"),
            score_gap=0.0,
            ambiguous=False,
            rejected=True,
        )
        self._log(query, decision)
        return decision

    # ── 2. Main decide() ─────────────────────────────────────────────────────

    def decide(
        self,
        query:            str,
        best_results:     List[Tuple[int, float]],
        fallback_results: Dict[str, List[Tuple[int, float]]],
        passages:         List[str],
        metadata:         List[dict],
        embeddings:       Optional[np.ndarray] = None,
    ) -> SearchDecision:
        """
        Run all decision checks and return a SearchDecision.

        Args:
            query:            User's query string (used only for telemetry here).
            best_results:     (idx, score) from Hybrid + Reranker (cross-encoder logits).
            fallback_results: {method: [(idx, score), ...]} for other retrieval methods.
            passages:         Full corpus text list.
            metadata:         Per-passage dicts; must have a "url" key.
            embeddings:       (N, D) L2-normalised corpus embeddings. Required for MMR;
                              when None, falls back to Jaccard diversification.
        """
        # 0. Empty guard
        if not best_results:
            decision = SearchDecision(
                confidence="none",
                should_generate_answer=False,
                final_results=[],
                warning="No results found. Try a different query.",
                fallback_triggered=False,
                fallback_method=None,
                top_score=float("-inf"),
                score_gap=0.0,
            )
            self._log(query, decision)
            return decision

        # 1. Read signals
        top_score = best_results[0][1]
        score_gap = (
            best_results[0][1] - best_results[1][1]
            if len(best_results) > 1
            else float("inf")
        )
        confidence = self._classify(top_score)

        # 2. Score-gap demotion: ambiguous wins should not look confident
        ambiguous = False
        if confidence in ("high", "medium") and score_gap < self.SCORE_GAP_AMBIGUOUS_THRESHOLD:
            ambiguous   = True
            confidence  = {"high": "medium", "medium": "low"}[confidence]

        # 3. Diversity filter on primary results
        diverse_results, diversity_filtered = self._diversify(
            best_results, passages, metadata, embeddings,
        )

        # 4. Fallback: confidence=none AND Dense has something genuinely useful
        fallback_triggered = False
        fallback_method    = None
        final_results      = diverse_results

        if confidence == "none" and "Dense only" in fallback_results:
            fb_raw = fallback_results["Dense only"]
            if fb_raw:
                dense_top         = fb_raw[0][1]
                dense_confidence  = self._classify_dense(dense_top)
                if dense_confidence != "none":
                    fb_diverse, _ = self._diversify(fb_raw, passages, metadata, embeddings)
                    if fb_diverse:
                        final_results      = fb_diverse
                        fallback_triggered = True
                        fallback_method    = "Dense only"
                        confidence         = dense_confidence  # use real dense conf, not hardcoded "low"

        # 5. Build decision
        decision = SearchDecision(
            confidence=confidence,
            should_generate_answer=confidence in ("high", "medium", "low"),
            final_results=final_results,
            warning=self._warning(
                confidence, diversity_filtered, fallback_triggered,
                fallback_method, ambiguous,
            ),
            fallback_triggered=fallback_triggered,
            fallback_method=fallback_method,
            top_score=top_score,
            score_gap=score_gap,
            ambiguous=ambiguous,
        )
        self._log(query, decision)
        return decision

    # ── 3. Classifiers ───────────────────────────────────────────────────────

    def _classify(self, score: float) -> str:
        """Confidence classifier for cross-encoder logits (Hybrid + Reranker path)."""
        if score >= self.HIGH_THRESHOLD:   return "high"
        if score >= self.MEDIUM_THRESHOLD: return "medium"
        if score >= self.LOW_THRESHOLD:    return "low"
        return "none"

    def _classify_dense(self, score: float) -> str:
        """Confidence classifier for dense cosine scores (Dense Only fallback path)."""
        if score >= self.DENSE_HIGH:   return "high"
        if score >= self.DENSE_MEDIUM: return "medium"
        if score >= self.DENSE_LOW:    return "low"
        return "none"

    # ── 4. Diversifiers ──────────────────────────────────────────────────────

    def _diversify(
        self,
        results:    List[Tuple[int, float]],
        passages:   List[str],
        metadata:   List[dict],
        embeddings: Optional[np.ndarray],
    ) -> Tuple[List[Tuple[int, float]], bool]:
        """Dispatch to the configured diversifier; fall back to Jaccard if no embeddings."""
        if self.diversifier == "mmr" and embeddings is not None:
            return self._diversify_mmr(results, metadata, embeddings)
        return self._diversify_jaccard(results, passages, metadata)

    def _diversify_jaccard(
        self,
        results:  List[Tuple[int, float]],
        passages: List[str],
        metadata: List[dict],
    ) -> Tuple[List[Tuple[int, float]], bool]:
        """
        Lexical diversifier. Two checks:
          (a) Hard cap: no more than MAX_PER_DOMAIN results from any one domain.
          (b) Drop any candidate whose Jaccard token-overlap with an already-kept
              result is >= DUPLICATE_THRESHOLD.

        Strengths:  zero ML dependency, catches near-exact duplicates instantly.
        Weakness:   misses paraphrases ("the sky is blue" vs "blue is the sky's colour").
                    Use MMR if you have embeddings on hand.
        """
        domain_counts:  Dict[str, int] = {}
        seen_word_sets: List[set]      = []
        filtered:       List[Tuple[int, float]] = []
        removed_any = False

        for idx, score in results:
            idx    = int(idx)
            domain = self._domain(metadata[idx].get("url", ""))

            if domain_counts.get(domain, 0) >= self.MAX_PER_DOMAIN:
                removed_any = True
                continue

            words = set(passages[idx].lower().split())
            is_dup = any(
                len(words | s) > 0
                and len(words & s) / len(words | s) >= self.DUPLICATE_THRESHOLD
                for s in seen_word_sets
            )
            if is_dup:
                removed_any = True
                continue

            filtered.append((idx, score))
            seen_word_sets.append(words)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        return filtered, removed_any

    def _diversify_mmr(
        self,
        results:    List[Tuple[int, float]],
        metadata:   List[dict],
        embeddings: np.ndarray,
    ) -> Tuple[List[Tuple[int, float]], bool]:
        """
        Maximal Marginal Relevance with cosine on dense embeddings.

        At every step pick the candidate that maximises
            λ * rerank_score(i) - (1 - λ) * max_cos_sim(i, already_picked)

        Plus two hard cuts: domain cap, and reject anything with cosine ≥
        MMR_HARD_DUP_COSINE to an already-picked passage.

        Strength:  catches semantic paraphrases Jaccard misses; actively
                   spreads coverage across distinct angles.
        Cost:      O(N · k) cosine ops where k = picked count. Negligible.
        Assumes:   embeddings are L2-normalised so dot product == cosine.
        """
        if not results:
            return results, False

        # Always keep the top-scored result
        picked: List[Tuple[int, float]] = [results[0]]
        pool:   List[Tuple[int, float]] = list(results[1:])

        domain_counts: Dict[str, int] = {}
        top_idx     = int(results[0][0])
        top_domain  = self._domain(metadata[top_idx].get("url", ""))
        domain_counts[top_domain] = 1

        while pool and len(picked) < self.MAX_RESULTS:
            best_i:   int   = -1
            best_mmr: float = -float("inf")

            for i, (cand_idx, cand_score) in enumerate(pool):
                cand_idx    = int(cand_idx)
                cand_domain = self._domain(metadata[cand_idx].get("url", ""))

                # Hard domain cap
                if domain_counts.get(cand_domain, 0) >= self.MAX_PER_DOMAIN:
                    continue

                # Max cosine similarity to anything already picked
                cand_vec = embeddings[cand_idx]
                max_sim  = 0.0
                for p_idx, _ in picked:
                    sim = float(np.dot(cand_vec, embeddings[int(p_idx)]))
                    if sim > max_sim:
                        max_sim = sim

                # Hard near-duplicate cap
                if max_sim >= self.MMR_HARD_DUP_COSINE:
                    continue

                mmr = self.MMR_LAMBDA * cand_score - (1 - self.MMR_LAMBDA) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_i   = i

            if best_i == -1:
                break  # no eligible candidates remain

            chosen = pool.pop(best_i)
            chosen_domain = self._domain(metadata[int(chosen[0])].get("url", ""))
            picked.append(chosen)
            domain_counts[chosen_domain] = domain_counts.get(chosen_domain, 0) + 1

        removed_any = len(picked) < len(results)
        return picked, removed_any

    # ── 5. Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _domain(url: str) -> str:
        """
        Normalise a URL to a comparable domain string. Strips 'www.' so
        'www.example.com' and 'example.com' collapse together. Does NOT
        fully strip subdomains (so en.wikipedia and simple.wikipedia stay
        distinct) — install tldextract if you need true registrable-domain
        normalisation.
        """
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc

    @staticmethod
    def _warning(
        confidence:        str,
        diversity_filtered: bool,
        fallback_triggered: bool,
        fallback_method:    Optional[str],
        ambiguous:          bool,
    ) -> Optional[str]:
        """
        Pick the single most informative warning. Order matters — the first
        match wins. Hierarchy:
          1. No results at all                  → highest signal
          2. Fallback fired                     → user should know we switched
          3. Ambiguous (high/medium got demoted) → distinct from "actually low"
          4. Genuinely low confidence
          5. Diversity filter trimmed results
        """
        if confidence == "none":
            return "No relevant results found. Try rephrasing your query."
        if fallback_triggered:
            return f"Low confidence in primary results — showing {fallback_method} results instead."
        if ambiguous:
            return "Results are ambiguous — top candidates have similar scores."
        if confidence == "low":
            return "Low confidence — results may not directly answer your query."
        if diversity_filtered:
            return "Some near-duplicate results were removed to improve diversity."
        return None

    def _log(self, query: str, decision: SearchDecision) -> None:
        """
        Append the decision to a JSONL telemetry file. Never crashes the request.

        Each line is a self-contained JSON object — easy to ingest with
        `pandas.read_json(..., lines=True)` or duckdb for offline analysis.
        """
        if not self.enable_telemetry:
            return
        try:
            entry = {
                "ts":                    datetime.now(timezone.utc).isoformat(),
                "query":                 (query or "")[:200],
                "confidence":            decision.confidence,
                "top_score":             None if decision.top_score == float("-inf") else decision.top_score,
                "score_gap":             None if decision.score_gap == float("inf") else decision.score_gap,
                "ambiguous":             decision.ambiguous,
                "fallback_triggered":    decision.fallback_triggered,
                "fallback_method":       decision.fallback_method,
                "n_results":             len(decision.final_results),
                "should_generate_answer": decision.should_generate_answer,
                "rejected":              decision.rejected,
                "diversifier":           self.diversifier,
            }
            self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.telemetry_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            # Telemetry must never break the request path
            pass
