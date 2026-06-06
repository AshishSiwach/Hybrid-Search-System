# Layer 2 — Implementation Plan

> A clean-slate planning document. Edit freely. We refer back to this
> file as the source of truth while building.
>
> **Revision history**
> - v1 — initial plan (single-signal confidence, single-criterion fallback,
>   one LLM gate)
> - v2 — review-pass update: split show-results vs generate-answer,
>   any-2-of-3 fallback gate, added phases for answerability calibration
>   and multi-signal confidence, named output-validation as out-of-scope
>   Layer 4 work

---

## 1. What Layer 2 is, in one paragraph

Layer 2 is the **decision layer** that sits between retrieval (Layer 1)
and LLM generation (Layer 3). It does NOT retrieve. It does NOT
generate. It examines the query going in, examines the scores coming
out of retrieval, and decides: *should we send anything to the LLM at
all, and if so, what*? Its purpose is to prevent the most common
production failure of RAG systems — passing weak or adversarial
evidence to an LLM that will confidently confabulate an answer.

---

## 2. Where Layer 2 sits in the larger project

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 0 — Data                                                  │
│  configs/config.yaml, data/ms_marco_*.json, data/*.npy, data/*.index │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  Layer 1 — Retrieval (already built)                             │
│  querylens/retrievers.py — BM25, Dense, Hybrid (RRF)             │
│  querylens/reranker.py   — Cross-encoder reranker                │
│  Output: (idx, score) tuples, top-N candidates                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  Layer 2 — Decision (THIS PLAN)                                  │
│  querylens/decision.py — new                                     │
│  querylens/safety.py   — new                                     │
│  querylens/prompts.py  — new                                     │
│  Output: SearchDecision dataclass                                │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  Layer 3 — Generation (already built)                            │
│  streamlit_app.py:generate_answer() — Claude API call            │
│  Receives SearchDecision, gates the LLM call on it               │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                          User UI
                       (Streamlit page)
```

**Adjacent components Layer 2 connects to:**

| Connection | Purpose |
|---|---|
| `configs/config.yaml` | Pulls operational knobs (timeouts, paths) |
| `results/decisions.jsonl` | Writes per-query telemetry here |
| `scripts/calibrate_thresholds.py` | Derives Layer 2's threshold values from labelled data |
| `tests/test_decision.py` | Verifies Layer 2's logic without loading any models |
| `streamlit_app.py` | Instantiates Layer 2 and consumes its output |
| `pages/observability.py` | Reads Layer 2's telemetry JSONL and renders aggregate production behaviour — closes the observability loop |

---

## 3. Runtime flow — where each mechanism fires

A query travels through Layer 2 in five checkpoints. Each checkpoint
is one of the four mechanisms the user asked for:

```
┌─────────────────────────────────────────────────────────────────────┐
│  USER QUERY ARRIVES                                                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CHECKPOINT 1 — INPUT GUARDRAIL                                     │
│  ──────────────────────────────                                     │
│  Mechanism: guardrail                                               │
│  Where:     querylens/decision.py :: validate_query()               │
│  Checks:                                                            │
│    a) Empty / whitespace-only                                       │
│    b) Length below MIN_QUERY_LENGTH                                 │
│    c) Length above MAX_QUERY_LENGTH                                 │
│    d) No alphanumeric content                                       │
│    e) Matches an injection pattern (from querylens/safety.py)       │
│  On failure:                                                        │
│    - Skip Layer 1 entirely (zero compute spent)                     │
│    - LOG to results/decisions.jsonl (mechanism: logging)            │
│    - Return SearchDecision with rejected=True                       │
│    - Layer 3 sees should_generate_answer=False → no LLM call        │
└─────────────────────────────────────────────────────────────────────┘
                              │   pass
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1 RUNS — returns (idx, score) top-N from each method         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CHECKPOINT 2 — CONFIDENCE THRESHOLD                                │
│  ────────────────────────────────                                   │
│  Mechanism: confidence threshold                                    │
│  Where:     querylens/decision.py :: _classify()                    │
│  Input:     top score from Hybrid + Reranker (cross-encoder logit)  │
│  Output:    "high" | "medium" | "low" | "none"                      │
│  Buckets:   HIGH_THRESHOLD / MEDIUM_THRESHOLD / LOW_THRESHOLD       │
│             — calibrated values come from                           │
│             scripts/calibrate_thresholds.py                         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CHECKPOINT 3 — FALLBACK TRIGGER (multi-criterion)                  │
│  ──────────────────────────────────────                             │
│  Mechanism: fallback                                                │
│  Where:     querylens/decision.py :: decide() (fallback block)      │
│  Triggers when: confidence from Checkpoint 2 == "none"              │
│  Quality gate: swap to Dense Only only when ≥ 2 of 3 criteria pass: │
│    a) Top dense score ≥ DENSE_LOW (basic threshold)                 │
│    b) Top-3 dense results span ≥ 2 distinct normalised domains      │
│       (coverage check — guards against single-source dominance)     │
│    c) Mean of top-3 dense scores ≥ DENSE_LOW × 0.7                  │
│       (agreement check — not just one strong outlier)               │
│  Action:                                                            │
│    - If swap: confidence = dense classifier's verdict               │
│    - If no swap: confidence stays "none"                            │
│  Rationale: single-criterion swaps were brittle (a lone strong      │
│  outlier could pull a swap that delivers no real coverage).         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CHECKPOINT 4 — POST-PROCESSING (diversity, score-gap)              │
│  ─────────────────────────────────────                              │
│  Mechanism: result quality enforcement                              │
│  Where:     querylens/decision.py :: _diversify() + gap demotion    │
│  Actions:                                                           │
│    a) Diversify top-N (MMR with cosine OR Jaccard fallback)         │
│    b) Cap N results per domain                                      │
│    c) If score_gap between #1 and #2 < ε, demote confidence one     │
│       tier (catches "ambiguous winner" cases)                       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CHECKPOINT 5 — TELEMETRY                                           │
│  ─────────────────────                                              │
│  Mechanism: logging                                                 │
│  Where:     querylens/decision.py :: _log()                         │
│  Writes one JSON line per decision to                               │
│  results/decisions.jsonl with:                                      │
│    ts, query, confidence, top_score, score_gap,                     │
│    fallback_triggered, fallback_method,                             │
│    n_results, should_generate_answer,                               │
│    rejected, rejected_reason, diversifier, ...                      │
│  Telemetry MUST NEVER crash the request path                        │
│  (try/except wrap that swallows errors)                             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                  SearchDecision dataclass
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3 (streamlit_app.py) — two decisions, not one                │
│                                                                     │
│  if decision.should_show_results:                                   │
│      render decision.final_results[:N] as expandable passage list   │
│                                                                     │
│  if decision.should_generate_answer:                                │
│      call Claude with decision.final_results[:N]                    │
│      render the synthesized answer above the passage list           │
│  else if decision.should_show_results:                              │
│      render "We found these but couldn't confidently synthesize"    │
│                                                                     │
│  Always render: confidence badge, warning (if any)                  │
│                                                                     │
│  Rationale: showing results is a softer decision than synthesizing  │
│  an answer. Some queries warrant "here's what we found, you read"   │
│  rather than a hallucination-prone LLM summary.                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. The four mechanisms — detailed

### 4.1 Guardrails (input safety)

| Aspect | Detail |
|---|---|
| **Purpose** | Reject bad inputs before spending Layer 1 compute. Block adversarial inputs from reaching the LLM. |
| **Two sub-types** | (a) Cheap validation — length, format, alphanumeric. (b) Pattern-based injection detection. |
| **Location** | `querylens/decision.py :: validate_query()` calling `querylens/safety.py :: detect_injection()` |
| **Runs when** | At the very start of the request, BEFORE Layer 1 |
| **Output** | `None` (query is OK to proceed) or a rejected `SearchDecision` |
| **User-facing message** | Vague for safety rejections (don't fingerprint patterns), specific for validation rejections ("too short", etc.) |
| **Telemetry** | Records `rejected_reason` ∈ {"validation", "safety"}; matched pattern goes to log only |
| **Tied to project** | First defence layer. Reduces cost (compute + Claude API calls) by short-circuiting bad requests. |

**Open decisions:**

- `MIN_QUERY_LENGTH` and `MAX_QUERY_LENGTH` values (suggest: 2 and 500)
- Which injection-pattern families to include — propose covering: instruction-override, role-swap, system-prompt extraction, system-state probing
- User-facing copy for each rejection type

### 4.2 Confidence thresholds

| Aspect | Detail |
|---|---|
| **Purpose** | Translate the cross-encoder's continuous logit score into a discrete action policy. |
| **Buckets** | `"high"` / `"medium"` / `"low"` / `"none"` |
| **Location** | `querylens/decision.py :: _classify()` |
| **Runs when** | Immediately after Layer 1, on the top result of Hybrid + Reranker |
| **Bucket meanings** | high = generate answer with confidence badge<br>medium = generate with normal framing<br>low = generate with warning banner<br>none = DO NOT generate; show "no results" |
| **The LLM gate** | `should_generate_answer = (confidence != "none")` — single line, biggest production-safety lever |
| **Threshold values** | Live as class constants on `DecisionLayer`. Derived from `scripts/calibrate_thresholds.py` against labelled data. |
| **Tied to project** | This is what makes Layer 2 "smart." Without it, every query gets the same treatment regardless of evidence quality. |

**v1 → v2 evolution path: single-signal → multi-signal**

The initial confidence classifier uses **top score** and **score gap** only.
This is single-signal and known-fragile. After v1 ships and telemetry
accumulates, Phase 13 evolves the classifier to compose multiple weak
signals into a stronger combined judgement:

| Feature | Why it adds information beyond top score |
|---|---|
| **Top-3 mean score** | Catches "one strong outlier vs. genuinely strong set" |
| **Retrieval-method agreement** (Hybrid top-K ∩ Dense top-K) | Cross-method consensus is a strong relevance signal |
| Score decay (rank 1 → 5) | Steep decay = fragile #1; flat decay = robust evidence |
| Top-K unique-domain count | Coverage; one source dominating is a brittleness signal |

The first two are highest-leverage. Add them in Phase 13. The rest are
optional based on what telemetry reveals.

**Open decisions:**

- Initial threshold values (seeded vs calibrated)
- Whether to also threshold on the second-best score (Layer 2 demotion logic)
- Whether dense fallback uses the same buckets or a separate set
- Whether `should_generate_answer` and `should_show_results` use the same
  bucket cutoff or different ones (proposal: generate on `high`+`medium`,
  show on `high`+`medium`+`low`)

### 4.3 Fallback

| Aspect | Detail |
|---|---|
| **Purpose** | When the primary retrieval path returns nothing useful, try the secondary path before giving up. |
| **Primary** | Hybrid (BM25 + Dense via RRF) + Cross-encoder Reranker |
| **Fallback** | Dense Only (semantic search, no reranker) |
| **Trigger** | Primary confidence == `"none"` |
| **Quality gate (multi-criterion, any 2 of 3 must pass):** | <br>**(a)** Dense top score ≥ `DENSE_LOW` (basic threshold)<br>**(b)** Top-3 dense results span ≥ 2 distinct normalised domains (coverage)<br>**(c)** Mean of top-3 dense scores ≥ `DENSE_LOW × 0.7` (agreement) |
| **Why "any 2 of 3"** | All-3 risks fallback never firing; any-1 risks blind swaps on a single strong outlier. 2-of-3 enforces a real bar without being unreachable. |
| **Reported confidence** | After swap, confidence = dense classifier's verdict (could be high/medium/low, never silently capped) |
| **Location** | `querylens/decision.py :: decide()` (fallback block, after the primary classifier) |
| **Tied to project** | Uses Dense Only results that Layer 1 already computed — zero extra compute. Improves recall on queries the cross-encoder is uncertain about. |

**Open decisions:**

- Whether to fall back to BM25 as a final resort (third tier) — likely overkill given dense already covers semantic gap
- Dense threshold values (cosine similarity space, different from logit space)
- Whether the 0.7 coefficient in criterion (c) is right — tune from telemetry once available
- Should we log which of the 3 criteria each fallback decision passed/failed (helpful for telemetry analysis)

### 4.4 Logging (telemetry)

| Aspect | Detail |
|---|---|
| **Purpose** | Persist every decision so we can analyse the system's behaviour offline and refine thresholds, patterns, and copy. |
| **Format** | One JSON object per line in `results/decisions.jsonl` (JSONL) |
| **Fields** | timestamp, query, confidence, top_score, score_gap, ambiguous, fallback_triggered, fallback_method, n_results, should_generate_answer, rejected, rejected_reason, diversifier, plus optional `safety_pattern` when applicable |
| **Location** | `querylens/decision.py :: _log()` |
| **Runs when** | At the END of every call to `decide()` and `validate_query()` — including rejections |
| **Critical constraint** | MUST NEVER crash the request — wrapped in try/except that silently swallows errors |
| **Tied to project** | Feeds future calibration runs. Distribution of confidence buckets over time is the production signal we need to know if thresholds are still right. |

**Open decisions:**

- Whether to also log raw retrieval scores from each method (could be large)
- Truncation rules for very long queries (currently propose first 200 chars)
- Whether to also expose a "decision ID" for tracing into front-end logs

---

## 5. The output contract — `SearchDecision`

Layer 2 hands Layer 3 exactly one object — a dataclass:

```python
@dataclass
class SearchDecision:
    confidence: str                          # "high" | "medium" | "low" | "none"
    # TWO decisions, not one — keeps "search quality" separate from
    # "generation permission." See § 4.2 for cutoff proposal.
    should_show_results: bool                # SOFTER: show retrieved passages
    should_generate_answer: bool             # STRICTER: actually call the LLM
                                             # Invariant: should_generate_answer
                                             # implies should_show_results.
    final_results: List[Tuple[int, float]]   # what to pass to the LLM (or display)
    warning: Optional[str]                   # banner text for the UI
    fallback_triggered: bool                 # did we swap to dense?
    fallback_method: Optional[str]           # which method took over
    top_score: float                         # primary path's #1 score
    score_gap: float                         # #1 - #2 (used for ambiguity demote)
    ambiguous: bool = False                  # demoted from gap check?
    rejected: bool = False                   # validation/safety rejected?
    rejected_reason: Optional[str] = None    # "validation" | "safety" | None
```

**Layer 3's three rendering modes:**

| `should_show_results` | `should_generate_answer` | UI shows |
|:---:|:---:|---|
| ✗ | ✗ | Warning only ("no useful results / rejected") |
| ✓ | ✗ | Passages + "we found these but couldn't confidently synthesize an answer" |
| ✓ | ✓ | LLM answer + passages below |

The middle row is the new mode this revision enables. It's where weak
but non-trivial evidence lives — show the user, but don't put words in
the LLM's mouth.

**Layer 3's contract:** read these fields; render the badge; render the
warning if non-None; if `should_generate_answer` is True call Claude with
`final_results[:N]`; otherwise show the warning and stop. Layer 3 does NOT
look at raw scores, the corpus, or make any policy decisions.

---

## 6. Implementation phases

Order matters: each phase builds on the previous one. Plan to stop and
verify (tests + manual smoke test) at the end of each phase.

| Phase | What | Files touched | Mechanism focus |
|---|---|---|---|
| **0** | Define the contract — write the empty `SearchDecision` dataclass and the empty `DecisionLayer` class skeleton. No logic yet. | `querylens/decision.py` (new) | — |
| **1** | Confidence thresholds. Implement `_classify()` and a minimal `decide()` that just classifies and returns `SearchDecision`. Seed thresholds with placeholder values. | `querylens/decision.py` | thresholds |
| **2** | LLM gate. Wire `decide()` into `streamlit_app.py`: instantiate Layer 2, call `decide()` after retrieval, gate `generate_answer()` on `should_generate_answer`, render badge + warning. | `streamlit_app.py` | thresholds |
| **3** | Fallback. Add `_classify_dense()` and the fallback block in `decide()`. Add `DENSE_*` constants. | `querylens/decision.py` | fallback |
| **4** | Input guardrails — validation. Add `validate_query()` with the cheap checks. Call from `streamlit_app.py` before `decide()`. | `querylens/decision.py`, `streamlit_app.py` | guardrails |
| **5** | Input guardrails — safety. Create `querylens/safety.py` with `detect_injection()` and pattern list. Wire into `validate_query()`. | `querylens/safety.py` (new), `querylens/decision.py` | guardrails |
| **6** | Telemetry. Add `_log()` method. Write JSONL on every decision. Verify the file accumulates correctly. | `querylens/decision.py`, `.gitignore` | logging |
| **7** | Post-processing — diversity. Add MMR or Jaccard diversifier. Domain cap. | `querylens/decision.py` | result quality |
| **8** | Post-processing — score-gap demotion. Detect ambiguous winners. | `querylens/decision.py` | result quality |
| **9** | Tests. Add `tests/test_decision.py` with cases for every branch — zero models loaded. | `tests/test_decision.py` (new) | all |
| **10** | Calibration. Run `scripts/calibrate_thresholds.py`. Replace seeded thresholds with calibrated values. | `querylens/decision.py` (constants only) | thresholds |
| **11** | Prompts centralization. Create `querylens/prompts.py` with hardened system prompt. Refactor `streamlit_app.py:generate_answer()` to use it. | `querylens/prompts.py` (new), `streamlit_app.py` | guardrails (output side) |
| **12** | **Observability dashboard**. Streamlit `pages/observability.py` that reads `results/decisions.jsonl` and renders per-day rollups: confidence-bucket distribution, fallback rate, safety-rejection rate, median `top_score` (drift signal), `top_score` histogram, fallback-criteria breakdown (which of the 3 §4.3 criteria fired). Lives inside the existing Streamlit app — Streamlit auto-discovers `pages/` and adds a sidebar entry. No new infrastructure, no Grafana, no extra hosting. | `pages/observability.py` (new), maybe `.streamlit/config.toml` | logging (consumer side) |
| **13** | **Answerability calibration** (informed by Phase 12). Build a small "LLM-as-judge" script that, for each labelled test query, asks Claude "given these top-3 passages, can you answer fully / partially / not at all?" Treat that as a noisy answerability label. Re-derive thresholds against answerability rather than `is_selected`. | `scripts/probe_answerability.py` (new), `querylens/decision.py` (constants only) | thresholds (north-star) |
| **14** | **Multi-signal confidence**. Add `top3_mean_score` and `method_overlap` features to the classifier. Optionally combine via a small learned weight set. Recalibrate. | `querylens/decision.py`, `scripts/calibrate_thresholds.py` | thresholds (richer) |

**Phase ordering rationale.** Phases 1-11 ship the v1 Layer 2 (the
contract, the four mechanisms, calibration, prompts). Phase 12 makes
the system **visible to itself** — without it, every subsequent
improvement (Phase 13, Phase 14) is calibrated against the labelled
test set in the dark, with no idea what queries real users send. Phase
13 is the **single highest-leverage** improvement after observability —
it targets what we actually care about (answer quality), not what we
measure today (passage relevance). Crucially, Phase 13 is much
stronger when informed by Phase 12: instead of picking arbitrary test
queries to judge, we can sample from the actual production distribution
surfaced by `pages/observability.py`. Phase 14 follows because
multi-signal confidence only pays off once you have *the right target*
to calibrate against (which Phase 13 provides). Doing 14 first would
just multi-signal the wrong target; doing 13 without 12 would calibrate
against the wrong distribution.

---

## 7. File-change summary

After all phases, the following files exist or change:

| Path | Status | Purpose |
|---|---|---|
| `querylens/decision.py` | NEW | The DecisionLayer class and SearchDecision dataclass |
| `querylens/safety.py` | NEW | Injection-pattern detector |
| `querylens/prompts.py` | NEW | Central LLM prompts with safety rules |
| `streamlit_app.py` | MODIFIED | Wire validate_query → decide → render; use centralized prompts |
| `pages/observability.py` | NEW (Phase 12) | Streamlit observability page reading `decisions.jsonl` |
| `scripts/probe_answerability.py` | NEW (Phase 13) | LLM-as-judge answerability calibration |
| `tests/test_decision.py` | NEW | Unit tests, no models loaded |
| `scripts/calibrate_thresholds.py` | EXISTING (use) | Run once to derive threshold values |
| `results/decisions.jsonl` | NEW (runtime) | Telemetry output, gitignored |
| `docs/layer2_plan.md` | THIS FILE | Source of truth for the plan |
| `.gitignore` | MODIFIED | Add `results/decisions.jsonl` |

---

## 8. Connection to the larger project

| Layer 2 component | What it connects to |
|---|---|
| `DecisionLayer.decide()` | Receives Layer 1's output (cross-encoder scores) |
| `SearchDecision.should_generate_answer` | Gates Layer 3's Claude API call |
| `validate_query()` | Runs before Layer 1; saves compute on rejected queries |
| `_log()` → `results/decisions.jsonl` | Feeds `scripts/calibrate_thresholds.py` AND `pages/observability.py`. The same JSONL stream powers offline calibration (periodic) and live observability (continuous). |
| Threshold constants on `DecisionLayer` | Set by `scripts/calibrate_thresholds.py` after offline calibration |
| `querylens/safety.py` | Defends both Layer 1 (saves compute) and Layer 3 (prevents injection-driven LLM behaviour) |
| `querylens/prompts.py` | Used by Layer 3 to enforce safety at the LLM-prompt boundary |

Layer 2 is a **first-class system component**, not a few `if` statements
in app code. It has its own module, its own tests, its own contract,
its own calibration story.

---

## 9. Open decisions to make before/during build

A checklist of choices that will surface — list them here so they don't
get made silently during implementation.

- [ ] **Diversifier**: MMR (semantic, needs embeddings) or Jaccard (lexical, no deps)?
- [ ] **Number of confidence buckets**: 4 (high/medium/low/none) or different?
- [ ] **Fallback ladder depth**: stop at Dense Only, or also fall back to BM25 as third tier?
- [ ] **Seeded threshold values** before calibration (need something to ship the first version)
- [ ] **Score-gap ambiguity threshold** (default 0.5 in logit space — tunable)
- [ ] **Warning copy** for each confidence bucket and rejection reason
- [ ] **Telemetry retention**: rotate the JSONL file? Cap its size?
- [ ] **Whether to display a probability (sigmoid) alongside the confidence badge** for transparency
- [ ] **Pattern list scope** in safety.py (which categories of injection to cover initially)
- [ ] **What to log on rejected queries** vs accepted (truncation, format)
- [ ] **Cutoff between `should_show_results` and `should_generate_answer`** — proposal: show on high+medium+low, generate on high+medium only. Confirm or adjust.
- [ ] **Fallback gate coefficient** — the `× 0.7` in the agreement criterion §4.3(c). Tune from telemetry once available.
- [ ] **Whether to log per-criterion fallback decisions** (which of the 3 criteria passed/failed) — useful for analyzing fallback behaviour
- [ ] **Phase 13 timing** — run answerability calibration immediately after Phase 12 ships, or wait for a meaningful telemetry window (e.g., 1-2 weeks of decisions)? Proposal: wait so we can sample test queries from the actual production distribution surfaced by Phase 12.
- [ ] **LLM-judge model choice for Phase 13** — same Claude model that generates answers? Different model (e.g., a Sonnet judge for a Haiku generator) for orthogonality?
- [ ] **Observability page scope** (Phase 12) — read-only dashboard only, or also include filters (date range, confidence bucket) and a CSV download? Proposal: start read-only, add filters in a v2 once the page proves useful.
- [ ] **Observability page hosting** — same Streamlit Cloud app or separate? Proposal: same (zero ops, anyone with the URL can see it). If access control becomes a concern, gate behind a Streamlit secret.

---

## 10. Explicitly out of scope — Layer 4 (output validation)

Layer 2 ensures the LLM receives **good evidence**. It does NOT verify
that the LLM's **output** uses that evidence faithfully. The classic
RAG failure mode — *"the passage is relevant, but the LLM hallucinated
a specific number or misattributed a citation"* — slips through every
Layer 2 check no matter how strict, because Layer 2 stops before
generation.

That's a real gap, but it's a different layer's job. Naming it
explicitly so we don't confuse ourselves about what Layer 2 promises.

### What a future Layer 4 would cover

| Concern | What it does | Cost |
|---|---|---|
| **Citation validation** | After Claude generates, scan every `[N]` reference. Confirm the cited passage actually contains substring evidence for the surrounding claim. Flag or rewrite if not. | Low — pure text matching, no LLM call |
| **Faithfulness check** | Second LLM call asking "does this answer use only facts from these passages?" — score and gate. | Medium — extra LLM call per query |
| **Output guardrails** | Filter LLM output for PII leakage, refusal-bypass artefacts, model-identity reveals (mirror of `safety.py` for outputs). | Low — pattern matching |
| **Refusal training compliance** | Verify Claude actually refused when our LLM gate said it shouldn't have generated. Catches prompt-prompt-injection that bypassed Layer 2. | Low — output classifier |

### Why we're NOT building Layer 4 in this plan

1. **Scope discipline.** Layer 2 is already substantial. Conflating it
   with output validation makes both harder to reason about and harder
   to test.
2. **Different mechanism.** Layer 2 is rule-based logic on numpy. Layer
   4 is mostly text matching and possibly extra LLM calls. They share
   no infrastructure.
3. **Sequence matters.** Output validation only has signal once we have
   a working generation path. Layer 2 unblocks generation; Layer 4
   guards it. Building 4 before 2 ships is premature.

### Hand-off contract

When Layer 4 is built, it consumes:
- `SearchDecision` from Layer 2 (so it knows the confidence band)
- The LLM's raw output
- The passages that were passed to the LLM

And it produces:
- A validated output (possibly rewritten)
- A faithfulness score (logged to telemetry)
- A reject signal if the output is fundamentally untrustworthy

That contract is **forward-compatible** with the current `SearchDecision`
— no changes needed to add Layer 4 later.

---

## 11. How to use this plan

- **Adjustments**: edit this file directly. Mark resolved decisions in §9 with `[x]` and the chosen value.
- **Reference during build**: each phase in §6 maps to a specific section in §3 and §4. Cross-reference as you implement.
- **If a phase reveals a design issue**: stop, fix the plan first, then continue. Code should not diverge from the plan silently.
- **Where to put NEW phases**: insert into §6 in dependency order, update §7 file list and §3 runtime flow accordingly.
- **Doc rule**: this plan is a planning artefact, not a project log. After Layer 2 is shipped, this file can be replaced by an "implemented design" doc that describes the as-built system. The plan is for the journey, not the destination.
