# How to Think About Layer 2

> A thinking guide, not a build log. Re-read this in six months and you should
> be able to re-derive the QueryLens decision layer from scratch — or design
> one for a different system you've never seen before.

---

## 1. The pattern in one sentence

Any ML system whose output drives downstream action needs a **decision layer**
between prediction and action. Without it, your pipeline does whatever the
model says — including when the model is confidently wrong.

```
predict  →  decide  →  act
  (ML)      (rules)    (side effect)
```

- **predict**  produces raw signals: scores, logits, embeddings, generated text.
- **decide**   maps those signals to a course of action using explicit policy.
- **act**      executes the side effect: call an LLM, hit an API, update UI.

Most ML projects collapse this into **predict → act**. That's the mistake.

---

## 2. Why most projects skip Layer 2 (and why that's a bug)

In a research notebook or demo, the failure modes are invisible because you
only test on inputs that work. The pipeline looks like this:

```
query → retrieve → rerank → LLM → answer
```

Demo-day questions hit the happy path. The model is "right enough." You ship.

In production the long tail hits you:

| Input | predict → act result | What should have happened |
|---|---|---|
| "asdf" | LLM generates a confident, fabricated answer | Refuse — no evidence |
| "what is photosynthesis" | 5 near-identical paraphrases of one Wikipedia paragraph | Diversify sources |
| A rare query the model has no signal for | LLM fills the gap with hallucination | Fall back to keyword search, or refuse |
| Latency spike on dense encoder | Empty results, blank UI | Fall back to BM25, surface degradation |

Every one of those failures is a **policy decision** the predict layer can't
make on its own. Predict layers produce signals. Policy lives elsewhere.
That elsewhere is Layer 2.

---

## 3. How to find what YOUR Layer 2 should do

Don't start from "what features should I add to the decision layer." Start
from failure modes. Run this exercise on any ML system:

### Exercise: the three questions

**Q1. What can go wrong silently?**
What outputs *look* fine but *are* wrong?
→ These need **confidence checks**.

**Q2. What gives the same input two reasonable but redundant outputs?**
Where can the predict layer return correct-but-useless results?
→ These need **diversity / deduplication checks**.

**Q3. What's your plan when the primary method fails?**
Do you have a fallback or do you just emit bad output?
→ These need **fallback triggers**.

For QueryLens the answers were:

| Question | QueryLens answer |
|---|---|
| Silent failures? | Cross-encoder returns a score for any passage — even off-topic ones. We were piping those scores to Claude without checking magnitude. |
| Redundant outputs? | Reranker happily returns 5 near-duplicate passages from the same domain when the corpus has overlapping sources. |
| No-plan failures? | If reranker thinks nothing is relevant, we had no plan — just an empty results list with no UI signal. |

Those three answers map *exactly* to the three responsibilities the decision
layer ended up with: confidence thresholds, diversity filter, fallback trigger.

This is the core reasoning move: **map failure modes to layer responsibilities.**

---

## 4. The three primitives, explained from first principles

### Confidence thresholds

Your model produces a score. Score ranges are model-specific:

| Output type | Range | Example model |
|---|---|---|
| Probability | [0, 1] | Most classifiers |
| Logit | (−∞, +∞) | Cross-encoders, raw heads |
| Cosine similarity | [−1, 1] (often [0, 1]) | Sentence-transformer dense retrievers |
| Distance | [0, ∞) | FAISS L2 |

You need to map this continuous signal to **discrete actions**:

- "definitely yes" → proceed normally
- "probably yes" → proceed with a caveat in the UI
- "ambiguous" → use fallback method
- "no" → don't act, surface failure to user

The mapping is just N−1 thresholds. Picking those thresholds is where most
people lose. Two failure modes:

1. **Guess and ship.** Pick round numbers, forget to recalibrate. Six months
   later your "high confidence" results are 60% irrelevant because the corpus
   shifted.
2. **Skip the layer entirely.** Pass raw scores to the next stage and hope.

The cure for both: **derive thresholds from labelled data**. For QueryLens
this is [`calibrate_thresholds.py`](calibrate_thresholds.py). The recipe:

1. Collect (score, is_relevant) pairs from your eval data — one pair per
   (query, candidate) the model scored.
2. Plot the score distribution split by label (relevant vs irrelevant).
   Two distributions that barely overlap → thresholds are easy. Heavily
   overlapping distributions → your predict layer is weak, fix that first.
3. Plot precision, recall, F1 vs threshold.
4. Pick operating points:
   - HIGH at the smallest threshold where precision ≥ 0.80
   - MEDIUM at the F1 maximum
   - LOW at the largest threshold still achieving recall ≥ 0.80

Each operating point is a **business decision**, not a mathematical one:
- "I want HIGH to mean 8 in 10 results are actually relevant" → precision target
- "I want MEDIUM to balance both" → F1
- "I want LOW to catch most relevant items" → recall target

Tune the targets to match your product's risk profile. A medical-search
product wants precision ≥ 95%. A news-feed wants recall.

**Thresholds are not the only confidence signal.** A top score with a
microscopic lead over #2 is not really high confidence — it's a tie that
happens to have a winner. Use the **score gap** between #1 and #2 as a
second axis:

- If `score[0] - score[1] < ε` → demote one tier, flag as ambiguous.
- ε is calibratable the same way: sweep it on labelled data, find the
  point where "winners with small gaps" actually correlate with relevance.

This catches a failure mode raw thresholds miss: five equally-plausible
top results that all clear the HIGH bar. The model isn't confident; it's
indecisive. The user should see "ambiguous results" instead of a
falsely-confident answer.

**Different score scales need different threshold sets.** If your fallback
path uses a different model (say, a bi-encoder cosine instead of a
cross-encoder logit), you cannot reuse the primary thresholds. Calibrate
each path separately or use rank-only signals.

### Diversity / deduplication

Most retrieval models score each item independently. The cross-encoder
doesn't know that result #2 is a paraphrase of result #1 — it just scores
each query–passage pair on its own.

Result: a user asking "what is photosynthesis" gets 5 near-paraphrases of
the same Wikipedia paragraph, when they wanted 5 different angles.

Two distinct problems, two distinct solutions:

- **Source diversity**: cap N results per domain / author / publisher.
  Normalize the source first (strip subdomains, lowercase) or you'll be
  fooled by `en.wikipedia.org` vs `simple.wikipedia.org`.
- **Content diversity**: catch passages that say the same thing in
  different words. Three real options with very different trade-offs:

| Method | Catches | Misses | Cost |
|---|---|---|---|
| **Jaccard on tokens** | Near-exact duplicates (scraped web pages, copy-paste) | Paraphrases — two passages can convey identical info with almost no shared vocabulary | Microseconds, no model |
| **MinHash on shingles** | Same as Jaccard, plus near-paraphrases at scale | Pure semantic restatements | Microseconds-to-milliseconds |
| **MMR with cosine on dense embeddings** | Paraphrases, semantic duplicates, *and actively promotes coverage* across distinct angles | Almost nothing in practice | O(N·k) cosine ops; needs embeddings on hand |

**The default should be MMR** when you already have embeddings on hand
(most modern retrieval pipelines do). Redundancy in real corpora is mostly
paraphrase, not exact duplication, and MMR is the only one of the three
that does *more than filter* — it actively picks for coverage.

MMR's logic: at each step, pick the candidate that maximises
```
λ * relevance(i) - (1 - λ) * max_cosine(i, already_picked)
```
λ is another calibratable knob — same recipe as score thresholds. Sweep
it on labelled data, measure the relevance-vs-diversity trade-off, pick
the operating point that matches your product.

Keep Jaccard as a **zero-dependency fallback** for when embeddings aren't
available or affordable (CPU-only deployment, cost-sensitive batch jobs).
Don't ship a system that has nothing if the embeddings fail to load.

### Fallback triggers

When your primary path fails, what's plan B?

Two patterns, often combined:

- **Method ladder**: try Method A → if confidence low, try Method B →
  if still low, try Method C → if still low, refuse.
- **Confidence gating**: only invoke expensive downstream operations when
  confidence clears a bar. (For QueryLens, this is the LLM gate.)

The trap: blind fallback. "If primary failed, swap in secondary" without
checking if secondary is actually better. You can end up confidently
serving worse results. Always re-check confidence after the swap, using
thresholds appropriate to the new method's score type.

---

## 5. The QueryLens decision layer as a worked example

Putting it together for this specific project:

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 1 — Predict                                               │
│  Hybrid retrieval (BM25 + Dense via RRF) + cross-encoder rerank │
│  Output: top-50 (idx, score) pairs; scores are cross-encoder    │
│          logits, roughly [−6, +10] on MS MARCO                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  Layer 2 — Decide   (decision.py)                                │
│                                                                  │
│  ⓪ validate_query()  — cheap pre-flight before any retrieval.    │
│     Reject empty / too-short / too-long / pure-punctuation.      │
│     If rejected, skip Layer 1 entirely.                          │
│                                                                  │
│  ① Read top score + score_gap.                                   │
│     Classify into high/medium/low/none using calibrated          │
│     cross-encoder thresholds.                                    │
│                                                                  │
│  ② Score-gap demotion. If score[0]−score[1] < 0.5,               │
│     demote one tier and flag ambiguous=True.                     │
│                                                                  │
│  ③ Diversity filter. MMR with cosine on dense embeddings         │
│     (default) or Jaccard on tokens (fallback when embeddings     │
│     unavailable). Domain cap of 2 either way.                    │
│                                                                  │
│  ④ Dense-aware fallback. If confidence=none, classify Dense      │
│     Only top score using ITS OWN cosine thresholds. Only         │
│     swap if dense clears its own "none" bar; report the          │
│     actual dense confidence, not a hardcoded value.              │
│                                                                  │
│  ⑤ Telemetry. Append decision to results/decisions.jsonl —       │
│     never crashes the request even on disk failure.              │
│                                                                  │
│  Emit SearchDecision { confidence, should_generate_answer,       │
│     final_results, warning, fallback_triggered, ambiguous, ... } │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  Layer 3 — Act    (streamlit_app.py)                             │
│  if decision.should_generate_answer:                             │
│      call Claude with decision.final_results[:3]                 │
│  Always render: confidence badge, warning (if any), top results  │
└──────────────────────────────────────────────────────────────────┘
```

The contract between Layer 2 and Layer 3 is a single dataclass. Layer 3
does not look at scores, does not look at the corpus, does not make any
policy decisions. It just renders what Layer 2 decided.

**An architectural note worth naming.** Step ④ uses *different threshold
constants* (`DENSE_HIGH/MEDIUM/LOW`) than step ①. This is not a violation
of rule #1 ("no models in the decision layer") — both threshold sets are
pure numbers. It's the rule applied per score-type: cross-encoder logits
and dense cosines live in incompatible scales, so they need separate
threshold sets, calibrated independently. The cost of conflating them is
silent — your "low confidence" message hides whether the system is
genuinely confident or just guessing.

---

## 6. Architectural rules that keep this clean as you grow

These three rules apply to any layered system, not just QueryLens.

### Rule 1 — No models in the decision layer

[`decision.py`](decision.py) has zero ML dependencies. It operates on
numpy arrays and Python dicts. If your decision layer has to load a model,
that model is doing **prediction**, not decision-making, and it belongs in
a separate predict layer.

Test: can you `import` your decision module in milliseconds? If yes, good.
If it takes seconds because a transformer is loading, you've mixed layers.

### Rule 2 — Structured output, not flags

Return a dataclass, not a tuple of bools or a dict:

```python
# BAD — adding any field is a breaking change
return (confidence, should_generate, final_results, warning)

# GOOD — adding fields is non-breaking
return SearchDecision(
    confidence=...,
    should_generate_answer=...,
    final_results=...,
    warning=...,
    fallback_triggered=...,
    ...
)
```

Six months from now you'll want to add a `score_gap` field or a
`debug_info` blob. With a dataclass, every consumer keeps working.

### Rule 3 — Read upstream data, never mutate it

The cross-encoder scores in `best_results` are immutable to Layer 2. You
can filter them, re-rank them, replace them with fallback results — but
you don't change the underlying score values. If you need a transformed
view, copy it.

This keeps debugging tractable: when something is wrong, you can compare
"what Layer 1 produced" against "what Layer 2 chose" without wondering
whether Layer 2 silently rewrote the inputs.

---

## 7. Adding new layers in the future

The pattern is recursive. Each new layer follows the same
predict → decide → act structure with whatever came before as its predict.

### Examples of layers you might add to QueryLens

| Layer | Position | What it does |
|---|---|---|
| Query understanding | Before Layer 1 | Rewrite, expand, classify intent. Strip PII. |
| Answer verification | After Layer 3 | Check that every citation in the generated answer actually appears in the source passages. Re-generate or refuse if not. |
| Personalization | After Layer 1 or Layer 2 | Re-rank using user history / preferences. |
| Safety | Around Layer 1 / Layer 3 | Filter toxic queries pre-retrieval. Filter unsafe outputs post-generation. |
| Caching | Around Layer 1 | Memoize decisions for repeated queries. |
| Telemetry | Around Layer 2 | Log every decision for offline analysis. Not really a "layer" but a cross-cutting concern. |
| Testing | Cross-cutting | Not a layer. Lives in `tests/`. Crucial property: tests for the decision layer should load **zero models**. The whole suite runs in milliseconds on any laptop, on every commit. If your decision layer needs a GPU to test, you've mixed prediction into it — refactor. |

### Integration recipe for a new layer

1. **Define its inputs.** What does it consume? (Usually the output of
   the layer immediately upstream.)
2. **Define its output as a dataclass.** Make it the explicit contract.
3. **Place it in the chain.** Either before or after an existing layer;
   never inside one.
4. **Update only the adjacent layers.** A well-designed system means
   adding Layer 4 only touches Layer 3 (which feeds it) and the new
   Layer 4 itself. The original Layers 1 and 2 don't change.
5. **Calibrate any thresholds from data**, never from intuition.
6. **Add telemetry on day one.** You'll need it to know if the new layer
   is helping.

If adding a new layer requires you to modify every existing layer, your
contracts between layers are leaky. Fix the contracts first.

---

## 8. The bigger picture — where this fits in end-to-end ML engineering

A complete production ML system has more than three layers. Here's the
fuller stack, with QueryLens's three highlighted:

```
   Layer 0  Data         (dataset_builder.py — corpus, labels, splits)
   Layer 1  Predict      (retrievers.py + reranker.py)
   Layer 2  Decide       (decision.py — the focus of this guide)
   Layer 3  Act          (streamlit_app.py LLM call)
   Layer 4  Verify       (not yet — citation check would live here)
   Layer 5  Observe      (partial — JSONL telemetry from Layer 2 lands in
                          results/decisions.jsonl; no dashboards yet)
   Layer 6  Calibrate    (calibrate_thresholds.py — offline, periodic)
   Layer 7  Evaluate     (evaluator.py + main.py — offline, periodic)
   Layer X  Tests        (tests/test_decision.py — cross-cutting, runs on
                          every commit, no models loaded)
```

Layers 0–4 run in the live request path. Layers 6–7 run offline. Layer 5
spans both: it logs from the live path and feeds dashboards / alerts. Layer
X is not really a layer — it's a property of the others.

Most early-stage projects have 0, 1, 3, 7. Adding 2, 5, 6 is the jump from
demo to production. The thinking framework in this doc is what makes that
jump tractable.

**An honest note on partial layers.** Layer 5 above is "partial" — the
telemetry exists, the dashboards don't. That's a deliberate choice: the
JSONL gives you the *signal*, and the question "is it worth building a
dashboard?" can now be answered by reading the JSONL. Building the
dashboard before reading the data would be premature. This is what
"telemetry-driven engineering" actually looks like in practice — observe
first, optimise the observation second.

---

## 9. Replicating this on a different project — a checklist

When you start the next AI project, before writing the prompt or picking a
model, walk this list:

- [ ] **List failure modes.** Not "what could the model get wrong" but
      "what would the user actually see if it went wrong." Each is a
      candidate for a decision check.
- [ ] **Categorize failures** into the three buckets:
      silent / redundant / no-fallback.
- [ ] **Map each failure to a check** in Layer 2.
- [ ] **Define your output contract** as a dataclass before writing logic.
- [ ] **Implement the logic** with pure functions, no models loaded.
- [ ] **Calibrate thresholds from labelled data.** Never ship guesses for
      checks that fire on most queries. Deferral is fine for checks that
      fire on the tail — but document the deferral and its trigger.
- [ ] **One threshold set per score scale.** Different models live in
      different ranges. Don't reuse cross-encoder thresholds on cosine.
- [ ] **Add telemetry at the moment of decision**, not after. Structured
      JSONL is enough — you can build the dashboard later, but you can't
      reconstruct decisions you didn't log.
- [ ] **Gate the act layer** on the decision output, never on raw scores.
- [ ] **Write tests that load zero models.** Edge cases: empty input,
      single result, all-same-domain, all-below-threshold, NaN scores,
      each diversifier path, each fallback scenario. If your tests need a
      GPU, your decision layer has leaked into prediction. Refactor.
- [ ] **Document the layer's contract**, not its implementation. Code
      changes; contracts shouldn't.

If you can tick every box, you've built a Layer 2 worth shipping.

---

## 10. The single biggest lesson

**Treat the decision layer as a first-class system component, not as
a few `if` statements scattered through your app code.**

The moment you give it a name (`DecisionLayer`), a contract
(`SearchDecision`), and its own file (`decision.py`), three things happen:

1. You can test it without loading models.
2. You can swap it without touching predict or act.
3. You can reason about your system as a chain of well-typed transformations,
   not as a tangle of conditionals.

That shift — from ad-hoc rules to a named, contracted layer — is the
end-to-end engineering move. Everything else in this document is detail.
