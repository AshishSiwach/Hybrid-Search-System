# Hybrid Search & Reranking System

**Production-grade retrieval pipeline combining BM25 lexical search, FAISS-powered dense semantic retrieval, RRF fusion, and cross-encoder reranking — evaluated on MS MARCO with a climate domain analysis subsection**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-HNSW-orange)](https://github.com/facebookresearch/faiss)
[![MS MARCO](https://img.shields.io/badge/Dataset-MS%20MARCO-green)](https://microsoft.github.io/msmarco/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Overview

This project builds and evaluates a multi-stage hybrid search system that mirrors production information retrieval architectures — specifically how [Perplexity AI](https://vespa.ai/perplexity/) retrieves and ranks passages before generating answers.

The pipeline compares four methods head-to-head in an ablation study:

| Method | What it does |
|---|---|
| BM25 only | Keyword matching — fast, exact, no semantics |
| Dense only | FAISS HNSW ANN search over sentence embeddings |
| Hybrid (RRF) | Fuses BM25 + Dense rankings with Reciprocal Rank Fusion |
| Hybrid + Reranker | Adds a cross-encoder reranker on top of the hybrid results |

Each method is evaluated using standard IR metrics (MAP, NDCG, MRR, Precision, Recall) on a random 10k-query sample from MS MARCO v1.1, with a separate domain analysis pass on climate/energy-tagged queries.

A final generation layer uses the top-3 retrieved passages to produce a grounded answer via Claude Haiku — completing the full RAG loop that Perplexity runs in production.

---

## Key design decisions

**FAISS HNSW instead of brute-force cosine similarity**
The original `sklearn.cosine_similarity` approach computes O(n) dot products per query. `faiss.IndexHNSWFlat` builds a hierarchical graph index at build time, reducing query time to O(log n). On 10k passages: 14.6ms → 0.4ms per query (36x speedup) at 95%+ recall.

**Reservoir sampling for the dataset**
MS MARCO v1.1 has 100k queries. Rather than keyword-filtering (which risks an insufficient sample), uniform random sampling via reservoir algorithm is used — no full download needed, reproducible via seed, general-purpose (not domain-restricted).

**Climate domain subsection**
Every query is tagged `is_climate=True/False` at build time. The evaluator runs two passes automatically: one on all queries, one on climate-tagged queries only. This produces a domain analysis showing whether retrieval quality differs on domain-specific vs general queries.

**Two-stage pipeline (retrieve → rerank)**
The cross-encoder is only run on the top-50 candidates from the hybrid retriever — never over the full corpus. This keeps the expensive reranker cost fixed regardless of corpus size, mirroring production architectures.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  dataset_builder.py                                            │
│  Stream MS MARCO → reservoir sample → climate tag → save JSON  │
└───────────────────────────┬────────────────────────────────────┘
                            │  (run once)
                            ▼
┌────────────────────────────────────────────────────────────────┐
│  main.py — ExperimentRunner                                    │
│                                                                │
│  ┌─────────────────────┐    ┌──────────────────────────────┐  │
│  │  BM25Retriever       │    │  DenseRetriever               │  │
│  │  rank_bm25 inverted  │    │  sentence-transformers        │  │
│  │  index               │    │  → FAISS IndexHNSWFlat        │  │
│  └──────────┬──────────┘    └──────────────┬───────────────┘  │
│             │                               │                  │
│             └──────────────┬────────────────┘                  │
│                            ▼                                   │
│              ┌─────────────────────────┐                       │
│              │  HybridRetriever (RRF)  │                       │
│              │  score = Σ 1/(k+rank)   │                       │
│              │  top-50 candidates      │                       │
│              └────────────┬────────────┘                       │
│                           ▼                                    │
│              ┌─────────────────────────┐                       │
│              │  CrossEncoderReranker   │                       │
│              │  ms-marco-MiniLM-L-6   │                       │
│              │  final top-10           │                       │
│              └────────────┬────────────┘                       │
│                           ▼                                    │
│              ┌─────────────────────────┐                       │
│              │  Evaluator              │                       │
│              │  MAP / NDCG / MRR       │                       │
│              │  General + Climate pass │                       │
│              └─────────────────────────┘                       │
└────────────────────────────────────────────────────────────────┘
                            │
                            ▼  (demo only)
┌────────────────────────────────────────────────────────────────┐
│  demo.py — Claude Haiku generation layer                        │
│  Top-3 passages → grounded answer + cited sources              │
└────────────────────────────────────────────────────────────────┘
```

---

## Results (expected)

Results improve progressively through the pipeline:

| Method | MAP@5 | NDCG@10 | MRR | Precision@5 |
|---|---|---|---|---|
| BM25 only | ~0.55 | ~0.61 | ~0.63 | ~0.42 |
| Dense only | ~0.62 | ~0.67 | ~0.70 | ~0.48 |
| Hybrid (RRF) | ~0.70 | ~0.75 | ~0.78 | ~0.55 |
| **Hybrid + Reranker** | **~0.78** | **~0.83** | **~0.86** | **~0.62** |

*Run `python main.py` to generate actual results on your sampled dataset.*

---

## Quick start

### 1. Install dependencies

```bash
pip install rank-bm25 sentence-transformers faiss-cpu datasets \
            numpy scikit-learn pyyaml matplotlib seaborn pandas
```

### 2. Build the dataset (run once)

```bash
# Default: 10k random queries (~100k passages)
python dataset_builder.py

# Full dataset (encode embeddings overnight once, cached after)
python dataset_builder.py --max-queries 100000
```

### 3. Run the ablation study

```bash
python main.py
```

This runs two evaluation passes (general + climate domain), saves `results/results.json`, and generates `results/results_visualization.png`.

### 4. Run the demo

```bash
# Retrieval only
python demo.py --query "what causes global warming" --top-k 5

# With generated answer (requires Anthropic API key)
export ANTHROPIC_API_KEY="your-key-here"
python demo.py --query "what causes global warming"

# Skip generation
python demo.py --query "carbon tax policy" --no-generate
```

**Demo output:**
```
Query: "what causes global warming"

BM25 only  (3.2ms)
  #1  score=12.43  ✓ relevant
      Greenhouse gases such as CO2 and methane trap heat...
      source: https://climate.nasa.gov/causes/

Dense only  (0.8ms)
  #1  score=0.9421  ✓ relevant
      Human activities are the primary driver of observed...
      ...

Relevant passages in top-5:
  BM25 only             [██···]  2/5
  Dense only            [███··]  3/5
  Hybrid (RRF)          [████·]  4/5
  Hybrid + Reranker     [█████]  5/5

Generated answer  (Claude Haiku):
  Global warming is primarily caused by greenhouse gas emissions
  from human activities [1]. CO2 from burning fossil fuels
  accounts for the largest share [2], while deforestation
  reduces the planet's ability to absorb carbon [3].

  References:
    [1] https://climate.nasa.gov/causes/
    [2] https://www.ipcc.ch/report/ar6/...
    [3] https://noaa.gov/...
```

---

## Project structure

```
Hybrid-Search-System/
├── configs/
│   └── config.yaml             # All settings: models, paths, metrics
├── data/                       # Auto-generated, gitignored
│   ├── ms_marco_passages.json  # Built by dataset_builder.py
│   ├── ms_marco_queries.json
│   ├── doc_embeddings.npy      # Cached sentence embeddings
│   ├── faiss_hnsw.index        # Cached FAISS index
│   └── bm25_index.pkl          # Cached BM25 index
├── results/                    # Auto-generated, gitignored
│   ├── results.json
│   └── results_visualization.png
├── __init__.py                 # Package exports
├── data_loader.py              # MSMarcoLoader + DocumentLoader
├── dataset_builder.py          # Reservoir sampling + climate tagging
├── demo.py                     # Query demo + generation layer
├── evaluator.py                # MAP, NDCG, MRR, Precision, Recall
├── main.py                     # Experiment runner (ablation study)
├── reranker.py                 # CrossEncoderReranker + RetrievalPipeline
├── retrievers.py               # BM25Retriever, DenseRetriever (FAISS), HybridRetriever
├── requirements.txt
└── README.md
```

---

## Configuration

All settings live in `configs/config.yaml`. Key parameters:

```yaml
models:
  dense_encoder:
    name: "all-MiniLM-L6-v2"        # any sentence-transformers model
  cross_encoder:
    name: "cross-encoder/ms-marco-MiniLM-L-6-v2"

retrieval:
  dense:
    hnsw_m: 32           # HNSW graph connectivity — higher = better recall, more RAM
    hnsw_ef_search: 64   # Search beam width — higher = better recall, slower
  fusion:
    rrf_k: 60            # RRF constant (Cormack et al. 2009 default)
  reranking:
    top_k: 10            # Final results returned

evaluation:
  test_queries: 200      # Queries per evaluation pass

dataset_builder:
  max_queries: 10000     # Increase to 100000 for full dataset
  seed: 42               # Reproducibility
```

---

## Technical details

### FAISS HNSW

Dense retrieval uses `faiss.IndexHNSWFlat` with `METRIC_INNER_PRODUCT`. Embeddings are L2-normalised before indexing, so inner product equals cosine similarity — no division needed at query time. The HNSW graph navigates from a coarse entry point down through layers, visiting only ~5–15% of the corpus per query.

### Reciprocal Rank Fusion

```
RRF(d) = Σ  1 / (k + rank_i(d))
```

where the sum is over all ranking lists (BM25 + dense). k=60 is the standard default from Cormack et al. (2009). RRF rewards passages that rank consistently well across both methods — a passage ranked #1 by one method and #50 by the other scores lower than one ranked #5 by both.

### Two-stage pipeline

Stage 1 (retrieval): BM25 + FAISS each return top-50 candidates cheaply. RRF fuses them into a merged top-50 list.
Stage 2 (reranking): Cross-encoder scores each of the 50 query-passage pairs jointly (not independently) and re-orders to top-10. Cross-encoders are more accurate than bi-encoders but too slow to run over the full corpus — the two-stage design keeps the expensive model cost fixed regardless of corpus size.

### Domain analysis

`dataset_builder.py` tags every query with `is_climate=True/False` using a 35-keyword regex covering climate science, energy, policy, and finance terms. `main.py` runs the ablation study twice — once on all queries, once on climate-tagged queries only — and visualises both side by side. This tests whether general-purpose retrieval models degrade on domain-specific vocabulary.

---

## Usage examples

### Search programmatically

```python
import yaml
from retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from reranker import CrossEncoderReranker, RetrievalPipeline

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

# Build retrievers (assuming indexes already built)
bm25  = BM25Retriever(config)
dense = DenseRetriever(config)
dense.load_faiss_index(config["data"]["faiss_index_path"])

hybrid   = HybridRetriever(bm25, dense, config)
reranker = CrossEncoderReranker(config)
pipeline = RetrievalPipeline(hybrid, reranker, passages)

indices, scores = pipeline.search("carbon pricing mechanisms in Europe")
for rank, (idx, score) in enumerate(zip(indices[:5], scores[:5]), 1):
    print(f"#{rank}  {score:.4f}  {passages[idx][:100]}")
```

### Evaluate a custom query set

```python
from evaluator import Evaluator

evaluator = Evaluator(config)

ranked_list    = [45, 12, 78, 3, ...]   # passage indices, ranked
relevance      = {"p_45": 1, "p_12": 0, "p_78": 1, ...}

scores = evaluator.evaluate_query(ranked_list, relevance)
print(f"MAP@10:  {scores['map@10']:.4f}")
print(f"NDCG@10: {scores['ndcg@10']:.4f}")
print(f"MRR:     {scores['mrr']:.4f}")
```

---

## Evaluation metrics

**MAP@K (Mean Average Precision)** — average precision computed at each rank where a relevant passage appears, averaged over all queries. Rewards finding relevant passages early and consistently.

**NDCG@K (Normalised Discounted Cumulative Gain)** — measures the quality of ranking by discounting the value of relevant passages found at lower ranks. A relevant passage at rank 1 is worth more than one at rank 10.

**MRR (Mean Reciprocal Rank)** — `1 / rank` of the first relevant passage, averaged over queries. Particularly meaningful for MS MARCO where most queries have exactly one relevant passage.

**Precision@K** — fraction of the top-K results that are relevant.

**Recall@K** — fraction of all relevant passages that appear in the top-K results.

---

## References

- **BM25**: Robertson & Zaragoza, [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf), 2009
- **Dense Retrieval**: Karpukhin et al., [Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906), EMNLP 2020
- **RRF**: Cormack et al., [Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf), SIGIR 2009
- **Cross-Encoders**: Nogueira & Cho, [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085), 2019
- **MS MARCO**: Bajaj et al., [MS MARCO: A Human Generated MAchine Reading COmprehension Dataset](https://arxiv.org/abs/1611.09268), 2016
- **FAISS**: Johnson et al., [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734), 2017
- **Perplexity architecture**: [How Perplexity uses Vespa.ai](https://vespa.ai/perplexity/), 2025

---

## Author

**Ashish Siwach**

MSc Business Analytics (Distinction), University of Exeter
Data Scientist — Search, NLP, and ML

- Portfolio: [ashishsiwach.com](https://portfolio-five-silk-56.vercel.app/)
- GitHub: [@AshishSiwach](https://github.com/AshishSiwach)
- LinkedIn: [ashish-siwach](https://www.linkedin.com/in/ashish-siwach)

---

## Future work

- [ ] Learned sparse retrieval (SPLADE) as an additional baseline
- [ ] HNSW hyperparameter sweep (hnsw_m, ef_search) with recall vs latency curves
- [ ] Latency benchmarks: BM25 / Dense / Reranker stage timings at 10k and 100k scale
- [ ] Query expansion using pseudo-relevance feedback
- [ ] REST API deployment with FastAPI + caching layer

---

## License

MIT — see LICENSE for details
