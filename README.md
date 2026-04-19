# Hybrid Search & Reranking System

**Production-grade retrieval pipeline combining BM25 lexical search, FAISS-powered dense semantic retrieval, RRF fusion, and cross-encoder reranking — evaluated on MS MARCO with a climate domain analysis subsection**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-HNSW-orange)](https://github.com/facebookresearch/faiss)
[![MS MARCO](https://img.shields.io/badge/Dataset-MS%20MARCO-green)](https://microsoft.github.io/msmarco/)
[![Live Demo](https://img.shields.io/badge/Live-QueryLens-blueviolet)](https://ashishsiwach-querylens.streamlit.app)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Overview

This project builds and evaluates a multi-stage hybrid search system that mirrors production information retrieval architectures — specifically how [Perplexity AI](https://vespa.ai/perplexity/) retrieves and ranks passages before generating answers.

A live demo is deployed at **[ashishsiwach-querylens.streamlit.app](https://ashishsiwach-querylens.streamlit.app)** — search any query, get ranked passages and an AI-generated answer grounded in the retrieved results.

The pipeline compares **four methods** head-to-head in a full ablation study:

| Method | What it does |
|---|---|
| **BM25 only** | Keyword matching — fast, exact, no semantics |
| **Dense only** | FAISS HNSW ANN search over sentence embeddings |
| **Hybrid (RRF)** | Fuses BM25 + Dense rankings with Reciprocal Rank Fusion |
| **Hybrid + Reranker** | Adds a cross-encoder reranker on top of the hybrid results |

Each method is evaluated using standard IR metrics (**MAP, NDCG, MRR, Precision, Recall**) on **20,000 MS MARCO queries (~164,000 passages)**, with a separate climate domain analysis pass on climate/energy-tagged queries.

---

## Results

Evaluated on **200 randomly sampled queries** from the 20k-query corpus.

### General evaluation (all queries)

| Method | MAP@5 | NDCG@10 | MRR | Precision@5 | Recall@10 |
|---|---|---|---|---|---|
| BM25 only | 0.245 | 0.358 | 0.292 | 0.102 | 0.619 |
| Dense only | 0.444 | 0.570 | 0.483 | 0.158 | 0.873 |
| Hybrid (RRF) | 0.381 | 0.500 | 0.425 | 0.147 | 0.797 |
| **Hybrid + Reranker** | **0.510** | **0.626** | **0.540** | **0.177** | **0.914** |

### Climate domain subsection

| Method | MAP@5 | NDCG@10 | MRR | Precision@5 | Recall@10 |
|---|---|---|---|---|---|
| BM25 only | 0.255 | 0.351 | 0.291 | 0.101 | 0.592 |
| Dense only | 0.349 | 0.488 | 0.397 | 0.132 | 0.824 |
| Hybrid (RRF) | 0.323 | 0.445 | 0.374 | 0.118 | 0.729 |
| **Hybrid + Reranker** | **0.510** | **0.626** | **0.540** | **0.170** | **0.915** |

### Key findings

- **Dense > Hybrid RRF** — Dense consistently outperforms Hybrid RRF because RRF dilutes the stronger Dense signal when one method is significantly better than the other. Fusion hurts when the two inputs are asymmetric in quality.
- **Reranker fixes RRF dilution** — The cross-encoder reranker re-scores all candidates jointly from scratch, bypassing the RRF weakness entirely and delivering the best results on every metric.
- **91.4% Recall@10 on 164k passages** — The system surfaces 9 out of every 10 relevant passages in the top 10 results. For a RAG generation layer, this means near-complete information coverage on every query.
- **Precision@5 ceiling is 0.20** — MS MARCO has only 1 relevant passage per query by design, making 0.20 the theoretical maximum. The system achieves 0.177, meaning the relevant passage appears in the top 5 **~88% of the time**.
- **Climate domain matches general performance** — Hybrid + Reranker achieves identical MAP@5 (0.510) and near-identical Recall@10 (0.915) on climate-specific queries, confirming the pipeline generalises across domains without domain-specific tuning.

---

## Dataset & Build Methodology

The corpus was built on **Google Colab (T4 GPU)** for efficiency using `colab_build.py`.

**Corpus statistics:**
- **20,000 queries** from MS MARCO v1.1 train split
- **164,320 passages** indexed
- Climate/energy queries tagged using a **46-keyword regex** (35 topic areas)

**Build steps on Colab T4 GPU:**

| Step | Time |
|---|---|
| Stream 20k queries from HuggingFace | ~5 min |
| Encode 164k passages (`all-MiniLM-L6-v2`, `batch_size=512`) | ~12 min |
| Build FAISS HNSW index (`M=32`, `efConstruction=200`) | ~11 min |
| Build BM25Okapi index | ~3 min |
| **Total** | **~31 min** |

**Why GPU?** Encoding 164k passages at `batch_size=512` takes ~12 minutes on a T4 GPU vs ~2+ hours on CPU. The FAISS HNSW index build is CPU-based regardless.

Embeddings are **L2-normalised** before indexing so inner product equals cosine similarity at zero extra cost. All 5 artefacts are uploaded to a private HuggingFace dataset and downloaded automatically on Streamlit Cloud at first run.

To reproduce locally instead of Colab:
```bash
python dataset_builder.py --max-queries 20000
python main.py
```

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  colab_build.py  (or dataset_builder.py locally)               │
│  Stream MS MARCO → tag climate queries → encode on GPU         │
│  → passages JSON + embeddings + FAISS + BM25 indexes           │
└───────────────────────────┬────────────────────────────────────┘
                            │  (built once, stored on HuggingFace)
                            ▼
┌────────────────────────────────────────────────────────────────┐
│  main.py — ExperimentRunner                                    │
│                                                                │
│  ┌─────────────────────┐    ┌──────────────────────────────┐  │
│  │  BM25Retriever       │    │  DenseRetriever               │  │
│  │  rank_bm25 index     │    │  sentence-transformers        │  │
│  │                      │    │  → FAISS IndexHNSWFlat        │  │
│  └──────────┬──────────┘    └──────────────┬───────────────┘  │
│             └──────────────┬────────────────┘                  │
│                            ▼                                   │
│              ┌─────────────────────────┐                       │
│              │  HybridRetriever (RRF)  │                       │
│              │  score = Σ 1/(k+rank)   │                       │
│              └────────────┬────────────┘                       │
│                           ▼                                    │
│              ┌─────────────────────────┐                       │
│              │  CrossEncoderReranker   │                       │
│              │  ms-marco-MiniLM-L-6   │                       │
│              └────────────┬────────────┘                       │
│                           ▼                                    │
│              ┌─────────────────────────┐                       │
│              │  Evaluator              │                       │
│              │  MAP / NDCG / MRR       │                       │
│              │  General + Climate pass │                       │
│              └─────────────────────────┘                       │
└────────────────────────────────────────────────────────────────┘
                            │
                            ▼  (live demo)
┌────────────────────────────────────────────────────────────────┐
│  streamlit_app.py (QueryLens)                                  │
│  Downloads indexes from HuggingFace on first run               │
│  Top-5 results + Claude Haiku generated answer                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install rank-bm25 sentence-transformers faiss-cpu datasets \
            numpy scikit-learn pyyaml matplotlib seaborn pandas \
            huggingface_hub streamlit
```

### 2a. Build dataset on Colab (recommended)

Open `colab_build.py` in Google Colab with a **T4 GPU** runtime. Run all cells (~31 min). Downloads all 5 data files on completion — place them in your `data/` folder.

### 2b. Build dataset locally (CPU)

```bash
python dataset_builder.py --max-queries 20000  # match Colab build (~2hr CPU)
python dataset_builder.py --max-queries 10000  # quick local test (~12 min)
```

### 3. Run the ablation study

```bash
python main.py
```

Outputs: `results/results.json` and `results/results_visualization.png`

### 4. Run the live demo

```bash
# Optional: set API key for AI-generated answers
set ANTHROPIC_API_KEY=sk-ant-...

python -m streamlit run streamlit_app.py
```

Or visit the deployed version: **[ashishsiwach-querylens.streamlit.app](https://ashishsiwach-querylens.streamlit.app)**

---

## Project Structure

```
Hybrid-Search-System/
├── configs/
│   └── config.yaml             # All settings: models, paths, metrics
├── data/                       # Built by colab_build.py or dataset_builder.py
│   ├── ms_marco_passages.json  # 164k passages
│   ├── ms_marco_queries.json   # 20k queries with is_climate flag
│   ├── doc_embeddings.npy      # 384-dim sentence embeddings
│   ├── faiss_hnsw.index        # FAISS HNSW index (M=32, efConstruction=200)
│   └── bm25_index.pkl          # BM25Okapi index
├── results/
│   ├── results.json
│   └── results_visualization.png
├── __init__.py
├── colab_build.py              # GPU build script (Google Colab T4)
├── data_loader.py              # MSMarcoLoader + I/O helpers
├── dataset_builder.py          # Local CPU dataset builder
├── demo.py                     # CLI demo with generation
├── evaluator.py                # MAP, NDCG, MRR, Precision, Recall
├── find_queries.py             # Find good demo queries from dataset
├── main.py                     # Ablation study runner
├── reranker.py                 # CrossEncoderReranker + RetrievalPipeline
├── retrievers.py               # BM25, DenseRetriever (FAISS), HybridRetriever
├── streamlit_app.py            # QueryLens — deployed on Streamlit Cloud
├── upload_to_hf.py             # Upload data files to HuggingFace
└── requirements.txt
```

---

## Technical Details

### FAISS HNSW

Dense retrieval uses `faiss.IndexHNSWFlat` with `METRIC_INNER_PRODUCT`. Embeddings are **L2-normalised** before indexing so inner product equals cosine similarity. Build parameters: **`M=32`**, **`efConstruction=200`**. Query-time: `efSearch=64`. On 164k vectors: **~1ms per query**.

### Reciprocal Rank Fusion

```
RRF(d) = Σ  1 / (k + rank_i(d))
```

**k=60** is the standard default from Cormack et al. (2009). Rewards passages that rank consistently well across both BM25 and Dense — a passage ranked #1 by one method and #50 by the other scores lower than one ranked #5 by both.

### Two-Stage Pipeline

**Stage 1 (retrieval):** BM25 + FAISS each return top-50 candidates cheaply. RRF fuses into a merged top-50 list.

**Stage 2 (reranking):** Cross-encoder scores each of the 50 query-passage pairs jointly and re-orders to top-10. Cross-encoders are more accurate than bi-encoders but too slow over the full corpus — the two-stage design keeps reranker cost **fixed regardless of corpus size**.

---

## Author

**Ashish Siwach** — MSc Business Analytics (Distinction), University of Exeter

- Live demo: [ashishsiwach-querylens.streamlit.app](https://ashishsiwach-querylens.streamlit.app)
- Portfolio: [ashishsiwach.com](https://portfolio-five-silk-56.vercel.app/)
- GitHub: [@AshishSiwach](https://github.com/AshishSiwach)
- LinkedIn: [ashish-siwach](https://www.linkedin.com/in/ashish-siwach)

---

## License

MIT — see LICENSE for details
