"""
demo.py

Run a single query through the full retrieval pipeline and display
ranked results from all four methods, then generate a grounded answer
from the best method's top passages using Claude Haiku (Anthropic API).

This mirrors Perplexity's full pipeline:
  Query → Retrieval → Ranked passages → LLM answer + cited sources

Usage:
    python demo.py
    python demo.py --query "what causes global warming"
    python demo.py --query "carbon tax policy europe" --top-k 5
    python demo.py --query "sea level rise" --no-generate   # skip generation

Setup:
    export ANTHROPIC_API_KEY="your-key-here"
    python demo.py
"""

import argparse
import json
import logging
import os
import time
import urllib.request
import urllib.error
import yaml
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

# ------------------------------------------------------------------
# Config + Data
# ------------------------------------------------------------------

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_passages(config: dict):
    path = config["data"]["ms_marco_passages_path"]
    with open(path) as f:
        raw = json.load(f)
    passages    = [p["passage_text"] for p in raw]
    passage_ids = [p["passage_id"]   for p in raw]
    metadata    = [{"url": p["url"], "is_selected": p["is_selected"]} for p in raw]
    return passages, passage_ids, metadata


def build_retrievers(config, passages):
    from data_loader import load_embeddings, load_bm25_index, save_embeddings, save_bm25_index
    from retrievers import BM25Retriever, DenseRetriever, HybridRetriever

    print("  Loading BM25 index...")
    bm25       = BM25Retriever(config)
    index_path = config["data"]["index_path"]
    if Path(index_path).exists():
        bm25.index = load_bm25_index(index_path)
        bm25.tokenized_corpus = [p.lower().split() for p in passages]
    else:
        bm25.build_index(passages)
        save_bm25_index(bm25.index, index_path)

    print("  Loading dense encoder + FAISS index...")
    dense      = DenseRetriever(config)
    faiss_path = config["data"]["faiss_index_path"]
    emb_path   = config["data"]["embeddings_path"]
    if Path(faiss_path).exists() and Path(emb_path).exists():
        dense.load_faiss_index(faiss_path)
    else:
        embeddings = dense.encode_documents(passages)
        save_embeddings(embeddings, emb_path)
        dense.save_faiss_index(faiss_path)

    hybrid = HybridRetriever(bm25, dense, config)
    return bm25, dense, hybrid


def build_pipelines(bm25, dense, hybrid, passages, config):
    from reranker import CrossEncoderReranker, RetrievalPipeline
    reranker = CrossEncoderReranker(config)

    pipelines = {
        "BM25 only":         RetrievalPipeline(bm25,   reranker, passages),
        "Dense only":        RetrievalPipeline(dense,  reranker, passages),
        "Hybrid (RRF)":      RetrievalPipeline(hybrid, reranker, passages),
        "Hybrid + Reranker": RetrievalPipeline(hybrid, reranker, passages),
    }
    for method in ["BM25 only", "Dense only", "Hybrid (RRF)"]:
        orig = pipelines[method].search
        pipelines[method].search = lambda q, _o=orig: _o(q, rerank=False)

    return pipelines


# ------------------------------------------------------------------
# Generation layer  (mirrors Perplexity's answer synthesis step)
# ------------------------------------------------------------------

GENERATION_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

SYSTEM_PROMPT = (
    "You are a factual search assistant. Answer the user's query concisely "
    "using ONLY the provided passages. Cite each fact with [1], [2], or [3] "
    "corresponding to the passage number. If the passages do not contain "
    "enough information, say so. Do not add external knowledge."
)


def generate_answer(query: str, top_passages: list, top_urls: list) -> str:
    """
    Call Claude Haiku to generate a grounded answer from retrieved passages.

    Args:
        query:        The user's query string
        top_passages: List of up to 3 passage texts (from Hybrid + Reranker)
        top_urls:     Corresponding source URLs

    Returns:
        Generated answer string, or an error message if API key is missing.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return (
            "[Generation skipped — set ANTHROPIC_API_KEY environment variable]\n"
            "  export ANTHROPIC_API_KEY='your-key-here'"
        )

    # Build the user message with numbered passages
    passage_block = "\n\n".join(
        f"[{i+1}] {text}\nSource: {url}"
        for i, (text, url) in enumerate(zip(top_passages, top_urls))
    )
    user_message = f"Query: {query}\n\nPassages:\n{passage_block}"

    payload = json.dumps({
        "model":      GENERATION_MODEL,
        "max_tokens": 512,
        "system":     SYSTEM_PROMPT,
        "messages":   [{"role": "user", "content": user_message}],
    }).encode("utf-8")

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=payload,
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["content"][0]["text"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return f"[API error {e.code}]: {body[:200]}"
    except Exception as e:
        return f"[Generation error]: {e}"


# ------------------------------------------------------------------
# Display helpers
# ------------------------------------------------------------------

DIVIDER      = "─" * 80
THIN_DIVIDER = "·" * 80


def truncate(text: str, n: int = 130) -> str:
    return text if len(text) <= n else text[:n] + "…"


def relevance_tag(is_selected: int) -> str:
    return "  ✓ relevant" if is_selected else ""


def print_retrieval_results(query, all_results, passages, metadata, top_k):
    print()
    print(DIVIDER)
    print(f"  Query: \"{query}\"")
    print(DIVIDER)

    for method, (indices, scores, latency_ms) in all_results.items():
        print(f"\n  {method}  ({latency_ms:.1f}ms)")
        print(THIN_DIVIDER)
        for rank, (idx, score) in enumerate(zip(indices[:top_k], scores[:top_k]), 1):
            idx          = int(idx)
            passage_text = truncate(passages[idx])
            url          = metadata[idx]["url"]
            rel_tag      = relevance_tag(metadata[idx]["is_selected"])
            print(f"  #{rank}  score={score:.4f}{rel_tag}")
            print(f"       {passage_text}")
            print(f"       source: {url[:70]}")
            if rank < top_k:
                print()

    print()
    print(DIVIDER)
    print()


def print_generation_output(query, answer, top_passages, top_urls, latency_ms):
    print(DIVIDER)
    print("  Generated answer  (Claude Haiku — Hybrid + Reranker passages)")
    print(THIN_DIVIDER)
    print()
    # Indent the answer for readability
    for line in answer.strip().splitlines():
        print(f"  {line}")
    print()
    print("  References:")
    for i, url in enumerate(top_urls, 1):
        print(f"    [{i}] {url}")
    print()
    print(f"  Generation latency: {latency_ms:.0f}ms")
    print(DIVIDER)
    print()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run_demo(
    query: str,
    top_k: int = 5,
    config_path: str = "configs/config.yaml",
    generate: bool = True,
):
    print()
    print("=" * 80)
    print("  Hybrid Search Pipeline — Demo")
    print("=" * 80)
    print(f"\n  Query  : \"{query}\"")
    print(f"  Top-k  : {top_k}")
    print()

    config = load_config(config_path)

    print("Loading data and indexes...")
    passages, passage_ids, metadata = load_passages(config)
    print(f"  Corpus : {len(passages):,} passages loaded")

    bm25, dense, hybrid = build_retrievers(config, passages)
    pipelines           = build_pipelines(bm25, dense, hybrid, passages, config)

    print("\nRunning query through all four methods...")
    all_results = {}
    for method, pipeline in pipelines.items():
        t0 = time.perf_counter()
        indices, scores = pipeline.search(query)
        latency_ms = (time.perf_counter() - t0) * 1000
        all_results[method] = (indices, scores, latency_ms)
        print(f"  {method:<25} {latency_ms:6.1f}ms")

    # Print ranked passages for all four methods
    print_retrieval_results(query, all_results, passages, metadata, top_k)

    # Relevance summary bar
    print("  Relevant passages in top-k (is_selected=1):")
    for method, (indices, _, _) in all_results.items():
        n_rel = sum(metadata[int(i)]["is_selected"] for i in indices[:top_k])
        bar   = "█" * n_rel + "·" * (top_k - n_rel)
        print(f"    {method:<25} [{bar}]  {n_rel}/{top_k}")
    print()

    # ------------------------------------------------------------------
    # Generation layer — use top-3 passages from Hybrid + Reranker
    # ------------------------------------------------------------------
    if generate:
        best_indices, _, _ = all_results["Hybrid + Reranker"]
        top3_indices  = [int(i) for i in best_indices[:3]]
        top3_passages = [passages[i] for i in top3_indices]
        top3_urls     = [metadata[i]["url"] for i in top3_indices]

        print("Generating answer from top-3 retrieved passages (Hybrid + Reranker)...")
        t0 = time.perf_counter()
        answer = generate_answer(query, top3_passages, top3_urls)
        gen_latency = (time.perf_counter() - t0) * 1000

        print_generation_output(query, answer, top3_passages, top3_urls, gen_latency)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo: hybrid search pipeline with LLM answer generation"
    )
    parser.add_argument(
        "--query",
        default="what are the effects of global warming on sea levels",
        help="Query string to search",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results to show per method (default: 5)",
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--no-generate", action="store_true",
        help="Skip the generation step (retrieval only)",
    )
    args = parser.parse_args()
    run_demo(
        query=args.query,
        top_k=args.top_k,
        config_path=args.config,
        generate=not args.no_generate,
    )
