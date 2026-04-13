"""
Builds a small demo dataset on Streamlit Cloud at first startup.
Downloads 1000 queries from MS MARCO directly from HuggingFace.
"""
import json, logging
from pathlib import Path

def build_if_missing(config):
    passages_path = config["data"]["ms_marco_passages_path"]
    queries_path  = config["data"]["ms_marco_queries_path"]

    if Path(passages_path).exists() and Path(queries_path).exists():
        return  # already built

    logging.info("Cloud setup: downloading MS MARCO sample from HuggingFace...")
    from datasets import load_dataset

    Path(passages_path).parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "ms_marco", "v1.1", split="train",
        streaming=True, trust_remote_code=True,
    )

    CLIMATE_KEYWORDS = [
        "climate", "carbon", "emission", "renewable", "solar", "wind",
        "fossil", "greenhouse", "warming", "energy", "temperature",
    ]

    passages, queries = [], []
    seen_passages = set()
    MAX_QUERIES = 1000

    for item in dataset:
        if len(queries) >= MAX_QUERIES:
            break
        query_id   = str(item["query_id"])
        query_text = item["query"]
        pas_list   = item["passages"]["passage_text"]
        sel_list   = item["passages"]["is_selected"]
        url_list   = item["passages"]["url"]

        if sum(sel_list) == 0:
            continue

        is_climate = any(kw in query_text.lower() for kw in CLIMATE_KEYWORDS)
        relevance  = {}

        for pas, sel, url in zip(pas_list, sel_list, url_list):
            pid = f"p_{len(passages)}"
            if pid not in seen_passages:
                seen_passages.add(pid)
                passages.append({
                    "passage_id": pid,
                    "passage_text": pas,
                    "query_id": query_id,
                    "is_selected": int(sel),
                    "url": url,
                    "is_climate": is_climate,
                })
            relevance[pid] = int(sel)

        queries.append({
            "query_id":   query_id,
            "query":      query_text,
            "is_climate": is_climate,
            "relevance":  relevance,
        })

    with open(passages_path, "w") as f:
        json.dump(passages, f)
    with open(queries_path, "w") as f:
        json.dump(queries, f)

    logging.info(f"Cloud setup complete: {len(passages)} passages, {len(queries)} queries")