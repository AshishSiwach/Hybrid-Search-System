"""
dataset_builder.py

Downloads MS MARCO v1.1 from HuggingFace and saves a random sample
as a clean local dataset ready for the retrieval pipeline.

Each query is also tagged with is_climate=True/False so that
main.py can run a domain analysis subsection automatically.

Output files (saved to data/):
    ms_marco_passages.json   — all passage records
    ms_marco_queries.json    — all query records (with is_climate flag)
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Climate keyword tagger  (used for domain analysis only, not filtering)
# ------------------------------------------------------------------
CLIMATE_KEYWORDS = [
    "climate", "global warming", "greenhouse", "carbon", "co2",
    "emissions", "net zero", "decarbonisation", "decarbonization",
    "renewable energy", "solar", "wind energy", "wind turbine",
    "hydropower", "geothermal", "fossil fuel", "coal", "natural gas",
    "energy transition", "clean energy", "electricity grid",
    "paris agreement", "carbon tax", "carbon price", "carbon credit",
    "cap and trade", "emission trading", "climate policy", "climate finance",
    "green bond", "esg", "sustainability",
    "sea level", "ice melt", "arctic", "temperature rise", "ipcc",
    "tipping point", "extreme weather", "drought", "flood", "wildfire",
    "deforestation", "biodiversity", "pollution", "air quality",
    "methane", "nitrous oxide",
]

_CLIMATE_RE = re.compile(
    # re.escape ensures keywords with special regex chars (e.g. "+") are treated literally.
    # "|".join builds a single OR-pattern so the entire list is matched in one pass.
    "|".join(re.escape(kw) for kw in CLIMATE_KEYWORDS),
    re.IGNORECASE,
)


def is_climate_query(query: str) -> bool:
    """Return True if query contains a climate/energy keyword."""
    return bool(_CLIMATE_RE.search(query))


# ------------------------------------------------------------------
# Core build function
# ------------------------------------------------------------------

def build_dataset(
    split: str = "train",
    max_queries: int = 10_000,
    seed: int = 42,
    output_dir: str = "data",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stream MS MARCO, collect a random sample, tag climate queries,
    and save to disk.

    Args:
        split:       HuggingFace split — "train" (100k) or "validation"
        max_queries: How many queries to keep.
                     10 000 → ~100k passages, runs well on a laptop.
                     100 000 → full dataset, encode embeddings overnight once.
        seed:        Random seed for reproducibility.
        output_dir:  Where to save output JSON files.

    Returns:
        (queries, passages)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    logger.info(
        "Streaming MS MARCO v1.1 (%s split), targeting %d queries...",
        split, max_queries,
    )

    ds = load_dataset(
        "microsoft/ms_marco",
        "v1.1",
        split=split,
        streaming=True,   # streaming=True avoids downloading the full dataset upfront
    )

    # ------------------------------------------------------------------
    # Reservoir sampling — gives a uniform random sample without
    # loading the full dataset into memory.
    # Each incoming example replaces a random existing one with
    # probability max_queries / (index + 1), which guarantees
    # uniform probability for every example in the stream.
    # ------------------------------------------------------------------
    rng = random.Random(seed)
    reservoir: List[Dict] = []

    logger.info("Reservoir sampling in progress...")
    for i, example in enumerate(ds):
        if len(reservoir) < max_queries:
            reservoir.append(example)
        else:
            j = rng.randint(0, i)
            if j < max_queries:
                reservoir[j] = example

        if (i + 1) % 10_000 == 0:
            logger.info("  Scanned %d examples, reservoir size: %d", i + 1, len(reservoir))

    logger.info("Sampling complete. Processing %d examples...", len(reservoir))

    # ------------------------------------------------------------------
    # Convert reservoir to structured passages + queries
    # ------------------------------------------------------------------
    queries:  List[Dict] = []
    passages: List[Dict] = []
    passage_id = 0   # global counter across all passages — ensures unique passage IDs
    n_climate  = 0

    for example in reservoir:
        query_text = example["query"]
        query_id   = f"q_{example['query_id']}"   # "q_" prefix avoids ID collisions with passage IDs ("p_...")
        climate    = is_climate_query(query_text)
        if climate:
            n_climate += 1

        # MS MARCO stores passages as a dict-of-lists (columnar) rather than a list-of-dicts
        p_data    = example["passages"]
        texts     = p_data["passage_text"]   # list[str]
        selected  = p_data["is_selected"]    # list[int] — 1 = relevant, 0 = not relevant
        urls      = p_data["url"]            # list[str]

        # Skip if no relevant passage (rare edge case in MS MARCO)
        if sum(selected) == 0:
            continue

        relevance = {}   # ground-truth map: {passage_id → relevance_label} used by main.py for evaluation
        for text, rel, url in zip(texts, selected, urls):
            pid = f"p_{passage_id}"
            passages.append({
                "passage_id":   pid,
                "query_id":     query_id,
                "passage_text": text.strip(),
                "is_selected":  int(rel),
                "url":          url,
                "is_climate":   climate,
            })
            relevance[pid] = int(rel)
            passage_id += 1

        queries.append({
            "query_id":   query_id,
            "query":      query_text,
            "answers":    example.get("answers", []),
            "relevance":  relevance,
            "is_climate": climate,
        })

    logger.info(
        "Processed %d queries, %d passages.",
        len(queries), len(passages),
    )
    logger.info(
        "Climate queries: %d / %d (%.1f%%)",
        n_climate, len(queries),
        100 * n_climate / max(len(queries), 1),
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    queries_path  = Path(output_dir) / "ms_marco_queries.json"
    passages_path = Path(output_dir) / "ms_marco_passages.json"

    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)

    with open(passages_path, "w", encoding="utf-8") as f:
        json.dump(passages, f, indent=2, ensure_ascii=False)

    logger.info("Saved queries  → %s", queries_path)
    logger.info("Saved passages → %s", passages_path)
    _print_summary(queries, passages, n_climate)

    return queries, passages


def _print_summary(queries, passages, n_climate):
    n_relevant = sum(p["is_selected"] for p in passages)
    print("\n" + "=" * 60)
    print("MS MARCO SAMPLE — SUMMARY")
    print("=" * 60)
    print(f"  Total queries      : {len(queries)}")
    print(f"  Climate queries    : {n_climate} ({100*n_climate/max(len(queries),1):.1f}%)")
    print(f"  Total passages     : {len(passages)}")
    print(f"  Relevant passages  : {n_relevant}")
    print(f"  Avg passages/query : {len(passages)/max(len(queries),1):.1f}")
    print("=" * 60 + "\n")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a random MS MARCO sample with climate tagging."
    )
    parser.add_argument(
        "--split", default="train", choices=["train", "validation"],
    )
    parser.add_argument(
        "--max-queries", type=int, default=10_000,
        help="Queries to sample (default: 10 000). Use 100 000 for full dataset.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output-dir", default="data",
    )
    args = parser.parse_args()

    build_dataset(
        split=args.split,
        max_queries=args.max_queries,
        seed=args.seed,
        output_dir=args.output_dir,
    )
