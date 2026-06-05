"""
Data loading and preprocessing module for hybrid search system.

Primary dataset: MS MARCO random sample (passage-level retrieval)
Built by dataset_builder.py — run that first before main.py.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# MS MARCO loader  (primary)
# ------------------------------------------------------------------

class MSMarcoLoader:
    """
    Load the MS MARCO sample produced by dataset_builder.py.

    Passage-level retrieval mirrors Perplexity's actual pipeline:
    sub-document units scored individually against the query.

    Also exposes a domain split so main.py can run:
      - general evaluation  (all queries)
      - climate subsection  (is_climate=True queries only)
    """

    def __init__(self, config: Dict):
        self.config       = config
        self.passages:    List[str]  = []
        self.passage_ids: List[str]  = []
        self.metadata:    List[Dict] = []
        self.queries:     List[Dict] = []

    def load(self) -> Tuple[List[str], List[str], List[Dict], List[Dict]]:
        """
        Load passages and queries from the sampled MS MARCO JSON files.

        Returns:
            (passages, passage_ids, metadata, queries)
        """
        passages_path = self.config["data"]["ms_marco_passages_path"]
        queries_path  = self.config["data"]["ms_marco_queries_path"]

        logger.info("Loading passages from %s", passages_path)
        with open(passages_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        logger.info("Loading queries from %s", queries_path)
        with open(queries_path, "r", encoding="utf-8") as f:
            self.queries = json.load(f)

        self.passages    = [p["passage_text"] for p in raw]
        self.passage_ids = [p["passage_id"]   for p in raw]
        self.metadata    = [
            {
                "passage_id":  p["passage_id"],
                "query_id":    p["query_id"],
                "url":         p["url"],
                "is_selected": p["is_selected"],
                "is_climate":  p.get("is_climate", False),
            }
            for p in raw
        ]

        n_climate = sum(q.get("is_climate", False) for q in self.queries)
        logger.info(
            "Loaded %d passages from %d queries (%d climate)",
            len(self.passages), len(self.queries), n_climate,
        )
        return self.passages, self.passage_ids, self.metadata, self.queries

    def build_relevance_labels(
        self,
        climate_only: bool = False,
    ) -> Dict[str, Dict[int, int]]:
        """
        Build relevance labels keyed by INTEGER passage index.

        Retrievers return integer indices into the passages list (e.g. [4821, 203]).
        The evaluator needs labels keyed by the same integers, not string passage IDs.

        This method builds a passage_id → integer_index lookup from self.passage_ids,
        then converts every relevance dict from {passage_id: rel} to {int_index: rel}.

        Args:
            climate_only: If True, return labels only for climate queries.

        Returns:
            {query_id: {int_index: 0|1}}
        """
        # Build reverse lookup: "p_4821" -> 4821 (position in passages list)
        pid_to_idx = {pid: idx for idx, pid in enumerate(self.passage_ids)}

        labels: Dict[str, Dict[int, int]] = {}
        for q in self.queries:
            if climate_only and not q.get("is_climate", False):
                continue
            # Convert {passage_id: rel} → {int_index: rel}
            labels[q["query_id"]] = {
                pid_to_idx[pid]: rel
                for pid, rel in q["relevance"].items()
                if pid in pid_to_idx
            }

        logger.info(
            "Built relevance labels for %d queries (climate_only=%s)",
            len(labels), climate_only,
        )
        return labels

    def get_test_queries(
        self,
        max_queries: int,
        climate_only: bool = False,
    ) -> List[Tuple[str, str]]:
        """
        Return (query_id, query_text) tuples for evaluation.

        Args:
            max_queries:  Cap from config.
            climate_only: If True, return only climate-tagged queries.
        """
        pool = [
            q for q in self.queries
            if not climate_only or q.get("is_climate", False)
        ]
        return [
            (q["query_id"], q["query"])
            for q in pool[:max_queries]
        ]

    def climate_query_count(self) -> int:
        """Return number of climate-tagged queries in the loaded data."""
        return sum(q.get("is_climate", False) for q in self.queries)



# ------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------

def load_embeddings(filepath: str) -> np.ndarray:
    logger.info("Loading embeddings from %s", filepath)
    emb = np.load(filepath)
    logger.info("Embeddings shape: %s", emb.shape)
    return emb


def save_embeddings(embeddings: np.ndarray, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, embeddings)
    logger.info("Saved embeddings to %s", filepath)


def load_bm25_index(filepath: str):
    logger.info("Loading BM25 index from %s", filepath)
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_bm25_index(index, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(index, f)
    logger.info("Saved BM25 index to %s", filepath)
