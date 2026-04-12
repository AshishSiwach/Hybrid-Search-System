"""
Data loading and preprocessing module for hybrid search system.

Primary  : MS MARCO random sample (passage-level retrieval)
Secondary: Climate documents CSV (document-level, domain comparison)
"""

import csv
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
    ) -> Dict[str, Dict[str, int]]:
        """
        Build relevance labels from MS MARCO's is_selected field.

        Args:
            climate_only: If True, return labels only for climate queries.
                          Used for the domain analysis subsection.

        Returns:
            {query_id: {passage_id: 0|1}}
        """
        labels = {
            q["query_id"]: q["relevance"]
            for q in self.queries
            if not climate_only or q.get("is_climate", False)
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
# Climate CSV loader  (secondary / fallback)
# ------------------------------------------------------------------

DOCUMENT_TYPE_CATEGORY_MAP = {
    "Assessment Report":    "science_research",
    "Technical Summary":    "science_research",
    "Research Report":      "science_research",
    "Research Paper":       "science_research",
    "Special Report":       "science_research",
    "National Assessment":  "science_research",
    "Statistical Report":   "science_research",
    "Policy Document":      "policy_governance",
    "Policy Report":        "policy_governance",
    "Summary Report":       "policy_governance",
    "Synthesis Report":     "policy_governance",
    "Adaptation Plan":      "policy_governance",
    "Guide":                "policy_governance",
    "Financial Report":     "finance_economics",
    "Economic Report":      "finance_economics",
    "Annual Report":        "finance_economics",
    "Analysis Report":      "finance_economics",
    "Energy Report":        "energy",
    "Energy Analysis":      "energy",
    "Health Report":        "health_environment",
    "Agricultural Report":  "health_environment",
    "Employment Report":    "health_environment",
}


class DocumentLoader:
    """Climate CSV loader — kept for secondary domain comparison."""

    def __init__(self, config: Dict):
        self.config    = config
        self.documents: List[str]  = []
        self.doc_ids:   List[str]  = []
        self.metadata:  List[Dict] = []

    def load_from_csv(
        self,
        filepath: Optional[str] = None,
        save_json: bool = True,
    ) -> Tuple[List[str], List[str], List[Dict]]:
        filepath = filepath or self.config["data"].get(
            "csv_path", "data/updated_climate_documents_with_files.csv"
        )
        logger.info("Loading climate CSV from %s", filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.documents = []
        self.doc_ids   = []
        self.metadata  = []

        for i, row in enumerate(rows):
            doc_id   = f"doc_{i + 1:04d}"
            title    = row.get("title", "").strip()
            desc     = row.get("description", "").strip()
            keywords = row.get("keywords", "").strip()
            text     = f"{title}. {desc} {keywords}".strip()
            doc_type = row.get("document_type", "")

            self.doc_ids.append(doc_id)
            self.documents.append(text)
            self.metadata.append({
                "title":         title,
                "year":          int(row["year"]) if row.get("year") else None,
                "institution":   row.get("institution", ""),
                "document_type": doc_type,
                "url":           row.get("url", ""),
                "language":      row.get("language", "EN"),
                "keywords":      keywords,
                "file":          row.get("file", ""),
                "category":      DOCUMENT_TYPE_CATEGORY_MAP.get(
                    doc_type, "science_research"
                ),
            })

        logger.info("Loaded %d climate documents", len(self.documents))
        if save_json:
            self._save_as_json(self.config["data"]["documents_path"])
        return self.documents, self.doc_ids, self.metadata

    def load_documents(
        self, filepath: Optional[str] = None
    ) -> Tuple[List[str], List[str], List[Dict]]:
        filepath = filepath or self.config["data"]["documents_path"]
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.documents = [item["text"]             for item in data]
        self.doc_ids   = [item["id"]               for item in data]
        self.metadata  = [item.get("metadata", {}) for item in data]
        logger.info("Loaded %d documents from JSON", len(self.documents))
        return self.documents, self.doc_ids, self.metadata

    def _save_as_json(self, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data = [
            {"id": did, "text": text, "metadata": meta}
            for did, text, meta in zip(
                self.doc_ids, self.documents, self.metadata
            )
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Saved JSON cache to %s", output_path)


# ------------------------------------------------------------------
# Relevance labeler for climate CSV
# ------------------------------------------------------------------

class RelevanceLabeler:
    def __init__(self, documents, doc_ids, metadata):
        self.documents = documents
        self.doc_ids   = doc_ids
        self.metadata  = metadata

    def create_category_based_labels(self) -> Dict[str, Dict[str, int]]:
        category_docs: Dict[str, List[str]] = {}
        for doc_id, meta in zip(self.doc_ids, self.metadata):
            cat = meta.get(
                "category",
                DOCUMENT_TYPE_CATEGORY_MAP.get(
                    meta.get("document_type", ""), "science_research"
                ),
            )
            category_docs.setdefault(cat, []).append(doc_id)

        labels = {}
        for cat, docs in category_docs.items():
            docs_set = set(docs)
            labels[f"query_{cat}"] = {
                did: (1 if did in docs_set else 0)
                for did in self.doc_ids
            }
        logger.info("Created labels for %d category queries", len(labels))
        return labels

    def save_labels(self, labels: Dict, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(labels, f, indent=2)
        logger.info("Saved relevance labels to %s", filepath)


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
