"""
Retrieval modules: BM25 lexical and FAISS-powered dense semantic retrieval.

DenseRetriever now uses a FAISS IndexHNSWFlat index instead of brute-force
cosine similarity. This mirrors production ANN search (Perplexity, Pinecone,
Weaviate all use HNSW under the hood) and reduces query latency from O(n)
to O(log n).

Index type choice:
  IndexHNSWFlat  — graph-based ANN, best recall/speed tradeoff, no training
                   needed, works well from 10k to 100M+ vectors.
  IndexFlatIP    — exact inner product (kept as fallback for tiny corpora).

Both operate on L2-normalised vectors so inner product == cosine similarity.
"""

import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# BM25 retriever  (unchanged except minor logging tweak)
# ------------------------------------------------------------------

class BM25Retriever:
    """BM25 lexical retrieval using keyword matching."""

    def __init__(self, config: dict):
        self.config          = config
        self.top_k           = config["retrieval"]["bm25"]["top_k"]
        self.b               = config["retrieval"]["bm25"]["b"]
        self.k1              = config["retrieval"]["bm25"]["k1"]
        self.index           = None
        self.tokenized_corpus = None

    def build_index(self, documents: List[str]):
        logger.info("Building BM25 index for %d documents", len(documents))
        self.tokenized_corpus = [doc.lower().split() for doc in documents]
        self.index = BM25Okapi(
            self.tokenized_corpus, b=self.b, k1=self.k1
        )
        logger.info("BM25 index built")

    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise ValueError("Call build_index() first.")
        top_k = top_k or self.top_k
        scores      = self.index.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        return top_indices, scores[top_indices]


# ------------------------------------------------------------------
# FAISS dense retriever  (replaces brute-force cosine similarity)
# ------------------------------------------------------------------

class DenseRetriever:
    """
    Dense semantic retrieval using sentence-transformers + FAISS HNSW.

    Build time  : encode all passages once, build HNSW index, save to disk.
    Query time  : encode query (1 vector), FAISS searches in O(log n).

    HNSW parameters (tunable in config):
      hnsw_m          — number of neighbours per node (default 32).
                        Higher = better recall, more RAM, slower build.
      hnsw_ef_search  — search beam width at query time (default 64).
                        Higher = better recall, slower queries.
    """

    def __init__(self, config: dict):
        self.config     = config
        self.top_k      = config["retrieval"]["dense"]["top_k"]
        self.model_name = config["models"]["dense_encoder"]["name"]
        self.device     = config["models"]["dense_encoder"]["device"]
        self.batch_size = config["models"]["dense_encoder"]["batch_size"]

        # HNSW tuning params (with sensible defaults if absent from config)
        dense_cfg          = config["retrieval"]["dense"]
        self.hnsw_m        = dense_cfg.get("hnsw_m", 32)
        self.hnsw_ef_search = dense_cfg.get("hnsw_ef_search", 64)

        logger.info("Loading dense encoder: %s", self.model_name)
        self.model = SentenceTransformer(self.model_name, device=self.device)

        self._faiss_index  = None   # FAISS index (built or loaded)
        self.doc_embeddings = None  # raw numpy array (kept for saving)
        self._dim           = None  # embedding dimension

    # ------------------------------------------------------------------
    # Index build / load
    # ------------------------------------------------------------------

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Encode passages and build FAISS HNSW index."""
        logger.info("Encoding %d passages...", len(documents))
        t0 = time.perf_counter()

        embeddings = self.model.encode(
            documents,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2 normalise → inner product = cosine
        )

        elapsed = time.perf_counter() - t0
        logger.info(
            "Encoded %d passages in %.1fs → shape %s",
            len(documents), elapsed, embeddings.shape,
        )

        self.doc_embeddings = embeddings
        self._dim           = embeddings.shape[1]
        self._build_faiss_index(embeddings)
        return embeddings

    def _build_faiss_index(self, embeddings: np.ndarray):
        """Build an HNSW index from a numpy embedding matrix."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS not installed. Run: pip install faiss-cpu\n"
                "(or faiss-gpu if you have a CUDA GPU)"
            )

        dim = embeddings.shape[1]
        logger.info(
            "Building FAISS HNSW index (dim=%d, M=%d)...", dim, self.hnsw_m
        )
        t0 = time.perf_counter()

        # IndexHNSWFlat: no quantisation, exact distances within the graph
        index = faiss.IndexHNSWFlat(dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = self.hnsw_ef_search

        # FAISS requires float32
        index.add(embeddings.astype(np.float32))

        elapsed = time.perf_counter() - t0
        logger.info(
            "FAISS HNSW index built in %.1fs (%d vectors)", elapsed, index.ntotal
        )
        self._faiss_index = index

    def load_embeddings(self, embeddings: np.ndarray):
        """Load pre-computed embeddings and build FAISS index from them."""
        self.doc_embeddings = embeddings
        self._dim           = embeddings.shape[1]
        self._build_faiss_index(embeddings)
        logger.info("Loaded embeddings and built FAISS index, shape: %s", embeddings.shape)

    def save_faiss_index(self, filepath: str):
        """Persist the FAISS index to disk."""
        try:
            import faiss
        except ImportError:
            logger.warning("FAISS not available — index not saved.")
            return
        if self._faiss_index is None:
            logger.warning("No FAISS index to save.")
            return
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._faiss_index, filepath)
        logger.info("FAISS index saved to %s", filepath)

    def load_faiss_index(self, filepath: str):
        """Load a persisted FAISS index from disk."""
        try:
            import faiss
        except ImportError:
            raise ImportError("Run: pip install faiss-cpu")
        logger.info("Loading FAISS index from %s", filepath)
        self._faiss_index = faiss.read_index(filepath)
        self._faiss_index.hnsw.efSearch = self.hnsw_ef_search
        logger.info("FAISS index loaded (%d vectors)", self._faiss_index.ntotal)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top-k passages using FAISS HNSW ANN search.

        Steps:
          1. Encode query to a normalised vector (1 × dim)
          2. FAISS searches the HNSW graph in O(log n)
          3. Returns (indices, inner-product scores) ≡ cosine similarity

        Args:
            query: Query string
            top_k: Results to return (config default if None)

        Returns:
            (doc_indices, scores) — both shape (top_k,)
        """
        if self._faiss_index is None:
            raise ValueError(
                "FAISS index not built. Call encode_documents() or load_faiss_index()."
            )

        top_k = top_k or self.top_k

        # Encode + normalise query
        query_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # FAISS search: returns (scores, indices) shaped (1, top_k)
        scores, indices = self._faiss_index.search(query_vec, top_k)

        return indices[0], scores[0]


# ------------------------------------------------------------------
# Hybrid retriever  (RRF over BM25 + FAISS dense)
# ------------------------------------------------------------------

class HybridRetriever:
    """Hybrid retrieval combining BM25 and FAISS dense retrieval via RRF."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        config: dict,
    ):
        self.bm25          = bm25_retriever
        self.dense         = dense_retriever
        self.config        = config
        self.fusion_top_k  = config["retrieval"]["fusion"]["top_k"]
        self.rrf_k         = config["retrieval"]["fusion"]["rrf_k"]

    def reciprocal_rank_fusion(
        self,
        rankings: List[List[int]],
        k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Merge multiple ranked lists with Reciprocal Rank Fusion.

        RRF score for document d = Σ 1 / (k + rank(d))
        where the sum is over all ranking lists that contain d.

        k=60 is the standard default from the original RRF paper
        (Cormack et al., 2009).
        """
        k = k or self.rrf_k
        scores: dict = {}

        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        top_k = top_k or self.fusion_top_k

        bm25_indices,  _ = self.bm25.retrieve(query)
        dense_indices, _ = self.dense.retrieve(query)

        fused    = self.reciprocal_rank_fusion(
            [bm25_indices.tolist(), dense_indices.tolist()]
        )
        top      = fused[:top_k]
        indices  = np.array([d for d, _ in top])
        scores   = np.array([s for _, s in top])

        logger.debug("Hybrid RRF returned %d results", len(indices))
        return indices, scores
