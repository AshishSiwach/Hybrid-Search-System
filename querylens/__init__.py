"""
QueryLens — hybrid search with a decision layer.

Layer 2 imports (decision module) are eager because they have zero ML
dependencies. Layer 1 imports (retrievers, reranker, evaluator,
data_loader) are lazy so that importing this package does not pull in
torch / sentence-transformers / faiss — critical for tests.
"""

__version__ = "1.2.0-phase0"

# Eager — zero ML dependencies
from querylens.decision import DecisionLayer, SearchDecision

# Lazy — heavy ML imports loaded on first access
_LAZY_EXPORTS = {
    "BM25Retriever":        ("querylens.retrievers",  "BM25Retriever"),
    "DenseRetriever":       ("querylens.retrievers",  "DenseRetriever"),
    "HybridRetriever":      ("querylens.retrievers",  "HybridRetriever"),
    "CrossEncoderReranker": ("querylens.reranker",    "CrossEncoderReranker"),
    "RetrievalPipeline":    ("querylens.reranker",    "RetrievalPipeline"),
    "Evaluator":            ("querylens.evaluator",   "Evaluator"),
    "RankingMetrics":       ("querylens.evaluator",   "RankingMetrics"),
    "MSMarcoLoader":        ("querylens.data_loader", "MSMarcoLoader"),
}


def __getattr__(name):
    """PEP 562 lazy attribute loader — fires only on access."""
    if name in _LAZY_EXPORTS:
        import importlib
        module_path, attr = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DecisionLayer", "SearchDecision", *_LAZY_EXPORTS.keys()]
