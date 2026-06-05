"""
QueryLens — hybrid search with a decision layer.

Public API: import directly from `querylens` for the common types,
or from `querylens.<module>` for everything else.

Import design
-------------
Decision-layer types (DecisionLayer, SearchDecision) and the prompts /
safety modules are imported eagerly — they have no ML dependencies and
must be importable in test environments without torch / sentence-
transformers / faiss installed.

Retriever / reranker / data-loader / evaluator types are loaded
LAZILY via __getattr__ so the package import doesn't drag in heavy
ML deps. They become available as `querylens.BM25Retriever` etc. only
when accessed.
"""

__version__ = "1.1.0"

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
        globals()[name] = value      # cache on first access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DecisionLayer",
    "SearchDecision",
    *_LAZY_EXPORTS.keys(),
]
