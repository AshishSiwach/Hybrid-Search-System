"""
Reranking module using cross-encoder models.
"""

import logging
from typing import List, Tuple
import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Rerank retrieved documents using a cross-encoder model."""
    
    def __init__(self, config: dict):
        """
        Initialize cross-encoder reranker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_name = config['models']['cross_encoder']['name']
        self.device = config['models']['cross_encoder']['device']
        self.batch_size = config['models']['cross_encoder']['batch_size']
        self.top_k = config['retrieval']['reranking']['top_k']
        
        # Load cross-encoder model
        logger.info(f"Loading cross-encoder: {self.model_name}")
        self.model = CrossEncoder(self.model_name, device=self.device)
        logger.info("Cross-encoder loaded successfully")
    
    def rerank(
        self, 
        query: str, 
        candidate_docs: List[str], 
        candidate_indices: np.ndarray,
        top_k: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rerank candidate documents using cross-encoder.
        
        Args:
            query: Query string
            candidate_docs: List of candidate document texts
            candidate_indices: Original indices of candidate documents
            top_k: Number of top results to return (uses config default if None)
            
        Returns:
            Tuple of (reranked_indices, reranking_scores)
        """
        top_k = top_k or self.top_k
        
        if len(candidate_docs) == 0:
            logger.warning("No candidate documents to rerank")
            return np.array([]), np.array([])
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in candidate_docs]
        
        # Get reranking scores
        logger.debug(f"Reranking {len(pairs)} candidates")
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False
        )
        
        # Sort by score descending
        sorted_idx = np.argsort(scores)[::-1]
        
        # Get top-k
        top_sorted_idx = sorted_idx[:top_k]
        reranked_indices = candidate_indices[top_sorted_idx]
        reranking_scores = scores[top_sorted_idx]
        
        logger.debug(f"Reranking complete, returning top {len(reranked_indices)} results")
        return reranked_indices, reranking_scores


class RetrievalPipeline:
    """Complete retrieval pipeline: retrieve → rerank."""
    
    def __init__(self, retriever, reranker, documents: List[str]):
        """
        Initialize retrieval pipeline.
        
        Args:
            retriever: Retriever instance (BM25, Dense, or Hybrid)
            reranker: Reranker instance
            documents: Full list of documents
        """
        self.retriever = retriever
        self.reranker = reranker
        self.documents = documents
    
    def search(self, query: str, rerank: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute full search pipeline.
        
        Args:
            query: Query string
            rerank: Whether to apply reranking (default True)
            
        Returns:
            Tuple of (final_indices, final_scores)
        """
        # Stage 1: Retrieval
        candidate_indices, retrieval_scores = self.retriever.retrieve(query)
        
        if not rerank or len(candidate_indices) == 0:
            return candidate_indices, retrieval_scores
        
        # Get candidate documents
        candidate_docs = [self.documents[idx] for idx in candidate_indices]
        
        # Stage 2: Reranking
        final_indices, reranking_scores = self.reranker.rerank(
            query, 
            candidate_docs, 
            candidate_indices
        )
        
        return final_indices, reranking_scores
    
    def batch_search(self, queries: List[str], rerank: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Execute search for multiple queries.
        
        Args:
            queries: List of query strings
            rerank: Whether to apply reranking
            
        Returns:
            List of (indices, scores) tuples for each query
        """
        results = []
        for query in queries:
            indices, scores = self.search(query, rerank=rerank)
            results.append((indices, scores))
        return results
