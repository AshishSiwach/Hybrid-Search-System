"""
Evaluation metrics for ranking systems.
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class RankingMetrics:
    """Compute standard ranking evaluation metrics."""
    
    @staticmethod
    def precision_at_k(ranked_list: List[int], relevant_docs: List[int], k: int) -> float:
        """
        Compute Precision@K.
        
        Args:
            ranked_list: List of retrieved document indices (ranked)
            relevant_docs: List of relevant document indices
            k: Cutoff rank
            
        Returns:
            Precision@K score
        """
        if k == 0 or len(ranked_list) == 0:
            return 0.0
        
        top_k = ranked_list[:k]
        relevant_retrieved = len(set(top_k) & set(relevant_docs))
        return relevant_retrieved / k
    
    @staticmethod
    def recall_at_k(ranked_list: List[int], relevant_docs: List[int], k: int) -> float:
        """
        Compute Recall@K.
        
        Args:
            ranked_list: List of retrieved document indices (ranked)
            relevant_docs: List of relevant document indices
            k: Cutoff rank
            
        Returns:
            Recall@K score
        """
        if len(relevant_docs) == 0:
            return 0.0
        
        top_k = ranked_list[:k]
        relevant_retrieved = len(set(top_k) & set(relevant_docs))
        return relevant_retrieved / len(relevant_docs)
    
    @staticmethod
    def average_precision(ranked_list: List[int], relevant_docs: List[int]) -> float:
        """
        Compute Average Precision (AP).
        
        Args:
            ranked_list: List of retrieved document indices (ranked)
            relevant_docs: List of relevant document indices
            
        Returns:
            Average Precision score
        """
        if len(relevant_docs) == 0:
            return 0.0
        
        relevant_set = set(relevant_docs)
        score = 0.0
        num_relevant_seen = 0
        
        for i, doc_id in enumerate(ranked_list, start=1):
            if doc_id in relevant_set:
                num_relevant_seen += 1
                precision_at_i = num_relevant_seen / i
                score += precision_at_i
        
        return score / len(relevant_docs)
    
    @staticmethod
    def reciprocal_rank(ranked_list: List[int], relevant_docs: List[int]) -> float:
        """
        Compute Reciprocal Rank (RR).
        
        Args:
            ranked_list: List of retrieved document indices (ranked)
            relevant_docs: List of relevant document indices
            
        Returns:
            Reciprocal Rank score
        """
        relevant_set = set(relevant_docs)
        
        for i, doc_id in enumerate(ranked_list, start=1):
            if doc_id in relevant_set:
                return 1.0 / i
        
        return 0.0
    
    @staticmethod
    def dcg_at_k(ranked_list: List[int], relevance_scores: Dict[int, int], k: int) -> float:
        """
        Compute Discounted Cumulative Gain at K (DCG@K).
        
        Args:
            ranked_list: List of retrieved document indices (ranked)
            relevance_scores: Dict mapping doc_id -> relevance score
            k: Cutoff rank
            
        Returns:
            DCG@K score
        """
        dcg = 0.0
        for i, doc_id in enumerate(ranked_list[:k], start=1):
            rel = relevance_scores.get(doc_id, 0)
            dcg += rel / np.log2(i + 1)
        return dcg
    
    @staticmethod
    def ndcg_at_k(ranked_list: List[int], relevance_scores: Dict[int, int], k: int) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at K (NDCG@K).
        
        Args:
            ranked_list: List of retrieved document indices (ranked)
            relevance_scores: Dict mapping doc_id -> relevance score
            k: Cutoff rank
            
        Returns:
            NDCG@K score
        """
        dcg = RankingMetrics.dcg_at_k(ranked_list, relevance_scores, k)
        
        # Compute ideal DCG (IDCG)
        ideal_ranking = sorted(relevance_scores.keys(), 
                              key=lambda x: relevance_scores[x], 
                              reverse=True)
        idcg = RankingMetrics.dcg_at_k(ideal_ranking, relevance_scores, k)
        
        if idcg == 0.0:
            return 0.0
        
        return dcg / idcg


class Evaluator:
    """Evaluate retrieval system performance."""
    
    def __init__(self, config: dict):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics_config = config['evaluation']['metrics']
        self.metrics = RankingMetrics()
    
    def evaluate_query(
        self, 
        ranked_list: List[int], 
        relevance_labels: Dict[int, int]
    ) -> Dict[str, float]:
        """
        Evaluate a single query's results.
        
        Args:
            ranked_list: Ranked list of document indices
            relevance_labels: Dict mapping doc_id -> relevance (0 or 1)
            
        Returns:
            Dictionary of metric_name -> score
        """
        results = {}
        
        # Get relevant documents (relevance = 1)
        relevant_docs = [doc_id for doc_id, rel in relevance_labels.items() if rel > 0]
        
        # Compute each metric
        if "map@5" in self.metrics_config:
            ap = self.metrics.average_precision(ranked_list[:5], relevant_docs)
            results["map@5"] = ap
        
        if "map@10" in self.metrics_config:
            ap = self.metrics.average_precision(ranked_list[:10], relevant_docs)
            results["map@10"] = ap
        
        if "mrr" in self.metrics_config:
            results["mrr"] = self.metrics.reciprocal_rank(ranked_list, relevant_docs)
        
        if "precision@5" in self.metrics_config:
            results["precision@5"] = self.metrics.precision_at_k(ranked_list, relevant_docs, 5)
        
        if "recall@10" in self.metrics_config:
            results["recall@10"] = self.metrics.recall_at_k(ranked_list, relevant_docs, 10)
        
        if "ndcg@10" in self.metrics_config:
            results["ndcg@10"] = self.metrics.ndcg_at_k(ranked_list, relevance_labels, 10)
        
        return results
    
    def evaluate_system(
        self,
        predictions: Dict[str, List[int]],
        relevance_labels: Dict[str, Dict[int, int]]
    ) -> Dict[str, float]:
        """
        Evaluate system across all queries.
        
        Args:
            predictions: Dict mapping query_id -> ranked list of doc indices
            relevance_labels: Dict mapping query_id -> {doc_id: relevance}
            
        Returns:
            Dictionary of metric_name -> average score
        """
        all_scores = defaultdict(list)
        
        for query_id, ranked_list in predictions.items():
            if query_id not in relevance_labels:
                logger.warning(f"No relevance labels for query: {query_id}")
                continue
            
            query_scores = self.evaluate_query(ranked_list, relevance_labels[query_id])
            
            for metric, score in query_scores.items():
                all_scores[metric].append(score)
        
        # Compute averages
        avg_scores = {
            metric: np.mean(scores) 
            for metric, scores in all_scores.items()
        }
        
        return avg_scores
    
    def run_ablation_study(
        self,
        pipelines: Dict[str, any],
        queries: List[Tuple[str, str]],  # List of (query_id, query_text)
        relevance_labels: Dict[str, Dict[int, int]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Run ablation study comparing multiple retrieval methods.
        
        Args:
            pipelines: Dict mapping method_name -> pipeline instance
            queries: List of (query_id, query_text) tuples
            relevance_labels: Ground truth relevance labels
            
        Returns:
            Dict mapping method_name -> {metric: score}
        """
        results = {}
        
        for method_name, pipeline in pipelines.items():
            logger.info(f"Evaluating method: {method_name}")
            
            predictions = {}
            for query_id, query_text in queries:
                indices, scores = pipeline.search(query_text)
                predictions[query_id] = indices.tolist()
            
            method_scores = self.evaluate_system(predictions, relevance_labels)
            results[method_name] = method_scores
            
            logger.info(f"{method_name} results: {method_scores}")
        
        return results
