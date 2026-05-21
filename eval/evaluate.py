"""Evaluation module for RAG system.

Provides metrics calculation and benchmarking tools to measure retrieval quality.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def calculate_mrr(retrieved_chunks: list[str], relevant_chunks: list[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR).
    
    MRR measures how early the first relevant result appears in the ranked list.
    
    Args:
        retrieved_chunks: List of retrieved chunk IDs in ranked order
        relevant_chunks: List of ground truth relevant chunk IDs
    
    Returns:
        MRR score (0.0 to 1.0, higher is better)
    """
    if not relevant_chunks or not retrieved_chunks:
        return 0.0
    
    for rank, chunk_id in enumerate(retrieved_chunks, start=1):
        if chunk_id in relevant_chunks:
            return 1.0 / rank
    
    return 0.0


def calculate_ndcg_at_k(
    retrieved_chunks: list[str], 
    relevant_chunks: list[str],
    k: int = 10
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K (NDCG@K).
    
    NDCG measures ranking quality by considering both relevance and position.
    
    Args:
        retrieved_chunks: List of retrieved chunk IDs in ranked order
        relevant_chunks: List of ground truth relevant chunk IDs
        k: Number of top results to consider
    
    Returns:
        NDCG@K score (0.0 to 1.0, higher is better)
    """
    # Truncate to top-k
    retrieved_top_k = retrieved_chunks[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, chunk_id in enumerate(retrieved_top_k, start=1):
        if chunk_id in relevant_chunks:
            # Binary relevance: 1 if relevant, 0 otherwise
            rel = 1.0
            dcg += rel / (i + 1)  # log2(i+1) simplified
    
    # Calculate ideal DCG (all relevant chunks at top positions)
    ideal_relevant = min(len(relevant_chunks), k)
    idcg = sum(1.0 / (i + 1) for i in range(ideal_relevant))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_precision_at_k(
    retrieved_chunks: list[str],
    relevant_chunks: list[str],
    k: int = 10
) -> float:
    """Calculate Precision@K.
    
    Args:
        retrieved_chunks: List of retrieved chunk IDs in ranked order
        relevant_chunks: List of ground truth relevant chunk IDs
        k: Number of top results to consider
    
    Returns:
        Precision@K score (0.0 to 1.0, higher is better)
    """
    retrieved_top_k = retrieved_chunks[:k]
    if not retrieved_top_k:
        return 0.0
    
    relevant_retrieved = sum(1 for chunk in retrieved_top_k if chunk in relevant_chunks)
    return relevant_retrieved / len(retrieved_top_k)


def calculate_recall_at_k(
    retrieved_chunks: list[str],
    relevant_chunks: list[str],
    k: int = 10
) -> float:
    """Calculate Recall@K.
    
    Args:
        retrieved_chunks: List of retrieved chunk IDs in ranked order
        relevant_chunks: List of ground truth relevant chunk IDs
        k: Number of top results to consider
    
    Returns:
        Recall@K score (0.0 to 1.0, higher is better)
    """
    if not relevant_chunks:
        return 0.0
    
    retrieved_top_k = set(retrieved_chunks[:k])
    relevant_set = set(relevant_chunks)
    
    relevant_retrieved = len(retrieved_top_k & relevant_set)
    return relevant_retrieved / len(relevant_set)


class RAGEvaluator:
    """Evaluator for RAG retrieval quality."""
    
    def __init__(self, test_data_path: Path | None = None):
        """Initialize evaluator with optional test data.
        
        Args:
            test_data_path: Path to JSONL file with test queries and ground truth
                           Format: {"query": "...", "relevant_chunks": ["id1", "id2"]}
        """
        self.test_data: list[dict[str, Any]] = []
        if test_data_path and test_data_path.exists():
            self.load_test_data(test_data_path)
    
    def load_test_data(self, path: Path) -> None:
        """Load test data from JSONL file."""
        self.test_data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.test_data.append(json.loads(line))
        logger.info("Loaded %d test queries from %s", len(self.test_data), path)
    
    def evaluate_retrieval(
        self,
        retrieve_func,
        course_id: str,
        k: int = 5,
        search_mode: str = "hybrid",
    ) -> dict[str, float]:
        """Evaluate retrieval function on test data.
        
        Args:
            retrieve_func: Function to call for retrieval
                          Signature: retrieve_func(course_id, query, k, ...)
            course_id: Course ID for retrieval
            k: Number of results to retrieve
            search_mode: Search mode to use
        
        Returns:
            Dictionary of aggregated metrics
        """
        if not self.test_data:
            raise ValueError("No test data loaded. Call load_test_data() first.")
        
        metrics = {
            'mrr': [],
            'ndcg@5': [],
            'precision@5': [],
            'recall@5': [],
        }
        
        for test_case in self.test_data:
            query = test_case['query']
            relevant_chunks = test_case.get('relevant_chunks', [])
            
            # Perform retrieval
            try:
                results = retrieve_func(
                    course_id=course_id,
                    query=query,
                    k=k,
                    search_mode=search_mode,
                )
                retrieved_chunk_ids = [r['chunk_id'] for r in results]
                
                # Calculate metrics
                metrics['mrr'].append(calculate_mrr(retrieved_chunk_ids, relevant_chunks))
                metrics['ndcg@5'].append(calculate_ndcg_at_k(retrieved_chunk_ids, relevant_chunks, k=5))
                metrics['precision@5'].append(calculate_precision_at_k(retrieved_chunk_ids, relevant_chunks, k=5))
                metrics['recall@5'].append(calculate_recall_at_k(retrieved_chunk_ids, relevant_chunks, k=5))
                
            except Exception as e:
                logger.error("Failed to evaluate query '%s': %s", query, e)
                # Add zeros for failed queries
                metrics['mrr'].append(0.0)
                metrics['ndcg@5'].append(0.0)
                metrics['precision@5'].append(0.0)
                metrics['recall@5'].append(0.0)
        
        # Aggregate metrics
        aggregated = {
            metric: sum(values) / len(values) if values else 0.0
            for metric, values in metrics.items()
        }
        
        logger.info("Evaluation results on %d queries:", len(self.test_data))
        for metric_name, score in aggregated.items():
            logger.info("  %s: %.4f", metric_name, score)
        
        return aggregated
    
    def compare_modes(
        self,
        retrieve_func,
        course_id: str,
        k: int = 5,
    ) -> dict[str, dict[str, float]]:
        """Compare different search modes (dense, sparse, hybrid).
        
        Args:
            retrieve_func: Retrieval function to test
            course_id: Course ID for retrieval
            k: Number of results to retrieve
        
        Returns:
            Dictionary mapping search mode to metrics
        """
        results = {}
        
        for mode in ['dense', 'sparse', 'hybrid']:
            logger.info("Evaluating %s mode...", mode)
            try:
                metrics = self.evaluate_retrieval(
                    retrieve_func=retrieve_func,
                    course_id=course_id,
                    k=k,
                    search_mode=mode,
                )
                results[mode] = metrics
            except Exception as e:
                logger.error("Failed to evaluate %s mode: %s", mode, e)
        
        return results
