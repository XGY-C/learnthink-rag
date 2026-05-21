"""Cross-Encoder Reranker module for re-ranking retrieved candidates.

Uses BGE-Reranker-v2-m3 to compute query-document relevance scores
and re-rank the candidate set from Milvus retrieval.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Global state with thread-safe initialization
_reranker_model = None
_reranker_lock = threading.Lock()


def _load_reranker():
    """Load BGE-Reranker-v2-m3 model with double-checked locking.
    
    Returns:
        FlagReranker instance
    """
    global _reranker_model
    
    if _reranker_model is not None:
        return _reranker_model
    
    with _reranker_lock:
        # Double-check after acquiring lock
        if _reranker_model is not None:
            return _reranker_model
        
        try:
            from FlagEmbedding import FlagReranker
            from pathlib import Path
            
            # Use local model path to avoid network issues
            model_path = str(Path("./models/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e").resolve())
            
            logger.info("Loading BGE-Reranker-v2-m3 model from local path: %s", model_path)
            _reranker_model = FlagReranker(
                model_path,
                use_fp16=True,  # Use FP16 for faster inference
            )
            logger.info("BGE-Reranker-v2-m3 loaded successfully")
            return _reranker_model
            
        except Exception as e:
            logger.error("Failed to load reranker model: %s", e)
            raise


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 5,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Re-rank retrieved candidates using Cross-Encoder.
    
    Args:
        query: User query string
        candidates: List of candidate documents from Milvus retrieval
                   Each dict should have at least 'excerpt' and 'relevance' keys
        top_k: Number of top results to return after re-ranking
        batch_size: Batch size for computing scores (larger = faster but more memory)
    
    Returns:
        Re-ranked list of candidates sorted by reranker score (descending)
    """
    if not candidates:
        return []
    
    if len(candidates) <= top_k and all('rerank_score' in c for c in candidates):
        # Already re-ranked and filtered
        return candidates[:top_k]
    
    try:
        model = _load_reranker()
        
        # Prepare pairs for scoring
        pairs = [(query, doc.get('excerpt', '')) for doc in candidates]
        
        # Compute relevance scores
        logger.debug("Computing reranker scores for %d candidates", len(pairs))
        scores = model.compute_score(pairs, batch_size=batch_size)
        
        # Handle single score case (FlagEmbedding returns float for single pair)
        if isinstance(scores, (float, int)):
            scores = [scores]
        
        # Attach scores to candidates
        scored_candidates = []
        for doc, score in zip(candidates, scores):
            doc_copy = doc.copy()
            doc_copy['rerank_score'] = round(float(score), 4)
            # Update relevance to use reranker score (more accurate)
            doc_copy['relevance'] = round(float(score), 4)
            scored_candidates.append(doc_copy)
        
        # Sort by reranker score descending
        ranked = sorted(
            scored_candidates,
            key=lambda x: x.get('rerank_score', 0.0),
            reverse=True
        )
        
        # Return top-k
        result = ranked[:top_k]
        logger.info(
            "Re-ranked %d candidates, returning top %d (best score: %.4f)",
            len(candidates), len(result), result[0].get('rerank_score', 0.0) if result else 0.0
        )
        
        return result
        
    except Exception as e:
        logger.error("Reranking failed: %s. Returning original candidates.", e)
        # Fallback: return original candidates without re-ranking
        return candidates[:top_k]


def warmup_reranker():
    """Warmup the reranker model by running a dummy inference.
    
    Call this during application startup to avoid cold-start latency.
    """
    try:
        model = _load_reranker()
        # Run a dummy inference to warm up the model
        dummy_pairs = [("test query", "test document")]
        model.compute_score(dummy_pairs, batch_size=1)
        logger.info("Reranker model warmed up successfully")
    except Exception as e:
        logger.warning("Reranker warmup failed: %s", e)
