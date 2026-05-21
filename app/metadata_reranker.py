"""Metadata-aware reranking enhancement.

This module provides metadata-based boosting to improve retrieval quality
by considering source_type, heading_path, and topic relevance.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Source type weights - lecture materials should be prioritized
SOURCE_TYPE_WEIGHTS = {
    "讲义": 1.5,        # Primary teaching materials
    "术语": 1.2,        # Glossary terms (secondary)
    "题库": 0.8,        # Exercises (tertiary)
    "实践项目": 0.9,    # Projects (contextual)
}

def extract_keywords(query: str) -> list[str]:
    """Extract key terms from query for matching."""
    # Simple keyword extraction: remove common stop words
    stop_words = {"什么", "是", "的", "如何", "怎么", "哪些", "一个", "这种"}
    keywords = []
    
    # Split by common delimiters
    parts = re.split(r'[\s,，.。;；:：!?！？]+', query)
    
    for part in parts:
        part = part.strip()
        if part and part not in stop_words and len(part) >= 2:
            keywords.append(part)
    
    return keywords


def compute_metadata_boost(doc: dict[str, Any], query: str) -> float:
    """Compute metadata-based boost factor for a document.
    
    Args:
        doc: Document with metadata fields
        query: Original user query
    
    Returns:
        Boost factor (>= 1.0 means boost, < 1.0 means penalty)
    """
    boost = 1.0
    
    # 1. Source type weighting
    source_type = doc.get('source_type', '')
    source_weight = SOURCE_TYPE_WEIGHTS.get(source_type, 1.0)
    boost *= source_weight
    
    # 2. Heading path matching
    heading_path = doc.get('heading_path', '')
    if heading_path:
        keywords = extract_keywords(query)
        heading_lower = heading_path.lower()
        
        # Check if any query keyword appears in heading
        matches = sum(1 for kw in keywords if kw in heading_lower)
        if matches > 0:
            # Boost based on number of keyword matches
            heading_boost = 1.0 + (0.1 * matches)
            boost *= min(heading_boost, 1.3)  # Cap at 1.3x
    
    # 3. Topic exact match
    topic = doc.get('topic', '')
    if topic and topic in query:
        boost *= 1.2
    
    # 4. Chapter title matching
    chapter_title = doc.get('chapter_title', '') or doc.get('book_title', '')
    if chapter_title:
        keywords = extract_keywords(query)
        title_lower = chapter_title.lower()

        matches = sum(1 for kw in keywords if kw in title_lower)
        if matches > 0:
            title_boost = 1.0 + (0.15 * matches)
            boost *= min(title_boost, 1.4)  # Cap at 1.4x
    
    return boost


def apply_metadata_reranking(
    candidates: list[dict[str, Any]],
    query: str,
    top_k: int = 5,
    blend_weight: float = 0.7,
) -> list[dict[str, Any]]:
    """Apply metadata-aware re-ranking to candidates.
    
    This blends the original relevance score (from vector search or reranker)
    with metadata-based boosting to produce a final ranking.
    
    Args:
        candidates: List of candidate documents with 'relevance' scores
        query: Original user query
        top_k: Number of results to return
        blend_weight: Weight for original relevance (0.0-1.0)
                     Higher = trust vector/reranker more
                     Lower = trust metadata more
    
    Returns:
        Re-ranked list of top_k candidates
    """
    if not candidates:
        return []
    
    scored_candidates = []
    
    for doc in candidates:
        original_score = doc.get('relevance', 0.0)
        metadata_boost = compute_metadata_boost(doc, query)
        
        # Blend original score with metadata boost
        # Formula: final = original^blend_weight * boost^(1-blend_weight)
        # This ensures both factors contribute meaningfully
        if original_score >= 0:
            final_score = (original_score ** blend_weight) * (metadata_boost ** (1 - blend_weight))
        else:
            # For negative scores (irrelevant), apply boost as additive
            final_score = original_score + (metadata_boost - 1.0) * 0.1
        
        doc_copy = doc.copy()
        doc_copy['final_score'] = round(final_score, 4)
        doc_copy['metadata_boost'] = round(metadata_boost, 4)
        scored_candidates.append(doc_copy)
    
    # Sort by final score descending
    ranked = sorted(
        scored_candidates,
        key=lambda x: x.get('final_score', 0.0),
        reverse=True
    )
    
    result = ranked[:top_k]
    
    logger.info(
        "Applied metadata reranking: %d candidates -> top %d (best final_score: %.4f)",
        len(candidates), len(result), result[0].get('final_score', 0.0) if result else 0.0
    )
    
    return result
