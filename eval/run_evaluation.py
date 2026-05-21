#!/usr/bin/env python3
"""Run RAG evaluation on test data.

Usage:
    python eval/run_evaluation.py --course-id course-ai-001 --k 5
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from app.log_config import setup_logger

logger = setup_logger("evaluation")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument("--course-id", required=True, help="Course ID to evaluate")
    parser.add_argument("--kb-base-dir", default="kb", help="Knowledge base directory")
    parser.add_argument("--k", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--test-data", default="eval/test_data.jsonl", help="Path to test data JSONL file")
    parser.add_argument("--output", default="eval/results.json", help="Path to save results")
    
    args = parser.parse_args()
    
    # Import after argument parsing to avoid unnecessary imports
    import sys
    from pathlib import Path
    
    # Add project root to Python path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from eval.evaluate import RAGEvaluator
    from app.retrievers.milvus_retriever import retrieve as milvus_retrieve
    from app.retrievers.milvus_retriever import connect
    
    # Connect to Milvus
    logger.info("Connecting to Milvus...")
    connect()
    
    # Initialize evaluator
    test_data_path = Path(args.test_data)
    if not test_data_path.exists():
        logger.warning("Test data file not found: %s", test_data_path)
        logger.info("Creating sample test data...")
        test_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(test_data_path, 'w', encoding='utf-8') as f:
            f.write('{"query": "什么是贝叶斯分类器", "relevant_chunks": ["ch03_贝叶斯分类器#0"]}\n')
            f.write('{"query": "搜索算法有哪些", "relevant_chunks": ["ch02_搜索算法#0"]}\n')
        logger.info("Sample test data created at %s", test_data_path)
    
    evaluator = RAGEvaluator(test_data_path)
    
    # Evaluate all search modes
    logger.info("Starting evaluation for course=%s, k=%d", args.course_id, args.k)
    
    def retrieve_wrapper(course_id, query, k, search_mode):
        return milvus_retrieve(
            course_id=course_id,
            query=query,
            k=k,
            search_mode=search_mode,
            use_reranker=True,  # Disable reranker for baseline evaluation
        )
    
    results = evaluator.compare_modes(
        retrieve_func=retrieve_wrapper,
        course_id=args.course_id,
        k=args.k,
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info("Evaluation results saved to %s", output_path)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for mode, metrics in results.items():
        print(f"\n{mode.upper()} Mode:")
        for metric_name, score in metrics.items():
            print(f"  {metric_name:20s}: {score:.4f}")
    
    print("\n" + "="*60)
    print("Recommendation: Use the mode with highest MRR and NDCG@5")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
