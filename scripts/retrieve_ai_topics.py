#!/usr/bin/env python3
"""Script to retrieve specific topics from the AI course knowledge base."""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent  # Go up one level to reach project root
sys.path.insert(0, str(project_root))

from app.retrievers.milvus_retriever import retrieve as milvus_retrieve, connect

def main():
    # Connect to Milvus
    print("Connecting to Milvus...")
    connect()

    # Define the search queries based on user request
    search_queries = [
        "人工智能导论",
        "核心知识点", 
        "扩展阅读",
        "前沿进展",
        "相关领域"
    ]

    print("\n" + "="*80)
    print("RETRIEVING AI COURSE KNOWLEDGE BASE CONTENT")
    print("="*80)
    
    for query in search_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)
        
        try:
            results = milvus_retrieve(
                course_id="course-ai-001",
                query=query,
                k=5,  # Get top 5 results for each query
                search_mode="hybrid",
                use_reranker=True,
            )
            
            if results:
                print(f"\nTop {len(results)} Results:")
                for i, result in enumerate(results, 1):
                    print(f"{i}. [Chunk ID: {result['chunk_id']}] (Relevance: {result['relevance']:.4f})")
                    excerpt = result.get('excerpt', 'N/A')
                    source = result.get('book_title', result.get('source', 'Unknown'))
                    chapter = result.get('chapter_title', '')
                    print(f"   Source: {source}")
                    if chapter:
                        print(f"   Chapter: {chapter}")
                    print(f"   Excerpt: {str(excerpt)[:300]}...")
                    print()
            else:
                print("No results found for this query.")
                
        except Exception as e:
            print(f"Error retrieving results for '{query}': {e}")
    
    print("\n" + "="*80)
    print("RETRIEVAL COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()