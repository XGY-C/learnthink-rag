#!/usr/bin/env python3
"""Script to send a complete query via HTTP request to the RAG service."""
import requests
import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent  # Go up one level to reach project root
sys.path.insert(0, str(project_root))

def main():
    # Define the complete query as requested by user
    complete_query = "人工智能导论 核心知识点 扩展阅读 前沿进展 相关领域 综述"
    
    # API endpoint
    url = "http://localhost:8000/internal/rag/retrieve"
    
    # Request payload
    payload = {
        "course_id": "course-ai-001",
        "query": complete_query,
        "k": 10,  # Get top 10 results
        "search_mode": "hybrid",
        "sparse_weight": 0.3,
        "min_relevance": 0.0,  # No minimum relevance filter
        "min_sources": 1
    }
    
    print(f"Sending request to: {url}")
    print(f"Query: {complete_query}")
    print("="*80)
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            print("RETRIEVAL RESULTS:")
            print("="*80)
            print(f"Total sources returned: {result['stats']['returned']}")
            print(f"Search mode: {result['stats']['search_mode']}")
            print(f"K parameter: {result['stats']['k']}")
            print()
            
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. [Chunk ID: {source['chunk_id']}] (Relevance: {source['relevance']:.4f})")
                print(f"   Book Title: {source.get('book_title', 'N/A')}")
                print(f"   Chapter: {source.get('chapter_title', 'N/A')}")
                print(f"   Source Type: {source.get('source_type', 'N/A')}")
                print(f"   Excerpt: {source['excerpt'][:300]}...")
                print()
                
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the RAG service. Is it running?")
        print("Make sure the service is started with: uvicorn app.main:app --reload")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()