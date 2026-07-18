"""
Concurrency stress test for RAG retrieval endpoint.

Tests the semaphore-based concurrency control by sending multiple
simultaneous requests and measuring response times and success rates.
"""
import asyncio
import aiohttp
import time
import json
from typing import List, Dict
from dataclasses import dataclass, asdict


@dataclass
class TestResult:
    request_id: int
    status_code: int
    response_time_ms: float
    success: bool
    error: str = ""


async def send_retrieval_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    request_id: int,
    timeout: int = 60
) -> TestResult:
    """Send a single retrieval request and measure response time."""
    start_time = time.time()
    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            elapsed_ms = (time.time() - start_time) * 1000
            status_code = response.status
            
            if status_code == 200:
                return TestResult(
                    request_id=request_id,
                    status_code=status_code,
                    response_time_ms=elapsed_ms,
                    success=True
                )
            else:
                error_text = await response.text()
                return TestResult(
                    request_id=request_id,
                    status_code=status_code,
                    response_time_ms=elapsed_ms,
                    success=False,
                    error=f"HTTP {status_code}: {error_text[:100]}"
                )
    
    except asyncio.TimeoutError:
        elapsed_ms = (time.time() - start_time) * 1000
        return TestResult(
            request_id=request_id,
            status_code=0,
            response_time_ms=elapsed_ms,
            success=False,
            error="Request timeout"
        )
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return TestResult(
            request_id=request_id,
            status_code=0,
            response_time_ms=elapsed_ms,
            success=False,
            error=str(e)
        )


async def run_concurrency_test(
    num_requests: int = 20,
    concurrent_limit: int = 12,
    base_url: str = "http://localhost:19531",
    course_id: str = "course-ai-001",
    query: str = "什么是机器学习？"
):
    """
    Run concurrency stress test.
    
    Args:
        num_requests: Total number of requests to send
        concurrent_limit: Number of requests to send simultaneously
        base_url: RAG service URL
        course_id: Course ID for retrieval
        query: Query text
    """
    url = f"{base_url}/internal/rag/retrieve"
    payload = {
        "course_id": course_id,
        "query": query,
        "k": 5,
        "search_mode": "hybrid",
        "sparse_weight": 0.3,
        "min_relevance": 0.0,
        "min_sources": 0,
        "query_mode": "raw"
    }
    
    print("=" * 80)
    print("RAG Concurrency Stress Test")
    print("=" * 80)
    print(f"Target URL: {url}")
    print(f"Total requests: {num_requests}")
    print(f"Concurrent batch size: {concurrent_limit}")
    print(f"Expected concurrency limit: 12")
    print("-" * 80)
    
    all_results: List[TestResult] = []
    total_start_time = time.time()
    
    # Send requests in batches
    for batch_num in range(0, num_requests, concurrent_limit):
        batch_end = min(batch_num + concurrent_limit, num_requests)
        batch_size = batch_end - batch_num
        
        print(f"\n[Batch {batch_num // concurrent_limit + 1}] Sending {batch_size} requests...")
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                send_retrieval_request(session, url, payload, i + 1)
                for i in range(batch_num, batch_end)
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"  Request failed with exception: {result}")
                else:
                    all_results.append(result)
                    status_icon = "✓" if result.success else "✗"
                    print(f"  {status_icon} Request #{result.request_id}: "
                          f"{result.response_time_ms:.0f}ms "
                          f"(HTTP {result.status_code})")
        
        # Small delay between batches
        if batch_end < num_requests:
            await asyncio.sleep(1)
    
    total_elapsed = time.time() - total_start_time
    
    # Calculate statistics
    successful = [r for r in all_results if r.success]
    failed = [r for r in all_results if not r.success]
    timeouts = [r for r in all_results if "timeout" in r.error.lower()]
    busy_errors = [r for r in all_results if "503" in str(r.status_code)]
    
    response_times = [r.response_time_ms for r in successful]
    
    print("\n" + "=" * 80)
    print("Test Results Summary")
    print("=" * 80)
    print(f"Total requests: {num_requests}")
    print(f"Successful: {len(successful)} ({len(successful)/num_requests*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/num_requests*100:.1f}%)")
    print(f"  - Timeouts: {len(timeouts)}")
    print(f"  - Service Busy (503): {len(busy_errors)}")
    print(f"  - Other errors: {len(failed) - len(timeouts) - len(busy_errors)}")
    
    if response_times:
        print(f"\nResponse Time Statistics (successful requests):")
        print(f"  Min: {min(response_times):.0f}ms")
        print(f"  Max: {max(response_times):.0f}ms")
        print(f"  Avg: {sum(response_times)/len(response_times):.0f}ms")
        print(f"  Median: {sorted(response_times)[len(response_times)//2]:.0f}ms")
    
    print(f"\nTotal test duration: {total_elapsed:.2f}s")
    print(f"Throughput: {num_requests/total_elapsed:.2f} req/s")
    
    # Analyze concurrency behavior
    print("\n" + "-" * 80)
    print("Concurrency Analysis")
    print("-" * 80)
    
    if busy_errors:
        print(f"✓ Concurrency limit is working! Detected {len(busy_errors)} requests rejected (503)")
        print("  This indicates the semaphore is properly limiting concurrent requests.")
    else:
        print("⚠ No 503 errors detected. Either:")
        print("  1. All requests completed within the 30s queue timeout")
        print("  2. Concurrency limit may not be enforced correctly")
    
    # Check if response times increased under load
    if len(response_times) >= 10:
        first_half_avg = sum(response_times[:len(response_times)//2]) / (len(response_times)//2)
        second_half_avg = sum(response_times[len(response_times)//2:]) / (len(response_times) - len(response_times)//2)
        
        print(f"\nResponse time trend:")
        print(f"  First half avg: {first_half_avg:.0f}ms")
        print(f"  Second half avg: {second_half_avg:.0f}ms")
        
        if second_half_avg > first_half_avg * 1.5:
            print("  ⚠ Response times increased significantly under load")
        else:
            print("  ✓ Response times remained stable under load")
    
    # Save detailed results
    results_file = "test_concurrency_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_requests": num_requests,
                "successful": len(successful),
                "failed": len(failed),
                "throughput_req_per_sec": num_requests / total_elapsed,
                "avg_response_time_ms": sum(response_times) / len(response_times) if response_times else 0,
            },
            "results": [asdict(r) for r in all_results]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {results_file}")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    num_requests = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    concurrent = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 19531
    
    print(f"\nStarting test with {num_requests} requests, {concurrent} concurrent...")
    
    try:
        asyncio.run(run_concurrency_test(
            num_requests=num_requests,
            concurrent_limit=concurrent,
            base_url=f"http://localhost:{port}"
        ))
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nTest failed: {e}")
        import traceback
        traceback.print_exc()
