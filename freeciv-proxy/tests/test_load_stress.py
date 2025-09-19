#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Load and stress testing for FreeCiv State Extraction Service
Tests system performance under various load conditions
"""

import unittest
import asyncio
import aiohttp
import time
import json
import statistics
import concurrent.futures
import threading
import sys
import os
from typing import List, Dict, Any
from unittest.mock import Mock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state_extractor import StateExtractor, StateFormat, civcom_registry
from state_cache import StateCache
from api_rate_limiter import APIRateLimiter
from auth import SimpleAuthenticator


class LoadTestResults:
    """Container for load test results"""

    def __init__(self):
        self.response_times: List[float] = []
        self.successful_requests = 0
        self.failed_requests = 0
        self.rate_limited_requests = 0
        self.errors: List[str] = []
        self.start_time = 0
        self.end_time = 0

    @property
    def total_requests(self) -> int:
        return self.successful_requests + self.failed_requests + self.rate_limited_requests

    @property
    def success_rate(self) -> float:
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 0

    @property
    def average_response_time(self) -> float:
        return statistics.mean(self.response_times) if self.response_times else 0

    @property
    def p95_response_time(self) -> float:
        return statistics.quantiles(self.response_times, n=20)[18] if len(self.response_times) >= 20 else 0

    @property
    def p99_response_time(self) -> float:
        return statistics.quantiles(self.response_times, n=100)[98] if len(self.response_times) >= 100 else 0

    @property
    def requests_per_second(self) -> float:
        duration = self.end_time - self.start_time
        return self.total_requests / duration if duration > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'rate_limited_requests': self.rate_limited_requests,
            'success_rate': self.success_rate,
            'average_response_time_ms': self.average_response_time * 1000,
            'p95_response_time_ms': self.p95_response_time * 1000,
            'p99_response_time_ms': self.p99_response_time * 1000,
            'requests_per_second': self.requests_per_second,
            'duration_seconds': self.end_time - self.start_time,
            'error_count': len(self.errors)
        }


class LoadTestFramework:
    """Framework for running load tests"""

    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a single HTTP request and measure response time"""
        start_time = time.time()
        try:
            async with self.session.get(f"{self.base_url}{endpoint}", params=params) as response:
                response_time = time.time() - start_time

                if response.status == 200:
                    data = await response.json()
                    return {
                        'success': True,
                        'response_time': response_time,
                        'status_code': response.status,
                        'data': data
                    }
                elif response.status == 429:  # Rate limited
                    return {
                        'success': False,
                        'response_time': response_time,
                        'status_code': response.status,
                        'error': 'rate_limited'
                    }
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'response_time': response_time,
                        'status_code': response.status,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
        except Exception as e:
            response_time = time.time() - start_time
            return {
                'success': False,
                'response_time': response_time,
                'status_code': 0,
                'error': str(e)
            }

    async def run_concurrent_requests(self, endpoint: str, params: Dict[str, Any],
                                    concurrent_users: int, requests_per_user: int) -> LoadTestResults:
        """Run concurrent load test"""
        results = LoadTestResults()
        results.start_time = time.time()

        # Create tasks for all requests
        tasks = []
        for user_id in range(concurrent_users):
            user_params = params.copy()
            user_params['player_id'] = (user_id % 8) + 1  # Distribute across 8 players

            for _ in range(requests_per_user):
                task = asyncio.create_task(self.make_request(endpoint, user_params))
                tasks.append(task)

        # Wait for all requests to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        results.end_time = time.time()

        # Process results
        for response in responses:
            if isinstance(response, Exception):
                results.failed_requests += 1
                results.errors.append(str(response))
            elif response['success']:
                results.successful_requests += 1
                results.response_times.append(response['response_time'])
            elif response.get('error') == 'rate_limited':
                results.rate_limited_requests += 1
                results.response_times.append(response['response_time'])
            else:
                results.failed_requests += 1
                results.errors.append(response.get('error', 'Unknown error'))

        return results

    async def run_sustained_load(self, endpoint: str, params: Dict[str, Any],
                               target_rps: int, duration_seconds: int) -> LoadTestResults:
        """Run sustained load test with target requests per second"""
        results = LoadTestResults()
        results.start_time = time.time()
        end_time = results.start_time + duration_seconds

        request_interval = 1.0 / target_rps
        next_request_time = results.start_time

        tasks = []

        while time.time() < end_time:
            current_time = time.time()

            if current_time >= next_request_time:
                # Vary player IDs for realistic load distribution
                test_params = params.copy()
                test_params['player_id'] = (len(tasks) % 8) + 1

                task = asyncio.create_task(self.make_request(endpoint, test_params))
                tasks.append(task)

                next_request_time += request_interval
            else:
                # Sleep until next request time
                await asyncio.sleep(min(0.001, next_request_time - current_time))

        # Wait for all pending requests to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        results.end_time = time.time()

        # Process results
        for response in responses:
            if isinstance(response, Exception):
                results.failed_requests += 1
                results.errors.append(str(response))
            elif response['success']:
                results.successful_requests += 1
                results.response_times.append(response['response_time'])
            elif response.get('error') == 'rate_limited':
                results.rate_limited_requests += 1
                results.response_times.append(response['response_time'])
            else:
                results.failed_requests += 1
                results.errors.append(response.get('error', 'Unknown error'))

        return results


class TestLoadAndStress(unittest.TestCase):
    """Load and stress tests for the state extraction service"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment (normally the server would be running)"""
        # Set required environment variables
        os.environ['CACHE_HMAC_SECRET'] = 'test-secret-that-is-at-least-64-characters-long-for-testing-purposes-only'
        os.environ['AUTH_ENABLED'] = 'false'  # Disable auth for load testing

        # Create mock game environment
        cls.mock_civcom = Mock()
        cls.mock_civcom.get_full_state.return_value = {
            'turn': 100,
            'phase': 'movement',
            'units': [
                {'id': i, 'type': 'warrior', 'x': i % 80, 'y': i // 80, 'owner': (i % 8) + 1,
                 'hp': 10, 'moves': 1, 'moves_left': 1}
                for i in range(50)
            ],
            'cities': [
                {'id': i, 'name': f'City{i}', 'x': i * 10, 'y': i * 10,
                 'owner': (i % 8) + 1, 'population': 5 + i, 'production': 'warrior'}
                for i in range(20)
            ],
            'players': {i: {'name': f'Player{i}', 'score': 100 * i, 'gold': 50 * i}
                       for i in range(1, 9)},
            'visible_tiles': [
                {'x': x, 'y': y, 'terrain': 'grassland',
                 'resource': 'wheat' if (x + y) % 15 == 0 else None,
                 'city_id': x // 20 if y == 10 else None}
                for x in range(0, 80, 5) for y in range(0, 50, 5)
            ]
        }

        # Register test games
        for i in range(10):
            civcom_registry.register_game(f'load_test_game_{i}', cls.mock_civcom)

    def test_moderate_concurrent_load(self):
        """Test system under moderate concurrent load (10 users, 5 requests each)"""

        # Create a direct extractor for testing (bypassing HTTP server)
        cache = StateCache(ttl=300, max_cache_size_mb=10, max_entries=100)
        extractor = StateExtractor(cache=cache, registry=civcom_registry)

        results = LoadTestResults()
        results.start_time = time.time()

        def make_request(user_id: int, request_id: int) -> Dict[str, Any]:
            try:
                start_time = time.time()
                state = extractor.extract_state('load_test_game_0', (user_id % 8) + 1, StateFormat.FULL)
                response_time = time.time() - start_time

                return {
                    'success': True,
                    'response_time': response_time,
                    'user_id': user_id,
                    'request_id': request_id,
                    'state_size': len(json.dumps(state))
                }
            except Exception as e:
                return {
                    'success': False,
                    'response_time': time.time() - start_time,
                    'error': str(e),
                    'user_id': user_id,
                    'request_id': request_id
                }

        # Run concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for user_id in range(10):
                for request_id in range(5):
                    future = executor.submit(make_request, user_id, request_id)
                    futures.append(future)

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result['success']:
                    results.successful_requests += 1
                    results.response_times.append(result['response_time'])
                else:
                    results.failed_requests += 1
                    results.errors.append(result['error'])

        results.end_time = time.time()

        # Verify results
        print(f"\nModerate Load Test Results: {results.to_dict()}")

        self.assertGreater(results.success_rate, 0.95, "Success rate should be >95%")
        self.assertLess(results.average_response_time, 0.1, "Average response time should be <100ms")
        self.assertGreater(results.requests_per_second, 10, "Should handle >10 RPS")

    def test_high_concurrent_load(self):
        """Test system under high concurrent load (50 users, 10 requests each)"""

        cache = StateCache(ttl=300, max_cache_size_mb=50, max_entries=500)
        extractor = StateExtractor(cache=cache, registry=civcom_registry)

        results = LoadTestResults()
        results.start_time = time.time()

        def make_request(user_id: int, request_id: int) -> Dict[str, Any]:
            try:
                start_time = time.time()
                format_type = StateFormat.LLM_OPTIMIZED if request_id % 2 == 0 else StateFormat.FULL
                state = extractor.extract_state(f'load_test_game_{user_id % 10}',
                                              (user_id % 8) + 1, format_type)
                response_time = time.time() - start_time

                return {
                    'success': True,
                    'response_time': response_time,
                    'format': format_type.value,
                    'state_size': len(json.dumps(state))
                }
            except Exception as e:
                return {
                    'success': False,
                    'response_time': time.time() - start_time,
                    'error': str(e)
                }

        # Run high concurrency test
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = []
            for user_id in range(50):
                for request_id in range(10):
                    future = executor.submit(make_request, user_id, request_id)
                    futures.append(future)

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result['success']:
                    results.successful_requests += 1
                    results.response_times.append(result['response_time'])
                else:
                    results.failed_requests += 1
                    results.errors.append(result['error'])

        results.end_time = time.time()

        # Verify results
        print(f"\nHigh Load Test Results: {results.to_dict()}")

        self.assertGreater(results.success_rate, 0.9, "Success rate should be >90% under high load")
        self.assertLess(results.p95_response_time, 0.2, "95th percentile response time should be <200ms")
        self.assertGreater(results.requests_per_second, 50, "Should handle >50 RPS")

    def test_cache_performance_under_load(self):
        """Test cache performance and hit rates under sustained load"""

        cache = StateCache(ttl=300, max_cache_size_mb=20, max_entries=200)
        extractor = StateExtractor(cache=cache, registry=civcom_registry)

        # Prime the cache with some requests
        for game_id in range(5):
            for player_id in range(1, 5):
                extractor.extract_state(f'load_test_game_{game_id}', player_id, StateFormat.FULL)

        initial_stats = cache.get_cache_stats()

        # Run sustained requests that should mostly hit cache
        results = LoadTestResults()
        results.start_time = time.time()

        def make_cached_request(request_id: int) -> Dict[str, Any]:
            try:
                start_time = time.time()
                # Use same games/players to maximize cache hits
                game_id = request_id % 5
                player_id = (request_id % 4) + 1

                state = extractor.extract_state(f'load_test_game_{game_id}', player_id, StateFormat.FULL)
                response_time = time.time() - start_time

                return {
                    'success': True,
                    'response_time': response_time
                }
            except Exception as e:
                return {
                    'success': False,
                    'response_time': time.time() - start_time,
                    'error': str(e)
                }

        # Run 200 requests that should mostly be cache hits
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_cached_request, i) for i in range(200)]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result['success']:
                    results.successful_requests += 1
                    results.response_times.append(result['response_time'])
                else:
                    results.failed_requests += 1
                    results.errors.append(result['error'])

        results.end_time = time.time()
        final_stats = cache.get_cache_stats()

        print(f"\nCache Performance Test Results: {results.to_dict()}")
        print(f"Cache hit rate: {final_stats['hit_rate']:.2%}")
        print(f"Cache utilization: {final_stats['cache_utilization_percent']:.1f}%")

        # Verify cache performance
        self.assertGreater(final_stats['hit_rate'], 0.8, "Cache hit rate should be >80%")
        self.assertLess(results.average_response_time, 0.05, "Cache hits should be <50ms on average")
        self.assertEqual(results.failed_requests, 0, "No requests should fail with cache")

    def test_memory_usage_under_sustained_load(self):
        """Test memory usage doesn't grow excessively under sustained load"""
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        except ImportError:
            self.skipTest("psutil not available for memory testing")

        # Create cache with reasonable limits
        cache = StateCache(ttl=60, max_cache_size_mb=10, max_entries=100)
        extractor = StateExtractor(cache=cache, registry=civcom_registry)

        # Run sustained load for 30 seconds
        end_time = time.time() + 30
        request_count = 0

        while time.time() < end_time:
            try:
                game_id = request_count % 10
                player_id = (request_count % 8) + 1
                format_type = StateFormat.LLM_OPTIMIZED if request_count % 3 == 0 else StateFormat.FULL

                extractor.extract_state(f'load_test_game_{game_id}', player_id, format_type)
                request_count += 1

                # Small delay to prevent overwhelming
                time.sleep(0.01)

            except Exception as e:
                print(f"Request {request_count} failed: {e}")

        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_growth = final_memory - initial_memory
        cache_stats = cache.get_cache_stats()

        print(f"\nMemory Usage Test Results:")
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"Memory growth: {memory_growth:.1f} MB")
        print(f"Requests processed: {request_count}")
        print(f"Cache entries: {cache_stats['cache_entries']}")
        print(f"Cache evictions: {cache_stats['eviction_count']}")

        # Verify memory doesn't grow excessively (allow 50MB growth)
        self.assertLess(memory_growth, 50, f"Memory grew by {memory_growth:.1f}MB, should be <50MB")
        self.assertGreater(request_count, 1000, "Should process >1000 requests in 30 seconds")

    @classmethod
    def tearDownClass(cls):
        """Clean up after load tests"""
        # Clean up environment
        if 'CACHE_HMAC_SECRET' in os.environ:
            del os.environ['CACHE_HMAC_SECRET']
        if 'AUTH_ENABLED' in os.environ:
            del os.environ['AUTH_ENABLED']


if __name__ == '__main__':
    # Run load tests with verbose output
    unittest.main(verbosity=2)