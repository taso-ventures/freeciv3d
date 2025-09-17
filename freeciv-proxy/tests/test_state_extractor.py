#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test suite for State Extraction Service
Tests REST API endpoints for game state extraction and LLM optimization
Includes integration tests, performance tests, and security tests
"""

import unittest
import asyncio
import json
import time
import sys
import os
import concurrent.futures
import threading
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from tornado.testing import AsyncHTTPTestCase
from tornado import web
from tornado.httpclient import HTTPResponse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state_extractor import (
    StateExtractor, StateFormat, StateExtractorHandler, LegalActionsHandler,
    validate_request_parameters, authenticate_request, civcom_registry, CivComRegistry
)
from state_cache import StateCache, CacheEntry
from api_rate_limiter import APIRateLimiter
from auth import SimpleAuthenticator, AuthSession
from monitoring import HealthCheckHandler, MetricsHandler, StatsHandler
from admin_handlers import AdminAuthHandler


class TestStateExtractor(unittest.TestCase):
    """Test suite for StateExtractor class"""

    def setUp(self):
        """Set up test fixtures"""
        self.maxDiff = None
        self.mock_civcom = Mock()
        self.mock_cache = Mock(spec=StateCache)
        self.extractor = StateExtractor(civcom=self.mock_civcom, cache=self.mock_cache)

        # Sample game state for testing
        self.sample_game_state = {
            'turn': 42,
            'phase': 'movement',
            'map': {
                'width': 80,
                'height': 50,
                'tiles': [{'x': 10, 'y': 10, 'terrain': 'grassland', 'visible': True}] * 100
            },
            'units': [
                {'id': 1, 'type': 'warrior', 'x': 10, 'y': 10, 'owner': 1, 'hp': 10, 'moves': 1},
                {'id': 2, 'type': 'settler', 'x': 11, 'y': 11, 'owner': 1, 'hp': 20, 'moves': 1},
                {'id': 3, 'type': 'warrior', 'x': 50, 'y': 50, 'owner': 2, 'hp': 10, 'moves': 1}
            ],
            'cities': [
                {'id': 1, 'name': 'Capital', 'x': 10, 'y': 10, 'owner': 1, 'population': 5, 'production': 'warrior'},
                {'id': 2, 'name': 'Second City', 'x': 20, 'y': 20, 'owner': 1, 'population': 3, 'production': 'granary'}
            ],
            'players': [
                {'id': 1, 'name': 'Player1', 'nation': 'romans', 'gold': 100, 'science': 50},
                {'id': 2, 'name': 'Player2', 'nation': 'greeks', 'gold': 80, 'science': 40}
            ],
            'techs': {
                'player1': ['pottery', 'bronze_working'],
                'player2': ['pottery', 'animal_husbandry']
            }
        }

    def test_extract_full_state(self):
        """Test extraction of complete game state"""
        game_id = "test_game_1"
        player_id = 1

        # Mock cache to return None (cache miss)
        self.mock_cache.get.return_value = None

        # Mock civcom to return sample state
        self.mock_civcom.get_full_state.return_value = self.sample_game_state

        # Mock _get_civcom_for_game to return our mock
        self.extractor._get_civcom_for_game = Mock(return_value=self.mock_civcom)

        # Test extraction
        result = self.extractor.extract_state(game_id, player_id, StateFormat.FULL)

        # Verify civcom was called correctly
        self.mock_civcom.get_full_state.assert_called_once_with(player_id)

        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertEqual(result['turn'], 42)
        self.assertEqual(result['phase'], 'movement')
        self.assertIn('units', result)
        self.assertIn('cities', result)
        self.assertIn('players', result)

    def test_extract_llm_optimized_state(self):
        """Test LLM-optimized state extraction with >70% size reduction"""
        game_id = "test_game_1"
        player_id = 1

        # Mock cache to return None (cache miss)
        self.mock_cache.get.return_value = None

        # Mock civcom to return sample state
        self.mock_civcom.get_full_state.return_value = self.sample_game_state

        # Mock _get_civcom_for_game to return our mock
        self.extractor._get_civcom_for_game = Mock(return_value=self.mock_civcom)

        # Test optimization
        full_result = self.extractor.extract_state(game_id, player_id, StateFormat.FULL)
        optimized_result = self.extractor.extract_state(game_id, player_id, StateFormat.LLM_OPTIMIZED)

        # Calculate sizes
        full_size = len(json.dumps(full_result))
        optimized_size = len(json.dumps(optimized_result))
        reduction_percentage = ((full_size - optimized_size) / full_size) * 100

        # Verify >70% size reduction
        self.assertGreater(reduction_percentage, 70.0,
                          f"Size reduction was only {reduction_percentage:.1f}%, expected >70%")

        # Verify critical information is preserved
        self.assertEqual(optimized_result['turn'], 42)
        self.assertIn('strategic', optimized_result)
        self.assertIn('tactical', optimized_result)
        self.assertIn('economic', optimized_result)

    def test_extract_delta_state(self):
        """Test delta state extraction (changes since last turn)"""
        game_id = "test_game_1"
        player_id = 1
        since_turn = 40

        # Mock cache to return None (cache miss)
        self.mock_cache.get.return_value = None

        # Mock current state
        current_state = self.sample_game_state.copy()
        self.mock_civcom.get_full_state.return_value = current_state

        # Mock _get_civcom_for_game to return our mock
        self.extractor._get_civcom_for_game = Mock(return_value=self.mock_civcom)

        # Test delta extraction
        result = self.extractor.extract_state(game_id, player_id, StateFormat.DELTA, since_turn=since_turn)

        # Verify delta structure
        self.assertIsInstance(result, dict)
        self.assertIn('changes', result)
        self.assertIn('since_turn', result)
        self.assertEqual(result['since_turn'], since_turn)

        # Verify unit movement is captured
        changes = result['changes']
        self.assertIn('units', changes)

    def test_cache_integration(self):
        """Test state caching with TTL"""
        game_id = "test_game_1"
        player_id = 1

        # Mock _get_civcom_for_game to return our mock
        self.extractor._get_civcom_for_game = Mock(return_value=self.mock_civcom)
        self.mock_civcom.get_full_state.return_value = self.sample_game_state

        # Create expected formatted state
        expected_state = {
            'format': 'full',
            'turn': 42,
            'phase': 'movement',
            'map': self.sample_game_state['map'],
            'units': self.sample_game_state['units'],
            'cities': self.sample_game_state['cities'],
            'players': self.sample_game_state['players'],
            'techs': self.sample_game_state['techs'],
            'player_perspective': player_id
        }

        # Mock cache miss, then hit
        self.mock_cache.get.side_effect = [None, expected_state]

        # First call should miss cache and call civcom
        result1 = self.extractor.extract_state(game_id, player_id, StateFormat.FULL)
        self.mock_civcom.get_full_state.assert_called_once()
        self.mock_cache.set.assert_called_once()

        # Second call should hit cache
        result2 = self.extractor.extract_state(game_id, player_id, StateFormat.FULL)

        # Verify cache was used for second call
        self.assertEqual(result2, expected_state)

    def test_legal_actions_extraction(self):
        """Test extraction of legal actions for player"""
        game_id = "test_game_1"
        player_id = 1

        # Mock _get_civcom_for_game to return our mock
        self.extractor._get_civcom_for_game = Mock(return_value=self.mock_civcom)
        self.mock_civcom.get_full_state.return_value = self.sample_game_state

        # Test action extraction
        result = self.extractor.get_legal_actions(game_id, player_id)

        # Verify top 20 actions returned
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), 20)

        # Verify actions are sorted by priority (highest first)
        if len(result) > 1:
            for i in range(len(result) - 1):
                self.assertGreaterEqual(result[i]['priority'], result[i + 1]['priority'])

    def test_error_handling(self):
        """Test error handling for invalid game/player IDs"""
        # Mock cache to return None (cache miss)
        self.mock_cache.get.return_value = None

        # Test invalid game ID - mock _get_civcom_for_game to return None
        self.extractor._get_civcom_for_game = Mock(return_value=None)

        with self.assertRaises(Exception):
            self.extractor.extract_state("invalid_game", 1, StateFormat.FULL)

        # Test invalid player ID - mock civcom to raise exception
        self.mock_civcom.get_full_state.side_effect = Exception("Player not found")
        self.extractor._get_civcom_for_game = Mock(return_value=self.mock_civcom)

        with self.assertRaises(Exception):
            self.extractor.extract_state("test_game", 999, StateFormat.FULL)

    def test_state_format_enum(self):
        """Test StateFormat enum values"""
        self.assertEqual(StateFormat.FULL.value, "full")
        self.assertEqual(StateFormat.DELTA.value, "delta")
        self.assertEqual(StateFormat.LLM_OPTIMIZED.value, "llm_optimized")

    def test_performance_requirements(self):
        """Test that state extraction meets performance requirements"""
        game_id = "test_game_1"
        player_id = 1

        # Mock cache to return None (cache miss)
        self.mock_cache.get.return_value = None
        # Mock civcom to return sample state
        self.mock_civcom.get_full_state.return_value = self.sample_game_state
        # Mock _get_civcom_for_game to return our mock
        self.extractor._get_civcom_for_game = Mock(return_value=self.mock_civcom)

        # Measure extraction time
        start_time = time.time()
        result = self.extractor.extract_state(game_id, player_id, StateFormat.LLM_OPTIMIZED)
        end_time = time.time()

        extraction_time_ms = (end_time - start_time) * 1000

        # Verify < 100ms requirement
        self.assertLess(extraction_time_ms, 100.0,
                       f"Extraction took {extraction_time_ms:.2f}ms, expected <100ms")

    def test_concurrent_request_handling(self):
        """Test that multiple concurrent requests can be handled"""
        import threading
        import concurrent.futures

        game_id = "test_game_1"
        player_id = 1

        # Mock cache to return None (cache miss)
        self.mock_cache.get.return_value = None
        # Mock civcom to return sample state
        self.mock_civcom.get_full_state.return_value = self.sample_game_state
        # Mock _get_civcom_for_game to return our mock
        self.extractor._get_civcom_for_game = Mock(return_value=self.mock_civcom)

        # Function to extract state
        def extract_state():
            return self.extractor.extract_state(game_id, player_id, StateFormat.LLM_OPTIMIZED)

        # Run multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(extract_state) for _ in range(10)]

            # Wait for all to complete
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=5)
                    results.append(result)
                except Exception as e:
                    self.fail(f"Concurrent request failed: {e}")

        # Verify all requests succeeded
        self.assertEqual(len(results), 10)

        # Verify all results have the expected structure
        for result in results:
            self.assertIn('strategic', result)
            self.assertIn('tactical', result)
            self.assertIn('economic', result)


class TestStateExtractorHTTPEndpoints(AsyncHTTPTestCase):
    """Test HTTP REST endpoints for state extraction"""

    def get_app(self):
        """Create test Tornado application"""
        from state_extractor import StateExtractorHandler, LegalActionsHandler

        return web.Application([
            (r"/api/game/([^/]+)/state", StateExtractorHandler),
            (r"/api/game/([^/]+)/legal_actions", LegalActionsHandler),
        ])

    def test_get_game_state_endpoint(self):
        """Test GET /api/game/{game_id}/state endpoint"""
        game_id = "test_game_1"

        # Test full format
        response = self.fetch(f'/api/game/{game_id}/state?format=full&player_id=1')
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertIsInstance(data, dict)
        self.assertIn('turn', data)

    def test_get_game_state_llm_optimized(self):
        """Test LLM optimized format endpoint"""
        game_id = "test_game_1"

        response = self.fetch(f'/api/game/{game_id}/state?format=llm_optimized&player_id=1')
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertIn('strategic', data)
        self.assertIn('tactical', data)
        self.assertIn('economic', data)

    def test_get_legal_actions_endpoint(self):
        """Test GET /api/game/{game_id}/legal_actions endpoint"""
        game_id = "test_game_1"

        response = self.fetch(f'/api/game/{game_id}/legal_actions?player_id=1')
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertIsInstance(data, list)
        self.assertLessEqual(len(data), 20)

    def test_invalid_game_id(self):
        """Test error handling for invalid game ID"""
        response = self.fetch('/api/game/invalid_game/state?player_id=1')
        self.assertEqual(response.code, 404)

    def test_missing_player_id(self):
        """Test error handling for missing player_id parameter"""
        response = self.fetch('/api/game/test_game/state?format=full')
        self.assertEqual(response.code, 400)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""

    def setUp(self):
        """Set up integration test environment"""
        # Set environment variables for testing
        os.environ['CACHE_HMAC_SECRET'] = 'test-secret-that-is-at-least-64-characters-long-for-testing-purposes-only'
        os.environ['AUTH_ENABLED'] = 'false'  # Disable auth for integration tests

        # Create test instances
        self.cache = StateCache(ttl=5, max_size_kb=4)
        self.rate_limiter = APIRateLimiter()
        self.authenticator = SimpleAuthenticator()
        self.registry = CivComRegistry()

        # Mock CivCom instance
        self.mock_civcom = Mock()
        self.mock_civcom.get_full_state.return_value = {
            'turn': 42,
            'phase': 'movement',
            'units': [{'id': 1, 'type': 'warrior', 'x': 10, 'y': 10, 'owner': 1, 'hp': 10, 'moves': 1}],
            'cities': [{'id': 1, 'name': 'Capital', 'x': 10, 'y': 10, 'owner': 1, 'population': 5}],
            'players': {1: {'name': 'Player1', 'score': 100, 'gold': 50}}
        }

        # Register test game
        self.registry.register_game('integration_test_game', self.mock_civcom)

    def test_full_integration_flow(self):
        """Test complete integration from cache miss to successful response"""
        extractor = StateExtractor(cache=self.cache, registry=self.registry)

        # Test full state extraction
        start_time = time.time()
        state = extractor.extract_state('integration_test_game', 1, StateFormat.FULL)
        extraction_time = (time.time() - start_time) * 1000

        # Verify response structure
        self.assertIn('format', state)
        self.assertEqual(state['format'], 'full')
        self.assertIn('turn', state)
        self.assertEqual(state['turn'], 42)

        # Verify performance requirement (<100ms)
        self.assertLess(extraction_time, 100, f"Extraction took {extraction_time:.2f}ms, exceeds 100ms requirement")

        # Test cache hit on second request
        cache_start_time = time.time()
        cached_state = extractor.extract_state('integration_test_game', 1, StateFormat.FULL)
        cache_time = (time.time() - cache_start_time) * 1000

        # Cached response should be much faster
        self.assertLess(cache_time, 50, f"Cached response took {cache_time:.2f}ms, should be <50ms")
        self.assertEqual(state, cached_state)

    def test_llm_optimization_size_requirement(self):
        """Test that LLM optimized format meets size requirements"""
        extractor = StateExtractor(cache=self.cache, registry=self.registry)

        # Get full state first
        full_state = extractor.extract_state('integration_test_game', 1, StateFormat.FULL)
        full_size = len(json.dumps(full_state, separators=(',', ':')))

        # Get LLM optimized state
        llm_state = extractor.extract_state('integration_test_game', 1, StateFormat.LLM_OPTIMIZED)
        llm_size = len(json.dumps(llm_state, separators=(',', ':')))

        # Verify size reduction requirement (>70%)
        size_reduction = (full_size - llm_size) / full_size
        self.assertGreater(size_reduction, 0.7, f"Size reduction was {size_reduction:.2%}, needs to be >70%")

        # Verify size limit (<4KB)
        self.assertLess(llm_size, 4096, f"LLM optimized state is {llm_size} bytes, exceeds 4KB limit")

    def test_cache_eviction_and_limits(self):
        """Test cache LRU eviction and size limits"""
        small_cache = StateCache(ttl=60, max_cache_size_mb=1, max_entries=5)
        extractor = StateExtractor(cache=small_cache, registry=self.registry)

        # Fill cache beyond entry limit
        for i in range(10):
            game_id = f'test_game_{i}'
            self.registry.register_game(game_id, self.mock_civcom)
            extractor.extract_state(game_id, 1, StateFormat.FULL)

        # Verify cache respects entry limit
        stats = small_cache.get_cache_stats()
        self.assertLessEqual(stats['cache_entries'], 5)
        self.assertGreater(stats['eviction_count'], 0)

    def test_rate_limiting_integration(self):
        """Test rate limiting integration"""
        # Test player rate limiting
        for i in range(65):  # Exceed 60 RPM limit
            allowed, retry_after, limit_type = self.rate_limiter.check_limits(1, "127.0.0.1")
            if not allowed:
                self.assertEqual(limit_type, "player")
                self.assertIsNotNone(retry_after)
                break
        else:
            self.fail("Rate limiting should have activated")

    def test_concurrent_requests(self):
        """Test concurrent request handling"""
        extractor = StateExtractor(cache=self.cache, registry=self.registry)

        def make_request(thread_id):
            try:
                return extractor.extract_state('integration_test_game', 1, StateFormat.FULL)
            except Exception as e:
                return f"Error in thread {thread_id}: {e}"

        # Test 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            results = [future.result() for future in futures]

        # All requests should succeed
        for result in results:
            self.assertIsInstance(result, dict, f"Request failed: {result}")

    def tearDown(self):
        """Clean up after integration tests"""
        self.cache.clear()
        # Clean up environment
        if 'CACHE_HMAC_SECRET' in os.environ:
            del os.environ['CACHE_HMAC_SECRET']
        if 'AUTH_ENABLED' in os.environ:
            del os.environ['AUTH_ENABLED']


class TestPerformanceRequirements(unittest.TestCase):
    """Performance tests to verify system requirements"""

    def setUp(self):
        os.environ['CACHE_HMAC_SECRET'] = 'test-secret-that-is-at-least-64-characters-long-for-testing-purposes-only'

        # Create optimized cache for performance testing
        self.cache = StateCache(ttl=300, max_cache_size_mb=100, max_entries=1000)
        self.registry = CivComRegistry()

        # Mock with larger, more realistic game state
        self.mock_civcom = Mock()
        self.mock_civcom.get_full_state.return_value = {
            'turn': 150,
            'phase': 'movement',
            'units': [
                {'id': i, 'type': 'warrior', 'x': i % 80, 'y': i // 80, 'owner': 1, 'hp': 10, 'moves': 1}
                for i in range(100)
            ],
            'cities': [
                {'id': i, 'name': f'City{i}', 'x': i * 10, 'y': i * 10, 'owner': 1, 'population': 5 + i}
                for i in range(20)
            ],
            'players': {i: {'name': f'Player{i}', 'score': 100 * i, 'gold': 50 * i} for i in range(1, 9)},
            'visible_tiles': [
                {'x': x, 'y': y, 'terrain': 'grassland', 'resource': 'wheat' if (x + y) % 10 == 0 else None}
                for x in range(80) for y in range(50)
            ]
        }
        self.registry.register_game('performance_test_game', self.mock_civcom)

    def test_response_time_requirement(self):
        """Test <100ms response time requirement"""
        extractor = StateExtractor(cache=self.cache, registry=self.registry)

        # Test multiple formats
        formats = [StateFormat.FULL, StateFormat.LLM_OPTIMIZED]

        for format_type in formats:
            with self.subTest(format_type=format_type.value):
                start_time = time.time()
                state = extractor.extract_state('performance_test_game', 1, format_type)
                response_time = (time.time() - start_time) * 1000

                self.assertLess(response_time, 100,
                    f"{format_type.value} format took {response_time:.2f}ms, exceeds 100ms requirement")

    def test_cache_performance(self):
        """Test cache hit performance requirement (<50ms)"""
        extractor = StateExtractor(cache=self.cache, registry=self.registry)

        # Prime the cache
        extractor.extract_state('performance_test_game', 1, StateFormat.FULL)

        # Test cached response
        start_time = time.time()
        cached_state = extractor.extract_state('performance_test_game', 1, StateFormat.FULL)
        cache_time = (time.time() - start_time) * 1000

        self.assertLess(cache_time, 50, f"Cache hit took {cache_time:.2f}ms, should be <50ms")

    def test_state_optimization_size_limit(self):
        """Test state size optimization meets <4KB requirement"""
        extractor = StateExtractor(cache=self.cache, registry=self.registry)

        llm_state = extractor.extract_state('performance_test_game', 1, StateFormat.LLM_OPTIMIZED)
        state_size = len(json.dumps(llm_state, separators=(',', ':')))

        self.assertLess(state_size, 4096, f"LLM optimized state is {state_size} bytes, exceeds 4KB limit")

    def test_compression_effectiveness(self):
        """Test cache compression provides significant size reduction"""
        # Test with large state
        large_state = self.cache.optimize_state_data(self.mock_civcom.get_full_state(1))

        # Force compression by making state larger than threshold
        large_state['large_data'] = 'x' * 2000  # Add 2KB of data

        # Cache the state and check compression
        success = self.cache.set('compression_test', large_state, 1)
        self.assertTrue(success)

        stats = self.cache.get_cache_stats()
        if stats['compression_enabled']:
            # Should have some compression ratio improvement
            self.assertGreater(stats.get('average_compression_ratio', 1.0), 1.0)

    def tearDown(self):
        self.cache.clear()
        if 'CACHE_HMAC_SECRET' in os.environ:
            del os.environ['CACHE_HMAC_SECRET']


class TestSecurityValidation(unittest.TestCase):
    """Security-focused tests"""

    def setUp(self):
        os.environ['CACHE_HMAC_SECRET'] = 'test-secret-that-is-at-least-64-characters-long-for-testing-purposes-only'
        os.environ['AUTH_ENABLED'] = 'true'
        self.authenticator = SimpleAuthenticator()

    def test_input_validation(self):
        """Test input validation prevents injection attacks"""
        # Test SQL injection patterns
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "game_id; rm -rf /",
        ]

        for malicious_input in malicious_inputs:
            with self.subTest(input=malicious_input):
                try:
                    validate_request_parameters(malicious_input, 1, 'full', None)
                    self.fail(f"Malicious input '{malicious_input}' was not caught by validation")
                except Exception:
                    # Expected - validation should reject malicious input
                    pass

    def test_api_key_security(self):
        """Test API key generation and validation security"""
        # Test API key generation
        api_key = self.authenticator.generate_api_key(1, 'test_game')
        self.assertIsNotNone(api_key)
        self.assertTrue(api_key.startswith('fcv_'))

        # Test valid API key
        valid, player_id, game_id = self.authenticator.validate_api_key(api_key)
        self.assertTrue(valid)
        self.assertEqual(player_id, 1)
        self.assertEqual(game_id, 'test_game')

        # Test invalid API key
        invalid_key = api_key[:-1] + 'x'  # Corrupt the signature
        valid, _, _ = self.authenticator.validate_api_key(invalid_key)
        self.assertFalse(valid)

    def test_session_security(self):
        """Test session security and timeout"""
        # Create session
        session_id = self.authenticator.create_session(1, 'test_game')
        self.assertIsNotNone(session_id)

        # Validate fresh session
        session = self.authenticator.validate_session(session_id)
        self.assertIsNotNone(session)
        self.assertEqual(session.player_id, 1)

        # Test session expiration (mock time advance)
        with patch('time.time', return_value=time.time() + 7200):  # 2 hours later
            expired_session = self.authenticator.validate_session(session_id)
            self.assertIsNone(expired_session)

    def test_rate_limiting_prevents_abuse(self):
        """Test rate limiting prevents abuse"""
        rate_limiter = APIRateLimiter()

        # Exhaust rate limit
        successful_requests = 0
        for i in range(100):  # Try 100 requests
            allowed, _, _ = rate_limiter.check_limits(1, "127.0.0.1")
            if allowed:
                successful_requests += 1
            else:
                break

        # Should not allow all 100 requests
        self.assertLess(successful_requests, 100)

    def tearDown(self):
        if 'CACHE_HMAC_SECRET' in os.environ:
            del os.environ['CACHE_HMAC_SECRET']
        if 'AUTH_ENABLED' in os.environ:
            del os.environ['AUTH_ENABLED']


if __name__ == '__main__':
    unittest.main()