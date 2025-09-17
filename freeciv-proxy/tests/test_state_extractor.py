#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test suite for State Extraction Service
Tests REST API endpoints for game state extraction and LLM optimization
"""

import unittest
import asyncio
import json
import time
import sys
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from tornado.testing import AsyncHTTPTestCase
from tornado import web
from tornado.httpclient import HTTPResponse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state_extractor import StateExtractor, StateFormat
from state_cache import StateCache, CacheEntry


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


if __name__ == '__main__':
    unittest.main()