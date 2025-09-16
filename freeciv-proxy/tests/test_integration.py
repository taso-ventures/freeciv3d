#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integration tests for LLM agent support in FreeCiv proxy
Tests end-to-end functionality including concurrent agents
"""

import unittest
import asyncio
import json
import time
import sys
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_handler import LLMWSHandler, llm_agents
from state_cache import StateCache, state_cache
from action_validator import LLMActionValidator, ActionType
from config_loader import LLMConfig
from civcom import CivCom
from tornado import web

class TestLLMIntegration(unittest.TestCase):
    """Integration tests for LLM agent functionality"""

    def setUp(self):
        """Set up integration test fixtures"""
        # Clear global state
        llm_agents.clear()
        state_cache.clear()

        # Create mock WebSocket connections
        self.mock_ws1 = Mock()
        self.mock_ws1.write_message = Mock()
        self.mock_ws1.close = Mock()

        self.mock_ws2 = Mock()
        self.mock_ws2.write_message = Mock()
        self.mock_ws2.close = Mock()

    def create_mock_handler(self):
        """Helper to create mock LLM handler"""
        mock_app = Mock()
        mock_app.ui_methods = {}
        mock_app.ui_modules = {}
        mock_request = Mock()

        handler = LLMWSHandler(mock_app, mock_request)
        handler.write_message = Mock()
        handler.close = Mock()
        handler.set_nodelay = Mock()  # Mock the WebSocket method
        handler.ws_connection = Mock()  # Mock WebSocket connection
        return handler

    def tearDown(self):
        """Clean up after tests"""
        llm_agents.clear()
        state_cache.clear()

    def test_llm_agent_authentication_flow(self):
        """Test complete LLM agent authentication process"""
        # Create LLM handler
        handler = self.create_mock_handler()

        # Test connection opening
        handler.open()
        handler.write_message.assert_called_once()

        # Verify welcome message
        call_args = handler.write_message.call_args[0][0]
        welcome_msg = json.loads(call_args)
        self.assertEqual(welcome_msg['type'], 'welcome')
        self.assertIn('handler_id', welcome_msg)

        # Test authentication message
        auth_msg = {
            'type': 'llm_connect',
            'agent_id': 'test-agent-1',
            'api_token': 'test-token-123',
            'capabilities': ['unit_move', 'city_production'],
            'port': 6001
        }

        with patch.object(handler, '_connect_to_civserver'):
            handler._handle_llm_connect(auth_msg)

        # Verify authentication success
        self.assertTrue(handler.is_llm_agent)
        self.assertEqual(handler.agent_id, 'test-agent-1')
        self.assertIn('test-agent-1', llm_agents)

    def test_state_caching_with_optimization(self):
        """Test state caching and optimization for LLM agents"""
        cache = StateCache(ttl=1, max_size_kb=4)

        # Test large state that should be optimized
        large_state = {
            'turn': 50,
            'units': [{'id': i, 'type': 'warrior', 'x': i, 'y': i} for i in range(100)],
            'cities': [{'id': i, 'name': f'City{i}', 'population': 10} for i in range(50)],
            'visible_tiles': [{'x': i, 'y': j, 'terrain': 'grassland'} for i in range(20) for j in range(20)]
        }

        # Should successfully cache after optimization
        result = cache.set('large_state', large_state, player_id=1)
        self.assertTrue(result)

        # Should retrieve cached data
        cached = cache.get('large_state')
        self.assertIsNotNone(cached)
        self.assertIn('turn', cached)

        # Verify optimization reduced size
        original_size = len(json.dumps(large_state).encode('utf-8'))
        optimized_size = len(json.dumps(cached).encode('utf-8'))
        self.assertLess(optimized_size, original_size)
        self.assertLess(optimized_size, 4 * 1024)  # Under 4KB

    def test_action_validation_comprehensive(self):
        """Test comprehensive action validation scenarios"""
        validator = LLMActionValidator()

        # Test valid unit move
        valid_move = {
            'type': 'unit_move',
            'unit_id': 1,
            'dest_x': 10,
            'dest_y': 20,
            'player_id': 1
        }

        game_state = {
            'units': [{'id': 1, 'owner': 1, 'x': 9, 'y': 20, 'moves_left': 1}]
        }

        result = validator.validate_action(valid_move, 1, game_state)
        self.assertTrue(result.is_valid)

        # Test invalid action - wrong player
        invalid_move = valid_move.copy()
        invalid_move['player_id'] = 2

        result = validator.validate_action(invalid_move, 1, game_state)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.error_code, 'E005')

        # Test capability restriction
        restricted_action = {
            'type': 'declare_war',
            'target_player': 2,
            'player_id': 1
        }

        result = validator.validate_action(restricted_action, 1, game_state)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.error_code, 'E003')  # Unknown action type

    def test_concurrent_agent_isolation(self):
        """Test that concurrent agents have isolated state"""
        # Create two mock handlers
        handler1 = self.create_mock_handler()
        handler1.agent_id = 'agent-1'
        handler1.player_id = 1
        handler1.is_llm_agent = True

        handler2 = self.create_mock_handler()
        handler2.agent_id = 'agent-2'
        handler2.player_id = 2
        handler2.is_llm_agent = True

        # Register both agents
        llm_agents['agent-1'] = handler1
        llm_agents['agent-2'] = handler2

        # Test state isolation
        state1 = {'player_id': 1, 'units': [{'id': 1, 'owner': 1}]}
        state2 = {'player_id': 2, 'units': [{'id': 2, 'owner': 2}]}

        cache_key1 = 'state_1_llm_optimized_123'
        cache_key2 = 'state_2_llm_optimized_123'

        state_cache.set(cache_key1, state1, 1)
        state_cache.set(cache_key2, state2, 2)

        # Verify agents get their own state
        cached1 = state_cache.get(cache_key1)
        cached2 = state_cache.get(cache_key2)

        self.assertEqual(cached1['player_id'], 1)
        self.assertEqual(cached2['player_id'], 2)

        # Test cache invalidation by player
        state_cache.invalidate(player_id=1)

        cached1_after = state_cache.get(cache_key1)
        cached2_after = state_cache.get(cache_key2)

        self.assertIsNone(cached1_after)  # Player 1 cache cleared
        self.assertIsNotNone(cached2_after)  # Player 2 cache intact

    def test_agent_capacity_enforcement(self):
        """Test maximum agent capacity enforcement"""
        from llm_handler import MAX_LLM_AGENTS

        # Fill up to capacity
        for i in range(MAX_LLM_AGENTS):
            handler = self.create_mock_handler()
            handler.agent_id = f'agent-{i}'
            llm_agents[f'agent-{i}'] = handler

        # Try to add one more agent
        overflow_handler = self.create_mock_handler()

        # Should reject the connection
        overflow_handler.open()

        # Verify capacity enforcement
        self.assertEqual(len(llm_agents), MAX_LLM_AGENTS)
        overflow_handler.close.assert_called_once()

    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        handler = self.create_mock_handler()
        handler.rate_limit_tokens = 5  # Set low limit for testing

        # Should allow requests while tokens available
        for i in range(5):
            result = handler._check_rate_limit()
            self.assertTrue(result)

        # Should deny when tokens exhausted
        result = handler._check_rate_limit()
        self.assertFalse(result)

    def test_configuration_loading(self):
        """Test configuration loading and defaults"""
        config = LLMConfig()

        # Test default values
        self.assertTrue(config.is_enabled())
        self.assertEqual(config.get_max_agents(), 2)
        self.assertEqual(config.get_cache_ttl(), 5)
        self.assertTrue(config.is_strict_validation())

        # Test token validation
        self.assertTrue(config.validate_token('test-token-123'))
        self.assertFalse(config.validate_token('short'))

    def test_civcom_llm_state_methods(self):
        """Test CivCom LLM-optimized state methods"""
        # Create mock CivCom
        civcom = CivCom('test-user', 6001, 'test-key', None)

        # Set up mock game state
        civcom.game_turn = 25
        civcom.game_phase = 'movement'
        civcom.player_units = [
            {'id': 1, 'type': 'warrior', 'x': 10, 'y': 20, 'moves_left': 1},
            {'id': 2, 'type': 'settler', 'x': 11, 'y': 21, 'moves_left': 2}
        ]
        civcom.player_cities = [
            {'id': 1, 'name': 'Capital', 'x': 10, 'y': 20, 'population': 5}
        ]

        # Test LLM-optimized state building
        state = civcom.build_llm_optimized_state(player_id=1)

        self.assertEqual(state['turn'], 25)
        self.assertEqual(state['phase'], 'movement')
        self.assertIn('strategic', state)
        self.assertIn('tactical', state)
        self.assertIn('economic', state)
        self.assertIn('legal_actions', state)

        # Verify state is properly compressed
        state_json = json.dumps(state, separators=(',', ':'))
        state_size = len(state_json.encode('utf-8'))
        self.assertLess(state_size, 4 * 1024)  # Under 4KB

    def test_performance_requirements(self):
        """Test performance requirements are met"""
        cache = StateCache(ttl=5, max_size_kb=4)

        # Pre-populate cache
        test_state = {
            'turn': 1,
            'strategic': {'score': 100},
            'tactical': {'units': []},
            'economic': {'gold': 50}
        }
        cache.set('perf_test', test_state, player_id=1)

        # Test cache hit performance
        start_time = time.time()
        for _ in range(100):
            cached_state = cache.get('perf_test')
            self.assertIsNotNone(cached_state)
        end_time = time.time()

        # Should complete 100 cache hits in well under 50ms total
        total_time_ms = (end_time - start_time) * 1000
        avg_time_ms = total_time_ms / 100

        self.assertLess(avg_time_ms, 1.0)  # Average < 1ms per query (well under 50ms target)

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery scenarios"""
        handler = self.create_mock_handler()

        # Test malformed JSON message
        handler.on_message('invalid json')

        # Should send error response
        handler.write_message.assert_called()
        call_args = handler.write_message.call_args[0][0]
        error_msg = json.loads(call_args)
        self.assertEqual(error_msg['type'], 'error')
        self.assertEqual(error_msg['code'], 'E102')

        # Test unknown message type
        handler.write_message.reset_mock()
        handler.on_message(json.dumps({'type': 'unknown_message'}))

        handler.write_message.assert_called()
        call_args = handler.write_message.call_args[0][0]
        error_msg = json.loads(call_args)
        self.assertEqual(error_msg['code'], 'E103')

    def test_cleanup_on_disconnection(self):
        """Test proper cleanup when agents disconnect"""
        handler = self.create_mock_handler()
        handler.agent_id = 'cleanup-test-agent'
        handler.player_id = 1
        handler.civcom = Mock()
        handler.civcom.stopped = False
        handler.civcom.close_connection = Mock()

        # Register agent
        llm_agents['cleanup-test-agent'] = handler

        # Add some cache entries
        state_cache.set('cleanup_state_1', {'data': 'test'}, player_id=1)

        # Verify setup
        self.assertIn('cleanup-test-agent', llm_agents)
        self.assertIsNotNone(state_cache.get('cleanup_state_1'))

        # Trigger cleanup
        handler.on_close()

        # Verify cleanup
        self.assertNotIn('cleanup-test-agent', llm_agents)
        self.assertTrue(handler.civcom.stopped)
        handler.civcom.close_connection.assert_called_once()

if __name__ == '__main__':
    unittest.main(verbosity=2)