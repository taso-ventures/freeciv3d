#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for LLM WebSocket Protocol message handlers
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import the modules we're testing
try:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from llm_protocol import MessageType, LLMMessage
    from message_handlers import MessageHandlerRegistry, UnknownMessageTypeError
except ImportError:
    # Will fail initially until we implement the module
    MessageType = None
    LLMMessage = None
    MessageHandlerRegistry = None
    UnknownMessageTypeError = None


class TestMessageHandlerRegistry:
    """Test MessageHandlerRegistry class"""

    def test_handler_registry_initialization(self):
        """Test that registry initializes with all required handlers"""
        if MessageHandlerRegistry is None:
            pytest.skip("MessageHandlerRegistry not implemented yet")

        registry = MessageHandlerRegistry()

        # Should have handlers for all message types
        expected_handlers = [
            MessageType.LLM_CONNECT,
            MessageType.STATE_QUERY,
            MessageType.ACTION,
            MessageType.TURN_START,
            MessageType.TURN_END,
            MessageType.SPECTATOR_UPDATE
        ]

        for msg_type in expected_handlers:
            assert msg_type in registry.handlers

    def test_handler_registration(self):
        """Test manual handler registration"""
        if MessageHandlerRegistry is None:
            pytest.skip("MessageHandlerRegistry not implemented yet")

        registry = MessageHandlerRegistry()

        async def test_handler(message, connection):
            return {"status": "test"}

        # Register a custom handler
        registry.register_handler(MessageType.LLM_CONNECT, test_handler)

        assert registry.handlers[MessageType.LLM_CONNECT] == test_handler

    @pytest.mark.asyncio
    async def test_unknown_message_type_handling(self):
        """Test handling of unknown message types"""
        if MessageHandlerRegistry is None:
            pytest.skip("MessageHandlerRegistry not implemented yet")

        registry = MessageHandlerRegistry()
        mock_connection = Mock()

        # Create a message with unknown type
        unknown_msg = Mock()
        unknown_msg.type = "unknown_type"

        with pytest.raises(UnknownMessageTypeError):
            await registry.handle_message(unknown_msg, mock_connection)

    @pytest.mark.asyncio
    async def test_message_routing(self):
        """Test that messages are routed to correct handlers"""
        if MessageHandlerRegistry is None:
            pytest.skip("MessageHandlerRegistry not implemented yet")

        registry = MessageHandlerRegistry()
        mock_connection = Mock()

        # Mock a handler
        mock_handler = AsyncMock(return_value={"status": "success"})
        registry.handlers[MessageType.LLM_CONNECT] = mock_handler

        # Create test message
        message = Mock()
        message.type = MessageType.LLM_CONNECT

        result = await registry.handle_message(message, mock_connection)

        mock_handler.assert_called_once_with(message, mock_connection)
        assert result["status"] == "success"


class TestConnectHandler:
    """Test LLM_CONNECT message handler"""

    @pytest.mark.asyncio
    async def test_handle_connect_success(self):
        """Test successful LLM agent connection"""
        if MessageHandlerRegistry is None:
            pytest.skip("MessageHandlerRegistry not implemented yet")

        registry = MessageHandlerRegistry()
        mock_connection = Mock()

        # Mock successful authentication
        message = Mock()
        message.type = MessageType.LLM_CONNECT
        message.agent_id = "test-agent"
        message.data = {
            "api_token": "valid-token",
            "model": "gpt-4",
            "game_id": "game-123"
        }

        with patch('freeciv_proxy.protocol.message_handlers.validate_token', return_value=True):
            result = await registry.handle_connect(message, mock_connection)

        assert result["type"] == "auth_success"
        assert result["agent_id"] == "test-agent"
        assert "session_id" in result
        assert "player_id" in result

    @pytest.mark.asyncio
    async def test_handle_connect_invalid_token(self):
        """Test connection with invalid API token"""
        if MessageHandlerRegistry is None:
            pytest.skip("MessageHandlerRegistry not implemented yet")

        registry = MessageHandlerRegistry()
        mock_connection = Mock()

        message = Mock()
        message.type = MessageType.LLM_CONNECT
        message.agent_id = "test-agent"
        message.data = {
            "api_token": "invalid-token",
            "model": "gpt-4",
            "game_id": "game-123"
        }

        with patch('freeciv_proxy.protocol.message_handlers.validate_token', return_value=False):
            result = await registry.handle_connect(message, mock_connection)

        assert result["type"] == "error"
        assert "Invalid API token" in result["message"]

    @pytest.mark.asyncio
    async def test_handle_connect_missing_data(self):
        """Test connection with missing required data"""
        if MessageHandlerRegistry is None:
            pytest.skip("MessageHandlerRegistry not implemented yet")

        registry = MessageHandlerRegistry()
        mock_connection = Mock()

        message = Mock()
        message.type = MessageType.LLM_CONNECT
        message.agent_id = "test-agent"
        message.data = {
            # Missing api_token
            "model": "gpt-4",
            "game_id": "game-123"
        }

        result = await registry.handle_connect(message, mock_connection)

        assert result["type"] == "error"
        assert "api_token" in result["message"].lower()


class TestStateQueryHandler:
    """Test STATE_QUERY message handler"""

    @pytest.mark.asyncio
    async def test_handle_state_query_full_format(self):
        """Test state query with full format"""
        if MessageHandlerRegistry is None:
            pytest.skip("MessageHandlerRegistry not implemented yet")

        registry = MessageHandlerRegistry()
        mock_connection = Mock()
        mock_connection.is_llm_agent = True
        mock_connection.player_id = 1

        message = Mock()
        message.type = MessageType.STATE_QUERY
        message.agent_id = "test-agent"
        message.data = {
            "format": "full",
            "include_actions": True
        }
        message.correlation_id = "state-123"

        # Mock civcom
        mock_civcom = Mock()
        mock_civcom.get_full_state.return_value = {
            "turn": 1,
            "units": [],
            "cities": [],
            "players": {}
        }

        with patch.object(registry, '_get_game_state', return_value=mock_civcom.get_full_state.return_value):
            result = await registry.handle_state_query(message, mock_connection)

        assert result["type"] == "state_response"
        assert result["format"] == "full"
        assert result["correlation_id"] == "state-123"
        assert "data" in result

    @pytest.mark.asyncio
    async def test_handle_state_query_llm_optimized(self):
        """Test state query with LLM optimized format"""
        if MessageHandlerRegistry is None:
            pytest.skip("MessageHandlerRegistry not implemented yet")

        registry = MessageHandlerRegistry()
        mock_connection = Mock()
        mock_connection.is_llm_agent = True
        mock_connection.player_id = 1

        message = Mock()
        message.type = MessageType.STATE_QUERY
        message.data = {
            "format": "llm_optimized",
            "include_actions": False
        }
        message.correlation_id = "state-456"

        with patch.object(registry, '_get_optimized_state') as mock_optimized:
            mock_optimized.return_value = {
                "strategic_summary": {},
                "immediate_priorities": [],
                "threats": [],
                "opportunities": []
            }

            result = await registry.handle_state_query(message, mock_connection)

        assert result["type"] == "state_response"
        assert result["format"] == "llm_optimized"
        assert "strategic_summary" in result["data"]

    @pytest.mark.asyncio
    async def test_handle_state_query_unauthorized(self):
        """Test state query from unauthorized connection"""
        if MessageHandlerRegistry is None:
            pytest.skip("MessageHandlerRegistry not implemented yet")

        registry = MessageHandlerRegistry()
        mock_connection = Mock()
        mock_connection.is_llm_agent = False  # Not authenticated

        message = Mock()
        message.type = MessageType.STATE_QUERY
        message.data = {"format": "full"}

        result = await registry.handle_state_query(message, mock_connection)

        assert result["type"] == "error"
        assert "not authenticated" in result["message"].lower()


class TestActionHandler:
    """Test ACTION message handler"""

    @pytest.mark.asyncio
    async def test_handle_action_valid_move(self):
        """Test handling valid unit move action"""
        if MessageHandlerRegistry is None:
            pytest.skip("MessageHandlerRegistry not implemented yet")

        registry = MessageHandlerRegistry()
        mock_connection = Mock()
        mock_connection.is_llm_agent = True
        mock_connection.player_id = 1

        message = Mock()
        message.type = MessageType.ACTION
        message.data = {
            "action_type": "unit_move",
            "actor_id": 42,
            "target": {"x": 10, "y": 20},
            "parameters": {}
        }
        message.correlation_id = "action-789"

        # Mock action validator
        with patch('freeciv_proxy.protocol.message_handlers.validate_action', return_value=True):
            with patch.object(registry, '_execute_action') as mock_execute:
                mock_execute.return_value = {"success": True}

                result = await registry.handle_action(message, mock_connection)

        assert result["type"] == "action_result"
        assert result["success"] is True
        assert result["correlation_id"] == "action-789"

    @pytest.mark.asyncio
    async def test_handle_action_invalid(self):
        """Test handling invalid action"""
        if MessageHandlerRegistry is None:
            pytest.skip("MessageHandlerRegistry not implemented yet")

        registry = MessageHandlerRegistry()
        mock_connection = Mock()
        mock_connection.is_llm_agent = True
        mock_connection.player_id = 1

        message = Mock()
        message.type = MessageType.ACTION
        message.data = {
            "action_type": "invalid_action",
            "actor_id": 999,  # Non-existent unit
        }

        with patch('freeciv_proxy.protocol.message_handlers.validate_action', return_value=False):
            result = await registry.handle_action(message, mock_connection)

        assert result["type"] == "action_result"
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_handle_action_unauthorized(self):
        """Test action from unauthorized connection"""
        if MessageHandlerRegistry is None:
            pytest.skip("MessageHandlerRegistry not implemented yet")

        registry = MessageHandlerRegistry()
        mock_connection = Mock()
        mock_connection.is_llm_agent = False

        message = Mock()
        message.type = MessageType.ACTION
        message.data = {"action_type": "unit_move"}

        result = await registry.handle_action(message, mock_connection)

        assert result["type"] == "error"
        assert "not authenticated" in result["message"].lower()


class TestErrorHandling:
    """Test error handling in message handlers"""

    @pytest.mark.asyncio
    async def test_handler_exception_handling(self):
        """Test that handler exceptions are properly caught"""
        if MessageHandlerRegistry is None:
            pytest.skip("MessageHandlerRegistry not implemented yet")

        registry = MessageHandlerRegistry()
        mock_connection = Mock()

        # Create a handler that raises an exception
        async def failing_handler(message, connection):
            raise Exception("Test exception")

        registry.handlers[MessageType.LLM_CONNECT] = failing_handler

        message = Mock()
        message.type = MessageType.LLM_CONNECT

        result = await registry.handle_message(message, mock_connection)

        assert result["type"] == "error"
        assert "internal error" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_correlation_id_preservation(self):
        """Test that correlation IDs are preserved in responses"""
        if MessageHandlerRegistry is None:
            pytest.skip("MessageHandlerRegistry not implemented yet")

        registry = MessageHandlerRegistry()
        mock_connection = Mock()
        mock_connection.is_llm_agent = True

        message = Mock()
        message.type = MessageType.STATE_QUERY
        message.correlation_id = "preserve-me"
        message.data = {"format": "full"}

        with patch.object(registry, '_get_game_state', return_value={}):
            result = await registry.handle_state_query(message, mock_connection)

        assert result["correlation_id"] == "preserve-me"


class TestInputValidation:
    """Test input validation functionality"""

    @pytest.mark.asyncio
    async def test_validation_invalid_agent_id(self):
        """Test validation of invalid agent IDs"""
        if MessageHandlerRegistry is None:
            pytest.skip("MessageHandlerRegistry not implemented yet")

        registry = MessageHandlerRegistry()

        # Test with invalid characters
        message = Mock()
        message.type = MessageType.LLM_CONNECT
        message.agent_id = "invalid@agent"  # @ is not allowed
        message.data = {
            "api_token": "valid-token-123456789",
            "model": "gpt-4",
            "game_id": "test-game"
        }

        validation_error = registry._validate_message_data(message, ["api_token", "model", "game_id"])
        assert validation_error is not None
        assert "invalid characters" in validation_error.lower()

    @pytest.mark.asyncio
    async def test_validation_short_api_token(self):
        """Test validation of short API tokens"""
        if MessageHandlerRegistry is None:
            pytest.skip("MessageHandlerRegistry not implemented yet")

        registry = MessageHandlerRegistry()

        message = Mock()
        message.type = MessageType.LLM_CONNECT
        message.data = {
            "api_token": "short",  # Too short
            "model": "gpt-4",
            "game_id": "test-game"
        }

        validation_error = registry._validate_message_data(message, ["api_token", "model", "game_id"])
        assert validation_error is not None
        assert "too short" in validation_error.lower()

    @pytest.mark.asyncio
    async def test_error_message_handling(self):
        """Test ERROR message type handling"""
        if MessageHandlerRegistry is None:
            pytest.skip("MessageHandlerRegistry not implemented yet")

        registry = MessageHandlerRegistry()
        mock_connection = Mock()

        message = Mock()
        message.type = MessageType.ERROR
        message.agent_id = "test-agent"
        message.data = {
            "error_code": "E102",
            "error_message": "Test error"
        }
        message.correlation_id = "error-test"

        result = await registry.handle_error(message, mock_connection)

        assert result["type"] == "error_acknowledged"
        assert result["data"]["success"] is True
        assert result["correlation_id"] == "error-test"