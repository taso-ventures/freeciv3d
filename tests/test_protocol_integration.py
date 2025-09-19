#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integration tests for LLM protocol between Gateway and FreeCiv proxy
"""

import asyncio
import json
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import components from both FC-3 and FC-4
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'freeciv-proxy', 'protocol'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llm-gateway'))

    from llm_protocol import MessageType, LLMMessage, create_connect_message, create_state_query_message
    from message_handlers import MessageHandlerRegistry
    from main import LLMGateway
    from connection_manager import ConnectionManager
except ImportError as e:
    pytest.skip(f"Integration test modules not available: {e}", allow_module_level=True)


class TestProtocolIntegration:
    """Test end-to-end protocol integration"""

    @pytest.mark.asyncio
    async def test_agent_registration_flow(self):
        """Test complete agent registration flow from Gateway to FreeCiv"""
        # Setup gateway
        gateway = LLMGateway()
        await gateway.start()

        # Setup protocol handler
        handler_registry = MessageHandlerRegistry()

        try:
            # Test agent registration
            agent_config = {
                "agent_id": "test-agent",
                "api_token": "test-token-123456789",
                "model": "gpt-4",
                "game_id": "integration-game"
            }

            result = await gateway.register_agent("test-agent", agent_config)

            assert result["success"] is True
            assert result["agent_id"] == "test-agent"
            assert "session_id" in result

            # Verify agent is registered
            assert "test-agent" in gateway.active_agents
            assert gateway.active_agents["test-agent"]["config"] == agent_config

        finally:
            await gateway.stop()

    @pytest.mark.asyncio
    async def test_message_routing_gateway_to_freeciv(self):
        """Test message routing from Gateway to FreeCiv proxy"""
        # Setup components
        gateway = LLMGateway()
        handler_registry = MessageHandlerRegistry()

        await gateway.start()

        try:
            # Register agent first
            agent_config = {
                "agent_id": "routing-test",
                "api_token": "test-token-123456789",
                "model": "gpt-4",
                "game_id": "routing-game"
            }

            await gateway.register_agent("routing-test", agent_config)

            # Create LLM protocol message
            llm_message = create_state_query_message(
                "routing-test",
                format_type="llm_optimized",
                include_actions=True
            )

            # Mock FreeCiv proxy connection
            with patch.object(gateway, 'forward_to_proxy') as mock_forward:
                mock_forward.return_value = None

                # Route message through gateway
                result = await gateway.route_message("game_arena", "freeciv", llm_message.to_json())

                # Verify message was routed
                assert result["success"] is True
                mock_forward.assert_called_once()

        finally:
            await gateway.stop()

    @pytest.mark.asyncio
    async def test_state_query_end_to_end(self):
        """Test complete state query flow"""
        # Setup components
        gateway = LLMGateway()
        handler_registry = MessageHandlerRegistry()
        mock_connection = Mock()
        mock_connection.is_llm_agent = True
        mock_connection.player_id = 1

        await gateway.start()

        try:
            # Register agent
            await gateway.register_agent("state-test", {
                "agent_id": "state-test",
                "api_token": "test-token-123456789",
                "model": "gpt-4",
                "game_id": "state-game"
            })

            # Create game session
            game_result = await gateway.create_game({
                "ruleset": "classic",
                "map_size": "small"
            })

            assert game_result["success"] is True
            game_id = game_result["game_id"]

            # Test state query through gateway
            state_result = await gateway.get_game_state(game_id, 1, "llm_optimized")

            assert state_result["success"] is True
            assert state_result["format"] == "llm_optimized"
            assert "data" in state_result

            # Test protocol message handling
            state_query_message = create_state_query_message(
                "state-test",
                format_type="full",
                include_actions=False
            )

            # Mock civcom connection
            with patch('message_handlers.MessageHandlerRegistry._get_game_state') as mock_state:
                mock_state.return_value = {
                    "turn": 1,
                    "units": [],
                    "cities": []
                }

                result = await handler_registry.handle_state_query(state_query_message, mock_connection)

                assert result["data"]["type"] == "state_response"
                assert result["data"]["format"] == "full"

        finally:
            await gateway.stop()

    @pytest.mark.asyncio
    async def test_action_submission_end_to_end(self):
        """Test complete action submission flow"""
        gateway = LLMGateway()
        handler_registry = MessageHandlerRegistry()
        mock_connection = Mock()
        mock_connection.is_llm_agent = True
        mock_connection.player_id = 1

        await gateway.start()

        try:
            # Register agent and create game
            await gateway.register_agent("action-test", {
                "agent_id": "action-test",
                "api_token": "test-token-123456789",
                "model": "gpt-4",
                "game_id": "action-game"
            })

            game_result = await gateway.create_game({
                "ruleset": "classic",
                "map_size": "small"
            })

            game_id = game_result["game_id"]

            # Test action submission through gateway
            action = {
                "action_type": "unit_move",
                "actor_id": 42,
                "target": {"x": 11, "y": 21},
                "player_id": 1
            }

            action_result = await gateway.submit_action(game_id, action)

            assert action_result["success"] is True
            assert "action_id" in action_result

            # Test protocol message handling
            from llm_protocol import create_action_message

            action_message = create_action_message(
                "action-test",
                "unit_move",
                42,
                {"x": 11, "y": 21}
            )

            # Mock action validation and execution
            with patch('message_handlers.MessageHandlerRegistry._validate_action', return_value=True):
                with patch('message_handlers.MessageHandlerRegistry._execute_action') as mock_execute:
                    mock_execute.return_value = {"success": True}

                    result = await handler_registry.handle_action(action_message, mock_connection)

                    assert result["data"]["type"] == "action_result"
                    assert result["data"]["success"] is True

        finally:
            await gateway.stop()

    @pytest.mark.asyncio
    async def test_connection_recovery(self):
        """Test connection recovery and resilience"""
        gateway = LLMGateway()
        connection_manager = ConnectionManager()

        await gateway.start()
        await connection_manager.start()

        try:
            # Test gateway proxy connection recovery
            game_id = "recovery-test"

            # Simulate initial connection
            with patch('websockets.connect') as mock_connect:
                mock_ws = AsyncMock()
                mock_connect.return_value = mock_ws

                result = await gateway.connect_to_freeciv_proxy(game_id)

                assert result["success"] is True
                assert game_id in gateway.proxy_connections

                # Simulate connection failure and recovery
                gateway.proxy_connections[game_id] = None

                # Try to send message (should trigger reconnection)
                with patch.object(gateway, 'connect_to_freeciv_proxy') as mock_reconnect:
                    mock_reconnect.return_value = {"success": True}

                    await gateway.forward_to_proxy(game_id, {"test": "message"})

                    mock_reconnect.assert_called_once_with(game_id)

        finally:
            await connection_manager.stop()
            await gateway.stop()

    @pytest.mark.asyncio
    async def test_concurrent_agents(self):
        """Test multiple agents operating concurrently"""
        gateway = LLMGateway()
        await gateway.start()

        try:
            # Register multiple agents
            agents = []
            for i in range(3):
                agent_id = f"concurrent-agent-{i}"
                config = {
                    "agent_id": agent_id,
                    "api_token": f"test-token-{i}23456789",
                    "model": "gpt-4",
                    "game_id": f"concurrent-game-{i}"
                }

                result = await gateway.register_agent(agent_id, config)
                assert result["success"] is True
                agents.append(agent_id)

            # Verify all agents are registered
            assert len(gateway.active_agents) == 3

            # Test concurrent operations
            tasks = []
            for i, agent_id in enumerate(agents):
                # Create game for each agent
                task = gateway.create_game({
                    "ruleset": "classic",
                    "map_size": "small"
                })
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            # Verify all games created successfully
            for result in results:
                assert result["success"] is True

            assert len(gateway.game_sessions) == 3

        finally:
            await gateway.stop()

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery scenarios"""
        gateway = LLMGateway()
        handler_registry = MessageHandlerRegistry()

        await gateway.start()

        try:
            # Test invalid agent registration
            invalid_config = {
                "agent_id": "error-test",
                # Missing required fields
                "model": "gpt-4"
            }

            result = await gateway.register_agent("error-test", invalid_config)
            assert result["success"] is False
            assert "Missing required field" in result["error"]

            # Test invalid action handling
            mock_connection = Mock()
            mock_connection.is_llm_agent = True
            mock_connection.player_id = 1

            invalid_action = LLMMessage(
                type=MessageType.ACTION,
                agent_id="error-test",
                timestamp=time.time(),
                data={"invalid": "action"}  # Missing required fields
            )

            result = await handler_registry.handle_action(invalid_action, mock_connection)
            assert result["data"]["success"] is False
            assert "Missing required field" in result["data"]["error_message"]

        finally:
            await gateway.stop()

    @pytest.mark.asyncio
    async def test_message_correlation(self):
        """Test message correlation for async request/response"""
        handler_registry = MessageHandlerRegistry()
        mock_connection = Mock()
        mock_connection.is_llm_agent = True
        mock_connection.player_id = 1

        # Create message with correlation ID
        correlation_id = "test-correlation-123"
        state_query = LLMMessage(
            type=MessageType.STATE_QUERY,
            agent_id="correlation-test",
            timestamp=time.time(),
            data={"format": "llm_optimized"},
            correlation_id=correlation_id
        )

        # Mock state retrieval
        with patch('message_handlers.MessageHandlerRegistry._get_game_state') as mock_state:
            mock_state.return_value = {"turn": 1}

            result = await handler_registry.handle_state_query(state_query, mock_connection)

            # Verify correlation ID is preserved
            assert result["correlation_id"] == correlation_id
            assert result["data"]["type"] == "state_response"


class TestProtocolCompatibility:
    """Test protocol compatibility and version handling"""

    def test_message_serialization_compatibility(self):
        """Test that messages serialize/deserialize correctly"""
        # Test all message types
        test_messages = [
            create_connect_message("test-agent", "token123", "gpt-4", "game-123"),
            create_state_query_message("test-agent", "full", True),
            LLMMessage(
                type=MessageType.ACTION,
                agent_id="test-agent",
                timestamp=time.time(),
                data={"action_type": "unit_move", "actor_id": 42}
            )
        ]

        for original_message in test_messages:
            # Serialize and deserialize
            json_str = original_message.to_json()
            restored_message = LLMMessage.from_json(json_str)

            # Verify all fields are preserved
            assert restored_message.type == original_message.type
            assert restored_message.agent_id == original_message.agent_id
            assert restored_message.data == original_message.data
            assert restored_message.correlation_id == original_message.correlation_id

    def test_protocol_error_handling(self):
        """Test protocol-level error handling"""
        # Test invalid JSON
        with pytest.raises((ValueError, json.JSONDecodeError)):
            LLMMessage.from_json("invalid json")

        # Test missing required fields
        incomplete_data = {
            "type": "state_query",
            "agent_id": "test-agent"
            # Missing timestamp and data
        }

        with pytest.raises(ValueError):
            LLMMessage.from_json(json.dumps(incomplete_data))

    def test_message_size_limits(self):
        """Test handling of large messages"""
        # Create large message
        large_data = {"large_field": "x" * 100000}  # 100KB

        large_message = LLMMessage(
            type=MessageType.STATE_QUERY,
            agent_id="test-agent",
            timestamp=time.time(),
            data=large_data
        )

        # Should serialize without error
        json_str = large_message.to_json()
        assert len(json_str) > 100000

        # Should deserialize correctly
        restored = LLMMessage.from_json(json_str)
        assert restored.data == large_data


class TestLoadAndStress:
    """Test system under load and stress conditions"""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_connections(self):
        """Test handling multiple concurrent connections"""
        gateway = LLMGateway()
        await gateway.start()

        try:
            # Register many agents concurrently
            agent_count = 10
            tasks = []

            for i in range(agent_count):
                config = {
                    "agent_id": f"load-test-{i}",
                    "api_token": f"token-{i}123456789",
                    "model": "gpt-4",
                    "game_id": f"load-game-{i}"
                }
                task = gateway.register_agent(f"load-test-{i}", config)
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            # Verify all registrations succeeded
            successful = sum(1 for result in results if result["success"])
            assert successful == agent_count

        finally:
            await gateway.stop()

    @pytest.mark.asyncio
    async def test_rapid_message_processing(self):
        """Test rapid message processing"""
        handler_registry = MessageHandlerRegistry()
        mock_connection = Mock()
        mock_connection.is_llm_agent = True
        mock_connection.player_id = 1

        # Send many messages rapidly
        message_count = 100
        tasks = []

        with patch('message_handlers.MessageHandlerRegistry._get_game_state') as mock_state:
            mock_state.return_value = {"turn": 1}

            for i in range(message_count):
                message = create_state_query_message(f"rapid-test-{i}")
                task = handler_registry.handle_state_query(message, mock_connection)
                tasks.append(task)

            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()

            # Verify all messages processed successfully
            successful = sum(1 for result in results if result["data"]["type"] == "state_response")
            assert successful == message_count

            # Check processing time (should be reasonable)
            processing_time = end_time - start_time
            assert processing_time < 5.0  # Should process 100 messages in under 5 seconds