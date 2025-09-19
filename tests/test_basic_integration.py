#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Basic integration tests for FC-3 and FC-4 components
"""

import pytest
import json
import time

# Simple integration test without complex dependencies
class TestBasicIntegration:
    """Basic integration tests"""

    def test_protocol_message_creation(self):
        """Test that protocol messages can be created"""
        try:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'freeciv-proxy', 'protocol'))

            from llm_protocol import MessageType, LLMMessage, create_connect_message

            # Test message creation
            message = create_connect_message("test-agent", "token123", "gpt-4", "game-123")

            assert message.type == MessageType.LLM_CONNECT
            assert message.agent_id == "test-agent"
            assert message.data["api_token"] == "token123"
            assert message.data["model"] == "gpt-4"
            assert message.data["game_id"] == "game-123"

            # Test serialization
            json_str = message.to_json()
            assert isinstance(json_str, str)

            # Test deserialization
            restored = LLMMessage.from_json(json_str)
            assert restored.type == message.type
            assert restored.agent_id == message.agent_id
            assert restored.data == message.data

        except ImportError:
            pytest.skip("Protocol module not available")

    def test_gateway_initialization(self):
        """Test that gateway can be initialized"""
        try:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llm-gateway'))

            from main import LLMGateway

            # Test gateway creation
            gateway = LLMGateway()

            assert hasattr(gateway, 'active_agents')
            assert hasattr(gateway, 'game_sessions')
            assert hasattr(gateway, 'proxy_connections')
            assert isinstance(gateway.active_agents, dict)
            assert isinstance(gateway.game_sessions, dict)
            assert isinstance(gateway.proxy_connections, dict)

        except ImportError:
            pytest.skip("Gateway module not available")

    def test_protocol_handler_initialization(self):
        """Test that protocol handlers can be initialized"""
        try:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'freeciv-proxy', 'protocol'))

            from message_handlers import MessageHandlerRegistry
            from llm_protocol import MessageType

            # Test handler registry creation
            registry = MessageHandlerRegistry()

            assert hasattr(registry, 'handlers')
            assert isinstance(registry.handlers, dict)

            # Check that handlers are registered for required message types
            required_types = [
                MessageType.LLM_CONNECT,
                MessageType.STATE_QUERY,
                MessageType.ACTION
            ]

            for msg_type in required_types:
                assert msg_type in registry.handlers
                assert callable(registry.handlers[msg_type])

        except ImportError:
            pytest.skip("Message handlers module not available")

    def test_component_integration_readiness(self):
        """Test that both components can work together"""
        try:
            import sys
            import os

            # Add both component paths
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'freeciv-proxy', 'protocol'))
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llm-gateway'))

            # Import from both components
            from llm_protocol import MessageType, LLMMessage
            from message_handlers import MessageHandlerRegistry
            from main import LLMGateway

            # Test that components can be created together
            gateway = LLMGateway()
            registry = MessageHandlerRegistry()

            # Test message creation and handling compatibility
            message = LLMMessage(
                type=MessageType.STATE_QUERY,
                agent_id="integration-test",
                timestamp=time.time(),
                data={"format": "llm_optimized"}
            )

            # Verify message structure is compatible
            assert message.type in registry.handlers

            # Test JSON serialization round-trip
            json_str = message.to_json()
            parsed_data = json.loads(json_str)

            assert parsed_data["type"] == "state_query"
            assert parsed_data["agent_id"] == "integration-test"
            assert "data" in parsed_data
            assert "timestamp" in parsed_data

            print("✅ Basic integration test passed - components are compatible")

        except ImportError as e:
            pytest.skip(f"Component modules not available: {e}")
        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")

    def test_configuration_compatibility(self):
        """Test that configurations are compatible between components"""
        try:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llm-gateway'))

            from config import Settings, get_freeciv_proxy_url

            # Test configuration creation
            settings = Settings()

            # Test that proxy URL can be generated
            proxy_url = get_freeciv_proxy_url()
            assert proxy_url.startswith("ws://")
            assert "8002" in proxy_url  # Default FreeCiv proxy port

            # Test that settings have required fields
            assert hasattr(settings, 'freeciv_proxy_host')
            assert hasattr(settings, 'freeciv_proxy_port')
            assert hasattr(settings, 'max_concurrent_games')

        except ImportError:
            pytest.skip("Configuration module not available")