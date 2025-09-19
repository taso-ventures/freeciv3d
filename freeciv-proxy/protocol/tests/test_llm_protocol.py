#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for LLM WebSocket Protocol message types and serialization
"""

import json
import time
import pytest
from unittest.mock import patch

# Import the modules we're testing
try:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from llm_protocol import MessageType, LLMMessage
except ImportError:
    # Will fail initially until we implement the module
    MessageType = None
    LLMMessage = None


class TestMessageType:
    """Test MessageType enum"""

    def test_message_type_enum_values(self):
        """Test that all required message types are defined"""
        if MessageType is None:
            pytest.skip("MessageType not implemented yet")

        # From Linear issue FC-3 requirements
        expected_types = [
            'LLM_CONNECT',
            'LLM_DISCONNECT',
            'STATE_QUERY',
            'STATE_UPDATE',
            'ACTION',
            'ACTION_RESULT',
            'TURN_START',
            'TURN_END',
            'SPECTATOR_UPDATE',
            'ERROR'
        ]

        for msg_type in expected_types:
            assert hasattr(MessageType, msg_type), f"MessageType.{msg_type} should exist"

    def test_message_type_string_values(self):
        """Test that message types have correct string values"""
        if MessageType is None:
            pytest.skip("MessageType not implemented yet")

        assert MessageType.LLM_CONNECT.value == "llm_connect"
        assert MessageType.STATE_QUERY.value == "state_query"
        assert MessageType.ACTION.value == "action"
        assert MessageType.ERROR.value == "error"


class TestLLMMessage:
    """Test LLMMessage dataclass"""

    def test_llm_message_creation(self):
        """Test creating LLMMessage instance"""
        if LLMMessage is None:
            pytest.skip("LLMMessage not implemented yet")

        msg = LLMMessage(
            type=MessageType.LLM_CONNECT,
            agent_id="test-agent",
            timestamp=time.time(),
            data={"api_token": "test-token"}
        )

        assert msg.type == MessageType.LLM_CONNECT
        assert msg.agent_id == "test-agent"
        assert isinstance(msg.timestamp, float)
        assert msg.data["api_token"] == "test-token"
        assert msg.correlation_id is None  # Optional field

    def test_llm_message_with_correlation_id(self):
        """Test LLMMessage with correlation ID for async operations"""
        if LLMMessage is None:
            pytest.skip("LLMMessage not implemented yet")

        msg = LLMMessage(
            type=MessageType.STATE_QUERY,
            agent_id="test-agent",
            timestamp=time.time(),
            data={"format": "llm_optimized"},
            correlation_id="corr-123"
        )

        assert msg.correlation_id == "corr-123"

    def test_message_serialization_to_json(self):
        """Test LLMMessage serialization to JSON"""
        if LLMMessage is None:
            pytest.skip("LLMMessage not implemented yet")

        msg = LLMMessage(
            type=MessageType.ACTION,
            agent_id="test-agent",
            timestamp=1234567890.123,
            data={"action_type": "move", "unit_id": 42},
            correlation_id="corr-456"
        )

        json_str = msg.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)

        assert parsed["type"] == "action"
        assert parsed["agent_id"] == "test-agent"
        assert parsed["timestamp"] == 1234567890.123
        assert parsed["data"]["action_type"] == "move"
        assert parsed["data"]["unit_id"] == 42
        assert parsed["correlation_id"] == "corr-456"

    def test_message_deserialization_from_json(self):
        """Test LLMMessage deserialization from JSON"""
        if LLMMessage is None:
            pytest.skip("LLMMessage not implemented yet")

        json_data = {
            "type": "state_query",
            "agent_id": "test-agent",
            "timestamp": 1234567890.123,
            "data": {"format": "full", "include_actions": True},
            "correlation_id": "corr-789"
        }

        json_str = json.dumps(json_data)
        msg = LLMMessage.from_json(json_str)

        assert msg.type == MessageType.STATE_QUERY
        assert msg.agent_id == "test-agent"
        assert msg.timestamp == 1234567890.123
        assert msg.data["format"] == "full"
        assert msg.data["include_actions"] is True
        assert msg.correlation_id == "corr-789"

    def test_message_deserialization_invalid_json(self):
        """Test handling of invalid JSON during deserialization"""
        if LLMMessage is None:
            pytest.skip("LLMMessage not implemented yet")

        with pytest.raises((ValueError, json.JSONDecodeError)):
            LLMMessage.from_json("invalid json")

    def test_message_deserialization_missing_fields(self):
        """Test handling of missing required fields"""
        if LLMMessage is None:
            pytest.skip("LLMMessage not implemented yet")

        # Missing required 'type' field
        json_data = {
            "agent_id": "test-agent",
            "timestamp": 1234567890.123,
            "data": {}
        }

        with pytest.raises((ValueError, KeyError, TypeError)):
            LLMMessage.from_json(json.dumps(json_data))

    def test_message_roundtrip_serialization(self):
        """Test that serialization->deserialization preserves data"""
        if LLMMessage is None:
            pytest.skip("LLMMessage not implemented yet")

        original = LLMMessage(
            type=MessageType.TURN_START,
            agent_id="agent-123",
            timestamp=1234567890.555,
            data={"turn": 42, "phase": "movement"},
            correlation_id="test-correlation"
        )

        # Serialize then deserialize
        json_str = original.to_json()
        restored = LLMMessage.from_json(json_str)

        assert restored.type == original.type
        assert restored.agent_id == original.agent_id
        assert restored.timestamp == original.timestamp
        assert restored.data == original.data
        assert restored.correlation_id == original.correlation_id

    def test_message_validation_agent_id(self):
        """Test agent_id validation"""
        if LLMMessage is None:
            pytest.skip("LLMMessage not implemented yet")

        # Empty agent_id should raise error
        with pytest.raises((ValueError, TypeError)):
            LLMMessage(
                type=MessageType.LLM_CONNECT,
                agent_id="",
                timestamp=time.time(),
                data={}
            )

        # None agent_id should raise error
        with pytest.raises((ValueError, TypeError)):
            LLMMessage(
                type=MessageType.LLM_CONNECT,
                agent_id=None,
                timestamp=time.time(),
                data={}
            )


class TestMessageFormats:
    """Test specific message format requirements from Linear issue"""

    def test_llm_connect_message_format(self):
        """Test LLM_CONNECT message format matches specification"""
        if LLMMessage is None:
            pytest.skip("LLMMessage not implemented yet")

        msg = LLMMessage(
            type=MessageType.LLM_CONNECT,
            agent_id="test-agent",
            timestamp=time.time(),
            data={
                "model": "gpt-4",
                "game_id": "game-123",
                "api_key": "secret-key"
            }
        )

        json_data = json.loads(msg.to_json())

        assert json_data["type"] == "llm_connect"
        assert "model" in json_data["data"]
        assert "game_id" in json_data["data"]
        assert "api_key" in json_data["data"]

    def test_state_query_message_format(self):
        """Test STATE_QUERY message format"""
        if LLMMessage is None:
            pytest.skip("LLMMessage not implemented yet")

        msg = LLMMessage(
            type=MessageType.STATE_QUERY,
            agent_id="test-agent",
            timestamp=time.time(),
            data={
                "format": "llm_optimized",
                "since_turn": 10,
                "include_legal_actions": True
            },
            correlation_id="state-query-123"
        )

        json_data = json.loads(msg.to_json())

        assert json_data["type"] == "state_query"
        assert json_data["data"]["format"] == "llm_optimized"
        assert json_data["data"]["since_turn"] == 10
        assert json_data["data"]["include_legal_actions"] is True
        assert json_data["correlation_id"] == "state-query-123"

    def test_action_message_format(self):
        """Test ACTION message format"""
        if LLMMessage is None:
            pytest.skip("LLMMessage not implemented yet")

        msg = LLMMessage(
            type=MessageType.ACTION,
            agent_id="test-agent",
            timestamp=time.time(),
            data={
                "action_type": "unit_move",
                "actor_id": 42,
                "target": {"x": 10, "y": 20},
                "parameters": {"validate": True}
            },
            correlation_id="action-456"
        )

        json_data = json.loads(msg.to_json())

        assert json_data["type"] == "action"
        assert json_data["data"]["action_type"] == "unit_move"
        assert json_data["data"]["actor_id"] == 42
        assert json_data["data"]["target"]["x"] == 10
        assert json_data["data"]["target"]["y"] == 20
        assert json_data["correlation_id"] == "action-456"

    def test_error_message_format(self):
        """Test ERROR message format"""
        if LLMMessage is None:
            pytest.skip("LLMMessage not implemented yet")

        msg = LLMMessage(
            type=MessageType.ERROR,
            agent_id="test-agent",
            timestamp=time.time(),
            data={
                "type": "error",
                "success": False,
                "error_code": "E102",
                "error_message": "Invalid API token"
            },
            correlation_id="error-123"
        )

        json_data = json.loads(msg.to_json())

        assert json_data["type"] == "error"
        assert json_data["data"]["success"] is False
        assert json_data["data"]["error_code"] == "E102"
        assert json_data["data"]["error_message"] == "Invalid API token"
        assert json_data["correlation_id"] == "error-123"