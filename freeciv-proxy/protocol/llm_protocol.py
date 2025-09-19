#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM WebSocket Protocol for FreeCiv3D Integration
Defines message types and serialization for LLM agent communication
"""

import json
import time
import uuid
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


class MessageType(Enum):
    """Enumeration of LLM WebSocket message types"""

    # Connection management
    LLM_CONNECT = "llm_connect"
    LLM_DISCONNECT = "llm_disconnect"

    # Game state
    STATE_QUERY = "state_query"
    STATE_UPDATE = "state_update"

    # Actions
    ACTION = "action"
    ACTION_RESULT = "action_result"

    # Turn management
    TURN_START = "turn_start"
    TURN_END = "turn_end"

    # Spectator
    SPECTATOR_UPDATE = "spectator_update"

    # Error handling
    ERROR = "error"


@dataclass
class LLMMessage:
    """
    LLM WebSocket message with correlation support for async operations
    """
    type: MessageType
    agent_id: str
    timestamp: float
    data: Dict[str, Any]
    correlation_id: Optional[str] = None

    def __post_init__(self):
        """Validate message after initialization"""
        if not self.agent_id or not isinstance(self.agent_id, str):
            raise ValueError("agent_id must be a non-empty string")

        if not isinstance(self.timestamp, (int, float)):
            raise ValueError("timestamp must be a number")

        if not isinstance(self.data, dict):
            raise ValueError("data must be a dictionary")

    def to_json(self) -> str:
        """Serialize message to JSON string"""
        message_dict = {
            "type": self.type.value,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "data": self.data
        }

        if self.correlation_id is not None:
            message_dict["correlation_id"] = self.correlation_id

        return json.dumps(message_dict)

    @classmethod
    def from_json(cls, json_str: str) -> 'LLMMessage':
        """Deserialize message from JSON string"""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        # Validate required fields
        required_fields = ["type", "agent_id", "timestamp", "data"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Convert type string to MessageType enum
        try:
            message_type = MessageType(data["type"])
        except ValueError:
            raise ValueError(f"Unknown message type: {data['type']}")

        return cls(
            type=message_type,
            agent_id=data["agent_id"],
            timestamp=data["timestamp"],
            data=data["data"],
            correlation_id=data.get("correlation_id")
        )

    def create_response(self, response_type: MessageType, response_data: Dict[str, Any]) -> 'LLMMessage':
        """Create a response message with the same correlation ID"""
        return LLMMessage(
            type=response_type,
            agent_id=self.agent_id,
            timestamp=time.time(),
            data=response_data,
            correlation_id=self.correlation_id
        )

    @staticmethod
    def create_with_correlation(message_type: MessageType, agent_id: str,
                              data: Dict[str, Any], correlation_id: Optional[str] = None) -> 'LLMMessage':
        """Create a new message with automatic correlation ID generation"""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())

        return LLMMessage(
            type=message_type,
            agent_id=agent_id,
            timestamp=time.time(),
            data=data,
            correlation_id=correlation_id
        )


# Message creation helpers for common patterns
def create_connect_message(agent_id: str, api_token: str, model: str, game_id: str) -> LLMMessage:
    """Create LLM_CONNECT message"""
    return LLMMessage(
        type=MessageType.LLM_CONNECT,
        agent_id=agent_id,
        timestamp=time.time(),
        data={
            "api_token": api_token,
            "model": model,
            "game_id": game_id
        }
    )


def create_state_query_message(agent_id: str, format_type: str = "llm_optimized",
                              include_actions: bool = True, since_turn: Optional[int] = None) -> LLMMessage:
    """Create STATE_QUERY message with correlation ID"""
    data = {
        "format": format_type,
        "include_actions": include_actions
    }

    if since_turn is not None:
        data["since_turn"] = since_turn

    return LLMMessage.create_with_correlation(
        MessageType.STATE_QUERY,
        agent_id,
        data
    )


def create_action_message(agent_id: str, action_type: str, actor_id: int,
                         target: Any, parameters: Optional[Dict[str, Any]] = None) -> LLMMessage:
    """Create ACTION message with correlation ID"""
    data = {
        "action_type": action_type,
        "actor_id": actor_id,
        "target": target
    }

    if parameters:
        data["parameters"] = parameters

    return LLMMessage.create_with_correlation(
        MessageType.ACTION,
        agent_id,
        data
    )


def create_error_response(original_message: LLMMessage, error_code: str, error_message: str) -> LLMMessage:
    """Create error response for a message"""
    return original_message.create_response(
        MessageType.ERROR,
        {
            "type": "error",
            "success": False,
            "error_code": error_code,
            "error_message": error_message
        }
    )


def create_success_response(original_message: LLMMessage, response_data: Dict[str, Any]) -> LLMMessage:
    """Create success response for a message"""
    response_data["success"] = True

    # Determine response type based on original message type
    response_type_map = {
        MessageType.LLM_CONNECT: MessageType.LLM_CONNECT,  # auth_success
        MessageType.STATE_QUERY: MessageType.STATE_UPDATE,
        MessageType.ACTION: MessageType.ACTION_RESULT
    }

    response_type = response_type_map.get(original_message.type, MessageType.ACTION_RESULT)

    return original_message.create_response(response_type, response_data)