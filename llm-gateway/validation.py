#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Input validation models using Pydantic
Provides comprehensive validation for all LLM Gateway inputs
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator, constr, conint
import re
import time


class AgentIdValidator(BaseModel):
    """Validates agent IDs"""
    agent_id: constr(min_length=1, max_length=64, regex=r'^[a-zA-Z0-9\-_]+$') = Field(
        ...,
        description="Agent identifier (alphanumeric, hyphens, underscores only)"
    )


class GameIdValidator(BaseModel):
    """Validates game IDs"""
    game_id: constr(min_length=1, max_length=64, regex=r'^[a-zA-Z0-9\-_]+$') = Field(
        ...,
        description="Game identifier (alphanumeric, hyphens, underscores only)"
    )


class ApiTokenValidator(BaseModel):
    """Validates API tokens"""
    api_token: constr(min_length=16, max_length=256) = Field(
        ...,
        description="API authentication token"
    )

    @validator('api_token')
    def validate_token_format(cls, v):
        """Validate token format"""
        if not re.match(r'^[a-zA-Z0-9\-_]+$', v):
            raise ValueError('API token contains invalid characters')
        return v


class CoordinateValidator(BaseModel):
    """Validates game coordinates"""
    x: conint(ge=0, le=9999) = Field(..., description="X coordinate")
    y: conint(ge=0, le=9999) = Field(..., description="Y coordinate")


class LLMConnectData(BaseModel):
    """Validation for LLM_CONNECT message data"""
    api_token: str = Field(..., min_length=16, max_length=256)
    model: constr(min_length=1, max_length=50) = Field(..., description="LLM model name")
    game_id: constr(min_length=1, max_length=64, regex=r'^[a-zA-Z0-9\-_]+$') = Field(
        ..., description="Game identifier"
    )
    capabilities: Optional[List[str]] = Field(
        default=None,
        description="List of agent capabilities",
        max_items=20
    )

    @validator('api_token')
    def validate_token(cls, v):
        if not re.match(r'^[a-zA-Z0-9\-_]+$', v):
            raise ValueError('API token format invalid')
        return v

    @validator('capabilities', each_item=True)
    def validate_capability(cls, v):
        valid_capabilities = [
            'move', 'attack', 'build', 'research', 'diplomacy',
            'trade', 'explore', 'defend', 'city_management'
        ]
        if v not in valid_capabilities:
            raise ValueError(f'Invalid capability: {v}')
        return v


class StateQueryData(BaseModel):
    """Validation for STATE_QUERY message data"""
    format: str = Field("llm_optimized", regex=r'^(full|delta|llm_optimized)$')
    include_legal_actions: bool = Field(True, description="Include legal actions in response")
    since_turn: Optional[conint(ge=0)] = Field(None, description="Get changes since turn number")
    player_perspective: Optional[conint(ge=1, le=8)] = Field(
        None, description="Player perspective for state"
    )

    @validator('format')
    def validate_format(cls, v):
        valid_formats = ['full', 'delta', 'llm_optimized']
        if v not in valid_formats:
            raise ValueError(f'Invalid format: {v}. Must be one of {valid_formats}')
        return v


class ActionData(BaseModel):
    """Validation for ACTION message data"""
    action_type: str = Field(..., regex=r'^[a-z_]+$')
    actor_id: Optional[conint(ge=0)] = Field(None, description="ID of acting unit/city")
    target: Optional[Union[CoordinateValidator, int, str]] = Field(
        None, description="Action target (coordinates, ID, or name)"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional action parameters"
    )

    @validator('action_type')
    def validate_action_type(cls, v):
        valid_actions = [
            'unit_move', 'unit_attack', 'unit_build_city', 'unit_explore',
            'city_production', 'city_build_unit', 'city_build_improvement',
            'tech_research', 'diplomacy_message', 'end_turn'
        ]
        if v not in valid_actions:
            raise ValueError(f'Invalid action type: {v}')
        return v

    @validator('target')
    def validate_target(cls, v, values):
        action_type = values.get('action_type')

        if action_type in ['unit_move', 'unit_attack']:
            if not isinstance(v, dict) or 'x' not in v or 'y' not in v:
                raise ValueError('Movement/attack actions require coordinate target')
        elif action_type in ['city_production', 'city_build_unit']:
            if not isinstance(v, str):
                raise ValueError('Production actions require string target')
        elif action_type == 'tech_research':
            if not isinstance(v, str):
                raise ValueError('Research actions require technology name')

        return v


class LLMMessage(BaseModel):
    """Complete LLM message validation"""
    type: str = Field(..., regex=r'^[a-z_]+$')
    agent_id: constr(min_length=1, max_length=64, regex=r'^[a-zA-Z0-9\-_]+$')
    timestamp: float = Field(..., ge=0)
    data: Dict[str, Any] = Field(..., description="Message payload")
    correlation_id: Optional[constr(max_length=128)] = Field(None)

    @validator('type')
    def validate_message_type(cls, v):
        valid_types = [
            'llm_connect', 'llm_disconnect', 'state_query', 'state_update',
            'action', 'action_result', 'turn_start', 'turn_end',
            'spectator_update', 'ping', 'pong', 'error'
        ]
        if v not in valid_types:
            raise ValueError(f'Invalid message type: {v}')
        return v

    @validator('timestamp')
    def validate_timestamp(cls, v):
        # Check if timestamp is reasonable (within last 24 hours to 1 hour in future)
        now = time.time()
        if v < (now - 86400) or v > (now + 3600):
            raise ValueError('Timestamp out of reasonable range')
        return v

    @validator('data')
    def validate_data_content(cls, v, values):
        """Validate data based on message type"""
        msg_type = values.get('type')

        if msg_type == 'llm_connect':
            LLMConnectData(**v)
        elif msg_type == 'state_query':
            StateQueryData(**v)
        elif msg_type == 'action':
            ActionData(**v)

        return v


class GameCreationRequest(BaseModel):
    """Validation for game creation requests"""
    ruleset: str = Field("classic", regex=r'^[a-zA-Z0-9_]+$')
    map_size: str = Field("small", regex=r'^(tiny|small|medium|large|huge)$')
    difficulty: str = Field("normal", regex=r'^(easy|normal|hard|expert)$')
    max_players: conint(ge=2, le=8) = Field(4, description="Maximum number of players")
    time_limit: Optional[conint(ge=30, le=3600)] = Field(
        300, description="Turn time limit in seconds"
    )
    game_name: Optional[constr(min_length=1, max_length=100)] = Field(
        None, description="Optional game name"
    )

    @validator('game_name')
    def validate_game_name(cls, v):
        if v and not re.match(r'^[a-zA-Z0-9\s\-_]+$', v):
            raise ValueError('Game name contains invalid characters')
        return v


class FreeCivActionRequest(BaseModel):
    """Validation for FreeCiv action requests"""
    action_type: str = Field(..., regex=r'^[a-z_]+$')
    player_id: conint(ge=1, le=8) = Field(..., description="Player ID")
    actor_id: Optional[conint(ge=0)] = Field(None, description="Acting unit/city ID")
    target: Optional[Union[Dict[str, int], int, str]] = Field(None)
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @validator('action_type')
    def validate_action_type(cls, v):
        # Validate against FreeCiv action types
        valid_actions = [
            'unit_move', 'unit_attack', 'unit_build_city', 'unit_explore',
            'unit_fortify', 'unit_sentry', 'unit_disband',
            'city_production', 'city_build_unit', 'city_build_improvement',
            'city_sell_improvement', 'city_change_specialist',
            'tech_research', 'government_change', 'diplomacy_message',
            'trade_route', 'end_turn'
        ]
        if v not in valid_actions:
            raise ValueError(f'Invalid FreeCiv action: {v}')
        return v


class RateLimitValidator(BaseModel):
    """Validation for rate limiting parameters"""
    requests_per_minute: conint(ge=1, le=1000) = Field(100)
    burst_size: conint(ge=1, le=100) = Field(20)
    window_size: conint(ge=60, le=3600) = Field(60, description="Time window in seconds")


# Validation functions for common use cases

def validate_agent_id(agent_id: str) -> str:
    """Validate and sanitize agent ID"""
    validator = AgentIdValidator(agent_id=agent_id)
    return validator.agent_id


def validate_game_id(game_id: str) -> str:
    """Validate and sanitize game ID"""
    validator = GameIdValidator(game_id=game_id)
    return validator.game_id


def validate_api_token(api_token: str) -> str:
    """Validate API token"""
    validator = ApiTokenValidator(api_token=api_token)
    return validator.api_token


def validate_coordinates(x: int, y: int) -> Dict[str, int]:
    """Validate game coordinates"""
    validator = CoordinateValidator(x=x, y=y)
    return {"x": validator.x, "y": validator.y}


def validate_llm_message(message_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate complete LLM message"""
    validator = LLMMessage(**message_data)
    return validator.dict()


def sanitize_string_input(input_str: str, max_length: int = 256) -> str:
    """Sanitize string input by removing dangerous characters"""
    if not isinstance(input_str, str):
        raise ValueError("Input must be a string")

    # Remove null bytes and control characters
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', input_str)

    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    # Remove leading/trailing whitespace
    sanitized = sanitized.strip()

    if not sanitized:
        raise ValueError("Input cannot be empty after sanitization")

    return sanitized


class ValidationError(Exception):
    """Custom validation error"""
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message)
        self.field = field
        self.value = value


def safe_validate(validator_class, data: Dict[str, Any], field_name: str = None) -> Any:
    """Safely validate data with custom error handling"""
    try:
        return validator_class(**data)
    except Exception as e:
        error_msg = f"Validation failed for {field_name or 'data'}: {str(e)}"
        raise ValidationError(error_msg, field_name, data)