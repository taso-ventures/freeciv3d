#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Message validation system for LLM WebSocket connections
Provides input validation, size limits, and schema validation
"""

import json
import logging
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger("freeciv-proxy")

class ValidationError(Exception):
    """Custom exception for validation errors"""
    def __init__(self, message: str, error_code: str):
        self.message = message
        self.error_code = error_code
        super().__init__(message)

class MessageType(Enum):
    """Supported message types"""
    LLM_CONNECT = "llm_connect"
    STATE_QUERY = "state_query"
    ACTION = "action"
    PING = "ping"

class MessageValidator:
    """
    Validates WebSocket messages for security and structure
    Prevents DoS attacks via large payloads or deep JSON structures
    """

    # Security limits
    MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB default
    MAX_JSON_DEPTH = 10
    MAX_STRING_LENGTH = 1000
    MAX_ARRAY_LENGTH = 100
    MAX_OBJECT_KEYS = 50

    # Message schemas
    SCHEMAS = {
        MessageType.LLM_CONNECT: {
            'required_fields': ['type', 'agent_id', 'api_token'],
            'optional_fields': ['capabilities', 'port'],
            'field_types': {
                'type': str,
                'agent_id': str,
                'api_token': str,
                'capabilities': list,
                'port': int
            },
            'field_constraints': {
                'agent_id': {'max_length': 50, 'pattern': r'^[a-zA-Z0-9_-]+$'},
                'api_token': {'min_length': 10, 'max_length': 100},
                'capabilities': {'max_length': 20},
                'port': {'min_value': 1000, 'max_value': 65535}
            }
        },
        MessageType.STATE_QUERY: {
            'required_fields': ['type'],
            'optional_fields': ['format', 'include_actions', 'player_id'],
            'field_types': {
                'type': str,
                'format': str,
                'include_actions': bool,
                'player_id': int
            },
            'field_constraints': {
                'format': {'allowed_values': ['full', 'delta', 'llm_optimized']},
                'player_id': {'min_value': 1, 'max_value': 8}
            }
        },
        MessageType.ACTION: {
            'required_fields': ['type', 'action'],
            'optional_fields': ['timestamp'],
            'field_types': {
                'type': str,
                'action': dict,
                'timestamp': (int, float)
            },
            'field_constraints': {
                'action': {'max_keys': 20}
            }
        },
        MessageType.PING: {
            'required_fields': ['type'],
            'optional_fields': ['timestamp'],
            'field_types': {
                'type': str,
                'timestamp': (int, float)
            }
        }
    }

    def __init__(self, max_message_size: int = None):
        self.max_message_size = max_message_size or self.MAX_MESSAGE_SIZE
        self.validation_stats = {
            'total_messages': 0,
            'valid_messages': 0,
            'validation_errors': 0,
            'errors_by_type': {}
        }

    def validate_message(self, raw_message: str) -> Dict[str, Any]:
        """
        Validate a raw WebSocket message

        Args:
            raw_message: Raw message string

        Returns:
            Parsed and validated message dictionary

        Raises:
            ValidationError: If validation fails
        """
        self.validation_stats['total_messages'] += 1

        try:
            # Size validation
            self._validate_message_size(raw_message)

            # JSON parsing with depth validation
            message = self._parse_json_safely(raw_message)

            # Schema validation
            self._validate_message_schema(message)

            # Content validation
            self._validate_message_content(message)

            self.validation_stats['valid_messages'] += 1
            return message

        except ValidationError as e:
            self.validation_stats['validation_errors'] += 1
            error_type = e.error_code
            self.validation_stats['errors_by_type'][error_type] = (
                self.validation_stats['errors_by_type'].get(error_type, 0) + 1
            )
            logger.warning(f"Message validation failed: {e.error_code} - {e.message}")
            raise

    def _validate_message_size(self, message: str):
        """Validate message size limits"""
        size = len(message.encode('utf-8'))
        if size > self.max_message_size:
            raise ValidationError(
                f"Message too large: {size} bytes (max: {self.max_message_size})",
                "V001"
            )

    def _parse_json_safely(self, message: str) -> Dict[str, Any]:
        """Parse JSON with depth and structure validation"""
        try:
            parsed = json.loads(message)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON: {e}", "V002")

        if not isinstance(parsed, dict):
            raise ValidationError("Message must be a JSON object", "V003")

        # Validate JSON depth and structure
        self._validate_json_structure(parsed, depth=0)

        return parsed

    def _validate_json_structure(self, obj: Any, depth: int = 0):
        """Recursively validate JSON structure limits"""
        if depth > self.MAX_JSON_DEPTH:
            raise ValidationError(
                f"JSON too deep: {depth} levels (max: {self.MAX_JSON_DEPTH})",
                "V004"
            )

        if isinstance(obj, dict):
            if len(obj) > self.MAX_OBJECT_KEYS:
                raise ValidationError(
                    f"Too many object keys: {len(obj)} (max: {self.MAX_OBJECT_KEYS})",
                    "V005"
                )
            for key, value in obj.items():
                if not isinstance(key, str) or len(key) > self.MAX_STRING_LENGTH:
                    raise ValidationError(
                        f"Invalid object key: {key}",
                        "V006"
                    )
                self._validate_json_structure(value, depth + 1)

        elif isinstance(obj, list):
            if len(obj) > self.MAX_ARRAY_LENGTH:
                raise ValidationError(
                    f"Array too long: {len(obj)} (max: {self.MAX_ARRAY_LENGTH})",
                    "V007"
                )
            for item in obj:
                self._validate_json_structure(item, depth + 1)

        elif isinstance(obj, str):
            if len(obj) > self.MAX_STRING_LENGTH:
                raise ValidationError(
                    f"String too long: {len(obj)} (max: {self.MAX_STRING_LENGTH})",
                    "V008"
                )

    def _validate_message_schema(self, message: Dict[str, Any]):
        """Validate message against schema"""
        msg_type_str = message.get('type')
        if not msg_type_str:
            raise ValidationError("Missing 'type' field", "V009")

        try:
            msg_type = MessageType(msg_type_str)
        except ValueError:
            raise ValidationError(f"Unknown message type: {msg_type_str}", "V010")

        schema = self.SCHEMAS.get(msg_type)
        if not schema:
            raise ValidationError(f"No schema defined for type: {msg_type_str}", "V011")

        # Check required fields
        for field in schema['required_fields']:
            if field not in message:
                raise ValidationError(f"Missing required field: {field}", "V012")

        # Check field types
        field_types = schema.get('field_types', {})
        for field, expected_type in field_types.items():
            if field in message:
                value = message[field]
                if not isinstance(value, expected_type):
                    raise ValidationError(
                        f"Invalid type for field {field}: {type(value).__name__} "
                        f"(expected: {expected_type})",
                        "V013"
                    )

        # Check for unexpected fields
        allowed_fields = set(schema['required_fields'] + schema.get('optional_fields', []))
        for field in message:
            if field not in allowed_fields:
                raise ValidationError(f"Unexpected field: {field}", "V014")

    def _validate_message_content(self, message: Dict[str, Any]):
        """Validate message content against constraints"""
        msg_type = MessageType(message['type'])
        schema = self.SCHEMAS[msg_type]
        constraints = schema.get('field_constraints', {})

        for field, field_constraints in constraints.items():
            if field not in message:
                continue

            value = message[field]

            # String constraints
            if isinstance(value, str):
                if 'min_length' in field_constraints:
                    if len(value) < field_constraints['min_length']:
                        raise ValidationError(
                            f"Field {field} too short: {len(value)} < {field_constraints['min_length']}",
                            "V015"
                        )

                if 'max_length' in field_constraints:
                    if len(value) > field_constraints['max_length']:
                        raise ValidationError(
                            f"Field {field} too long: {len(value)} > {field_constraints['max_length']}",
                            "V016"
                        )

                if 'pattern' in field_constraints:
                    import re
                    if not re.match(field_constraints['pattern'], value):
                        raise ValidationError(
                            f"Field {field} doesn't match pattern: {field_constraints['pattern']}",
                            "V017"
                        )

                if 'allowed_values' in field_constraints:
                    if value not in field_constraints['allowed_values']:
                        raise ValidationError(
                            f"Field {field} has invalid value: {value}",
                            "V018"
                        )

            # Numeric constraints
            elif isinstance(value, (int, float)):
                if 'min_value' in field_constraints:
                    if value < field_constraints['min_value']:
                        raise ValidationError(
                            f"Field {field} too small: {value} < {field_constraints['min_value']}",
                            "V019"
                        )

                if 'max_value' in field_constraints:
                    if value > field_constraints['max_value']:
                        raise ValidationError(
                            f"Field {field} too large: {value} > {field_constraints['max_value']}",
                            "V020"
                        )

            # List constraints
            elif isinstance(value, list):
                if 'max_length' in field_constraints:
                    if len(value) > field_constraints['max_length']:
                        raise ValidationError(
                            f"Field {field} list too long: {len(value)} > {field_constraints['max_length']}",
                            "V021"
                        )

            # Dict constraints
            elif isinstance(value, dict):
                if 'max_keys' in field_constraints:
                    if len(value) > field_constraints['max_keys']:
                        raise ValidationError(
                            f"Field {field} object too many keys: {len(value)} > {field_constraints['max_keys']}",
                            "V022"
                        )

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        total = self.validation_stats['total_messages']
        valid_rate = (self.validation_stats['valid_messages'] / total * 100) if total > 0 else 0

        return {
            **self.validation_stats,
            'valid_rate_percent': round(valid_rate, 2)
        }

    def reset_stats(self):
        """Reset validation statistics"""
        self.validation_stats = {
            'total_messages': 0,
            'valid_messages': 0,
            'validation_errors': 0,
            'errors_by_type': {}
        }