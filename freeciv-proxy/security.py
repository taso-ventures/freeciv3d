#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Security utilities for FreeCiv proxy
Provides input sanitization, validation, and security helpers
"""

import re
import logging
import os
import time
from typing import Any, Union, List, Dict

logger = logging.getLogger("freeciv-proxy")

class SecurityError(Exception):
    """Custom exception for security-related errors"""
    pass

class InputSanitizer:
    """
    Input sanitization utilities to prevent injection attacks
    """

    # Patterns for validation
    PATTERNS = {
        'player_id': r'^[1-8]$',  # Player IDs 1-8
        'unit_id': r'^[0-9]{1,6}$',  # Unit IDs up to 6 digits
        'city_id': r'^[0-9]{1,6}$',  # City IDs up to 6 digits
        'coordinates': r'^-?[0-9]{1,4}$',  # Coordinates -9999 to 9999
        'agent_id': r'^[a-zA-Z0-9_-]{1,50}$',  # Agent IDs alphanumeric with _ and -
        'tech_name': r'^[a-zA-Z0-9_\s]{1,50}$',  # Technology names
        'unit_type': r'^[a-zA-Z0-9_\s]{1,30}$',  # Unit types
        'production_type': r'^[a-zA-Z0-9_\s]{1,30}$',  # Production types
        'game_phase': r'^[a-zA-Z0-9_]{1,20}$',  # Game phases
    }

    # SQL injection patterns to detect
    SQL_INJECTION_PATTERNS = [
        r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)',
        r'(--|\#|/\*|\*/)',  # SQL comments
        r'(\b(OR|AND)\s+\d+\s*=\s*\d+)',  # OR 1=1, AND 1=1
        r'(\bUNION\s+SELECT)',  # UNION SELECT
        r'(\bINTO\s+OUTFILE)',  # INTO OUTFILE
        r'(\bLOAD_FILE\s*\()',  # LOAD_FILE
    ]

    @classmethod
    def sanitize_player_id(cls, value: Any) -> int:
        """Sanitize and validate player ID"""
        try:
            player_id = int(value)
            if not (1 <= player_id <= 8):
                raise SecurityError(f"Invalid player_id range: {player_id}")
            return player_id
        except (ValueError, TypeError):
            raise SecurityError(f"Invalid player_id type: {value}")

    @classmethod
    def sanitize_unit_id(cls, value: Any) -> int:
        """Sanitize and validate unit ID"""
        try:
            unit_id = int(value)
            if not (0 <= unit_id <= 999999):
                raise SecurityError(f"Invalid unit_id range: {unit_id}")
            return unit_id
        except (ValueError, TypeError):
            raise SecurityError(f"Invalid unit_id type: {value}")

    @classmethod
    def sanitize_city_id(cls, value: Any) -> int:
        """Sanitize and validate city ID"""
        try:
            city_id = int(value)
            if not (0 <= city_id <= 999999):
                raise SecurityError(f"Invalid city_id range: {city_id}")
            return city_id
        except (ValueError, TypeError):
            raise SecurityError(f"Invalid city_id type: {value}")

    @classmethod
    def sanitize_coordinates(cls, x: Any, y: Any) -> tuple:
        """Sanitize and validate coordinates"""
        try:
            x_val = int(x)
            y_val = int(y)
            if not (-9999 <= x_val <= 9999) or not (-9999 <= y_val <= 9999):
                raise SecurityError(f"Invalid coordinate range: ({x_val}, {y_val})")
            return (x_val, y_val)
        except (ValueError, TypeError):
            raise SecurityError(f"Invalid coordinate type: ({x}, {y})")

    @classmethod
    def sanitize_game_id(cls, value: Any) -> str:
        """Sanitize and validate game ID"""
        if not isinstance(value, str):
            raise SecurityError(f"Invalid game_id type: {type(value)}")
        
        # Check for SQL injection patterns
        cls._check_sql_injection(value, 'game_id')
        
        # Game ID pattern: alphanumeric, underscore, hyphen, max 50 chars
        pattern = r'^[a-zA-Z0-9_-]{1,50}$'
        if not re.match(pattern, value):
            raise SecurityError(f"Invalid game_id format: {value}")
        
        return value
    
    @classmethod
    def sanitize_string(cls, value: Any, max_length: int = 100) -> str:
        """Sanitize and validate general string input"""
        if not isinstance(value, str):
            raise SecurityError(f"Invalid string type: {type(value)}")
        
        # Check for SQL injection patterns
        cls._check_sql_injection(value, 'string')
        
        # Basic alphanumeric with underscore, max length
        if len(value) > max_length:
            raise SecurityError(f"String too long: {len(value)} > {max_length}")
        
        # Allow basic characters for action names
        pattern = r'^[a-zA-Z0-9_]{1,' + str(max_length) + '}$'
        if not re.match(pattern, value):
            raise SecurityError(f"Invalid string format: {value}")
        
        return value

    @classmethod
    def sanitize_string_field(cls, value: Any, field_type: str, max_length: int = None) -> str:
        """Sanitize and validate string fields"""
        if not isinstance(value, str):
            raise SecurityError(f"Invalid {field_type} type: {type(value)}")

        # Check for SQL injection patterns
        cls._check_sql_injection(value, field_type)

        # Pattern validation
        if field_type in cls.PATTERNS:
            if not re.match(cls.PATTERNS[field_type], value):
                raise SecurityError(f"Invalid {field_type} format: {value}")

        # Length validation
        if max_length and len(value) > max_length:
            raise SecurityError(f"{field_type} too long: {len(value)} > {max_length}")

        # Basic sanitization
        sanitized = value.strip()

        # Remove null bytes and control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\t\n\r')

        return sanitized

    @classmethod
    def _check_sql_injection(cls, value: str, field_name: str):
        """Check for SQL injection patterns"""
        value_upper = value.upper()

        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value_upper, re.IGNORECASE):
                logger.error(f"Potential SQL injection detected in {field_name}: {value}")
                raise SecurityError(f"Potential SQL injection in {field_name}")

    @classmethod
    def sanitize_action_data(cls, action: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize action data dictionary"""
        if not isinstance(action, dict):
            raise SecurityError("Action must be a dictionary")

        sanitized = {}
        action_type = action.get('type', '')

        # Validate action type
        if action_type:
            sanitized['type'] = cls.sanitize_string_field(action_type, 'action_type', 50)

        # Sanitize based on action type
        if action_type == 'unit_move':
            if 'unit_id' in action:
                sanitized['unit_id'] = cls.sanitize_unit_id(action['unit_id'])
            if 'dest_x' in action and 'dest_y' in action:
                sanitized['dest_x'], sanitized['dest_y'] = cls.sanitize_coordinates(
                    action['dest_x'], action['dest_y']
                )
            if 'player_id' in action:
                sanitized['player_id'] = cls.sanitize_player_id(action['player_id'])

        elif action_type == 'city_production':
            if 'city_id' in action:
                sanitized['city_id'] = cls.sanitize_city_id(action['city_id'])
            if 'production_type' in action:
                sanitized['production_type'] = cls.sanitize_string_field(
                    action['production_type'], 'production_type', 30
                )
            if 'player_id' in action:
                sanitized['player_id'] = cls.sanitize_player_id(action['player_id'])

        elif action_type == 'tech_research':
            if 'tech_name' in action:
                sanitized['tech_name'] = cls.sanitize_string_field(
                    action['tech_name'], 'tech_name', 50
                )
            if 'player_id' in action:
                sanitized['player_id'] = cls.sanitize_player_id(action['player_id'])

        elif action_type == 'unit_build_city':
            if 'unit_id' in action:
                sanitized['unit_id'] = cls.sanitize_unit_id(action['unit_id'])
            if 'player_id' in action:
                sanitized['player_id'] = cls.sanitize_player_id(action['player_id'])

        # Copy other safe fields
        safe_fields = ['timestamp', 'priority']
        for field in safe_fields:
            if field in action:
                if field == 'timestamp':
                    try:
                        sanitized[field] = float(action[field])
                    except (ValueError, TypeError):
                        raise SecurityError(f"Invalid timestamp: {action[field]}")
                elif field == 'priority':
                    if action[field] in ['low', 'medium', 'high']:
                        sanitized[field] = action[field]

        return sanitized

    @classmethod
    def sanitize_state_query_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize state query parameters"""
        sanitized = {}

        if 'format' in params:
            format_val = params['format']
            if format_val in ['full', 'delta', 'llm_optimized']:
                sanitized['format'] = format_val
            else:
                raise SecurityError(f"Invalid format: {format_val}")

        if 'player_id' in params:
            sanitized['player_id'] = cls.sanitize_player_id(params['player_id'])

        if 'include_actions' in params:
            if isinstance(params['include_actions'], bool):
                sanitized['include_actions'] = params['include_actions']
            else:
                raise SecurityError("include_actions must be boolean")

        return sanitized

class SecurityLogger:
    """
    Enhanced security event logging utility with structured logging
    """

    @staticmethod
    def log_authentication_attempt(agent_id: str, success: bool, ip_address: str = None,
                                 session_id: str = None, details: str = None):
        """Log authentication attempts with enhanced context"""
        status = "SUCCESS" if success else "FAILED"
        log_data = {
            'event': 'authentication',
            'status': status,
            'agent_id': agent_id,
            'ip_address': ip_address,
            'session_id': session_id,
            'details': details,
            'timestamp': time.time()
        }

        if success:
            logger.info(f"AUTH_{status}: {SecurityLogger._format_log_data(log_data)}")
        else:
            logger.warning(f"AUTH_{status}: {SecurityLogger._format_log_data(log_data)}")

    @staticmethod
    def log_rate_limit_exceeded(agent_id: str, ip_address: str = None,
                              limit_type: str = 'default', current_count: int = None):
        """Log rate limit violations with detailed metrics"""
        log_data = {
            'event': 'rate_limit_exceeded',
            'agent_id': agent_id,
            'ip_address': ip_address,
            'limit_type': limit_type,
            'current_count': current_count,
            'timestamp': time.time()
        }
        logger.warning(f"RATE_LIMIT_EXCEEDED: {SecurityLogger._format_log_data(log_data)}")

    @staticmethod
    def log_validation_error(agent_id: str, error_code: str, message: str,
                           session_id: str = None, input_data: str = None):
        """Log validation errors with context"""
        log_data = {
            'event': 'validation_error',
            'agent_id': agent_id,
            'error_code': error_code,
            'message': message,
            'session_id': session_id,
            'input_sample': input_data[:100] if input_data else None,  # Only log first 100 chars
            'timestamp': time.time()
        }
        logger.warning(f"VALIDATION_ERROR: {SecurityLogger._format_log_data(log_data)}")

    @staticmethod
    def log_security_violation(agent_id: str, violation_type: str, details: str,
                             severity: str = 'medium', session_id: str = None):
        """Log security violations with severity levels"""
        log_data = {
            'event': 'security_violation',
            'agent_id': agent_id,
            'violation_type': violation_type,
            'severity': severity,
            'details': details,
            'session_id': session_id,
            'timestamp': time.time()
        }

        if severity in ['high', 'critical']:
            logger.error(f"SECURITY_VIOLATION: {SecurityLogger._format_log_data(log_data)}")
        else:
            logger.warning(f"SECURITY_VIOLATION: {SecurityLogger._format_log_data(log_data)}")

    @staticmethod
    def log_connection_event(agent_id: str, event_type: str, details: str = None,
                           session_id: str = None, ip_address: str = None):
        """Log connection events with session context"""
        log_data = {
            'event': 'connection',
            'event_type': event_type,
            'agent_id': agent_id,
            'session_id': session_id,
            'ip_address': ip_address,
            'details': details,
            'timestamp': time.time()
        }
        logger.info(f"CONNECTION_{event_type}: {SecurityLogger._format_log_data(log_data)}")

    @staticmethod
    def log_session_event(session_id: str, agent_id: str, event_type: str,
                         details: str = None, metadata: dict = None):
        """Log session-specific events"""
        log_data = {
            'event': 'session',
            'event_type': event_type,
            'session_id': session_id,
            'agent_id': agent_id,
            'details': details,
            'metadata': metadata,
            'timestamp': time.time()
        }
        logger.info(f"SESSION_{event_type}: {SecurityLogger._format_log_data(log_data)}")

    @staticmethod
    def log_action_attempt(agent_id: str, action_type: str, success: bool,
                          session_id: str = None, validation_errors: list = None):
        """Log action execution attempts"""
        status = "SUCCESS" if success else "FAILED"
        log_data = {
            'event': 'action',
            'status': status,
            'agent_id': agent_id,
            'action_type': action_type,
            'session_id': session_id,
            'validation_errors': validation_errors,
            'timestamp': time.time()
        }

        if success:
            logger.info(f"ACTION_{status}: {SecurityLogger._format_log_data(log_data)}")
        else:
            logger.warning(f"ACTION_{status}: {SecurityLogger._format_log_data(log_data)}")

    @staticmethod
    def log_cache_event(event_type: str, cache_key: str, player_id: int = None,
                       success: bool = True, error: str = None):
        """Log cache-related security events"""
        log_data = {
            'event': 'cache',
            'event_type': event_type,
            'cache_key': cache_key,
            'player_id': player_id,
            'success': success,
            'error': error,
            'timestamp': time.time()
        }

        if success:
            logger.debug(f"CACHE_{event_type}: {SecurityLogger._format_log_data(log_data)}")
        else:
            logger.error(f"CACHE_{event_type}: {SecurityLogger._format_log_data(log_data)}")

    @staticmethod
    def log_performance_warning(operation: str, duration_ms: float, threshold_ms: float,
                              agent_id: str = None, details: str = None):
        """Log performance warnings for security monitoring"""
        log_data = {
            'event': 'performance_warning',
            'operation': operation,
            'duration_ms': duration_ms,
            'threshold_ms': threshold_ms,
            'agent_id': agent_id,
            'details': details,
            'timestamp': time.time()
        }
        logger.warning(f"PERFORMANCE_WARNING: {SecurityLogger._format_log_data(log_data)}")

    @staticmethod
    def _format_log_data(log_data: dict) -> str:
        """Format log data for consistent structured logging"""
        return ' | '.join([f"{k}={v}" for k, v in log_data.items() if v is not None])

def validate_environment_setup():
    """Validate that required security environment variables are set"""
    required_vars = ['LLM_API_TOKENS']
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        raise SecurityError(f"Missing required environment variables: {missing_vars}")

    logger.info("Security environment validation passed")
