#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Centralized error handling for FreeCiv proxy
Provides consistent error responses, recovery strategies, and security event logging
"""

import json
import logging
import traceback
import time
from typing import Dict, Any, Optional, Callable
from enum import Enum
from security import SecurityLogger

logger = logging.getLogger("freeciv-proxy")

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    SYSTEM = "system"
    SECURITY = "security"
    CONFIGURATION = "configuration"

class ErrorCode:
    """Standardized error codes"""
    # Authentication errors (E100-E119)
    AUTH_INVALID_TOKEN = "E110"
    AUTH_FAILED = "E111"
    AUTH_SESSION_CAPACITY = "E112"
    AUTH_SESSION_EXPIRED = "E102"

    # Rate limiting errors (E120-E139)
    RATE_LIMIT_EXCEEDED = "E101"

    # Validation errors (E140-E179)
    VALIDATION_FAILED = "E116"
    VALIDATION_MESSAGE_SIZE = "V001"
    VALIDATION_JSON_INVALID = "V002"
    VALIDATION_JSON_STRUCTURE = "V003"

    # System errors (E180-E199)
    SYSTEM_CAPACITY = "E100"
    SYSTEM_INTERNAL = "E199"
    SYSTEM_CIVSERVER_CONNECTION = "E140"

    # Security errors (E200-E219)
    SECURITY_VIOLATION = "E200"
    SECURITY_INJECTION_ATTEMPT = "E201"
    SECURITY_CACHE_POISONING = "E202"

class ErrorResponse:
    """Standardized error response structure"""

    def __init__(self, code: str, message: str, category: ErrorCategory = None,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 retry_after: int = None, details: Dict[str, Any] = None):
        self.code = code
        self.message = message
        self.category = category
        self.severity = severity
        self.retry_after = retry_after
        self.details = details or {}
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        response = {
            'type': 'error',
            'code': self.code,
            'message': self.message,
            'timestamp': self.timestamp
        }

        if self.retry_after:
            response['retry_after'] = self.retry_after

        if self.details:
            response['details'] = self.details

        return response

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())

class ErrorHandler:
    """
    Centralized error handling with security logging and recovery strategies
    """

    def __init__(self):
        self.error_counts = {}  # Track error frequencies
        self.last_cleanup = time.time()

    def handle_authentication_error(self, agent_id: str, reason: str,
                                  session_id: str = None) -> ErrorResponse:
        """Handle authentication-related errors"""
        SecurityLogger.log_authentication_attempt(
            agent_id, False, session_id=session_id, details=reason
        )

        if "token" in reason.lower():
            code = ErrorCode.AUTH_INVALID_TOKEN
        elif "session" in reason.lower():
            code = ErrorCode.AUTH_SESSION_EXPIRED
        else:
            code = ErrorCode.AUTH_FAILED

        return ErrorResponse(
            code=code,
            message=f"Authentication failed: {reason}",
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH
        )

    def handle_rate_limit_error(self, agent_id: str, limit_type: str = "default",
                              retry_after: int = 60) -> ErrorResponse:
        """Handle rate limiting errors"""
        SecurityLogger.log_rate_limit_exceeded(agent_id, limit_type=limit_type)

        return ErrorResponse(
            code=ErrorCode.RATE_LIMIT_EXCEEDED,
            message=f"Rate limit exceeded for {limit_type}",
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            retry_after=retry_after
        )

    def handle_validation_error(self, agent_id: str, validation_code: str,
                              message: str, session_id: str = None,
                              input_data: str = None) -> ErrorResponse:
        """Handle input validation errors"""
        SecurityLogger.log_validation_error(
            agent_id, validation_code, message,
            session_id=session_id, input_data=input_data
        )

        severity = ErrorSeverity.HIGH if validation_code.startswith('V00') else ErrorSeverity.MEDIUM

        return ErrorResponse(
            code=validation_code,
            message=f"Validation failed: {message}",
            category=ErrorCategory.VALIDATION,
            severity=severity
        )

    def handle_security_violation(self, agent_id: str, violation_type: str,
                                details: str, session_id: str = None) -> ErrorResponse:
        """Handle security violations"""
        SecurityLogger.log_security_violation(
            agent_id, violation_type, details,
            severity='high', session_id=session_id
        )

        return ErrorResponse(
            code=ErrorCode.SECURITY_VIOLATION,
            message="Security violation detected",
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            details={'violation_type': violation_type}
        )

    def handle_system_error(self, operation: str, error: Exception,
                          agent_id: str = None, session_id: str = None) -> ErrorResponse:
        """Handle system-level errors with appropriate logging"""
        error_msg = str(error)
        error_type = type(error).__name__

        # Log detailed error for debugging
        logger.error(f"System error in {operation}: {error_type}: {error_msg}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")

        # Track error frequency
        self._track_error(operation, error_type)

        # Determine appropriate response based on error type
        if "connection" in error_msg.lower() or "socket" in error_msg.lower():
            code = ErrorCode.SYSTEM_CIVSERVER_CONNECTION
            message = "Game server connection error"
            severity = ErrorSeverity.HIGH
        else:
            code = ErrorCode.SYSTEM_INTERNAL
            message = "Internal server error"
            severity = ErrorSeverity.CRITICAL

        return ErrorResponse(
            code=code,
            message=message,
            category=ErrorCategory.SYSTEM,
            severity=severity,
            details={'operation': operation}
        )

    def handle_capacity_error(self, resource_type: str, current: int,
                            maximum: int) -> ErrorResponse:
        """Handle capacity/resource limit errors"""
        return ErrorResponse(
            code=ErrorCode.SYSTEM_CAPACITY,
            message=f"Server capacity exceeded: {current}/{maximum} {resource_type}",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            retry_after=300,  # 5 minutes
            details={
                'resource_type': resource_type,
                'current': current,
                'maximum': maximum
            }
        )

    def should_circuit_break(self, operation: str, error_threshold: int = 10,
                           time_window: int = 300) -> bool:
        """Determine if circuit breaker should activate"""
        now = time.time()

        # Clean old entries periodically
        if now - self.last_cleanup > 60:  # Cleanup every minute
            self._cleanup_error_counts()
            self.last_cleanup = now

        # Check error count in time window
        if operation in self.error_counts:
            recent_errors = [
                timestamp for timestamp in self.error_counts[operation]
                if now - timestamp < time_window
            ]

            if len(recent_errors) >= error_threshold:
                logger.warning(f"Circuit breaker activated for {operation}: "
                             f"{len(recent_errors)} errors in {time_window}s")
                return True

        return False

    def _track_error(self, operation: str, error_type: str):
        """Track error occurrences for circuit breaker logic"""
        key = f"{operation}:{error_type}"
        now = time.time()

        if key not in self.error_counts:
            self.error_counts[key] = []

        self.error_counts[key].append(now)

        # Keep only recent errors (last hour)
        cutoff = now - 3600
        self.error_counts[key] = [
            timestamp for timestamp in self.error_counts[key]
            if timestamp > cutoff
        ]

    def _cleanup_error_counts(self):
        """Clean up old error tracking data"""
        now = time.time()
        cutoff = now - 3600  # Keep last hour

        for operation in list(self.error_counts.keys()):
            self.error_counts[operation] = [
                timestamp for timestamp in self.error_counts[operation]
                if timestamp > cutoff
            ]

            # Remove empty entries
            if not self.error_counts[operation]:
                del self.error_counts[operation]

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error handling statistics"""
        now = time.time()
        total_errors = sum(len(errors) for errors in self.error_counts.values())

        # Recent errors (last 5 minutes)
        recent_cutoff = now - 300
        recent_errors = sum(
            len([e for e in errors if e > recent_cutoff])
            for errors in self.error_counts.values()
        )

        return {
            'total_tracked_errors': total_errors,
            'recent_errors_5min': recent_errors,
            'tracked_operations': len(self.error_counts),
            'error_types': list(self.error_counts.keys())
        }

# Global error handler instance
error_handler = ErrorHandler()

def with_error_handling(operation: str, agent_id: str = None,
                       session_id: str = None):
    """Decorator for automatic error handling"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_response = error_handler.handle_system_error(
                    operation, e, agent_id, session_id
                )
                logger.exception(f"Error in {operation}: {e}")
                return error_response
        return wrapper
    return decorator