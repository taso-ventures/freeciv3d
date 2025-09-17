#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Security-focused tests for FreeCiv proxy
Tests input validation, authentication security, rate limiting, and cache integrity
"""

import unittest
import json
import time
import hmac
import hashlib
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
from message_validator import MessageValidator, ValidationError
from security import InputSanitizer, SecurityError, SecurityLogger
from rate_limiter import DistributedRateLimiter, InMemoryRateLimiter
from session_manager import SessionManager, SessionState
from state_cache import StateCache
from error_handler import ErrorHandler, ErrorSeverity, ErrorCategory


class TestInputSanitization(unittest.TestCase):
    """Test input sanitization and validation"""

    def setUp(self):
        self.sanitizer = InputSanitizer()

    def test_sql_injection_prevention(self):
        """Test prevention of SQL injection attempts"""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1 OR 1=1",
            "UNION SELECT * FROM passwords",
            "/* malicious comment */",
            "admin'--",
            "1; DELETE FROM game_state WHERE 1=1"
        ]

        for malicious_input in malicious_inputs:
            with self.assertRaises(SecurityError):
                self.sanitizer.sanitize_string_field(malicious_input, 'agent_id')

    def test_xss_prevention(self):
        """Test prevention of XSS attacks"""
        xss_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "<img src=x onerror=alert(1)>",
            "<%2fscript%3e"
        ]

        for xss_input in xss_inputs:
            with self.assertRaises(SecurityError):
                self.sanitizer.sanitize_string_field(xss_input, 'agent_id')

    def test_valid_input_acceptance(self):
        """Test that valid inputs are accepted"""
        valid_inputs = [
            ("test-agent-1", "agent_id"),
            ("Warriors", "unit_type"),
            ("Alphabet", "tech_name"),
            ("Capital", "production_type")
        ]

        for valid_input, field_type in valid_inputs:
            try:
                result = self.sanitizer.sanitize_string_field(valid_input, field_type)
                self.assertEqual(result, valid_input)
            except SecurityError:
                self.fail(f"Valid input '{valid_input}' was rejected for field '{field_type}'")

    def test_coordinate_validation(self):
        """Test coordinate sanitization"""
        # Valid coordinates
        x, y = self.sanitizer.sanitize_coordinates(10, 20)
        self.assertEqual((x, y), (10, 20))

        x, y = self.sanitizer.sanitize_coordinates(-5, -10)
        self.assertEqual((x, y), (-5, -10))

        # Invalid coordinates
        with self.assertRaises(SecurityError):
            self.sanitizer.sanitize_coordinates(10000, 10000)

        with self.assertRaises(SecurityError):
            self.sanitizer.sanitize_coordinates("invalid", 20)

    def test_action_data_sanitization(self):
        """Test comprehensive action data sanitization"""
        # Valid action
        valid_action = {
            'type': 'unit_move',
            'unit_id': 123,
            'dest_x': 10,
            'dest_y': 20,
            'player_id': 1
        }

        sanitized = self.sanitizer.sanitize_action_data(valid_action)
        self.assertEqual(sanitized['type'], 'unit_move')
        self.assertEqual(sanitized['unit_id'], 123)

        # Action with SQL injection attempt
        malicious_action = {
            'type': "unit_move'; DROP TABLE units; --",
            'unit_id': 123
        }

        with self.assertRaises(SecurityError):
            self.sanitizer.sanitize_action_data(malicious_action)


class TestMessageValidation(unittest.TestCase):
    """Test WebSocket message validation"""

    def setUp(self):
        self.validator = MessageValidator(max_message_size=1024)

    def test_message_size_limits(self):
        """Test message size validation"""
        # Large message
        large_message = json.dumps({'type': 'test', 'data': 'x' * 2000})

        with self.assertRaises(ValidationError) as context:
            self.validator.validate_message(large_message)

        self.assertEqual(context.exception.error_code, 'V001')

    def test_json_depth_limits(self):
        """Test JSON depth validation"""
        # Create deeply nested JSON
        nested = {}
        current = nested
        for i in range(15):  # Exceeds MAX_JSON_DEPTH
            current['nested'] = {}
            current = current['nested']

        deep_message = json.dumps({'type': 'test', 'data': nested})

        with self.assertRaises(ValidationError) as context:
            self.validator.validate_message(deep_message)

        self.assertEqual(context.exception.error_code, 'V004')

    def test_invalid_json_handling(self):
        """Test invalid JSON handling"""
        invalid_messages = [
            "not json at all",
            '{"incomplete": json',
            '{"type": "test" missing comma "data": "value"}',
            ""
        ]

        for invalid_msg in invalid_messages:
            with self.assertRaises(ValidationError) as context:
                self.validator.validate_message(invalid_msg)

            self.assertEqual(context.exception.error_code, 'V002')

    def test_schema_validation(self):
        """Test message schema validation"""
        # Valid connect message
        valid_connect = json.dumps({
            'type': 'llm_connect',
            'agent_id': 'test-agent',
            'api_token': 'valid-token-123'
        })

        try:
            result = self.validator.validate_message(valid_connect)
            self.assertEqual(result['type'], 'llm_connect')
        except ValidationError:
            self.fail("Valid connect message was rejected")

        # Missing required field
        invalid_connect = json.dumps({
            'type': 'llm_connect',
            'agent_id': 'test-agent'
            # Missing api_token
        })

        with self.assertRaises(ValidationError) as context:
            self.validator.validate_message(invalid_connect)

        self.assertEqual(context.exception.error_code, 'V012')

    def test_validation_statistics(self):
        """Test validation statistics tracking"""
        # Reset stats
        self.validator.reset_stats()

        # Process some messages
        valid_msg = json.dumps({'type': 'ping'})
        invalid_msg = "invalid"

        try:
            self.validator.validate_message(valid_msg)
        except ValidationError:
            pass

        try:
            self.validator.validate_message(invalid_msg)
        except ValidationError:
            pass

        stats = self.validator.get_validation_stats()
        self.assertEqual(stats['total_messages'], 2)
        self.assertEqual(stats['valid_messages'], 1)
        self.assertEqual(stats['validation_errors'], 1)


class TestRateLimiting(unittest.TestCase):
    """Test rate limiting functionality"""

    def setUp(self):
        self.rate_limiter = InMemoryRateLimiter()

    def test_token_bucket_algorithm(self):
        """Test token bucket rate limiting"""
        key = "test-agent"
        limit = 5
        window = 10

        # Should allow requests up to limit
        for i in range(limit):
            self.assertTrue(self.rate_limiter.check_limit(key, limit, window))

        # Should reject additional requests
        self.assertFalse(self.rate_limiter.check_limit(key, limit, window))

    def test_token_refill_over_time(self):
        """Test token refill over time"""
        key = "test-agent"
        limit = 2
        window = 1  # 1 second

        # Consume tokens
        self.assertTrue(self.rate_limiter.check_limit(key, limit, window))
        self.assertTrue(self.rate_limiter.check_limit(key, limit, window))
        self.assertFalse(self.rate_limiter.check_limit(key, limit, window))

        # Wait for refill (simulate time passage)
        import time
        time.sleep(1.1)

        # Should allow requests again
        self.assertTrue(self.rate_limiter.check_limit(key, limit, window))

    def test_remaining_tokens(self):
        """Test remaining token calculation"""
        key = "test-agent"
        limit = 5
        window = 10

        # Initially should have full tokens
        remaining = self.rate_limiter.get_remaining(key, limit, window)
        self.assertEqual(remaining, limit)

        # After consuming one token
        self.rate_limiter.check_limit(key, limit, window)
        remaining = self.rate_limiter.get_remaining(key, limit, window)
        self.assertEqual(remaining, limit - 1)

    def test_distributed_rate_limiter_fallback(self):
        """Test distributed rate limiter fallback to in-memory"""
        # Mock Redis failure
        with patch('redis.Redis') as mock_redis:
            mock_redis.side_effect = Exception("Redis connection failed")

            limiter = DistributedRateLimiter({'host': 'localhost', 'port': 6379})

            # Should fall back to in-memory limiter
            self.assertTrue(limiter.check_limit('test-agent', 'default'))


class TestSessionManagement(unittest.TestCase):
    """Test session management security"""

    def setUp(self):
        self.session_manager = SessionManager(session_timeout=60)

    def test_session_creation_and_validation(self):
        """Test secure session creation and validation"""
        agent_id = "test-agent"
        api_token = "test-token-123"
        capabilities = {'unit_move', 'city_production'}

        # Create session
        session = self.session_manager.create_session(agent_id, api_token, capabilities)
        self.assertIsNotNone(session)
        self.assertEqual(session.agent_id, agent_id)
        self.assertEqual(session.capabilities, capabilities)

        # Validate session
        validated_session = self.session_manager.validate_session(session.session_id, api_token)
        self.assertIsNotNone(validated_session)
        self.assertEqual(validated_session.session_id, session.session_id)

    def test_session_expiration(self):
        """Test session expiration"""
        agent_id = "test-agent"
        api_token = "test-token-123"

        # Create session with short timeout
        short_session_manager = SessionManager(session_timeout=1)
        session = short_session_manager.create_session(agent_id, api_token)
        self.assertIsNotNone(session)

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        validated_session = short_session_manager.validate_session(session.session_id, api_token)
        self.assertIsNone(validated_session)

    def test_invalid_token_rejection(self):
        """Test rejection of invalid tokens"""
        agent_id = "test-agent"
        api_token = "valid-token"
        wrong_token = "wrong-token"

        session = self.session_manager.create_session(agent_id, api_token)
        self.assertIsNotNone(session)

        # Should reject wrong token
        validated_session = self.session_manager.validate_session(session.session_id, wrong_token)
        self.assertIsNone(validated_session)

    def test_concurrent_session_limits(self):
        """Test concurrent session limits and thread safety"""
        import threading
        import concurrent.futures

        # Create session manager with low limit for testing
        limited_session_manager = SessionManager(max_concurrent_sessions=3)

        # Function to create sessions concurrently
        def create_session_worker(agent_id: str, api_token: str):
            return limited_session_manager.create_session(agent_id, api_token)

        # Create multiple sessions up to the limit
        sessions = []
        for i in range(3):
            session = limited_session_manager.create_session(f"agent-{i}", f"token-{i}")
            self.assertIsNotNone(session)
            sessions.append(session)

        # Should reject additional session
        overflow_session = limited_session_manager.create_session("agent-overflow", "token-overflow")
        self.assertIsNone(overflow_session)

        # Test concurrent session creation with threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Try to create 5 sessions concurrently when limit is 3
            futures = []
            for i in range(5, 10):
                future = executor.submit(create_session_worker, f"concurrent-agent-{i}", f"token-{i}")
                futures.append(future)

            # Collect results
            results = [future.result() for future in futures]

            # Should have no successful sessions (all slots taken)
            successful_sessions = [r for r in results if r is not None]
            self.assertEqual(len(successful_sessions), 0)

        # Clean up by terminating a session
        terminated = limited_session_manager.terminate_session(sessions[0].session_id)
        self.assertTrue(terminated)

        # Now should be able to create one more
        new_session = limited_session_manager.create_session("agent-new", "token-new")
        self.assertIsNotNone(new_session)

    def test_session_race_condition_prevention(self):
        """Test prevention of race conditions in session creation"""
        import threading
        import concurrent.futures

        # Create session manager with limit of 1 for clear testing
        race_session_manager = SessionManager(max_concurrent_sessions=1)

        results = []
        errors = []

        def concurrent_session_creator(agent_id: str):
            try:
                return race_session_manager.create_session(agent_id, f"token-{agent_id}")
            except Exception as e:
                errors.append(e)
                return None

        # Launch multiple threads trying to create sessions simultaneously
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(10):
                future = executor.submit(concurrent_session_creator, f"race-agent-{i}")
                futures.append(future)

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)

        # Should have exactly one successful session and 9 None results
        successful_sessions = [r for r in results if r is not None]
        failed_sessions = [r for r in results if r is None]

        self.assertEqual(len(successful_sessions), 1)
        self.assertEqual(len(failed_sessions), 9)
        self.assertEqual(len(errors), 0)  # No exceptions should occur

    def test_session_cleanup(self):
        """Test expired session cleanup"""
        agent_id = "test-agent"
        api_token = "test-token-123"

        # Create session with short timeout
        short_session_manager = SessionManager(session_timeout=1, cleanup_interval=0)
        session = short_session_manager.create_session(agent_id, api_token)

        # Wait for expiration
        time.sleep(1.1)

        # Trigger cleanup
        cleaned_count = short_session_manager.cleanup_expired_sessions()
        self.assertGreater(cleaned_count, 0)

    def test_session_capacity_limits(self):
        """Test session capacity enforcement"""
        limited_session_manager = SessionManager(max_concurrent_sessions=2)

        # Create up to limit
        session1 = limited_session_manager.create_session("agent1", "token1")
        session2 = limited_session_manager.create_session("agent2", "token2")

        self.assertIsNotNone(session1)
        self.assertIsNotNone(session2)

        # Should reject additional sessions
        session3 = limited_session_manager.create_session("agent3", "token3")
        self.assertIsNone(session3)


class TestCacheIntegrity(unittest.TestCase):
    """Test cache integrity and security"""

    def setUp(self):
        self.cache = StateCache(ttl=60, max_size_kb=4)

    def test_cache_hmac_integrity(self):
        """Test HMAC-based cache integrity"""
        test_data = {'test': 'data', 'number': 123}
        cache_key = 'integrity_test'
        player_id = 1

        # Set data in cache
        success = self.cache.set(cache_key, test_data, player_id)
        self.assertTrue(success)

        # Retrieve data
        retrieved_data = self.cache.get(cache_key)
        self.assertIsNotNone(retrieved_data)
        self.assertEqual(retrieved_data['test'], 'data')

    def test_cache_poisoning_detection(self):
        """Test detection of cache poisoning attempts"""
        test_data = {'legitimate': 'data'}
        cache_key = 'poisoning_test'
        player_id = 1

        # Set legitimate data
        self.cache.set(cache_key, test_data, player_id)

        # Manually tamper with cache entry
        if cache_key in self.cache.cache:
            entry = self.cache.cache[cache_key]
            entry.data['malicious'] = 'injected_data'

            # Should detect tampering and reject
            retrieved_data = self.cache.get(cache_key)
            self.assertIsNone(retrieved_data)

    def test_cache_size_limits(self):
        """Test cache size enforcement"""
        large_data = {'data': 'x' * 5000}  # Larger than 4KB limit
        cache_key = 'size_test'
        player_id = 1

        # Should reject oversized data
        success = self.cache.set(cache_key, large_data, player_id)
        self.assertFalse(success)

    def test_cache_ttl_enforcement(self):
        """Test TTL enforcement"""
        test_data = {'ttl': 'test'}
        cache_key = 'ttl_test'
        player_id = 1

        # Create cache with short TTL
        short_cache = StateCache(ttl=1)
        success = short_cache.set(cache_key, test_data, player_id)
        self.assertTrue(success)

        # Should be available immediately
        retrieved = short_cache.get(cache_key)
        self.assertIsNotNone(retrieved)

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        retrieved = short_cache.get(cache_key)
        self.assertIsNone(retrieved)


class TestErrorHandling(unittest.TestCase):
    """Test security-focused error handling"""

    def setUp(self):
        self.error_handler = ErrorHandler()

    def test_authentication_error_handling(self):
        """Test authentication error responses"""
        agent_id = "test-agent"
        error_response = self.error_handler.handle_authentication_error(
            agent_id, "Invalid API token"
        )

        self.assertEqual(error_response.category, ErrorCategory.AUTHENTICATION)
        self.assertEqual(error_response.severity, ErrorSeverity.HIGH)
        self.assertIn("authentication", error_response.message.lower())

    def test_security_violation_handling(self):
        """Test security violation responses"""
        agent_id = "malicious-agent"
        error_response = self.error_handler.handle_security_violation(
            agent_id, "SQL injection attempt", "Detected malicious SQL in input"
        )

        self.assertEqual(error_response.category, ErrorCategory.SECURITY)
        self.assertEqual(error_response.severity, ErrorSeverity.CRITICAL)

    def test_rate_limit_error_handling(self):
        """Test rate limit error responses"""
        agent_id = "spamming-agent"
        error_response = self.error_handler.handle_rate_limit_error(
            agent_id, "message", 60
        )

        self.assertEqual(error_response.category, ErrorCategory.RATE_LIMIT)
        self.assertEqual(error_response.retry_after, 60)

    def test_error_frequency_tracking(self):
        """Test error frequency tracking for circuit breaker"""
        operation = "test_operation"

        # Simulate multiple errors
        for i in range(5):
            try:
                raise Exception(f"Test error {i}")
            except Exception as e:
                self.error_handler.handle_system_error(operation, e)

        # Should not trigger circuit breaker yet (threshold is 10)
        self.assertFalse(self.error_handler.should_circuit_break(operation, 10))

        # Simulate more errors
        for i in range(6):
            try:
                raise Exception(f"Test error {i + 5}")
            except Exception as e:
                self.error_handler.handle_system_error(operation, e)

        # Should trigger circuit breaker
        self.assertTrue(self.error_handler.should_circuit_break(operation, 10))


class TestSecurityLogging(unittest.TestCase):
    """Test security event logging"""

    def setUp(self):
        # Mock the logger to capture log messages
        self.log_messages = []

        def mock_log(level, message):
            self.log_messages.append((level, message))

        self.original_info = SecurityLogger.logger.info
        self.original_warning = SecurityLogger.logger.warning
        self.original_error = SecurityLogger.logger.error

        SecurityLogger.logger.info = lambda msg: mock_log('INFO', msg)
        SecurityLogger.logger.warning = lambda msg: mock_log('WARNING', msg)
        SecurityLogger.logger.error = lambda msg: mock_log('ERROR', msg)

    def tearDown(self):
        # Restore original logger methods
        SecurityLogger.logger.info = self.original_info
        SecurityLogger.logger.warning = self.original_warning
        SecurityLogger.logger.error = self.original_error

    def test_authentication_logging(self):
        """Test authentication event logging"""
        SecurityLogger.log_authentication_attempt(
            "test-agent", True, "192.168.1.100", "session-123"
        )

        # Check if log message was created
        self.assertTrue(len(self.log_messages) > 0)
        level, message = self.log_messages[-1]
        self.assertEqual(level, 'INFO')
        self.assertIn('AUTH_SUCCESS', message)
        self.assertIn('test-agent', message)

    def test_security_violation_logging(self):
        """Test security violation logging"""
        SecurityLogger.log_security_violation(
            "malicious-agent", "injection_attempt", "SQL injection detected", "high"
        )

        level, message = self.log_messages[-1]
        self.assertEqual(level, 'ERROR')
        self.assertIn('SECURITY_VIOLATION', message)
        self.assertIn('malicious-agent', message)

    def test_rate_limit_logging(self):
        """Test rate limit violation logging"""
        SecurityLogger.log_rate_limit_exceeded(
            "spamming-agent", "192.168.1.200", "message", 50
        )

        level, message = self.log_messages[-1]
        self.assertEqual(level, 'WARNING')
        self.assertIn('RATE_LIMIT_EXCEEDED', message)
        self.assertIn('spamming-agent', message)


if __name__ == '__main__':
    unittest.main()