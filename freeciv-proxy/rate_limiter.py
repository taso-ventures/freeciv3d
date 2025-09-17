#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Distributed rate limiting system using Redis
Implements sliding window and token bucket algorithms
"""

import time
import logging
import json
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger("freeciv-proxy")

class RateLimiter(ABC):
    """Abstract base class for rate limiters"""

    @abstractmethod
    def check_limit(self, key: str, limit: int, window: int) -> bool:
        """Check if request is within rate limit"""
        pass

    @abstractmethod
    def get_remaining(self, key: str, limit: int, window: int) -> int:
        """Get remaining requests in current window"""
        pass

class InMemoryRateLimiter(RateLimiter):
    """
    In-memory rate limiter using token bucket algorithm
    Used when Redis is not available
    """

    def __init__(self):
        self.buckets: Dict[str, Dict[str, float]] = {}

    def check_limit(self, key: str, limit: int, window: int) -> bool:
        """Check if request is within rate limit using token bucket"""
        now = time.time()

        if key not in self.buckets:
            self.buckets[key] = {
                'tokens': limit,
                'last_refill': now
            }

        bucket = self.buckets[key]

        # Calculate tokens to add based on time passed
        time_passed = now - bucket['last_refill']
        tokens_to_add = time_passed * (limit / window)

        bucket['tokens'] = min(limit, bucket['tokens'] + tokens_to_add)
        bucket['last_refill'] = now

        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            return True

        return False

    def get_remaining(self, key: str, limit: int, window: int) -> int:
        """Get remaining tokens in bucket"""
        if key not in self.buckets:
            return limit

        return max(0, int(self.buckets[key]['tokens']))

class RedisRateLimiter(RateLimiter):
    """
    Redis-backed rate limiter using sliding window log algorithm
    More accurate but requires Redis connection
    """

    def __init__(self, redis_client=None, redis_config: Dict[str, Any] = None):
        self.redis = redis_client
        self.redis_config = redis_config or {}

        if not self.redis and redis_config:
            try:
                import redis
                # Use connection pool for better performance and reliability
                pool = redis.ConnectionPool(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6379),
                    password=redis_config.get('password'),
                    db=redis_config.get('db', 0),
                    decode_responses=True,
                    socket_timeout=2.0,
                    socket_connect_timeout=2.0,
                    retry_on_timeout=True,
                    max_connections=redis_config.get('max_connections', 20),
                    health_check_interval=redis_config.get('health_check_interval', 30)
                )
                self.redis = redis.Redis(connection_pool=pool)
                # Test connection
                self.redis.ping()
                logger.info("Connected to Redis with connection pool for rate limiting")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self.redis = None

    def is_available(self) -> bool:
        """Check if Redis is available"""
        if not self.redis:
            return False

        try:
            self.redis.ping()
            return True
        except Exception:
            return False

    def check_limit(self, key: str, limit: int, window: int) -> bool:
        """Check rate limit using sliding window log"""
        if not self.is_available():
            logger.warning("Redis unavailable, rate limiting disabled")
            return True

        try:
            now = time.time()
            pipeline = self.redis.pipeline()

            # Remove expired entries
            cutoff = now - window
            pipeline.zremrangebyscore(key, 0, cutoff)

            # Count current requests
            pipeline.zcard(key)

            # Add current request
            pipeline.zadd(key, {str(now): now})

            # Set expiration
            pipeline.expire(key, window + 1)

            results = pipeline.execute()

            current_count = results[1]  # Count from zcard

            if current_count < limit:
                logger.debug(f"Rate limit check passed for {key}: {current_count}/{limit}")
                return True
            else:
                logger.warning(f"Rate limit exceeded for {key}: {current_count}/{limit}")
                # Remove the request we just added since it's over limit
                self.redis.zrem(key, str(now))
                return False

        except Exception as e:
            logger.error(f"Redis rate limit error: {e}")
            # Fail open - allow request if Redis fails
            return True

    def get_remaining(self, key: str, limit: int, window: int) -> int:
        """Get remaining requests in current window"""
        if not self.is_available():
            return limit

        try:
            now = time.time()
            cutoff = now - window

            # Clean expired entries and count current
            pipeline = self.redis.pipeline()
            pipeline.zremrangebyscore(key, 0, cutoff)
            pipeline.zcard(key)
            results = pipeline.execute()

            current_count = results[1]
            return max(0, limit - current_count)

        except Exception as e:
            logger.error(f"Redis rate limit error: {e}")
            return limit

class DistributedRateLimiter:
    """
    Main rate limiter that attempts Redis first, falls back to in-memory
    """

    def __init__(self, redis_config: Dict[str, Any] = None):
        self.redis_limiter = RedisRateLimiter(redis_config=redis_config)
        self.memory_limiter = InMemoryRateLimiter()

        # Default rate limits
        self.default_limits = {
            'requests_per_second': 10,
            'burst_capacity': 100,
            'window_seconds': 60
        }

    def check_limit(self, agent_id: str, operation: str = 'default',
                   custom_limit: int = None, custom_window: int = None) -> bool:
        """
        Check if agent is within rate limits

        Args:
            agent_id: Unique identifier for the agent
            operation: Type of operation (e.g., 'message', 'action', 'state_query')
            custom_limit: Override default rate limit
            custom_window: Override default time window

        Returns:
            True if within limits, False if exceeded
        """
        # Use Redis if available, otherwise fall back to in-memory
        limiter = (self.redis_limiter if self.redis_limiter.is_available()
                  else self.memory_limiter)

        # Generate rate limit key
        key = f"rate_limit:{agent_id}:{operation}"

        # Get limits
        limit = custom_limit or self.default_limits['requests_per_second']
        window = custom_window or self.default_limits['window_seconds']

        return limiter.check_limit(key, limit, window)

    def get_remaining(self, agent_id: str, operation: str = 'default',
                     custom_limit: int = None, custom_window: int = None) -> int:
        """Get remaining requests for agent"""
        limiter = (self.redis_limiter if self.redis_limiter.is_available()
                  else self.memory_limiter)

        key = f"rate_limit:{agent_id}:{operation}"
        limit = custom_limit or self.default_limits['requests_per_second']
        window = custom_window or self.default_limits['window_seconds']

        return limiter.get_remaining(key, limit, window)

    def check_burst_limit(self, agent_id: str) -> bool:
        """Check burst rate limit (short-term high volume)"""
        return self.check_limit(
            agent_id,
            'burst',
            custom_limit=self.default_limits['burst_capacity'],
            custom_window=60  # 1 minute burst window
        )

    def get_rate_limit_info(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive rate limit information for agent"""
        return {
            'requests_remaining': self.get_remaining(agent_id, 'default'),
            'burst_remaining': self.get_remaining(agent_id, 'burst',
                                                custom_limit=self.default_limits['burst_capacity']),
            'using_redis': self.redis_limiter.is_available(),
            'limits': self.default_limits
        }

    def reset_limits(self, agent_id: str):
        """Reset rate limits for an agent (admin function)"""
        if self.redis_limiter.is_available():
            try:
                # Delete all rate limit keys for this agent
                pattern = f"rate_limit:{agent_id}:*"
                keys = self.redis_limiter.redis.keys(pattern)
                if keys:
                    self.redis_limiter.redis.delete(*keys)
                logger.info(f"Reset rate limits for agent: {agent_id}")
            except Exception as e:
                logger.error(f"Failed to reset rate limits for {agent_id}: {e}")
        else:
            # Reset in-memory limits
            keys_to_remove = [key for key in self.memory_limiter.buckets.keys()
                            if key.startswith(f"rate_limit:{agent_id}:")]
            for key in keys_to_remove:
                del self.memory_limiter.buckets[key]

class CircuitBreaker:
    """
    Circuit breaker to protect against overload
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
                logger.info("Circuit breaker: transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
                logger.info("Circuit breaker: transitioning to CLOSED")
            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
                logger.warning(f"Circuit breaker: transitioning to OPEN after {self.failure_count} failures")

            raise e