#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API Rate limiting for FreeCiv proxy state extraction endpoints
Provides per-player and per-IP rate limiting with token bucket algorithm
"""

import time
import logging
import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("freeciv-proxy")


@dataclass
class TokenBucket:
    """Token bucket for rate limiting"""
    capacity: int  # Maximum tokens
    tokens: float  # Current tokens
    fill_rate: float  # Tokens per second
    last_update: float  # Last time tokens were added

    def consume(self, tokens: int = 1) -> bool:
        """Consume tokens from bucket, return True if successful"""
        current_time = time.time()

        # Add tokens based on elapsed time
        elapsed = current_time - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
        self.last_update = current_time

        # Check if we have enough tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def time_until_tokens(self, tokens: int = 1) -> float:
        """Return time in seconds until specified tokens are available"""
        if self.tokens >= tokens:
            return 0.0

        needed_tokens = tokens - self.tokens
        return needed_tokens / self.fill_rate


class APIRateLimiter:
    """
    Rate limiter with per-player and per-IP limits for API endpoints
    Uses token bucket algorithm for smooth rate limiting
    """

    def __init__(self):
        # Player-based rate limiting (60 requests/minute per player)
        self.player_buckets: Dict[int, TokenBucket] = {}
        self.player_requests_per_minute = int(os.getenv('API_RATE_LIMIT_PLAYER_RPM', '60'))
        self.player_capacity = self.player_requests_per_minute
        self.player_fill_rate = self.player_requests_per_minute / 60.0  # tokens per second

        # IP-based rate limiting (120 requests/minute per IP)
        self.ip_buckets: Dict[str, TokenBucket] = {}
        self.ip_requests_per_minute = int(os.getenv('API_RATE_LIMIT_IP_RPM', '120'))
        self.ip_capacity = self.ip_requests_per_minute
        self.ip_fill_rate = self.ip_requests_per_minute / 60.0  # tokens per second

        # Cleanup tracking
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # Clean up every 5 minutes

    def check_player_limit(self, player_id: int) -> Tuple[bool, Optional[float]]:
        """
        Check if player request is within rate limit

        Returns:
            tuple: (allowed: bool, retry_after: Optional[float])
        """
        # Create bucket if it doesn't exist
        if player_id not in self.player_buckets:
            current_time = time.time()
            self.player_buckets[player_id] = TokenBucket(
                capacity=self.player_capacity,
                tokens=self.player_capacity,
                fill_rate=self.player_fill_rate,
                last_update=current_time
            )

        bucket = self.player_buckets[player_id]

        if bucket.consume():
            return True, None
        else:
            retry_after = bucket.time_until_tokens()
            logger.warning(f"Player {player_id} rate limited, retry after {retry_after:.1f}s")
            return False, retry_after

    def check_ip_limit(self, ip_address: str) -> Tuple[bool, Optional[float]]:
        """
        Check if IP request is within rate limit

        Returns:
            tuple: (allowed: bool, retry_after: Optional[float])
        """
        # Create bucket if it doesn't exist
        if ip_address not in self.ip_buckets:
            current_time = time.time()
            self.ip_buckets[ip_address] = TokenBucket(
                capacity=self.ip_capacity,
                tokens=self.ip_capacity,
                fill_rate=self.ip_fill_rate,
                last_update=current_time
            )

        bucket = self.ip_buckets[ip_address]

        if bucket.consume():
            return True, None
        else:
            retry_after = bucket.time_until_tokens()
            logger.warning(f"IP {ip_address} rate limited, retry after {retry_after:.1f}s")
            return False, retry_after

    def check_limits(self, player_id: int, ip_address: str) -> Tuple[bool, Optional[float], str]:
        """
        Check both player and IP limits

        Returns:
            tuple: (allowed: bool, retry_after: Optional[float], limit_type: str)
        """
        # Perform cleanup periodically
        self.cleanup_old_buckets()

        # Check IP limit first (more restrictive)
        ip_allowed, ip_retry_after = self.check_ip_limit(ip_address)
        if not ip_allowed:
            return False, ip_retry_after, "ip"

        # Check player limit
        player_allowed, player_retry_after = self.check_player_limit(player_id)
        if not player_allowed:
            return False, player_retry_after, "player"

        return True, None, ""

    def cleanup_old_buckets(self):
        """Remove inactive buckets to prevent memory leaks"""
        current_time = time.time()

        if current_time - self.last_cleanup < self.cleanup_interval:
            return

        # Remove player buckets that haven't been used in 1 hour
        inactive_threshold = current_time - 3600
        inactive_players = [
            player_id for player_id, bucket in self.player_buckets.items()
            if bucket.last_update < inactive_threshold
        ]

        for player_id in inactive_players:
            del self.player_buckets[player_id]

        # Remove IP buckets that haven't been used in 1 hour
        inactive_ips = [
            ip for ip, bucket in self.ip_buckets.items()
            if bucket.last_update < inactive_threshold
        ]

        for ip in inactive_ips:
            del self.ip_buckets[ip]

        if inactive_players or inactive_ips:
            logger.info(f"Cleaned up {len(inactive_players)} player buckets and {len(inactive_ips)} IP buckets")

        self.last_cleanup = current_time

    def get_stats(self) -> Dict[str, int]:
        """Get rate limiter statistics"""
        return {
            'active_player_buckets': len(self.player_buckets),
            'active_ip_buckets': len(self.ip_buckets),
            'player_rpm_limit': self.player_requests_per_minute,
            'ip_rpm_limit': self.ip_requests_per_minute,
            'last_cleanup': int(self.last_cleanup)
        }


# Global API rate limiter instance
api_rate_limiter = APIRateLimiter()