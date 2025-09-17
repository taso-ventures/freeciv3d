#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authentication and authorization for FreeCiv proxy API
Provides API key validation and session-based authentication
"""

import time
import hmac
import hashlib
import logging
import os
from typing import Dict, Optional, Tuple, Set
from dataclasses import dataclass

logger = logging.getLogger("freeciv-proxy")


@dataclass
class AuthSession:
    """Represents an authenticated session"""
    session_id: str
    player_id: int
    game_id: str
    authenticated_at: float
    expires_at: float
    permissions: Set[str]


class AuthenticationError(Exception):
    """Raised when authentication fails"""
    pass


class AuthorizationError(Exception):
    """Raised when authorization fails"""
    pass


class SimpleAuthenticator:
    """
    Simple authentication system for API endpoints
    Supports API keys and session-based authentication
    """

    def __init__(self):
        # API key configuration
        self.api_key_secret = os.getenv('API_KEY_SECRET')
        if not self.api_key_secret or len(self.api_key_secret) < 32:
            logger.warning("API_KEY_SECRET not set or too short, API key authentication disabled")
            self.api_key_secret = None

        # Session storage (in production, use Redis or database)
        self.active_sessions: Dict[str, AuthSession] = {}

        # Session configuration
        self.session_timeout = int(os.getenv('SESSION_TIMEOUT_SECONDS', '3600'))  # 1 hour
        self.max_sessions_per_player = int(os.getenv('MAX_SESSIONS_PER_PLAYER', '3'))

        # Cleanup tracking
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes

    def generate_api_key(self, player_id: int, game_id: str = "global") -> Optional[str]:
        """Generate an API key for a player (admin function)"""
        if not self.api_key_secret:
            return None

        # Create payload with player info and timestamp
        payload = f"{player_id}:{game_id}:{int(time.time())}"

        # Generate HMAC signature
        signature = hmac.new(
            self.api_key_secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        # Encode as API key
        api_key = f"fcv_{payload}_{signature[:16]}"
        return api_key

    def validate_api_key(self, api_key: str) -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Validate API key and extract player info

        Returns:
            tuple: (valid: bool, player_id: Optional[int], game_id: Optional[str])
        """
        if not self.api_key_secret or not api_key:
            return False, None, None

        try:
            # Parse API key format: fcv_{player_id}:{game_id}:{timestamp}_{signature}
            if not api_key.startswith('fcv_'):
                return False, None, None

            parts = api_key[4:].rsplit('_', 1)
            if len(parts) != 2:
                return False, None, None

            payload, provided_signature = parts

            # Generate expected signature
            expected_signature = hmac.new(
                self.api_key_secret.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()[:16]

            # Verify signature
            if not hmac.compare_digest(provided_signature, expected_signature):
                logger.warning(f"Invalid API key signature")
                return False, None, None

            # Extract player info
            payload_parts = payload.split(':')
            if len(payload_parts) != 3:
                return False, None, None

            player_id = int(payload_parts[0])
            game_id = payload_parts[1]
            timestamp = int(payload_parts[2])

            # Check if key is too old (optional expiration)
            max_age = int(os.getenv('API_KEY_MAX_AGE_DAYS', '365')) * 24 * 3600
            if time.time() - timestamp > max_age:
                logger.warning(f"Expired API key for player {player_id}")
                return False, None, None

            return True, player_id, game_id

        except (ValueError, IndexError) as e:
            logger.warning(f"Malformed API key: {e}")
            return False, None, None

    def create_session(self, player_id: int, game_id: str,
                      permissions: Optional[Set[str]] = None) -> str:
        """Create a new authenticated session"""
        # Clean up old sessions first
        self._cleanup_expired_sessions()

        # Check session limits per player
        player_sessions = [
            s for s in self.active_sessions.values()
            if s.player_id == player_id and s.expires_at > time.time()
        ]

        if len(player_sessions) >= self.max_sessions_per_player:
            # Remove oldest session
            oldest_session = min(player_sessions, key=lambda s: s.authenticated_at)
            del self.active_sessions[oldest_session.session_id]
            logger.info(f"Removed oldest session for player {player_id} due to limit")

        # Generate session ID
        import uuid
        session_id = str(uuid.uuid4())

        # Create session
        current_time = time.time()
        session = AuthSession(
            session_id=session_id,
            player_id=player_id,
            game_id=game_id,
            authenticated_at=current_time,
            expires_at=current_time + self.session_timeout,
            permissions=permissions or {'state_read', 'actions_read'}
        )

        self.active_sessions[session_id] = session
        logger.info(f"Created session {session_id} for player {player_id}")

        return session_id

    def validate_session(self, session_id: str) -> Optional[AuthSession]:
        """Validate and return session if valid"""
        if not session_id or session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]

        # Check expiration
        if time.time() > session.expires_at:
            del self.active_sessions[session_id]
            logger.debug(f"Expired session {session_id}")
            return None

        return session

    def check_permission(self, session: AuthSession, required_permission: str) -> bool:
        """Check if session has required permission"""
        return required_permission in session.permissions

    def authenticate_request(self, api_key: Optional[str] = None,
                           session_id: Optional[str] = None,
                           required_permission: str = 'state_read') -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Authenticate a request using API key or session

        Returns:
            tuple: (authenticated: bool, player_id: Optional[int], game_id: Optional[str])
        """
        # Try API key authentication first
        if api_key:
            valid, player_id, game_id = self.validate_api_key(api_key)
            if valid:
                logger.debug(f"API key authentication successful for player {player_id}")
                return True, player_id, game_id

        # Try session authentication
        if session_id:
            session = self.validate_session(session_id)
            if session and self.check_permission(session, required_permission):
                logger.debug(f"Session authentication successful for player {session.player_id}")
                return True, session.player_id, session.game_id

        return False, None, None

    def _cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = time.time()

        if current_time - self.last_cleanup < self.cleanup_interval:
            return

        expired_sessions = [
            session_id for session_id, session in self.active_sessions.items()
            if session.expires_at < current_time
        ]

        for session_id in expired_sessions:
            del self.active_sessions[session_id]

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

        self.last_cleanup = current_time

    def revoke_session(self, session_id: str) -> bool:
        """Revoke a specific session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Revoked session {session_id}")
            return True
        return False

    def revoke_all_sessions(self, player_id: int) -> int:
        """Revoke all sessions for a player"""
        sessions_to_revoke = [
            session_id for session_id, session in self.active_sessions.items()
            if session.player_id == player_id
        ]

        for session_id in sessions_to_revoke:
            del self.active_sessions[session_id]

        logger.info(f"Revoked {len(sessions_to_revoke)} sessions for player {player_id}")
        return len(sessions_to_revoke)

    def get_auth_stats(self) -> Dict[str, int]:
        """Get authentication statistics"""
        self._cleanup_expired_sessions()

        return {
            'active_sessions': len(self.active_sessions),
            'api_key_enabled': bool(self.api_key_secret),
            'session_timeout': self.session_timeout,
            'max_sessions_per_player': self.max_sessions_per_player
        }


# Global authenticator instance
authenticator = SimpleAuthenticator()