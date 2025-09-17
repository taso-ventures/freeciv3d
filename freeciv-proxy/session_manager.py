#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Session management for LLM agents in FreeCiv proxy
Provides secure session tracking, token expiration, and session validation
"""

import time
import uuid
import hmac
import hashlib
import logging
import os
import secrets
import threading
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
try:
    import bcrypt
except ImportError:
    bcrypt = None

logger = logging.getLogger("freeciv-proxy")

class SessionState(Enum):
    """Session states"""
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"

@dataclass
class SessionInfo:
    """Session information container"""
    session_id: str
    agent_id: str
    api_token_hash: str  # Hashed API token for verification
    created_at: float
    last_activity: float
    expires_at: float
    player_id: Optional[int] = None
    capabilities: Set[str] = field(default_factory=set)
    connection_count: int = 0
    state: SessionState = SessionState.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)

class SessionManager:
    """
    Manages LLM agent sessions with secure token handling and expiration
    """

    def __init__(self,
                 session_timeout: int = 3600,  # 1 hour default
                 max_concurrent_sessions: int = 100,
                 cleanup_interval: int = 300):  # 5 minutes
        self.session_timeout = session_timeout
        self.max_concurrent_sessions = max_concurrent_sessions
        self.cleanup_interval = cleanup_interval

        # Session storage
        self.sessions: Dict[str, SessionInfo] = {}
        self.agent_to_session: Dict[str, str] = {}  # agent_id -> session_id mapping

        # Session security
        self.session_secret = self._get_secure_session_secret()

        # Thread safety for session management
        self._session_lock = threading.RLock()

        # Cleanup tracking
        self.last_cleanup = time.time()

        # Statistics
        self.stats = {
            'sessions_created': 0,
            'sessions_expired': 0,
            'sessions_terminated': 0,
            'authentication_attempts': 0,
            'authentication_failures': 0
        }

    def create_session(self, agent_id: str, api_token: str,
                      capabilities: Optional[Set[str]] = None) -> Optional[SessionInfo]:
        """
        Create a new session for an agent

        Args:
            agent_id: Unique agent identifier
            api_token: API token for authentication
            capabilities: Set of allowed capabilities

        Returns:
            SessionInfo if successful, None if failed
        """
        with self._session_lock:
            try:
                # Check session limits with thread safety
                if len(self.sessions) >= self.max_concurrent_sessions:
                    logger.warning(f"Maximum concurrent sessions reached: {self.max_concurrent_sessions}")
                    return None

                # Terminate existing session for this agent if any
                self.terminate_agent_session(agent_id)

                # Generate secure session ID
                session_id = self._generate_session_id(agent_id)

                # Hash the API token for storage
                api_token_hash = self._hash_token(api_token)

                now = time.time()
                session = SessionInfo(
                    session_id=session_id,
                    agent_id=agent_id,
                    api_token_hash=api_token_hash,
                    created_at=now,
                    last_activity=now,
                    expires_at=now + self.session_timeout,
                    capabilities=capabilities or set(),
                    state=SessionState.ACTIVE
                )

                # Store session
                self.sessions[session_id] = session
                self.agent_to_session[agent_id] = session_id

                self.stats['sessions_created'] += 1
                logger.info(f"Created session for agent {agent_id}: {session_id}")

                return session

            except Exception as e:
                logger.error(f"Failed to create session for agent {agent_id}: {e}")
                return None

    def validate_session(self, session_id: str, api_token: Optional[str] = None) -> Optional[SessionInfo]:
        """
        Validate an existing session

        Args:
            session_id: Session identifier
            api_token: Optional API token for additional verification

        Returns:
            SessionInfo if valid, None if invalid/expired
        """
        self.stats['authentication_attempts'] += 1

        try:
            session = self.sessions.get(session_id)
            if not session:
                self.stats['authentication_failures'] += 1
                return None

            # Check session state
            if session.state != SessionState.ACTIVE:
                logger.warning(f"Session {session_id} is not active: {session.state}")
                self.stats['authentication_failures'] += 1
                return None

            # Check expiration
            now = time.time()
            if now > session.expires_at:
                logger.info(f"Session {session_id} expired")
                session.state = SessionState.EXPIRED
                self.stats['sessions_expired'] += 1
                self.stats['authentication_failures'] += 1
                return None

            # Verify API token if provided
            if api_token:
                if not self._verify_token(api_token, session.api_token_hash):
                    logger.warning(f"Invalid API token for session {session_id}")
                    self.stats['authentication_failures'] += 1
                    return None

            # Update last activity
            session.last_activity = now

            logger.debug(f"Validated session {session_id} for agent {session.agent_id}")
            return session

        except Exception as e:
            logger.error(f"Error validating session {session_id}: {e}")
            self.stats['authentication_failures'] += 1
            return None

    def update_session_activity(self, session_id: str) -> bool:
        """Update session last activity timestamp"""
        session = self.sessions.get(session_id)
        if session and session.state == SessionState.ACTIVE:
            session.last_activity = time.time()
            return True
        return False

    def extend_session(self, session_id: str, extension_seconds: int = None) -> bool:
        """
        Extend session expiration time

        Args:
            session_id: Session to extend
            extension_seconds: Additional seconds (default: session_timeout)

        Returns:
            True if extended, False if failed
        """
        session = self.sessions.get(session_id)
        if not session or session.state != SessionState.ACTIVE:
            return False

        extension = extension_seconds or self.session_timeout
        session.expires_at = max(session.expires_at, time.time()) + extension

        logger.debug(f"Extended session {session_id} by {extension} seconds")
        return True

    def terminate_session(self, session_id: str, reason: str = "manual") -> bool:
        """
        Terminate a specific session

        Args:
            session_id: Session to terminate
            reason: Reason for termination

        Returns:
            True if terminated, False if not found
        """
        session = self.sessions.get(session_id)
        if not session:
            return False

        session.state = SessionState.TERMINATED

        # Remove from agent mapping
        if session.agent_id in self.agent_to_session:
            del self.agent_to_session[session.agent_id]

        # Remove from sessions dictionary to free up space
        del self.sessions[session_id]

        self.stats['sessions_terminated'] += 1
        logger.info(f"Terminated session {session_id} for agent {session.agent_id}: {reason}")

        return True

    def terminate_agent_session(self, agent_id: str, reason: str = "new_session") -> bool:
        """Terminate all sessions for a specific agent"""
        session_id = self.agent_to_session.get(agent_id)
        if session_id:
            return self.terminate_session(session_id, reason)
        return False

    def get_session_by_agent(self, agent_id: str) -> Optional[SessionInfo]:
        """Get active session for an agent"""
        session_id = self.agent_to_session.get(agent_id)
        if session_id:
            return self.sessions.get(session_id)
        return None

    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired and terminated sessions

        Returns:
            Number of sessions cleaned up
        """
        now = time.time()

        # Only run cleanup periodically
        if now - self.last_cleanup < self.cleanup_interval:
            return 0

        self.last_cleanup = now

        sessions_to_remove = []
        for session_id, session in self.sessions.items():
            should_remove = False

            if session.state in [SessionState.EXPIRED, SessionState.TERMINATED]:
                should_remove = True
            elif now > session.expires_at:
                session.state = SessionState.EXPIRED
                self.stats['sessions_expired'] += 1
                should_remove = True

            if should_remove:
                sessions_to_remove.append(session_id)

        # Remove expired sessions
        for session_id in sessions_to_remove:
            session = self.sessions[session_id]
            if session.agent_id in self.agent_to_session:
                del self.agent_to_session[session.agent_id]
            del self.sessions[session_id]

        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} expired sessions")

        return len(sessions_to_remove)

    def get_active_session_count(self) -> int:
        """Get count of active sessions"""
        return len([s for s in self.sessions.values() if s.state == SessionState.ACTIVE])

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session management statistics"""
        active_sessions = self.get_active_session_count()

        return {
            **self.stats,
            'active_sessions': active_sessions,
            'total_sessions': len(self.sessions),
            'session_timeout': self.session_timeout,
            'max_concurrent_sessions': self.max_concurrent_sessions
        }

    def suspend_session(self, session_id: str, reason: str = "admin") -> bool:
        """Suspend a session temporarily"""
        session = self.sessions.get(session_id)
        if session and session.state == SessionState.ACTIVE:
            session.state = SessionState.SUSPENDED
            logger.info(f"Suspended session {session_id}: {reason}")
            return True
        return False

    def resume_session(self, session_id: str) -> bool:
        """Resume a suspended session"""
        session = self.sessions.get(session_id)
        if session and session.state == SessionState.SUSPENDED:
            # Check if not expired
            if time.time() <= session.expires_at:
                session.state = SessionState.ACTIVE
                session.last_activity = time.time()
                logger.info(f"Resumed session {session_id}")
                return True
            else:
                session.state = SessionState.EXPIRED
                logger.info(f"Cannot resume expired session {session_id}")
        return False

    def _generate_session_id(self, agent_id: str) -> str:
        """Generate secure session ID"""
        # Include timestamp and agent_id for uniqueness
        data = f"{agent_id}:{time.time()}:{uuid.uuid4()}"

        # Create HMAC-based session ID
        session_id = hmac.new(
            self.session_secret.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return f"sess_{session_id[:32]}"

    def _hash_token(self, token: str) -> str:
        """Hash API token for secure storage using bcrypt or fallback to PBKDF2"""
        if bcrypt:
            # Use bcrypt for secure password hashing
            salt = bcrypt.gensalt(rounds=12)
            return bcrypt.hashpw(token.encode('utf-8'), salt).decode('utf-8')
        else:
            # Fallback to PBKDF2 with salt (more secure than plain HMAC-SHA256)
            salt = secrets.token_bytes(32)
            key = hashlib.pbkdf2_hmac('sha256', token.encode('utf-8'), salt, 100000)
            # Store salt + hash together
            return salt.hex() + ':' + key.hex()

    def _verify_token(self, token: str, stored_hash: str) -> bool:
        """Verify token against stored hash using constant-time comparison"""
        try:
            if bcrypt:
                # bcrypt handles salt and timing attacks internally
                return bcrypt.checkpw(token.encode('utf-8'), stored_hash.encode('utf-8'))
            
            # PBKDF2 verification - requires salt:hash format
            if ':' not in stored_hash:
                # Legacy format detected - reject for security
                logger.warning("Legacy hash format detected - rejecting for security")
                return False

            salt_hex, key_hex = stored_hash.split(':', 1)
            salt = bytes.fromhex(salt_hex)
            stored_key = bytes.fromhex(key_hex)
            new_key = hashlib.pbkdf2_hmac('sha256', token.encode('utf-8'), salt, 100000)
            return hmac.compare_digest(stored_key, new_key)
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return False

    def _get_secure_session_secret(self) -> str:
        """Get or generate secure session secret"""
        secret = os.getenv('SESSION_SECRET')

        if not secret:
            # Check if we're in production environment
            env = os.getenv('ENVIRONMENT', 'development').lower()
            if env == 'production':
                raise ValueError(
                    "SESSION_SECRET environment variable is required in production. "
                    "Generate with: python -c 'import secrets; print(secrets.token_urlsafe(64))'"
                )

            # Generate random secret for development
            secret = secrets.token_urlsafe(64)
            logger.warning(
                "No SESSION_SECRET set. Generated random secret for development. "
                "Set SESSION_SECRET environment variable for production."
            )

        if len(secret) < 32:
            raise ValueError("SESSION_SECRET must be at least 32 characters long")

        return secret

# Global session manager instance
session_manager = SessionManager()
