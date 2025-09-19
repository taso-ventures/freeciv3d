#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Admin handlers for FreeCiv proxy
Provides administration endpoints for authentication and monitoring
"""

import json
import logging
import os
import time
import hmac
import hashlib
from tornado import web
from auth import authenticator
from security import InputSanitizer, SecurityError

logger = logging.getLogger("freeciv-proxy")


def validate_admin_token(provided_token: str) -> bool:
    """
    Validate admin authentication token using HMAC

    Args:
        provided_token: Token provided by admin user

    Returns:
        bool: True if token is valid and not expired
    """
    admin_secret = os.getenv('ADMIN_KEY_SECRET')
    if not admin_secret:
        logger.error("ADMIN_KEY_SECRET environment variable not set")
        return False

    if len(admin_secret) < 32:
        logger.error("ADMIN_KEY_SECRET must be at least 32 characters")
        return False

    if not provided_token or len(provided_token) < 10:
        return False

    try:
        # Token format: timestamp_signature
        parts = provided_token.rsplit('_', 1)
        if len(parts) != 2:
            return False

        timestamp_str, provided_signature = parts
        timestamp = int(timestamp_str)

        # Check if token is expired (15 minutes = 900 seconds)
        current_time = int(time.time())
        if current_time - timestamp > 900:
            logger.warning("Expired admin token attempted")
            return False

        # Generate expected signature
        expected_signature = hmac.new(
            admin_secret.encode('utf-8'),
            timestamp_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()  # Use full HMAC signature for security

        # Constant-time comparison
        return hmac.compare_digest(provided_signature, expected_signature)

    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid admin token format: {e}")
        return False


def log_admin_access(request_handler, action: str, success: bool, details: str = ""):
    """Log all admin access attempts with security context"""
    client_ip = request_handler.request.remote_ip or "unknown"
    user_agent = request_handler.request.headers.get('User-Agent', 'unknown')

    log_entry = {
        'action': action,
        'success': success,
        'client_ip': client_ip,
        'user_agent': user_agent,
        'timestamp': time.time(),
        'details': details
    }

    if success:
        logger.info(f"Admin access SUCCESS: {action} from {client_ip}")
    else:
        logger.warning(f"Admin access FAILED: {action} from {client_ip} - {details}")


class AdminAuthHandler(web.RequestHandler):
    """Admin endpoint for authentication management"""

    async def post(self):
        """Create API key or session"""
        try:
            # Secure admin authentication with HMAC tokens
            admin_token = self.get_argument('admin_token', None)
            if not validate_admin_token(admin_token):
                log_admin_access(self, "post_auth", False, "Invalid admin token")
                self.set_status(403)
                self.set_header("Content-Type", "application/json")
                self.write({"error": "Admin access required"})
                return

            self.set_header("Content-Type", "application/json")
            try:
                action = InputSanitizer.sanitize_string(self.get_argument('action'))
                player_id = InputSanitizer.sanitize_player_id(self.get_argument('player_id'))
                game_id = InputSanitizer.sanitize_game_id(self.get_argument('game_id', 'global'))
            except SecurityError as e:
                log_admin_access(self, "post_validation", False, f"Input validation failed: {e}")
                logger.warning(f"Input validation failed: {e}")
                self.set_status(400)
                self.write({"error": "Invalid input parameters"})
                return

            log_admin_access(self, f"post_{action}", True, f"player_id={player_id}, game_id={game_id}")

            if action == 'create_api_key':
                api_key = authenticator.generate_api_key(player_id, game_id)
                if api_key:
                    self.write({
                        "api_key": api_key,
                        "player_id": player_id,
                        "game_id": game_id
                    })
                else:
                    self.set_status(500)
                    self.write({"error": "API key generation disabled"})

            elif action == 'create_session':
                session_id = authenticator.create_session(player_id, game_id)
                self.write({
                    "session_id": session_id,
                    "player_id": player_id,
                    "game_id": game_id,
                    "expires_in": authenticator.session_timeout
                })

            elif action == 'revoke_sessions':
                count = authenticator.revoke_all_sessions(player_id)
                self.write({
                    "revoked_sessions": count,
                    "player_id": player_id
                })

            else:
                self.set_status(400)
                self.write({"error": "Invalid action"})

            self.set_header("Content-Type", "application/json")
        except Exception as e:
            log_admin_access(self, "post_error", False, f"Exception: {str(e)}")
            logger.error(f"Admin auth error: {e}")
            self.set_status(500)
            self.write({"error": "Internal server error"})

    async def get(self):
        """Get authentication statistics"""
        try:
            # Secure admin authentication with HMAC tokens
            admin_token = self.get_argument('admin_token', None)
            if not validate_admin_token(admin_token):
                log_admin_access(self, "get_auth", False, "Invalid admin token")
                self.set_status(403)
                self.write({"error": "Admin access required"})
                return

            log_admin_access(self, "get_stats", True, "Retrieved auth stats")
            stats = authenticator.get_auth_stats()
            self.set_header("Content-Type", "application/json")
            self.write(stats)

        except Exception as e:
            log_admin_access(self, "get_error", False, f"Exception: {str(e)}")
            logger.error(f"Admin auth stats error: {e}")
            self.set_status(500)
            self.write({"error": "Internal server error"})
