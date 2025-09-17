#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Admin handlers for FreeCiv proxy
Provides administration endpoints for authentication and monitoring
"""

import json
import logging
from tornado import web
from auth import authenticator
from security import InputSanitizer, SecurityError

logger = logging.getLogger("freeciv-proxy")


class AdminAuthHandler(web.RequestHandler):
    """Admin endpoint for authentication management"""

    async def post(self):
        """Create API key or session"""
        try:
            # Simple admin authentication (in production, use proper admin auth)
            admin_key = self.get_argument('admin_key', None)
            if admin_key != "admin-secret-key":  # Replace with proper admin auth
                self.set_status(403)
                self.write({"error": "Admin access required"})
                return

            try:
                action = InputSanitizer.sanitize_string(self.get_argument('action'))
                player_id = InputSanitizer.sanitize_player_id(self.get_argument('player_id'))
                game_id = InputSanitizer.sanitize_game_id(self.get_argument('game_id', 'global'))
            except SecurityError as e:
                logger.warning(f"Input validation failed: {e}")
                self.set_status(400)
                self.write({"error": "Invalid input parameters"})
                return

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

        except Exception as e:
            logger.error(f"Admin auth error: {e}")
            self.set_status(500)
            self.write({"error": "Internal server error"})

    async def get(self):
        """Get authentication statistics"""
        try:
            # Simple admin authentication
            admin_key = self.get_argument('admin_key', None)
            if admin_key != "admin-secret-key":  # Replace with proper admin auth
                self.set_status(403)
                self.write({"error": "Admin access required"})
                return

            stats = authenticator.get_auth_stats()
            self.set_header("Content-Type", "application/json")
            self.write(stats)

        except Exception as e:
            logger.error(f"Admin auth stats error: {e}")
            self.set_status(500)
            self.write({"error": "Internal server error"})
