#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WebSocket handlers for LLM Gateway
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect, FastAPI

try:
    from .connection_manager import connection_manager
    from .config import settings
    from .main import gateway
except ImportError:
    from connection_manager import connection_manager
    from config import settings
    # gateway will be available when main.py imports this

logger = logging.getLogger("llm-gateway")


class AgentWebSocketHandler:
    """Handler for agent WebSocket connections"""

    def __init__(self, websocket: WebSocket, agent_id: str):
        self.websocket = websocket
        self.agent_id = agent_id
        self.connection_id: Optional[str] = None
        self.authenticated = False
        self.player_id: Optional[int] = None
        self.game_id: Optional[str] = None

    async def handle_connection(self):
        """Handle the WebSocket connection lifecycle"""
        await self.websocket.accept()

        try:
            # Add to connection manager
            self.connection_id = await connection_manager.add_connection(
                self.websocket, "agent", self.agent_id
            )

            # Send welcome message
            await self._send_welcome()

            # Message handling loop
            while True:
                try:
                    data = await self.websocket.receive_text()
                    message = json.loads(data)
                    await self._handle_message(message)

                except WebSocketDisconnect:
                    logger.info(f"Agent {self.agent_id} disconnected")
                    break

                except json.JSONDecodeError:
                    await self._send_error("Invalid JSON format")

                except Exception as e:
                    logger.error(f"Error handling message from {self.agent_id}: {e}")
                    await self._send_error("Message processing failed")

        except Exception as e:
            logger.error(f"Error in agent connection {self.agent_id}: {e}")

        finally:
            # Cleanup
            if self.connection_id:
                await connection_manager.handle_disconnect(self.connection_id)

    async def _send_welcome(self):
        """Send welcome message"""
        welcome = {
            "type": "welcome",
            "handler_id": self.connection_id,
            "message": "LLM agent gateway ready. Send llm_connect message to authenticate."
        }
        await self.websocket.send_text(json.dumps(welcome))

    async def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming message from agent"""
        message_type = message.get("type")

        if message_type == "llm_connect":
            await self._handle_connect(message)
        elif message_type == "state_query":
            await self._handle_state_query(message)
        elif message_type == "action":
            await self._handle_action(message)
        elif message_type == "ping":
            await self._handle_ping(message)
        else:
            await self._send_error(f"Unknown message type: {message_type}")

    async def _handle_connect(self, message: Dict[str, Any]):
        """Handle authentication message"""
        try:
            data = message.get("data", {})
            api_token = data.get("api_token")
            model = data.get("model")
            game_id = data.get("game_id")

            if not all([api_token, model, game_id]):
                await self._send_error("Missing required fields: api_token, model, game_id")
                return

            # Register agent with gateway
            config = {
                "agent_id": self.agent_id,
                "api_token": api_token,
                "model": model,
                "game_id": game_id
            }

            if hasattr(gateway, 'register_agent'):
                result = await gateway.register_agent(self.agent_id, config)
            else:
                # Fallback for testing
                result = {
                    "success": True,
                    "session_id": f"session-{self.agent_id}",
                    "player_id": 1
                }

            if result["success"]:
                self.authenticated = True
                self.game_id = game_id
                self.player_id = result.get("player_id", 1)

                # Update connection info
                if self.connection_id and self.connection_id in connection_manager.connections:
                    connection_info = connection_manager.connections[self.connection_id]
                    connection_info.authenticated = True
                    connection_info.metadata.update({
                        "game_id": game_id,
                        "player_id": self.player_id,
                        "model": model
                    })

                response = {
                    "type": "llm_connect",
                    "agent_id": self.agent_id,
                    "timestamp": time.time(),
                    "data": {
                        "type": "auth_success",
                        "success": True,
                        "agent_id": self.agent_id,
                        "session_id": result.get("session_id"),
                        "player_id": self.player_id,
                        "game_id": game_id,
                        "model": model
                    }
                }

                logger.info(f"Agent {self.agent_id} authenticated for game {game_id}")

            else:
                response = {
                    "type": "llm_connect",
                    "agent_id": self.agent_id,
                    "timestamp": time.time(),
                    "data": {
                        "type": "error",
                        "success": False,
                        "error_code": "E102",
                        "error_message": result.get("error", "Authentication failed")
                    }
                }

            await self.websocket.send_text(json.dumps(response))

        except Exception as e:
            logger.error(f"Error in connect handler: {e}")
            await self._send_error("Authentication failed")

    async def _handle_state_query(self, message: Dict[str, Any]):
        """Handle state query message"""
        if not self.authenticated:
            await self._send_error("Not authenticated")
            return

        try:
            data = message.get("data", {})
            format_type = data.get("format", "llm_optimized")
            correlation_id = message.get("correlation_id")

            if hasattr(gateway, 'get_game_state'):
                result = await gateway.get_game_state(self.game_id, self.player_id, format_type)
            else:
                # Fallback for testing
                result = {
                    "success": True,
                    "format": format_type,
                    "data": {
                        "turn": 1,
                        "strategic_summary": {"cities_count": 1}
                    }
                }

            response = {
                "type": "state_update",
                "agent_id": self.agent_id,
                "timestamp": time.time(),
                "correlation_id": correlation_id,
                "data": {
                    "type": "state_response",
                    "format": format_type,
                    "data": result.get("data", {}),
                    "timestamp": time.time()
                }
            }

            await self.websocket.send_text(json.dumps(response))

        except Exception as e:
            logger.error(f"Error in state query handler: {e}")
            await self._send_error("State query failed")

    async def _handle_action(self, message: Dict[str, Any]):
        """Handle action message"""
        if not self.authenticated:
            await self._send_error("Not authenticated")
            return

        try:
            data = message.get("data", {})
            correlation_id = message.get("correlation_id")

            # Add player_id if not present
            if "player_id" not in data:
                data["player_id"] = self.player_id

            if hasattr(gateway, 'submit_action'):
                result = await gateway.submit_action(self.game_id, data)
            else:
                # Fallback for testing
                result = {
                    "success": True,
                    "action_id": f"action-{int(time.time())}"
                }

            response = {
                "type": "action_result",
                "agent_id": self.agent_id,
                "timestamp": time.time(),
                "correlation_id": correlation_id,
                "data": {
                    "type": "action_result",
                    "success": result["success"],
                    "action_type": data.get("action_type"),
                    "result": result
                }
            }

            await self.websocket.send_text(json.dumps(response))

        except Exception as e:
            logger.error(f"Error in action handler: {e}")
            await self._send_error("Action failed")

    async def _handle_ping(self, message: Dict[str, Any]):
        """Handle ping message"""
        pong = {
            "type": "pong",
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
        await self.websocket.send_text(json.dumps(pong))

    async def _send_error(self, error_message: str):
        """Send error message"""
        error = {
            "type": "error",
            "agent_id": self.agent_id,
            "timestamp": time.time(),
            "data": {
                "type": "error",
                "success": False,
                "error_code": "E500",
                "error_message": error_message
            }
        }
        await self.websocket.send_text(json.dumps(error))


class SpectatorWebSocketHandler:
    """Handler for spectator WebSocket connections"""

    def __init__(self, websocket: WebSocket, game_id: str):
        self.websocket = websocket
        self.game_id = game_id
        self.connection_id: Optional[str] = None

    async def handle_connection(self):
        """Handle the WebSocket connection lifecycle"""
        await self.websocket.accept()

        try:
            # Check if game exists
            if hasattr(gateway, 'game_sessions') and self.game_id not in gateway.game_sessions:
                await self._send_error("Game not found")
                return

            # Add to connection manager
            self.connection_id = await connection_manager.add_connection(
                self.websocket, "spectator", self.game_id
            )

            # Send welcome message
            await self._send_welcome()

            # Keep connection alive and handle messages
            while True:
                try:
                    # Spectators mainly receive updates, minimal message handling
                    data = await self.websocket.receive_text()
                    message = json.loads(data)

                    # Handle basic messages like ping
                    if message.get("type") == "ping":
                        await self._handle_ping()

                except WebSocketDisconnect:
                    logger.info(f"Spectator disconnected from game {self.game_id}")
                    break

                except json.JSONDecodeError:
                    await self._send_error("Invalid JSON format")

                except Exception as e:
                    logger.error(f"Error handling spectator message: {e}")

        except Exception as e:
            logger.error(f"Error in spectator connection for game {self.game_id}: {e}")

        finally:
            # Cleanup
            if self.connection_id:
                await connection_manager.handle_disconnect(self.connection_id)

    async def _send_welcome(self):
        """Send welcome message to spectator"""
        welcome = {
            "type": "spectator_welcome",
            "game_id": self.game_id,
            "message": f"Connected as spectator to game {self.game_id}"
        }
        await self.websocket.send_text(json.dumps(welcome))

    async def _handle_ping(self):
        """Handle ping from spectator"""
        pong = {
            "type": "pong",
            "game_id": self.game_id,
            "timestamp": time.time()
        }
        await self.websocket.send_text(json.dumps(pong))

    async def _send_error(self, error_message: str):
        """Send error message"""
        error = {
            "type": "error",
            "game_id": self.game_id,
            "message": error_message
        }
        await self.websocket.send_text(json.dumps(error))


# WebSocket route registration
def register_websocket_routes(app: FastAPI):
    """Register WebSocket routes with the FastAPI app"""

    @app.websocket("/ws/agent/{agent_id}")
    async def agent_websocket_endpoint(websocket: WebSocket, agent_id: str):
        """WebSocket endpoint for LLM agents"""
        handler = AgentWebSocketHandler(websocket, agent_id)
        await handler.handle_connection()

    @app.websocket("/ws/spectator/{game_id}")
    async def spectator_websocket_endpoint(websocket: WebSocket, game_id: str):
        """WebSocket endpoint for game spectators"""
        if not settings.enable_spectator_mode:
            await websocket.close(code=1008, reason="Spectator mode disabled")
            return

        handler = SpectatorWebSocketHandler(websocket, game_id)
        await handler.handle_connection()

    logger.info("WebSocket routes registered")