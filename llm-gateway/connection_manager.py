#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Connection Manager for LLM API Gateway
Handles WebSocket connections, heartbeats, and cleanup
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, Optional, Set, List
from fastapi import WebSocket, WebSocketDisconnect
try:
    from .config import settings
except ImportError:
    from config import settings

logger = logging.getLogger("llm-gateway")


class ConnectionInfo:
    """Information about a WebSocket connection"""

    def __init__(self, websocket: WebSocket, connection_type: str, identifier: str):
        self.websocket = websocket
        self.connection_type = connection_type  # "agent" or "spectator"
        self.identifier = identifier  # agent_id or game_id
        self.connection_id = str(uuid.uuid4())
        self.connected_at = time.time()
        self.last_seen = time.time()
        self.authenticated = False
        self.metadata: Dict[str, Any] = {}

    def update_activity(self):
        """Update last seen timestamp"""
        self.last_seen = time.time()

    def is_expired(self, timeout: int) -> bool:
        """Check if connection has expired"""
        return (time.time() - self.last_seen) > timeout

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/monitoring"""
        return {
            "connection_id": self.connection_id,
            "type": self.connection_type,
            "identifier": self.identifier,
            "connected_at": self.connected_at,
            "last_seen": self.last_seen,
            "authenticated": self.authenticated,
            "duration": time.time() - self.connected_at
        }


class ConnectionManager:
    """Manages WebSocket connections with heartbeat and cleanup"""

    def __init__(self):
        self.connections: Dict[str, ConnectionInfo] = {}
        self.agent_connections: Dict[str, Set[str]] = {}  # agent_id -> set of connection_ids
        self.spectator_connections: Dict[str, Set[str]] = {}  # game_id -> set of connection_ids
        self.heartbeat_interval = settings.heartbeat_interval
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """Start the connection manager"""
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("Connection manager started")

    async def stop(self):
        """Stop the connection manager"""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        await self._close_all_connections()
        logger.info("Connection manager stopped")

    async def add_connection(self, websocket: WebSocket, connection_type: str, identifier: str) -> str:
        """Add a new WebSocket connection"""
        connection_info = ConnectionInfo(websocket, connection_type, identifier)
        connection_id = connection_info.connection_id

        # Check connection limits
        if connection_type == "agent":
            if identifier in self.agent_connections:
                if len(self.agent_connections[identifier]) >= settings.max_connections_per_agent:
                    raise ValueError(f"Too many connections for agent {identifier}")
            else:
                self.agent_connections[identifier] = set()

            self.agent_connections[identifier].add(connection_id)

        elif connection_type == "spectator":
            if identifier not in self.spectator_connections:
                self.spectator_connections[identifier] = set()
            self.spectator_connections[identifier].add(connection_id)

        self.connections[connection_id] = connection_info

        logger.info(f"Added {connection_type} connection {connection_id} for {identifier}")
        return connection_id

    async def remove_connection(self, connection_id: str):
        """Remove a WebSocket connection"""
        if connection_id not in self.connections:
            return

        connection_info = self.connections[connection_id]
        identifier = connection_info.identifier
        connection_type = connection_info.connection_type

        # Remove from type-specific tracking
        if connection_type == "agent" and identifier in self.agent_connections:
            self.agent_connections[identifier].discard(connection_id)
            if not self.agent_connections[identifier]:
                del self.agent_connections[identifier]

        elif connection_type == "spectator" and identifier in self.spectator_connections:
            self.spectator_connections[identifier].discard(connection_id)
            if not self.spectator_connections[identifier]:
                del self.spectator_connections[identifier]

        # Close WebSocket if still open
        try:
            await connection_info.websocket.close()
        except Exception as e:
            logger.warning(f"Error closing WebSocket {connection_id}: {e}")

        del self.connections[connection_id]
        logger.info(f"Removed {connection_type} connection {connection_id} for {identifier}")

    async def handle_disconnect(self, connection_id: str):
        """Handle graceful disconnection"""
        if connection_id in self.connections:
            connection_info = self.connections[connection_id]
            logger.info(f"Handling disconnect for {connection_info.connection_type} {connection_info.identifier}")

            # Perform cleanup based on connection type
            if connection_info.connection_type == "agent":
                await self._cleanup_agent_disconnect(connection_info)
            elif connection_info.connection_type == "spectator":
                await self._cleanup_spectator_disconnect(connection_info)

            await self.remove_connection(connection_id)

    async def maintain_connections(self):
        """Perform connection maintenance (heartbeat, cleanup)"""
        expired_connections = []

        for connection_id, connection_info in self.connections.items():
            try:
                # Check if connection has expired
                if connection_info.is_expired(settings.agent_timeout):
                    expired_connections.append(connection_id)
                    continue

                # Send heartbeat ping
                if connection_info.authenticated:
                    await self._send_heartbeat(connection_info)

            except Exception as e:
                logger.warning(f"Error maintaining connection {connection_id}: {e}")
                expired_connections.append(connection_id)

        # Remove expired connections
        for connection_id in expired_connections:
            await self.handle_disconnect(connection_id)

    async def get_agent_connections(self, agent_id: str) -> List[ConnectionInfo]:
        """Get all connections for a specific agent"""
        if agent_id not in self.agent_connections:
            return []

        connections = []
        for connection_id in self.agent_connections[agent_id]:
            if connection_id in self.connections:
                connections.append(self.connections[connection_id])

        return connections

    async def get_spectator_connections(self, game_id: str) -> List[ConnectionInfo]:
        """Get all spectator connections for a game"""
        if game_id not in self.spectator_connections:
            return []

        connections = []
        for connection_id in self.spectator_connections[game_id]:
            if connection_id in self.connections:
                connections.append(self.connections[connection_id])

        return connections

    async def broadcast_to_spectators(self, game_id: str, message: Dict[str, Any]):
        """Broadcast message to all spectators of a game"""
        spectator_connections = await self.get_spectator_connections(game_id)

        for connection_info in spectator_connections:
            try:
                await connection_info.websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Error broadcasting to spectator {connection_info.connection_id}: {e}")
                # Schedule for removal
                asyncio.create_task(self.handle_disconnect(connection_info.connection_id))

    async def send_to_agent(self, agent_id: str, message: Dict[str, Any]) -> bool:
        """Send message to a specific agent (first available connection)"""
        agent_connections = await self.get_agent_connections(agent_id)

        if not agent_connections:
            return False

        # Use the first authenticated connection
        for connection_info in agent_connections:
            if connection_info.authenticated:
                try:
                    await connection_info.websocket.send_text(json.dumps(message))
                    return True
                except Exception as e:
                    logger.warning(f"Error sending to agent {agent_id}: {e}")
                    asyncio.create_task(self.handle_disconnect(connection_info.connection_id))

        return False

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        total_connections = len(self.connections)
        agent_count = len(self.agent_connections)
        spectator_count = sum(len(connections) for connections in self.spectator_connections.values())

        return {
            "total_connections": total_connections,
            "agent_connections": agent_count,
            "spectator_connections": spectator_count,
            "active_games": len(self.spectator_connections),
            "connections_by_type": {
                "agent": agent_count,
                "spectator": spectator_count
            }
        }

    async def _heartbeat_loop(self):
        """Background task for connection maintenance"""
        while self._running:
            try:
                await self.maintain_connections()
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying

    async def _send_heartbeat(self, connection_info: ConnectionInfo):
        """Send heartbeat ping to a connection"""
        try:
            ping_message = {
                "type": "ping",
                "timestamp": time.time(),
                "connection_id": connection_info.connection_id
            }
            await connection_info.websocket.send_text(json.dumps(ping_message))
            connection_info.update_activity()

        except Exception as e:
            logger.warning(f"Heartbeat failed for {connection_info.connection_id}: {e}")
            raise

    async def _cleanup_agent_disconnect(self, connection_info: ConnectionInfo):
        """Cleanup when an agent disconnects"""
        agent_id = connection_info.identifier

        # Log agent disconnection
        logger.info(f"Agent {agent_id} disconnected after {time.time() - connection_info.connected_at:.1f}s")

        # Additional agent-specific cleanup can be added here
        # e.g., save game state, notify other players, etc.

    async def _cleanup_spectator_disconnect(self, connection_info: ConnectionInfo):
        """Cleanup when a spectator disconnects"""
        game_id = connection_info.identifier

        # Log spectator disconnection
        logger.info(f"Spectator disconnected from game {game_id}")

        # Minimal cleanup needed for spectators

    async def _close_all_connections(self):
        """Close all connections during shutdown"""
        connection_ids = list(self.connections.keys())

        for connection_id in connection_ids:
            try:
                await self.handle_disconnect(connection_id)
            except Exception as e:
                logger.warning(f"Error closing connection {connection_id}: {e}")


# Global connection manager instance
connection_manager = ConnectionManager()