#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for LLM API Gateway main functionality
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

# Import the modules we're testing (these don't exist yet - TDD!)
try:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from main import app, LLMGateway
    from connection_manager import ConnectionManager
    from config import Settings
except ImportError:
    # Will fail initially until we implement the module
    app = None
    LLMGateway = None
    ConnectionManager = None
    Settings = None


class TestLLMGateway:
    """Test LLMGateway class"""

    def test_gateway_initialization(self):
        """Test LLMGateway initializes properly"""
        if LLMGateway is None:
            pytest.skip("LLMGateway not implemented yet")

        gateway = LLMGateway()

        assert hasattr(gateway, 'active_agents')
        assert hasattr(gateway, 'game_sessions')
        assert hasattr(gateway, 'proxy_connections')
        assert isinstance(gateway.active_agents, dict)
        assert isinstance(gateway.game_sessions, dict)
        assert isinstance(gateway.proxy_connections, dict)

    @pytest.mark.asyncio
    async def test_register_agent(self):
        """Test agent registration"""
        if LLMGateway is None:
            pytest.skip("LLMGateway not implemented yet")

        gateway = LLMGateway()
        agent_config = {
            "agent_id": "test-agent",
            "api_token": "test-token",
            "model": "gpt-4",
            "game_id": "game-123"
        }

        result = await gateway.register_agent("test-agent", agent_config)

        assert result["success"] is True
        assert "test-agent" in gateway.active_agents
        assert gateway.active_agents["test-agent"]["config"] == agent_config

    @pytest.mark.asyncio
    async def test_register_agent_duplicate(self):
        """Test registering duplicate agent fails gracefully"""
        if LLMGateway is None:
            pytest.skip("LLMGateway not implemented yet")

        gateway = LLMGateway()
        agent_config = {
            "agent_id": "test-agent",
            "api_token": "test-token",
            "model": "gpt-4",
            "game_id": "game-123"
        }

        # Register once
        await gateway.register_agent("test-agent", agent_config)

        # Try to register again
        result = await gateway.register_agent("test-agent", agent_config)

        assert result["success"] is False
        assert "already registered" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_route_message_game_arena_to_freeciv(self):
        """Test message routing from Game Arena to FreeCiv"""
        if LLMGateway is None:
            pytest.skip("LLMGateway not implemented yet")

        gateway = LLMGateway()

        # Mock FreeCiv proxy connection
        mock_proxy = AsyncMock()
        gateway.proxy_connections["game-123"] = mock_proxy

        message = {
            "type": "state_query",
            "agent_id": "test-agent",
            "data": {"format": "llm_optimized"}
        }

        result = await gateway.route_message("game_arena", "freeciv", message)

        assert result["success"] is True
        mock_proxy.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_message_no_connection(self):
        """Test message routing when no connection exists"""
        if LLMGateway is None:
            pytest.skip("LLMGateway not implemented yet")

        gateway = LLMGateway()

        message = {
            "type": "state_query",
            "agent_id": "test-agent",
            "data": {"format": "llm_optimized"}
        }

        result = await gateway.route_message("game_arena", "freeciv", message)

        assert result["success"] is False
        assert "no connection" in result["error"].lower()


class TestFastAPIApp:
    """Test FastAPI application setup"""

    def test_app_initialization(self):
        """Test that FastAPI app initializes"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        assert app is not None
        assert hasattr(app, 'routes')

    def test_cors_middleware(self):
        """Test CORS middleware is configured"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        # Check that CORS middleware is added
        middleware_types = [type(middleware.cls) for middleware in app.user_middleware]
        from fastapi.middleware.cors import CORSMiddleware
        assert CORSMiddleware in middleware_types

    def test_endpoints_registered(self):
        """Test that required endpoints are registered"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        routes = [route.path for route in app.routes]

        # Check required API endpoints
        assert "/api/game/create" in routes
        assert "/api/game/{game_id}/state" in routes
        assert "/api/game/{game_id}/action" in routes
        assert "/health" in routes

    def test_websocket_endpoints_registered(self):
        """Test that WebSocket endpoints are registered"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        routes = [route.path for route in app.routes]

        # Check WebSocket endpoints
        assert "/ws/agent/{agent_id}" in routes
        assert "/ws/spectator/{game_id}" in routes


class TestConnectionManager:
    """Test ConnectionManager class"""

    def test_connection_manager_initialization(self):
        """Test ConnectionManager initializes properly"""
        if ConnectionManager is None:
            pytest.skip("ConnectionManager not implemented yet")

        manager = ConnectionManager()

        assert hasattr(manager, 'connections')
        assert hasattr(manager, 'heartbeat_interval')
        assert isinstance(manager.connections, dict)
        assert manager.heartbeat_interval > 0

    @pytest.mark.asyncio
    async def test_add_connection(self):
        """Test adding a connection"""
        if ConnectionManager is None:
            pytest.skip("ConnectionManager not implemented yet")

        manager = ConnectionManager()
        mock_websocket = AsyncMock()

        connection_id = await manager.add_connection("test-agent", mock_websocket)

        assert connection_id is not None
        assert connection_id in manager.connections
        assert manager.connections[connection_id]["websocket"] == mock_websocket
        assert manager.connections[connection_id]["agent_id"] == "test-agent"

    @pytest.mark.asyncio
    async def test_remove_connection(self):
        """Test removing a connection"""
        if ConnectionManager is None:
            pytest.skip("ConnectionManager not implemented yet")

        manager = ConnectionManager()
        mock_websocket = AsyncMock()

        connection_id = await manager.add_connection("test-agent", mock_websocket)
        await manager.remove_connection(connection_id)

        assert connection_id not in manager.connections

    @pytest.mark.asyncio
    async def test_maintain_connections_heartbeat(self):
        """Test connection heartbeat maintenance"""
        if ConnectionManager is None:
            pytest.skip("ConnectionManager not implemented yet")

        manager = ConnectionManager()
        mock_websocket = AsyncMock()

        connection_id = await manager.add_connection("test-agent", mock_websocket)

        # Run one heartbeat cycle
        await manager.maintain_connections()

        # Should have sent ping
        mock_websocket.ping.assert_called()

    @pytest.mark.asyncio
    async def test_handle_disconnect_graceful(self):
        """Test graceful disconnection handling"""
        if ConnectionManager is None:
            pytest.skip("ConnectionManager not implemented yet")

        manager = ConnectionManager()
        mock_websocket = AsyncMock()

        connection_id = await manager.add_connection("test-agent", mock_websocket)

        await manager.handle_disconnect(connection_id)

        # Connection should be removed
        assert connection_id not in manager.connections

        # WebSocket should be closed gracefully
        mock_websocket.close.assert_called()


class TestSettings:
    """Test configuration settings"""

    def test_settings_initialization(self):
        """Test Settings class initializes with defaults"""
        if Settings is None:
            pytest.skip("Settings not implemented yet")

        settings = Settings()

        assert hasattr(settings, 'freeciv_proxy_host')
        assert hasattr(settings, 'freeciv_proxy_port')
        assert hasattr(settings, 'max_concurrent_games')
        assert hasattr(settings, 'agent_timeout')

        # Check default values
        assert settings.freeciv_proxy_host == "localhost"
        assert settings.freeciv_proxy_port == 8002
        assert settings.max_concurrent_games >= 1
        assert settings.agent_timeout > 0

    def test_settings_from_env(self):
        """Test Settings can be configured from environment"""
        if Settings is None:
            pytest.skip("Settings not implemented yet")

        with patch.dict('os.environ', {
            'FREECIV_PROXY_HOST': 'test-host',
            'FREECIV_PROXY_PORT': '9000',
            'MAX_CONCURRENT_GAMES': '20'
        }):
            settings = Settings()

            assert settings.freeciv_proxy_host == "test-host"
            assert settings.freeciv_proxy_port == 9000
            assert settings.max_concurrent_games == 20


class TestIntegrationWithFreeCivProxy:
    """Test integration with FreeCiv proxy"""

    @pytest.mark.asyncio
    async def test_proxy_connection_establishment(self):
        """Test establishing connection to FreeCiv proxy"""
        if LLMGateway is None:
            pytest.skip("LLMGateway not implemented yet")

        gateway = LLMGateway()

        with patch('websockets.connect') as mock_connect:
            mock_websocket = AsyncMock()
            mock_connect.return_value = mock_websocket

            result = await gateway.connect_to_freeciv_proxy("game-123")

            assert result["success"] is True
            assert "game-123" in gateway.proxy_connections
            mock_connect.assert_called_with("ws://localhost:8002/llmsocket/8002")

    @pytest.mark.asyncio
    async def test_proxy_connection_failure(self):
        """Test handling FreeCiv proxy connection failure"""
        if LLMGateway is None:
            pytest.skip("LLMGateway not implemented yet")

        gateway = LLMGateway()

        with patch('websockets.connect', side_effect=ConnectionError("Connection failed")):
            result = await gateway.connect_to_freeciv_proxy("game-123")

            assert result["success"] is False
            assert "connection failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_proxy_message_forwarding(self):
        """Test forwarding messages to FreeCiv proxy"""
        if LLMGateway is None:
            pytest.skip("LLMGateway not implemented yet")

        gateway = LLMGateway()

        # Setup mock proxy connection
        mock_proxy = AsyncMock()
        gateway.proxy_connections["game-123"] = mock_proxy

        message = {
            "type": "llm_connect",
            "agent_id": "test-agent",
            "data": {"api_token": "test-token"}
        }

        await gateway.forward_to_proxy("game-123", message)

        # Should forward message as JSON
        mock_proxy.send.assert_called_once()
        sent_data = mock_proxy.send.call_args[0][0]
        parsed_message = json.loads(sent_data)
        assert parsed_message["type"] == "llm_connect"
        assert parsed_message["agent_id"] == "test-agent"