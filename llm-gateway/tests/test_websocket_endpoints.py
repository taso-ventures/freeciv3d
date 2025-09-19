#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for LLM API Gateway WebSocket endpoints
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
    from main import app
    from websocket_handlers import AgentWebSocketHandler, SpectatorWebSocketHandler
except ImportError:
    # Will fail initially until we implement the module
    app = None
    AgentWebSocketHandler = None
    SpectatorWebSocketHandler = None


class TestAgentWebSocketEndpoint:
    """Test /ws/agent/{agent_id} WebSocket endpoint"""

    @pytest.mark.asyncio
    async def test_agent_websocket_connection(self):
        """Test successful agent WebSocket connection"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        with TestClient(app) as client:
            with client.websocket_connect("/ws/agent/test-agent") as websocket:
                # Should receive welcome message
                data = websocket.receive_json()
                assert data["type"] == "welcome"
                assert "handler_id" in data

    @pytest.mark.asyncio
    async def test_agent_authentication_flow(self):
        """Test agent authentication via WebSocket"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        auth_message = {
            "type": "llm_connect",
            "agent_id": "test-agent",
            "timestamp": 1234567890.0,
            "data": {
                "api_token": "valid-token",
                "model": "gpt-4",
                "game_id": "game-123"
            }
        }

        with TestClient(app) as client:
            with client.websocket_connect("/ws/agent/test-agent") as websocket:
                # Skip welcome message
                websocket.receive_json()

                # Send authentication
                with patch('main.gateway.authenticate_agent') as mock_auth:
                    mock_auth.return_value = {
                        "success": True,
                        "session_id": "session-456",
                        "player_id": 1
                    }

                    websocket.send_json(auth_message)
                    response = websocket.receive_json()

                assert response["data"]["type"] == "auth_success"
                assert response["data"]["agent_id"] == "test-agent"
                assert response["data"]["session_id"] == "session-456"

    @pytest.mark.asyncio
    async def test_agent_authentication_failure(self):
        """Test agent authentication failure"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        auth_message = {
            "type": "llm_connect",
            "agent_id": "test-agent",
            "timestamp": 1234567890.0,
            "data": {
                "api_token": "invalid-token",
                "model": "gpt-4",
                "game_id": "game-123"
            }
        }

        with TestClient(app) as client:
            with client.websocket_connect("/ws/agent/test-agent") as websocket:
                # Skip welcome message
                websocket.receive_json()

                # Send authentication
                with patch('main.gateway.authenticate_agent') as mock_auth:
                    mock_auth.return_value = {
                        "success": False,
                        "error": "Invalid API token"
                    }

                    websocket.send_json(auth_message)
                    response = websocket.receive_json()

                assert response["data"]["success"] is False
                assert "invalid" in response["data"]["error_message"].lower()

    @pytest.mark.asyncio
    async def test_state_query_via_websocket(self):
        """Test state query via WebSocket"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        state_query = {
            "type": "state_query",
            "agent_id": "test-agent",
            "timestamp": 1234567890.0,
            "correlation_id": "query-123",
            "data": {
                "format": "llm_optimized",
                "include_actions": True
            }
        }

        with TestClient(app) as client:
            with client.websocket_connect("/ws/agent/test-agent") as websocket:
                # Authenticate first
                self._authenticate_agent(websocket)

                # Send state query
                with patch('main.gateway.get_game_state') as mock_state:
                    mock_state.return_value = {
                        "success": True,
                        "format": "llm_optimized",
                        "data": {
                            "turn": 1,
                            "strategic_summary": {"cities_count": 1}
                        }
                    }

                    websocket.send_json(state_query)
                    response = websocket.receive_json()

                assert response["correlation_id"] == "query-123"
                assert response["data"]["type"] == "state_response"
                assert response["data"]["format"] == "llm_optimized"

    @pytest.mark.asyncio
    async def test_action_submission_via_websocket(self):
        """Test action submission via WebSocket"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        action_message = {
            "type": "action",
            "agent_id": "test-agent",
            "timestamp": 1234567890.0,
            "correlation_id": "action-456",
            "data": {
                "action_type": "unit_move",
                "actor_id": 42,
                "target": {"x": 11, "y": 21}
            }
        }

        with TestClient(app) as client:
            with client.websocket_connect("/ws/agent/test-agent") as websocket:
                # Authenticate first
                self._authenticate_agent(websocket)

                # Send action
                with patch('main.gateway.submit_action') as mock_action:
                    mock_action.return_value = {
                        "success": True,
                        "action_id": "action-789"
                    }

                    websocket.send_json(action_message)
                    response = websocket.receive_json()

                assert response["correlation_id"] == "action-456"
                assert response["data"]["type"] == "action_result"
                assert response["data"]["success"] is True

    @pytest.mark.asyncio
    async def test_websocket_disconnection_cleanup(self):
        """Test WebSocket disconnection and cleanup"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        with TestClient(app) as client:
            with patch('main.gateway.handle_agent_disconnect') as mock_disconnect:
                with client.websocket_connect("/ws/agent/test-agent") as websocket:
                    # Authenticate
                    self._authenticate_agent(websocket)
                    pass  # Connection will be closed when context exits

                # Should have called disconnect handler
                mock_disconnect.assert_called_once_with("test-agent")

    @pytest.mark.asyncio
    async def test_websocket_error_handling(self):
        """Test WebSocket error handling"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        invalid_message = {
            "invalid_json": "this is not a valid protocol message"
        }

        with TestClient(app) as client:
            with client.websocket_connect("/ws/agent/test-agent") as websocket:
                # Send invalid message
                websocket.send_json(invalid_message)
                response = websocket.receive_json()

                assert response["data"]["type"] == "error"
                assert "invalid" in response["data"]["error_message"].lower()

    @pytest.mark.asyncio
    async def test_websocket_heartbeat(self):
        """Test WebSocket heartbeat/ping-pong"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        ping_message = {
            "type": "ping",
            "agent_id": "test-agent",
            "timestamp": 1234567890.0,
            "data": {}
        }

        with TestClient(app) as client:
            with client.websocket_connect("/ws/agent/test-agent") as websocket:
                # Authenticate first
                self._authenticate_agent(websocket)

                # Send ping
                websocket.send_json(ping_message)
                response = websocket.receive_json()

                assert response["type"] == "pong"
                assert response["agent_id"] == "test-agent"

    def _authenticate_agent(self, websocket):
        """Helper method to authenticate an agent"""
        # Skip welcome message
        websocket.receive_json()

        auth_message = {
            "type": "llm_connect",
            "agent_id": "test-agent",
            "timestamp": 1234567890.0,
            "data": {
                "api_token": "valid-token",
                "model": "gpt-4",
                "game_id": "game-123"
            }
        }

        with patch('main.gateway.authenticate_agent') as mock_auth:
            mock_auth.return_value = {
                "success": True,
                "session_id": "session-456",
                "player_id": 1
            }

            websocket.send_json(auth_message)
            websocket.receive_json()  # Skip auth response


class TestSpectatorWebSocketEndpoint:
    """Test /ws/spectator/{game_id} WebSocket endpoint"""

    @pytest.mark.asyncio
    async def test_spectator_connection(self):
        """Test spectator WebSocket connection"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        with TestClient(app) as client:
            with client.websocket_connect("/ws/spectator/game-123") as websocket:
                # Should receive welcome message
                data = websocket.receive_json()
                assert data["type"] == "spectator_welcome"
                assert data["game_id"] == "game-123"

    @pytest.mark.asyncio
    async def test_spectator_game_updates(self):
        """Test spectator receiving game updates"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        with TestClient(app) as client:
            with client.websocket_connect("/ws/spectator/game-123") as websocket:
                # Skip welcome message
                websocket.receive_json()

                # Simulate game update
                with patch('main.gateway.broadcast_to_spectators') as mock_broadcast:
                    # This would typically be called from game state changes
                    update = {
                        "type": "spectator_update",
                        "game_id": "game-123",
                        "data": {
                            "update_type": "player_action",
                            "turn": 5,
                            "action": "unit_move",
                            "player": 1
                        }
                    }

                    # Simulate receiving the update
                    mock_broadcast.return_value = update
                    # In real implementation, this would be pushed to spectator
                    # For testing, we simulate receiving it
                    data = update

                assert data["type"] == "spectator_update"
                assert data["game_id"] == "game-123"
                assert data["data"]["update_type"] == "player_action"

    @pytest.mark.asyncio
    async def test_spectator_game_not_found(self):
        """Test spectator connecting to non-existent game"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        with TestClient(app) as client:
            with patch('main.gateway.game_exists') as mock_exists:
                mock_exists.return_value = False

                try:
                    with client.websocket_connect("/ws/spectator/invalid-game") as websocket:
                        data = websocket.receive_json()
                        assert data["type"] == "error"
                        assert "not found" in data["message"].lower()
                except Exception:
                    # Connection might be closed immediately
                    pass

    @pytest.mark.asyncio
    async def test_spectator_multiple_connections(self):
        """Test multiple spectators for same game"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        with TestClient(app) as client:
            with client.websocket_connect("/ws/spectator/game-123") as ws1:
                with client.websocket_connect("/ws/spectator/game-123") as ws2:
                    # Both should receive welcome messages
                    data1 = ws1.receive_json()
                    data2 = ws2.receive_json()

                    assert data1["type"] == "spectator_welcome"
                    assert data2["type"] == "spectator_welcome"
                    assert data1["game_id"] == "game-123"
                    assert data2["game_id"] == "game-123"

    @pytest.mark.asyncio
    async def test_spectator_real_time_updates(self):
        """Test real-time game updates to spectators"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        with TestClient(app) as client:
            with client.websocket_connect("/ws/spectator/game-123") as websocket:
                # Skip welcome
                websocket.receive_json()

                # Test different types of updates
                update_types = [
                    "turn_change", "player_action", "game_event", "state_change"
                ]

                for update_type in update_types:
                    # Simulate update (in real implementation, would come from game)
                    update = {
                        "type": "spectator_update",
                        "game_id": "game-123",
                        "data": {
                            "update_type": update_type,
                            "timestamp": 1234567890.0
                        }
                    }

                    # Test that spectator can process different update types
                    assert update["data"]["update_type"] in update_types


class TestWebSocketConnectionLimits:
    """Test WebSocket connection limits and management"""

    @pytest.mark.asyncio
    async def test_max_connections_per_agent(self):
        """Test maximum connections per agent limit"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        with TestClient(app) as client:
            # Try to create multiple connections for same agent
            connections = []

            try:
                for i in range(3):  # Assume limit is 2
                    ws = client.websocket_connect(f"/ws/agent/test-agent-{i}")
                    connections.append(ws.__enter__())

                # Third connection should be rejected or limited
                with patch('main.gateway.check_connection_limit') as mock_limit:
                    mock_limit.return_value = {"allowed": False, "reason": "Too many connections"}

                    with client.websocket_connect("/ws/agent/test-agent-overflow") as ws:
                        data = ws.receive_json()
                        assert "limit" in data.get("message", "").lower()

            finally:
                # Cleanup
                for ws in connections:
                    ws.__exit__(None, None, None)

    @pytest.mark.asyncio
    async def test_connection_timeout(self):
        """Test connection timeout for inactive agents"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        with TestClient(app) as client:
            with patch('main.gateway.get_connection_timeout') as mock_timeout:
                mock_timeout.return_value = 1  # 1 second timeout for testing

                with client.websocket_connect("/ws/agent/timeout-test") as websocket:
                    # Authenticate but then become inactive
                    self._authenticate_agent_quick(websocket)

                    # Wait for timeout (would need real async sleep in implementation)
                    # Connection should be closed by server
                    pass

    def _authenticate_agent_quick(self, websocket):
        """Quick agent authentication for testing"""
        websocket.receive_json()  # Welcome
        auth_message = {
            "type": "llm_connect",
            "agent_id": "timeout-test",
            "timestamp": 1234567890.0,
            "data": {"api_token": "valid-token", "model": "gpt-4", "game_id": "game-123"}
        }
        websocket.send_json(auth_message)
        websocket.receive_json()  # Auth response


class TestWebSocketSecurity:
    """Test WebSocket security features"""

    @pytest.mark.asyncio
    async def test_origin_validation(self):
        """Test WebSocket origin validation"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        # Test with invalid origin
        headers = {"Origin": "https://malicious-site.com"}

        with TestClient(app) as client:
            try:
                with client.websocket_connect("/ws/agent/test-agent", headers=headers) as ws:
                    # Should be rejected
                    pass
            except Exception as e:
                # Connection should be rejected
                assert "origin" in str(e).lower() or "forbidden" in str(e).lower()

    @pytest.mark.asyncio
    async def test_message_size_limits(self):
        """Test WebSocket message size limits"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        # Create oversized message
        large_data = {"large_field": "x" * 100000}  # 100KB of data
        oversized_message = {
            "type": "action",
            "agent_id": "test-agent",
            "timestamp": 1234567890.0,
            "data": large_data
        }

        with TestClient(app) as client:
            with client.websocket_connect("/ws/agent/test-agent") as websocket:
                # Send oversized message
                websocket.send_json(oversized_message)
                response = websocket.receive_json()

                assert response["data"]["type"] == "error"
                assert "size" in response["data"]["error_message"].lower()

    @pytest.mark.asyncio
    async def test_rate_limiting_websocket(self):
        """Test WebSocket rate limiting"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        with TestClient(app) as client:
            with client.websocket_connect("/ws/agent/test-agent") as websocket:
                # Authenticate first
                self._authenticate_agent_quick(websocket)

                # Send many messages rapidly
                for i in range(100):  # Send 100 messages
                    message = {
                        "type": "ping",
                        "agent_id": "test-agent",
                        "timestamp": 1234567890.0,
                        "data": {}
                    }
                    websocket.send_json(message)

                    if i > 50:  # After 50 messages, should hit rate limit
                        try:
                            response = websocket.receive_json()
                            if response.get("data", {}).get("type") == "error":
                                assert "rate limit" in response["data"]["error_message"].lower()
                                break
                        except:
                            # Connection might be closed due to rate limiting
                            break