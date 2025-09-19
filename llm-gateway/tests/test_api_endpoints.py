#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for LLM API Gateway REST API endpoints
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

# Import the modules we're testing (these don't exist yet - TDD!)
try:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from main import app
    from api_endpoints import GameConfig, FreeCivAction
except ImportError:
    # Will fail initially until we implement the module
    app = None
    GameConfig = None
    FreeCivAction = None


class TestGameCreationEndpoint:
    """Test /api/game/create endpoint"""

    def test_create_game_success(self):
        """Test successful game creation"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        client = TestClient(app)

        game_config = {
            "ruleset": "classic",
            "map_size": "small",
            "max_players": 4,
            "ai_level": "easy"
        }

        with patch('main.gateway.create_game') as mock_create:
            mock_create.return_value = {
                "success": True,
                "game_id": "game-123",
                "connection_details": {
                    "ws_url": "ws://localhost:8002/llmsocket/8002",
                    "game_port": 6001
                }
            }

            response = client.post("/api/game/create", json=game_config)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["game_id"] == "game-123"
        assert "connection_details" in data

    def test_create_game_invalid_config(self):
        """Test game creation with invalid configuration"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        client = TestClient(app)

        invalid_config = {
            "ruleset": "invalid_ruleset",
            "map_size": "huge_invalid",
            "max_players": 20  # Too many players
        }

        response = client.post("/api/game/create", json=invalid_config)

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_create_game_server_error(self):
        """Test game creation when server has issues"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        client = TestClient(app)

        game_config = {
            "ruleset": "classic",
            "map_size": "small"
        }

        with patch('main.gateway.create_game') as mock_create:
            mock_create.side_effect = Exception("FreeCiv server unavailable")

            response = client.post("/api/game/create", json=game_config)

        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "server unavailable" in data["error"].lower()

    def test_create_game_capacity_exceeded(self):
        """Test game creation when capacity is exceeded"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        client = TestClient(app)

        game_config = {
            "ruleset": "classic",
            "map_size": "small"
        }

        with patch('main.gateway.create_game') as mock_create:
            mock_create.return_value = {
                "success": False,
                "error": "Maximum concurrent games (10) exceeded"
            }

            response = client.post("/api/game/create", json=game_config)

        assert response.status_code == 503
        data = response.json()
        assert "capacity" in data["error"].lower()


class TestGameStateEndpoint:
    """Test /api/game/{game_id}/state endpoint"""

    def test_get_game_state_success(self):
        """Test successful game state retrieval"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        client = TestClient(app)

        with patch('main.gateway.get_game_state') as mock_get_state:
            mock_get_state.return_value = {
                "success": True,
                "format": "llm_optimized",
                "data": {
                    "turn": 1,
                    "phase": "movement",
                    "strategic_summary": {
                        "cities_count": 1,
                        "units_count": 2,
                        "tech_progress": "early"
                    }
                }
            }

            response = client.get("/api/game/game-123/state?player_id=1&format=llm_optimized")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["format"] == "llm_optimized"
        assert "strategic_summary" in data["data"]

    def test_get_game_state_full_format(self):
        """Test getting full format game state"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        client = TestClient(app)

        with patch('main.gateway.get_game_state') as mock_get_state:
            mock_get_state.return_value = {
                "success": True,
                "format": "full",
                "data": {
                    "turn": 1,
                    "units": [{"id": 1, "type": "warrior", "x": 10, "y": 20}],
                    "cities": [{"id": 1, "name": "Capital", "population": 3}],
                    "players": {"1": {"name": "Player 1", "score": 100}}
                }
            }

            response = client.get("/api/game/game-123/state?player_id=1&format=full")

        assert response.status_code == 200
        data = response.json()
        assert data["format"] == "full"
        assert "units" in data["data"]
        assert "cities" in data["data"]

    def test_get_game_state_missing_player_id(self):
        """Test game state request without player_id"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        client = TestClient(app)

        response = client.get("/api/game/game-123/state")

        assert response.status_code == 400
        data = response.json()
        assert "player_id" in data["error"].lower()

    def test_get_game_state_invalid_game_id(self):
        """Test game state request for non-existent game"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        client = TestClient(app)

        with patch('main.gateway.get_game_state') as mock_get_state:
            mock_get_state.return_value = {
                "success": False,
                "error": "Game not found: invalid-game"
            }

            response = client.get("/api/game/invalid-game/state?player_id=1")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["error"].lower()

    def test_get_game_state_unauthorized_player(self):
        """Test game state request for unauthorized player"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        client = TestClient(app)

        with patch('main.gateway.get_game_state') as mock_get_state:
            mock_get_state.return_value = {
                "success": False,
                "error": "Player 999 not authorized for game game-123"
            }

            response = client.get("/api/game/game-123/state?player_id=999")

        assert response.status_code == 403
        data = response.json()
        assert "not authorized" in data["error"].lower()

    def test_get_game_state_with_fog_of_war(self):
        """Test game state with fog of war applied"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        client = TestClient(app)

        with patch('main.gateway.get_game_state') as mock_get_state:
            mock_get_state.return_value = {
                "success": True,
                "format": "full",
                "data": {
                    "turn": 1,
                    "visible_tiles": [{"x": 10, "y": 20, "terrain": "grassland"}],
                    "fog_of_war_applied": True,
                    "player_perspective": 1
                }
            }

            response = client.get("/api/game/game-123/state?player_id=1&format=full")

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["fog_of_war_applied"] is True
        assert data["data"]["player_perspective"] == 1


class TestActionSubmissionEndpoint:
    """Test /api/game/{game_id}/action endpoint"""

    def test_submit_action_success(self):
        """Test successful action submission"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        client = TestClient(app)

        action = {
            "action_type": "unit_move",
            "actor_id": 42,
            "target": {"x": 11, "y": 21},
            "player_id": 1
        }

        with patch('main.gateway.submit_action') as mock_submit:
            mock_submit.return_value = {
                "success": True,
                "action_id": "action-456",
                "result": "Action executed successfully"
            }

            response = client.post("/api/game/game-123/action", json=action)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "action_id" in data

    def test_submit_action_invalid_type(self):
        """Test submitting action with invalid type"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        client = TestClient(app)

        invalid_action = {
            "action_type": "invalid_action_type",
            "actor_id": 42,
            "target": {"x": 11, "y": 21}
        }

        response = client.post("/api/game/game-123/action", json=invalid_action)

        assert response.status_code == 400
        data = response.json()
        assert "invalid action type" in data["error"].lower()

    def test_submit_action_missing_fields(self):
        """Test submitting action with missing required fields"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        client = TestClient(app)

        incomplete_action = {
            "action_type": "unit_move"
            # Missing actor_id and target
        }

        response = client.post("/api/game/game-123/action", json=incomplete_action)

        assert response.status_code == 400
        data = response.json()
        assert "required field" in data["error"].lower()

    def test_submit_action_validation_failure(self):
        """Test action submission that fails validation"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        client = TestClient(app)

        action = {
            "action_type": "unit_move",
            "actor_id": 999,  # Non-existent unit
            "target": {"x": 11, "y": 21},
            "player_id": 1
        }

        with patch('main.gateway.submit_action') as mock_submit:
            mock_submit.return_value = {
                "success": False,
                "error": "Unit 999 does not exist or is not owned by player 1"
            }

            response = client.post("/api/game/game-123/action", json=action)

        assert response.status_code == 400
        data = response.json()
        assert "unit 999" in data["error"].lower()

    def test_submit_action_game_not_found(self):
        """Test action submission for non-existent game"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        client = TestClient(app)

        action = {
            "action_type": "unit_move",
            "actor_id": 42,
            "target": {"x": 11, "y": 21},
            "player_id": 1
        }

        with patch('main.gateway.submit_action') as mock_submit:
            mock_submit.return_value = {
                "success": False,
                "error": "Game not found: invalid-game"
            }

            response = client.post("/api/game/invalid-game/action", json=action)

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["error"].lower()

    def test_submit_multiple_actions_batch(self):
        """Test submitting multiple actions in a batch"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        client = TestClient(app)

        actions = [
            {
                "action_type": "unit_move",
                "actor_id": 42,
                "target": {"x": 11, "y": 21},
                "player_id": 1
            },
            {
                "action_type": "city_production",
                "actor_id": 1,
                "target": "warrior",
                "player_id": 1
            }
        ]

        with patch('main.gateway.submit_actions_batch') as mock_submit_batch:
            mock_submit_batch.return_value = {
                "success": True,
                "results": [
                    {"action_id": "action-1", "success": True},
                    {"action_id": "action-2", "success": True}
                ]
            }

            response = client.post("/api/game/game-123/actions", json={"actions": actions})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) == 2


class TestHealthEndpoint:
    """Test /health endpoint"""

    def test_health_check_healthy(self):
        """Test health check when everything is healthy"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        client = TestClient(app)

        with patch('main.gateway.get_health_status') as mock_health:
            mock_health.return_value = {
                "status": "healthy",
                "active_games": 3,
                "active_agents": 5,
                "proxy_connections": 2,
                "uptime": 3600
            }

            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "active_games" in data
        assert "uptime" in data

    def test_health_check_degraded(self):
        """Test health check when service is degraded"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        client = TestClient(app)

        with patch('main.gateway.get_health_status') as mock_health:
            mock_health.return_value = {
                "status": "degraded",
                "active_games": 10,  # At capacity
                "active_agents": 15,
                "proxy_connections": 1,  # Some connections down
                "issues": ["High load", "FreeCiv proxy connection unstable"]
            }

            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert "issues" in data

    def test_health_check_unhealthy(self):
        """Test health check when service is unhealthy"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        client = TestClient(app)

        with patch('main.gateway.get_health_status') as mock_health:
            mock_health.return_value = {
                "status": "unhealthy",
                "error": "Cannot connect to FreeCiv proxy server"
            }

            response = client.get("/health")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert "error" in data


class TestAuthenticationAndAuthorization:
    """Test API authentication and authorization"""

    def test_api_key_authentication(self):
        """Test API key authentication"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        client = TestClient(app)

        headers = {"Authorization": "Bearer valid-api-key"}

        with patch('main.authenticate_api_key') as mock_auth:
            mock_auth.return_value = {"valid": True, "agent_id": "test-agent"}

            response = client.get("/api/game/game-123/state?player_id=1", headers=headers)

        # Should not get authentication error
        assert response.status_code != 401

    def test_api_key_authentication_invalid(self):
        """Test API key authentication with invalid key"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        client = TestClient(app)

        headers = {"Authorization": "Bearer invalid-api-key"}

        with patch('main.authenticate_api_key') as mock_auth:
            mock_auth.return_value = {"valid": False, "error": "Invalid API key"}

            response = client.get("/api/game/game-123/state?player_id=1", headers=headers)

        assert response.status_code == 401
        data = response.json()
        assert "invalid" in data["error"].lower()

    def test_rate_limiting(self):
        """Test API rate limiting"""
        if app is None:
            pytest.skip("FastAPI app not implemented yet")

        client = TestClient(app)

        headers = {"Authorization": "Bearer valid-api-key"}

        with patch('main.check_rate_limit') as mock_rate_limit:
            # First request should pass
            mock_rate_limit.return_value = {"allowed": True}
            response1 = client.get("/api/game/game-123/state?player_id=1", headers=headers)
            assert response1.status_code != 429

            # Subsequent request should be rate limited
            mock_rate_limit.return_value = {"allowed": False, "retry_after": 60}
            response2 = client.get("/api/game/game-123/state?player_id=1", headers=headers)
            assert response2.status_code == 429
            assert "retry_after" in response2.headers