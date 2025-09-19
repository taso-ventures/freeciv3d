#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
REST API endpoints for LLM Gateway
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends, Header, Request
from pydantic import BaseModel, Field, ValidationError

# Rate limiting imports
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    HAS_SLOWAPI = True
except ImportError:
    HAS_SLOWAPI = False

try:
    from .config import settings
except ImportError:
    from config import settings

# Gateway will be injected from main.py to avoid circular imports
gateway = None

def get_gateway():
    """Get the gateway instance (dependency injection)"""
    if gateway is None:
        raise HTTPException(status_code=500, detail="Gateway not initialized")
    return gateway

logger = logging.getLogger("llm-gateway")

# Create API router
router = APIRouter()

# Rate limiter setup
if HAS_SLOWAPI:
    try:
        from config import settings
        limiter = Limiter(key_func=get_remote_address, storage_uri=settings.redis_url)
    except:
        limiter = Limiter(key_func=get_remote_address)
else:
    limiter = None


# Pydantic models for request/response validation
class GameConfig(BaseModel):
    """Configuration for creating a new game"""
    ruleset: str = Field(default="classic", description="Game ruleset")
    map_size: str = Field(default="small", description="Map size")
    max_players: int = Field(default=4, ge=2, le=8, description="Maximum players")
    ai_level: str = Field(default="easy", description="AI difficulty level")
    turn_timeout: Optional[int] = Field(default=300, description="Turn timeout in seconds")

    class Config:
        schema_extra = {
            "example": {
                "ruleset": "classic",
                "map_size": "small",
                "max_players": 4,
                "ai_level": "easy",
                "turn_timeout": 300
            }
        }


class FreeCivAction(BaseModel):
    """FreeCiv game action"""
    action_type: str = Field(description="Type of action")
    actor_id: int = Field(description="ID of the acting unit/city")
    target: Any = Field(description="Action target (coordinates, unit ID, etc.)")
    player_id: int = Field(description="Player ID performing the action")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Additional parameters")

    class Config:
        schema_extra = {
            "example": {
                "action_type": "unit_move",
                "actor_id": 42,
                "target": {"x": 11, "y": 21},
                "player_id": 1,
                "parameters": {"validate": True}
            }
        }


class BatchActions(BaseModel):
    """Batch action submission"""
    actions: List[FreeCivAction] = Field(description="List of actions to execute")

    class Config:
        schema_extra = {
            "example": {
                "actions": [
                    {
                        "action_type": "unit_move",
                        "actor_id": 42,
                        "target": {"x": 11, "y": 21},
                        "player_id": 1
                    }
                ]
            }
        }


# Dependency for API key authentication
async def verify_api_key(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Verify API key from Authorization header"""
    if not settings.require_api_key:
        return {"valid": True, "agent_id": "anonymous"}

    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")

    api_key = authorization[7:]  # Remove "Bearer " prefix

    # Simple API key validation (extend with proper validation)
    if len(api_key) < 10:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return {"valid": True, "agent_id": f"agent-{api_key[:8]}"}


# Game management endpoints
@router.post("/game/create")
async def create_game(
    config: GameConfig,
    auth: Dict[str, Any] = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Create a new game session"""
    try:
        # Validate configuration
        if config.ruleset not in ["classic", "civ2civ3", "experimental"]:
            raise HTTPException(status_code=400, detail=f"Invalid ruleset: {config.ruleset}")

        if config.map_size not in ["tiny", "small", "medium", "large", "huge"]:
            raise HTTPException(status_code=400, detail=f"Invalid map size: {config.map_size}")

        # Create game via gateway
        if gateway is None:
            raise HTTPException(status_code=500, detail="Gateway not initialized")

        result = await gateway.create_game(config.dict())

        if not result["success"]:
            if "capacity" in result["error"].lower():
                raise HTTPException(status_code=503, detail=result["error"])
            else:
                raise HTTPException(status_code=500, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating game: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/game/{game_id}/state")
async def get_game_state(
    game_id: str,
    player_id: int = Query(description="Player ID for perspective"),
    format_type: str = Query(default="llm_optimized", alias="format", description="State format"),
    auth: Dict[str, Any] = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Get current game state"""
    try:
        # Validate format
        if format_type not in ["full", "delta", "llm_optimized"]:
            raise HTTPException(status_code=400, detail=f"Invalid format: {format_type}")

        # Validate player_id
        if not (1 <= player_id <= 8):
            raise HTTPException(status_code=400, detail="Player ID must be between 1 and 8")

        # Get state via gateway
        if gateway is None:
            raise HTTPException(status_code=500, detail="Gateway not initialized")

        result = await gateway.get_game_state(game_id, player_id, format_type)

        if not result["success"]:
            if "not found" in result["error"].lower():
                raise HTTPException(status_code=404, detail=result["error"])
            elif "not authorized" in result["error"].lower():
                raise HTTPException(status_code=403, detail=result["error"])
            else:
                raise HTTPException(status_code=500, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting game state for {game_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/game/{game_id}/action")
async def submit_action(
    game_id: str,
    action: FreeCivAction,
    auth: Dict[str, Any] = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Submit a game action"""
    try:
        # Validate action type
        valid_actions = [
            "unit_move", "unit_attack", "unit_build_city", "unit_explore",
            "city_production", "city_build_unit", "city_build_improvement",
            "tech_research", "diplomacy_message", "end_turn"
        ]

        if action.action_type not in valid_actions:
            raise HTTPException(status_code=400, detail=f"Invalid action type: {action.action_type}")

        # Submit action via gateway
        if gateway is None:
            raise HTTPException(status_code=500, detail="Gateway not initialized")

        result = await gateway.submit_action(game_id, action.dict())

        if not result["success"]:
            if "not found" in result["error"].lower():
                raise HTTPException(status_code=404, detail=result["error"])
            elif "unit" in result["error"].lower() and "does not exist" in result["error"].lower():
                raise HTTPException(status_code=400, detail=result["error"])
            else:
                raise HTTPException(status_code=500, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting action for {game_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/game/{game_id}/actions")
async def submit_actions_batch(
    game_id: str,
    batch: BatchActions,
    auth: Dict[str, Any] = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Submit multiple actions in a batch"""
    try:
        if not settings.enable_batch_actions:
            raise HTTPException(status_code=403, detail="Batch actions are disabled")

        if len(batch.actions) > 10:  # Limit batch size
            raise HTTPException(status_code=400, detail="Too many actions in batch (max 10)")

        results = []

        for i, action in enumerate(batch.actions):
            try:
                result = await submit_action(game_id, action, auth)
                results.append({"index": i, "action_id": result.get("action_id"), "success": True})
            except HTTPException as e:
                results.append({
                    "index": i,
                    "success": False,
                    "error": e.detail,
                    "status_code": e.status_code
                })

        return {
            "success": True,
            "batch_size": len(batch.actions),
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting batch actions for {game_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Game information endpoints
@router.get("/game/{game_id}/info")
async def get_game_info(
    game_id: str,
    auth: Dict[str, Any] = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Get game information"""
    try:
        if gateway is None:
            raise HTTPException(status_code=500, detail="Gateway not initialized")

        if game_id not in gateway.game_sessions:
            raise HTTPException(status_code=404, detail=f"Game not found: {game_id}")

        game_session = gateway.game_sessions[game_id]

        return {
            "game_id": game_id,
            "config": game_session["config"],
            "status": game_session["status"],
            "created_at": game_session["created_at"],
            "players": game_session.get("players", {}),
            "spectators": len(await connection_manager.get_spectator_connections(game_id))
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting game info for {game_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/games")
@limiter.limit(f"{settings.rate_limit_requests_per_minute}/minute") if limiter else lambda x: x
async def list_games(
    request: Request,
    auth: Dict[str, Any] = Depends(verify_api_key),
    gateway_instance = Depends(get_gateway)
) -> Dict[str, Any]:
    """List all active games"""
    try:

        games = []
        for game_id, session in gateway_instance.game_sessions.items():
            games.append({
                "game_id": game_id,
                "status": session["status"],
                "created_at": session["created_at"],
                "player_count": len(session.get("players", {})),
                "config": session["config"]
            })

        return {
            "games": games,
            "total": len(games),
            "capacity": settings.max_concurrent_games
        }

    except Exception as e:
        logger.error(f"Error listing games: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Metrics and monitoring endpoints
@router.get("/metrics")
async def get_metrics(auth: Dict[str, Any] = Depends(verify_api_key)) -> Dict[str, Any]:
    """Get gateway metrics"""
    try:
        if not settings.enable_metrics:
            raise HTTPException(status_code=403, detail="Metrics are disabled")

        if gateway is None:
            raise HTTPException(status_code=500, detail="Gateway not initialized")

        health_status = gateway.get_health_status()

        return {
            "timestamp": time.time(),
            "metrics": {
                "active_games": health_status["active_games"],
                "active_agents": health_status["active_agents"],
                "proxy_connections": health_status["proxy_connections"],
                "uptime": health_status["uptime"],
                "connection_stats": health_status["connection_stats"]
            },
            "status": health_status["status"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Helper functions
def authenticate_api_key(api_key: str) -> Dict[str, Any]:
    """Authenticate API key (extend with proper implementation)"""
    # Simple validation for now
    if len(api_key) >= 10:
        return {"valid": True, "agent_id": f"agent-{api_key[:8]}"}
    else:
        return {"valid": False, "error": "Invalid API key"}


def check_rate_limit(agent_id: str) -> Dict[str, Any]:
    """Check rate limit for agent (extend with proper implementation)"""
    # Placeholder implementation
    return {"allowed": True}


# Import time for metrics
import time