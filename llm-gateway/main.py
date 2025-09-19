#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM API Gateway Server for FreeCiv3D Integration
Main FastAPI application with WebSocket and REST endpoints
"""

import asyncio
import json
import logging
import random
import time
import uuid
from typing import Dict, Any, Optional, List
import websockets
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

# Rate limiting imports
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    HAS_SLOWAPI = True
except ImportError:
    HAS_SLOWAPI = False

try:
    from .config import settings, get_cors_origins, get_freeciv_proxy_url, validate_settings
    from .connection_manager import connection_manager, ConnectionInfo
    from .api_endpoints import GameConfig, FreeCivAction
    from .websocket_handlers import AgentWebSocketHandler, SpectatorWebSocketHandler
except ImportError:
    from config import settings, get_cors_origins, get_freeciv_proxy_url, validate_settings
    from connection_manager import connection_manager, ConnectionInfo

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format=settings.log_format
)
logger = logging.getLogger("llm-gateway")

# Create FastAPI app
app = FastAPI(
    title="FreeCiv LLM Gateway",
    description="API Gateway for LLM agent integration with FreeCiv3D",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting setup
if HAS_SLOWAPI:
    # Use Redis for distributed rate limiting if available, otherwise in-memory
    try:
        import redis
        redis_client = redis.from_url(settings.redis_url, db=settings.redis_db)
        limiter = Limiter(key_func=get_remote_address, storage_uri=settings.redis_url)
        logger.info("Rate limiting enabled with Redis backend")
    except Exception as e:
        logger.warning(f"Redis not available for rate limiting: {e}")
        limiter = Limiter(key_func=get_remote_address)
        logger.info("Rate limiting enabled with in-memory backend")

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
else:
    logger.warning("slowapi not installed - rate limiting disabled")
    limiter = None


class LLMGateway:
    """Main gateway class for coordinating between Game Arena and FreeCiv3D"""

    def __init__(self):
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        self.game_sessions: Dict[str, Dict[str, Any]] = {}
        self.proxy_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.connection_retry_counts: Dict[str, int] = {}
        self.failed_connections: Dict[str, float] = {}  # game_id -> last_failure_time
        self.pending_requests: Dict[str, asyncio.Future] = {}  # correlation_id -> Future
        self._running = False

    async def start(self):
        """Start the gateway"""
        self._running = True
        await connection_manager.start()
        logger.info("LLM Gateway started")

    async def stop(self):
        """Stop the gateway"""
        self._running = False
        await connection_manager.stop()

        # Close proxy connections
        for game_id, proxy_ws in self.proxy_connections.items():
            try:
                await proxy_ws.close()
            except Exception as e:
                logger.warning(f"Error closing proxy connection for {game_id}: {e}")

        self.proxy_connections.clear()
        logger.info("LLM Gateway stopped")

    async def register_agent(self, agent_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new LLM agent"""
        try:
            if agent_id in self.active_agents:
                return {
                    "success": False,
                    "error": f"Agent {agent_id} already registered"
                }

            # Validate configuration
            required_fields = ["api_token", "model", "game_id"]
            for field in required_fields:
                if field not in config:
                    return {
                        "success": False,
                        "error": f"Missing required field: {field}"
                    }

            # Store agent configuration
            self.active_agents[agent_id] = {
                "config": config,
                "registered_at": time.time(),
                "session_id": str(uuid.uuid4()),
                "status": "registered"
            }

            logger.info(f"Agent {agent_id} registered for game {config['game_id']}")

            return {
                "success": True,
                "agent_id": agent_id,
                "session_id": self.active_agents[agent_id]["session_id"]
            }

        except Exception as e:
            logger.error(f"Error registering agent {agent_id}: {e}")
            return {
                "success": False,
                "error": f"Registration failed: {str(e)}"
            }

    async def route_message(self, source: str, target: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Route message between Game Arena and FreeCiv"""
        try:
            if target == "freeciv":
                return await self._route_to_freeciv(message)
            elif target == "game_arena":
                return await self._route_to_game_arena(message)
            else:
                return {
                    "success": False,
                    "error": f"Unknown target: {target}"
                }

        except Exception as e:
            logger.error(f"Error routing message from {source} to {target}: {e}")
            return {
                "success": False,
                "error": f"Message routing failed: {str(e)}"
            }

    async def connect_to_freeciv_proxy(self, game_id: str) -> Dict[str, Any]:
        """Establish connection to FreeCiv proxy for a game"""
        try:
            if game_id in self.proxy_connections:
                return {"success": True, "message": "Already connected"}

            proxy_url = get_freeciv_proxy_url()

            # Connect to FreeCiv proxy
            websocket = await websockets.connect(proxy_url)
            self.proxy_connections[game_id] = websocket

            logger.info(f"Connected to FreeCiv proxy for game {game_id}")

            return {
                "success": True,
                "game_id": game_id,
                "proxy_url": proxy_url
            }

        except Exception as e:
            logger.error(f"Failed to connect to FreeCiv proxy for game {game_id}: {e}")
            return {
                "success": False,
                "error": f"Connection failed: {str(e)}"
            }

    async def forward_to_proxy(self, game_id: str, message: Dict[str, Any]) -> bool:
        """Forward message to FreeCiv proxy with retry logic"""
        try:
            # Check if we have a working connection
            if game_id not in self.proxy_connections or not await self._is_connection_healthy(game_id):
                success = await self._ensure_proxy_connection(game_id)
                if not success:
                    logger.error(f"Failed to establish proxy connection for game {game_id}")
                    return False

            proxy_ws = self.proxy_connections[game_id]
            await proxy_ws.send(json.dumps(message))
            logger.debug(f"Message forwarded to proxy for game {game_id}")
            return True

        except Exception as e:
            logger.error(f"Error forwarding message to proxy for game {game_id}: {e}")
            # Mark connection as failed and remove it
            await self._handle_connection_failure(game_id)
            return False

    async def _ensure_proxy_connection(self, game_id: str) -> bool:
        """Ensure proxy connection exists with retry logic"""
        # Check if connection is in cooldown period
        if game_id in self.failed_connections:
            last_failure = self.failed_connections[game_id]
            cooldown_period = 60  # 60 seconds cooldown
            if time.time() - last_failure < cooldown_period:
                logger.debug(f"Connection for game {game_id} in cooldown period")
                return False

        retry_count = self.connection_retry_counts.get(game_id, 0)

        for attempt in range(settings.max_retry_attempts):
            try:
                logger.info(f"Attempting to connect to proxy for game {game_id} (attempt {attempt + 1})")

                success = await self.connect_to_freeciv_proxy(game_id)
                if success.get("success", False):
                    # Reset retry count on successful connection
                    self.connection_retry_counts[game_id] = 0
                    if game_id in self.failed_connections:
                        del self.failed_connections[game_id]
                    logger.info(f"Successfully connected to proxy for game {game_id}")
                    return True

            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed for game {game_id}: {e}")

            # Calculate exponential backoff delay
            if attempt < settings.max_retry_attempts - 1:  # Don't wait after last attempt
                delay = min(
                    settings.initial_retry_delay * (settings.retry_backoff_multiplier ** attempt),
                    settings.max_retry_delay
                )
                # Add jitter to prevent thundering herd
                jitter = random.uniform(0.1, 0.3) * delay
                total_delay = delay + jitter

                logger.info(f"Waiting {total_delay:.2f}s before retry for game {game_id}")
                await asyncio.sleep(total_delay)

        # All attempts failed
        self.connection_retry_counts[game_id] = retry_count + 1
        await self._handle_connection_failure(game_id)
        return False

    async def _is_connection_healthy(self, game_id: str) -> bool:
        """Check if connection is healthy"""
        if game_id not in self.proxy_connections:
            return False

        try:
            proxy_ws = self.proxy_connections[game_id]
            # Check if WebSocket is closed
            if proxy_ws.closed:
                return False

            # Send a ping to test connection
            await proxy_ws.ping()
            return True

        except Exception as e:
            logger.debug(f"Connection health check failed for game {game_id}: {e}")
            return False

    async def _handle_connection_failure(self, game_id: str):
        """Handle connection failure"""
        logger.warning(f"Handling connection failure for game {game_id}")

        # Record failure time
        self.failed_connections[game_id] = time.time()

        # Remove the failed connection
        if game_id in self.proxy_connections:
            try:
                await self.proxy_connections[game_id].close()
            except:
                pass
            del self.proxy_connections[game_id]

        # Implement circuit breaker pattern
        retry_count = self.connection_retry_counts.get(game_id, 0)
        if retry_count >= settings.max_retry_attempts:
            logger.error(f"Max retry attempts reached for game {game_id}. Implementing circuit breaker.")
            # Could trigger alerts, disable game, etc.

    async def _handle_proxy_message(self, game_id: str, message: Dict[str, Any]):
        """Handle incoming message from FreeCiv proxy"""
        try:
            correlation_id = message.get("correlation_id")
            if correlation_id and correlation_id in self.pending_requests:
                # Resolve the pending request
                future = self.pending_requests.pop(correlation_id)
                if not future.done():
                    future.set_result(message)
                logger.debug(f"Resolved pending request {correlation_id}")
            else:
                # Handle non-request messages (broadcasts, events, etc.)
                logger.debug(f"Received non-correlated message from proxy: {message.get('type', 'unknown')}")
                # Could forward to all connected agents, etc.

        except Exception as e:
            logger.error(f"Error handling proxy message: {e}")

    async def _send_request_and_wait(self, game_id: str, message: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        """Send request to proxy and wait for response"""
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        message["correlation_id"] = correlation_id

        # Create future for response
        future = asyncio.Future()
        self.pending_requests[correlation_id] = future

        try:
            # Send message
            success = await self.forward_to_proxy(game_id, message)
            if not success:
                raise Exception("Failed to send message to proxy")

            # Wait for response with timeout
            try:
                response = await asyncio.wait_for(future, timeout=timeout)
                return response
            except asyncio.TimeoutError:
                logger.warning(f"Request {correlation_id} timed out after {timeout}s")
                return {
                    "success": False,
                    "error": "Request timed out",
                    "correlation_id": correlation_id
                }

        except Exception as e:
            logger.error(f"Error sending request {correlation_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "correlation_id": correlation_id
            }
        finally:
            # Clean up pending request
            if correlation_id in self.pending_requests:
                del self.pending_requests[correlation_id]

    async def create_game(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new game session"""
        try:
            game_id = f"game-{uuid.uuid4()}"

            # Check capacity
            if len(self.game_sessions) >= settings.max_concurrent_games:
                return {
                    "success": False,
                    "error": f"Maximum concurrent games ({settings.max_concurrent_games}) exceeded"
                }

            # Store game session
            self.game_sessions[game_id] = {
                "config": config,
                "created_at": time.time(),
                "status": "active",
                "players": {}
            }

            # Connect to FreeCiv proxy
            proxy_result = await self.connect_to_freeciv_proxy(game_id)
            if not proxy_result["success"]:
                del self.game_sessions[game_id]
                return proxy_result

            return {
                "success": True,
                "game_id": game_id,
                "connection_details": {
                    "ws_url": f"ws://localhost:{settings.port}/ws/agent/{{agent_id}}",
                    "game_port": settings.freeciv_proxy_port
                }
            }

        except Exception as e:
            logger.error(f"Error creating game: {e}")
            return {
                "success": False,
                "error": f"Game creation failed: {str(e)}"
            }

    async def get_game_state(self, game_id: str, player_id: int, format_type: str = "llm_optimized") -> Dict[str, Any]:
        """Get game state from FreeCiv proxy"""
        try:
            if game_id not in self.game_sessions:
                return {
                    "success": False,
                    "error": f"Game not found: {game_id}"
                }

            # Forward request to FreeCiv proxy
            state_request = {
                "type": "state_query",
                "agent_id": f"gateway-{uuid.uuid4()}",
                "timestamp": time.time(),
                "data": {
                    "format": format_type,
                    "player_id": player_id
                }
            }

            # Send request and wait for response
            response = await self._send_request_and_wait(game_id, state_request, timeout=15.0)

            if response.get("success", False):
                return {
                    "success": True,
                    "format": format_type,
                    "data": response.get("data", {})
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error", "Unknown error from proxy")
                }

        except Exception as e:
            logger.error(f"Error getting game state for {game_id}: {e}")
            return {
                "success": False,
                "error": f"State query failed: {str(e)}"
            }

    async def submit_action(self, game_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """Submit action to FreeCiv proxy"""
        try:
            if game_id not in self.game_sessions:
                return {
                    "success": False,
                    "error": f"Game not found: {game_id}"
                }

            # Forward action to FreeCiv proxy
            action_message = {
                "type": "action",
                "agent_id": f"gateway-{uuid.uuid4()}",
                "timestamp": time.time(),
                "data": action
            }

            # Send action and wait for response
            response = await self._send_request_and_wait(game_id, action_message, timeout=10.0)

            if response.get("success", False):
                return {
                    "success": True,
                    "action_id": response.get("action_id", str(uuid.uuid4())),
                    "result": response.get("result", "Action executed successfully")
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error", "Action execution failed")
                }

        except Exception as e:
            logger.error(f"Error submitting action for {game_id}: {e}")
            return {
                "success": False,
                "error": f"Action submission failed: {str(e)}"
            }

    def get_health_status(self) -> Dict[str, Any]:
        """Get gateway health status"""
        try:
            stats = connection_manager.get_connection_stats()
            active_games = len(self.game_sessions)
            active_agents = len(self.active_agents)

            status = "healthy"
            issues = []

            # Check for issues
            if active_games >= settings.max_concurrent_games:
                status = "degraded"
                issues.append("At maximum game capacity")

            if len(self.proxy_connections) < active_games:
                status = "degraded"
                issues.append("Some FreeCiv proxy connections missing")

            return {
                "status": status,
                "active_games": active_games,
                "active_agents": active_agents,
                "proxy_connections": len(self.proxy_connections),
                "connection_stats": stats,
                "uptime": time.time() - gateway_start_time,
                "issues": issues if issues else None
            }

        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def _route_to_freeciv(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Route message to FreeCiv proxy"""
        agent_id = message.get("agent_id")

        if not agent_id or agent_id not in self.active_agents:
            return {
                "success": False,
                "error": "Agent not registered"
            }

        game_id = self.active_agents[agent_id]["config"]["game_id"]

        if game_id not in self.proxy_connections:
            result = await self.connect_to_freeciv_proxy(game_id)
            if not result["success"]:
                return result

        await self.forward_to_proxy(game_id, message)

        return {"success": True}

    async def _route_to_game_arena(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Route message to Game Arena (not implemented in this scope)"""
        return {
            "success": False,
            "error": "Game Arena routing not implemented in this scope"
        }


# Global gateway instance
gateway = LLMGateway()
gateway_start_time = time.time()


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup"""
    if not validate_settings():
        raise RuntimeError("Invalid configuration settings")

    await gateway.start()
    logger.info(f"LLM Gateway started on {settings.host}:{settings.port}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    await gateway.stop()
    logger.info("LLM Gateway shutdown complete")


# Health check endpoint
@app.get("/health")
@limiter.limit(f"{settings.rate_limit_requests_per_minute}/minute") if limiter else lambda x: x
async def health_check(request: Request):
    """Health check endpoint"""
    health_status = gateway.get_health_status()

    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)

    return health_status


# Import API endpoints and WebSocket handlers after app creation
def setup_routes():
    """Setup API routes and WebSocket handlers"""
    try:
        from api_endpoints import router as api_router
        # Update the gateway reference in api_endpoints
        import api_endpoints
        api_endpoints.gateway = gateway
        app.include_router(api_router, prefix="/api")
        logger.info("API endpoints registered")
    except ImportError as e:
        logger.warning(f"API endpoints not available: {e}")

    try:
        from websocket_handlers import register_websocket_routes
        register_websocket_routes(app)
        logger.info("WebSocket handlers registered")
    except ImportError as e:
        logger.warning(f"WebSocket handlers not available: {e}")

# Setup routes after gateway is created
setup_routes()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=True
    )