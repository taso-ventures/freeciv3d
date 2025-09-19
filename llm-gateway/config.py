#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration settings for LLM API Gateway
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration settings with environment variable support"""

    # FreeCiv Proxy connection settings
    freeciv_proxy_host: str = "localhost"
    freeciv_proxy_port: int = 8002
    freeciv_proxy_ws_path: str = "/llmsocket/8002"

    # Redis connection for state management
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0

    # Gateway server settings
    host: str = "0.0.0.0"
    port: int = 8003
    max_concurrent_games: int = 10
    max_agents_per_game: int = 8

    # Agent connection settings
    agent_timeout: int = 120  # seconds
    max_connections_per_agent: int = 2
    heartbeat_interval: int = 30  # seconds

    # Security settings
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "https://localhost:3000",
        "https://localhost:8080"
    ]
    api_key_header: str = "Authorization"
    require_api_key: bool = True

    # Rate limiting
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst_size: int = 20

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Connection retry settings
    max_retry_attempts: int = 3
    initial_retry_delay: float = 1.0  # seconds
    max_retry_delay: float = 30.0  # seconds
    retry_backoff_multiplier: float = 2.0
    connection_timeout: float = 10.0  # seconds

    # Feature flags
    enable_spectator_mode: bool = True
    enable_batch_actions: bool = True
    enable_metrics: bool = True

    class Config:
        env_file = ".env"
        env_prefix = "GATEWAY_"


# Global settings instance
settings = Settings()


def get_freeciv_proxy_url() -> str:
    """Get the complete FreeCiv proxy WebSocket URL"""
    return f"ws://{settings.freeciv_proxy_host}:{settings.freeciv_proxy_port}{settings.freeciv_proxy_ws_path}"


def get_cors_origins() -> List[str]:
    """Get CORS allowed origins"""
    return settings.allowed_origins


def validate_settings() -> bool:
    """Validate configuration settings"""
    try:
        # Validate ports
        if not (1 <= settings.port <= 65535):
            raise ValueError(f"Invalid port: {settings.port}")

        if not (1 <= settings.freeciv_proxy_port <= 65535):
            raise ValueError(f"Invalid FreeCiv proxy port: {settings.freeciv_proxy_port}")

        # Validate limits
        if settings.max_concurrent_games <= 0:
            raise ValueError("max_concurrent_games must be positive")

        if settings.agent_timeout <= 0:
            raise ValueError("agent_timeout must be positive")

        # Validate Redis URL format
        if not settings.redis_url.startswith(("redis://", "rediss://")):
            raise ValueError("Invalid Redis URL format")

        # Test Redis connection
        try:
            import redis
            redis_client = redis.from_url(
                settings.redis_url,
                db=settings.redis_db,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection with a simple ping
            redis_client.ping()
            print(f"✅ Redis connection successful: {settings.redis_url}")
        except ImportError:
            print("⚠️  Redis package not installed - falling back to in-memory storage")
        except Exception as e:
            print(f"⚠️  Redis connection failed: {e}")
            print("⚠️  Falling back to in-memory storage (not recommended for production)")

        return True

    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False