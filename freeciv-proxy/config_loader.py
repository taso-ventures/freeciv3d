#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration loader for LLM support in FreeCiv proxy
"""

import json
import logging
import os
from typing import Dict, Any, List
from dotenv import load_dotenv

logger = logging.getLogger("freeciv-proxy")

class LLMConfig:
    """Configuration manager for LLM agent support"""

    DEFAULT_CONFIG = {
        "enabled": True,
        "endpoint": "/llmsocket",
        "max_agents": 2,
        "cache_config": {
            "ttl": 5,
            "max_size_kb": 4
        },
        "validation": {
            "strict": True,
            "log_failures": True
        },
        "capabilities": {
            "default": ["unit_move", "city_production", "tech_research"]
        },
        "authentication": {
            "require_api_token": True,
            "token_min_length": 10
        }
    }

    def __init__(self, config_file: str = "llm_config.json"):
        self.config_file = config_file
        self.config = self.DEFAULT_CONFIG.copy()

        # Load environment variables first
        load_dotenv()

        self.load_config()
        self._load_env_overrides()

    def load_config(self):
        """Load configuration from file, falling back to defaults"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)

                # Merge with defaults (file config takes precedence)
                self._merge_config(self.config, file_config)
                logger.info(f"Loaded LLM configuration from {self.config_file}")
            else:
                logger.warning(f"Config file {self.config_file} not found, using defaults")

        except Exception as e:
            logger.error(f"Error loading LLM config: {e}, using defaults")

    def _load_env_overrides(self):
        """Load configuration overrides from environment variables"""
        # Load API tokens from environment
        tokens_env = os.getenv('LLM_API_TOKENS')
        if tokens_env:
            separator = self.get('authentication.token_separator', ',')
            tokens = [token.strip() for token in tokens_env.split(separator) if token.strip()]
            if tokens:
                self.config['authentication']['valid_tokens'] = tokens
                logger.info(f"Loaded {len(tokens)} API tokens from environment")

        # Load other security settings
        env_mappings = {
            'SESSION_TIMEOUT_MINUTES': 'authentication.session_timeout_minutes',
            'MAX_MESSAGE_SIZE_MB': 'validation.max_message_size_mb',
            'RATE_LIMIT_ENABLED': 'validation.rate_limit.enabled',
            'CACHE_HMAC_SECRET': 'cache_config.hmac_secret',
            'LOG_LEVEL': 'logging.level',
            'DEBUG_MODE': 'debug_mode',
            'REDIS_HOST': 'redis.host',
            'REDIS_PORT': 'redis.port',
            'REDIS_PASSWORD': 'redis.password',
            'REDIS_DB': 'redis.db'
        }

        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if env_var.endswith('_ENABLED') or env_var == 'DEBUG_MODE':
                    env_value = env_value.lower() in ('true', '1', 'yes', 'on')
                elif env_var.endswith('_MINUTES') or env_var.endswith('_PORT') or env_var.endswith('_DB'):
                    try:
                        env_value = int(env_value)
                    except ValueError:
                        logger.warning(f"Invalid integer value for {env_var}: {env_value}")
                        continue
                elif env_var.endswith('_MB'):
                    try:
                        env_value = float(env_value)
                    except ValueError:
                        logger.warning(f"Invalid float value for {env_var}: {env_value}")
                        continue

                # Set the value using dot notation
                self._set_nested_value(config_path, env_value)
                logger.debug(f"Set {config_path} from environment variable {env_var}")

    def _set_nested_value(self, path: str, value):
        """Set a nested configuration value using dot notation"""
        keys = path.split('.')
        current = self.config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the final value
        current[keys[-1]] = value

    def _merge_config(self, default: Dict, override: Dict):
        """Recursively merge configuration dictionaries"""
        for key, value in override.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value

    def get(self, key: str, default=None):
        """Get configuration value with dot notation support"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def is_enabled(self) -> bool:
        """Check if LLM support is enabled"""
        return self.get('enabled', True)

    def get_max_agents(self) -> int:
        """Get maximum number of concurrent LLM agents"""
        return self.get('max_agents', 2)

    def get_cache_ttl(self) -> int:
        """Get cache TTL in seconds"""
        return self.get('cache_config.ttl', 5)

    def get_cache_max_size_kb(self) -> int:
        """Get maximum cache size in KB"""
        return self.get('cache_config.max_size_kb', 4)

    def is_strict_validation(self) -> bool:
        """Check if strict validation is enabled"""
        return self.get('validation.strict', True)

    def get_default_capabilities(self) -> List[str]:
        """Get default capabilities for LLM agents"""
        return self.get('capabilities.default', [])

    def get_restricted_capabilities(self) -> List[str]:
        """Get restricted capabilities that require special permissions"""
        return self.get('capabilities.restricted', [])

    def is_token_required(self) -> bool:
        """Check if API token is required for authentication"""
        return self.get('authentication.require_api_token', True)

    def get_token_min_length(self) -> int:
        """Get minimum required token length"""
        return self.get('authentication.token_min_length', 10)

    def get_valid_tokens(self) -> List[str]:
        """Get list of valid API tokens"""
        return self.get('authentication.valid_tokens', [])

    def get_rate_limit_rps(self) -> int:
        """Get rate limit requests per second"""
        return self.get('validation.rate_limit.requests_per_second', 10)

    def get_rate_limit_burst(self) -> int:
        """Get rate limit burst capacity"""
        return self.get('validation.rate_limit.burst_capacity', 100)

    def get_state_query_timeout_ms(self) -> int:
        """Get state query timeout in milliseconds"""
        return self.get('performance.state_query_timeout_ms', 50)

    def get_action_timeout_ms(self) -> int:
        """Get action timeout in milliseconds"""
        return self.get('performance.action_timeout_ms', 30)

    def should_log_actions(self) -> bool:
        """Check if actions should be logged"""
        return self.get('logging.log_actions', True)

    def should_log_state_queries(self) -> bool:
        """Check if state queries should be logged"""
        return self.get('logging.log_state_queries', True)

    def should_log_performance(self) -> bool:
        """Check if performance metrics should be logged"""
        return self.get('logging.log_performance', True)

    def validate_token(self, token: str) -> bool:
        """Validate API token"""
        if not self.is_token_required():
            return True

        if len(token) < self.get_token_min_length():
            logger.warning(f"Token too short: {len(token)} < {self.get_token_min_length()}")
            return False

        valid_tokens = self.get_valid_tokens()
        if not valid_tokens:
            logger.error("No valid tokens configured - check LLM_API_TOKENS environment variable")
            return False

        if token not in valid_tokens:
            logger.warning(f"Invalid token provided")
            return False

        return True

    def get_endpoint_path(self, port: int) -> str:
        """Get the full endpoint path for LLM connections"""
        endpoint = self.get('endpoint', '/llmsocket')
        return f"{endpoint}/{port}"

    def to_dict(self) -> Dict[str, Any]:
        """Get the full configuration as a dictionary"""
        return self.config.copy()

# Global configuration instance
llm_config = LLMConfig()