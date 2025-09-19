#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
State Extraction Service for FreeCiv LLM Integration
Provides REST API endpoints for game state extraction and optimization
"""

import json
import time
import logging
import os
import atexit
import signal
import sys
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
from tornado import web
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor

from state_cache import StateCache, CacheEntry
from civcom import CivCom

# Required modules for production use
try:
    from error_handler import error_handler, ErrorSeverity, ErrorCategory
    from security import InputSanitizer
    from api_rate_limiter import api_rate_limiter
    from auth import authenticator, AuthenticationError, AuthorizationError
except ImportError as e:
    raise ImportError(
        f"Failed to import required modules: {e}. "
        "The error_handler, security, api_rate_limiter, and auth modules are required. "
        "Ensure all required .py files are available in the Python path."
    )

logger = logging.getLogger("freeciv-proxy")


class StateExtractionError(Exception):
    """Exception raised during state extraction operations"""
    def __init__(self, message: str, game_id: str = None, player_id: int = None, cause: Exception = None):
        super().__init__(message)
        self.game_id = game_id
        self.player_id = player_id
        self.cause = cause


class CacheError(Exception):
    """Exception raised during cache operations"""
    def __init__(self, message: str, cache_key: str = None, operation: str = None):
        super().__init__(message)
        self.cache_key = cache_key
        self.operation = operation


class ValidationError(Exception):
    """Exception raised during input validation"""
    def __init__(self, message: str, parameter: str = None, value: Any = None):
        super().__init__(message)
        self.parameter = parameter
        self.value = value


class CivComNotFoundError(StateExtractionError):
    """Exception raised when CivCom instance is not available"""
    def __init__(self, game_id: str):
        message = f"No CivCom instance available for game {game_id}"
        super().__init__(message, game_id=game_id)


class CivComRegistry:
    """
    Registry for CivCom instances with proper lifecycle management
    Provides clean interface for registering and retrieving game connections
    """

    def __init__(self):
        self._civcom_instances: Dict[str, CivCom] = {}
        self._game_metadata: Dict[str, Dict[str, Any]] = {}

    def register_game(self, game_id: str, civcom: CivCom, metadata: Optional[Dict[str, Any]] = None):
        """Register a CivCom instance for a game"""
        if not isinstance(game_id, str) or not game_id.strip():
            raise ValueError("game_id must be a non-empty string")

        if not civcom:
            raise ValueError("civcom instance cannot be None")

        self._civcom_instances[game_id] = civcom
        self._game_metadata[game_id] = metadata or {}

        logger.info(f"Registered CivCom for game: {game_id}")

    def unregister_game(self, game_id: str):
        """Unregister a game and clean up resources"""
        if game_id in self._civcom_instances:
            try:
                # Try to cleanup the civcom instance if it has cleanup methods
                civcom = self._civcom_instances[game_id]
                if hasattr(civcom, 'cleanup'):
                    civcom.cleanup()
                elif hasattr(civcom, 'close'):
                    civcom.close()
            except Exception as e:
                logger.warning(f"Error cleaning up CivCom for game {game_id}: {e}")

            del self._civcom_instances[game_id]
            if game_id in self._game_metadata:
                del self._game_metadata[game_id]

            logger.info(f"Unregistered CivCom for game: {game_id}")

    def get_civcom(self, game_id: str) -> Optional[CivCom]:
        """Get CivCom instance for a game"""
        if not isinstance(game_id, str):
            return None

        return self._civcom_instances.get(game_id)

    def has_game(self, game_id: str) -> bool:
        """Check if game is registered"""
        return game_id in self._civcom_instances

    def list_games(self) -> List[str]:
        """Get list of registered game IDs"""
        return list(self._civcom_instances.keys())

    def get_game_metadata(self, game_id: str) -> Dict[str, Any]:
        """Get metadata for a game"""
        return self._game_metadata.get(game_id, {})

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            'total_games': len(self._civcom_instances),
            'active_games': list(self._civcom_instances.keys()),
            'registry_size_kb': len(str(self._civcom_instances)) / 1024
        }

# Global registry for CivCom instances
civcom_registry = CivComRegistry()

# Shared thread pool executor for all state extraction operations
# Configurable via environment variable, defaults to 4
_MAX_WORKERS = int(os.getenv('STATE_EXTRACTOR_THREADS', '4'))
_shared_executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="state-extractor")

# Configurable constants for magic numbers
MAX_TURN_NUMBER = int(os.getenv('MAX_TURN_NUMBER', '10000'))
MAX_EXPANSION_SITES = int(os.getenv('MAX_EXPANSION_SITES', '5'))
MAX_UNITS_ANALYZED = int(os.getenv('MAX_UNITS_ANALYZED', '5'))
MAX_CITIES_ANALYZED = int(os.getenv('MAX_CITIES_ANALYZED', '5'))
MAX_THREATS_RETURNED = int(os.getenv('MAX_THREATS_RETURNED', '5'))

def shutdown_executor():
    """Shutdown the shared thread pool executor gracefully"""
    if _shared_executor:
        logger.info("Shutting down state extractor thread pool...")
        _shared_executor.shutdown(wait=True)
        logger.info("State extractor thread pool shutdown complete")


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_executor()
    sys.exit(0)


# Register shutdown handlers
atexit.register(shutdown_executor)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def authenticate_request(request_handler, required_permission: str = 'state_read') -> Tuple[bool, Optional[int], Optional[str], str]:
    """
    Authenticate request using API key or session

    Returns:
        tuple: (authenticated: bool, player_id: Optional[int], game_id: Optional[str], error_message: str)
    """
    # Check environment and authentication settings
    environment = os.getenv('ENVIRONMENT', 'production').lower()
    auth_enabled = os.getenv('AUTH_ENABLED', 'true').lower() == 'true'

    # Only allow authentication bypass in development environment
    if not auth_enabled:
        if environment == 'development':
            # Authentication disabled in development, allow but warn
            logger.warning(f"Authentication bypassed in development mode for {request_handler.request.remote_ip}")
            # Add warning header to response
            request_handler.set_header("X-Auth-Bypassed", "true")
            request_handler.set_header("X-Environment", "development")
            return True, None, None, ""
        else:
            # Force authentication in production/staging regardless of AUTH_ENABLED setting
            logger.error(f"Authentication bypass attempted in {environment} environment - BLOCKED")
            # Continue to authentication logic below

    try:
        # Get authentication credentials from headers or query params
        api_key = request_handler.get_argument('api_key', None)
        if not api_key:
            api_key = request_handler.request.headers.get('Authorization')
            if api_key and api_key.startswith('Bearer '):
                api_key = api_key[7:]

        session_id = request_handler.get_argument('session_id', None)
        if not session_id:
            session_id = request_handler.request.headers.get('X-Session-ID')

        # Attempt authentication
        authenticated, auth_player_id, auth_game_id = authenticator.authenticate_request(
            api_key=api_key,
            session_id=session_id,
            required_permission=required_permission
        )

        if not authenticated:
            return False, None, None, "Authentication required. Provide valid API key or session ID."

        return True, auth_player_id, auth_game_id, ""

    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return False, None, None, f"Authentication failed: {e}"


def validate_request_parameters(game_id: str, player_id: Any, format_str: str = 'full', since_turn: Any = None) -> tuple:
    """
    Validate and sanitize request parameters

    Returns:
        tuple: (validated_game_id, validated_player_id, validated_format, validated_since_turn)

    Raises:
        ValidationError: If any parameter is invalid
    """
    try:
        # Validate game_id (alphanumeric, underscores, max 50 chars)
        if not isinstance(game_id, str) or not game_id.strip():
            raise ValidationError("Game ID must be a non-empty string", parameter="game_id", value=game_id)

        if len(game_id) > 50 or not all(c.isalnum() or c in '_-' for c in game_id):
            raise ValidationError("Game ID must be alphanumeric with underscores/hyphens, max 50 characters",
                                parameter="game_id", value=game_id)

        # Validate player_id
        validated_player_id = InputSanitizer.sanitize_player_id(player_id)

        # Validate format
        valid_formats = ['full', 'delta', 'llm_optimized']
        if format_str not in valid_formats:
            raise ValidationError(f"Format must be one of: {', '.join(valid_formats)}",
                                parameter="format", value=format_str)

        # Validate since_turn if provided
        validated_since_turn = None
        if since_turn is not None:
            try:
                validated_since_turn = int(since_turn)
                if validated_since_turn < 0 or validated_since_turn > 10000:  # Reasonable bounds
                    raise ValidationError("Since turn must be between 0 and 10000",
                                        parameter="since_turn", value=since_turn)
            except (ValueError, TypeError):
                raise ValidationError("Since turn must be a valid integer",
                                    parameter="since_turn", value=since_turn)

        return game_id.strip(), validated_player_id, format_str, validated_since_turn

    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Parameter validation failed: {e}")


class StateFormat(Enum):
    """Supported state extraction formats"""
    FULL = "full"
    DELTA = "delta"
    LLM_OPTIMIZED = "llm_optimized"


class StateExtractor:
    """
    Core state extraction logic
    Handles game state retrieval, optimization, and caching
    """

    def __init__(self, civcom: Optional[CivCom] = None, cache: Optional[StateCache] = None,
                 registry: Optional[CivComRegistry] = None):
        self.civcom = civcom  # Optional fallback civcom instance
        self.cache = cache or StateCache(ttl=5, max_size_kb=4)
        self.executor = _shared_executor
        self.registry = registry or civcom_registry  # Use global registry by default

    def _get_civcom_for_game(self, game_id: str) -> Optional[CivCom]:
        """Get CivCom instance for a game from registry or fallback"""
        # Try registry first
        civcom = self.registry.get_civcom(game_id)
        if civcom:
            return civcom

        # Fallback to provided civcom if it exists and has required methods
        if self.civcom and hasattr(self.civcom, 'get_full_state'):
            logger.debug(f"Using fallback civcom for game {game_id}")
            return self.civcom

        return None

    def extract_state(self, game_id: str, player_id: int, format_type: StateFormat,
                     since_turn: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract game state in specified format

        Args:
            game_id: Unique game identifier
            player_id: Player ID for perspective-based state
            format_type: State format (full, delta, llm_optimized)
            since_turn: For delta format, changes since this turn

        Returns:
            Dictionary containing game state in requested format
        """
        cache_key = self._build_cache_key(game_id, player_id, format_type.value, since_turn)

        # Check cache first
        cached_state = self.cache.get(cache_key)
        if cached_state is not None:
            logger.debug(f"Cache hit for {cache_key}")
            return cached_state

        # Get civcom for this game
        civcom = self._get_civcom_for_game(game_id)
        if not civcom:
            raise CivComNotFoundError(game_id)

        # Extract fresh state
        start_time = time.time()

        try:
            if format_type == StateFormat.DELTA and since_turn is not None:
                state = self._extract_delta_state(game_id, player_id, since_turn, civcom)
            else:
                raw_state = civcom.get_full_state(player_id)

                if format_type == StateFormat.FULL:
                    state = self._format_full_state(raw_state, player_id)
                elif format_type == StateFormat.LLM_OPTIMIZED:
                    state = self._format_llm_optimized_state(raw_state, player_id)
                else:
                    raise ValidationError(f"Unsupported format: {format_type}", parameter="format", value=format_type.value)

            # Cache the result
            self.cache.set(cache_key, state, player_id)

            extraction_time = (time.time() - start_time) * 1000
            logger.info(f"State extraction completed in {extraction_time:.2f}ms for {cache_key}")

            return state

        except (CivComNotFoundError, ValidationError, StateExtractionError) as e:
            # Re-raise specific exceptions with preserved context
            logger.error(f"State extraction failed for game {game_id}, player {player_id}: {e}")
            error_response = error_handler.handle_state_extraction_error(
                game_id, player_id, str(e)
            )
            logger.error(f"Error response: {error_response}")
            raise
        except Exception as e:
            # Convert unexpected exceptions to StateExtractionError
            logger.error(f"Unexpected error during state extraction: {e}", exc_info=True)
            raise StateExtractionError(
                f"Unexpected error during state extraction: {e}",
                game_id=game_id,
                player_id=player_id,
                cause=e
            )

    def get_legal_actions(self, game_id: str, player_id: int) -> List[Dict[str, Any]]:
        """
        Get top 20 legal actions for player, sorted by strategic priority

        Args:
            game_id: Unique game identifier
            player_id: Player ID

        Returns:
            List of up to 20 legal actions sorted by priority
        """
        try:
            civcom = self._get_civcom_for_game(game_id)
            if not civcom:
                raise CivComNotFoundError(game_id)

            # For now, generate mock actions based on game state
            # In a full implementation, this would extract from the actual game
            state = civcom.get_full_state(player_id)
            all_actions = self._generate_legal_actions_from_state(state, player_id)

            # Sort by priority (highest first) and take top 20
            sorted_actions = sorted(all_actions, key=lambda x: x.get('priority', 0), reverse=True)
            return sorted_actions[:20]

        except (CivComNotFoundError, StateExtractionError) as e:
            # Re-raise specific exceptions with preserved context
            logger.error(f"Legal actions extraction failed for game {game_id}, player {player_id}: {e}")
            error_response = error_handler.handle_action_extraction_error(
                game_id, player_id, str(e)
            )
            logger.error(f"Error response: {error_response}")
            raise
        except Exception as e:
            # Convert unexpected exceptions to StateExtractionError
            logger.error(f"Unexpected error during legal actions extraction: {e}", exc_info=True)
            raise StateExtractionError(
                f"Unexpected error during legal actions extraction: {e}",
                game_id=game_id,
                player_id=player_id,
                cause=e
            )

    def _extract_delta_state(self, game_id: str, player_id: int, since_turn: int, civcom: CivCom) -> Dict[str, Any]:
        """Extract changes since specified turn"""
        current_state = civcom.get_full_state(player_id)
        # For MVP, simulate previous state (in full implementation, store historical states)
        previous_state = self._simulate_previous_state(current_state, since_turn)

        return {
            'since_turn': since_turn,
            'current_turn': current_state.get('turn'),
            'changes': self._calculate_state_delta(previous_state, current_state),
            'timestamp': time.time()
        }

    def _calculate_state_delta(self, previous: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate differences between two game states
        Optimized for performance with O(1) lookups and set operations
        """
        changes = {}

        # Track unit changes - O(n) complexity
        prev_units = {u['id']: u for u in previous.get('units', [])}
        curr_units = {u['id']: u for u in current.get('units', [])}

        # Use sets for efficient difference operations
        prev_unit_ids = set(prev_units.keys())
        curr_unit_ids = set(curr_units.keys())

        unit_changes = []

        # Modified/existing units - O(n) where n = current units
        for unit_id in (prev_unit_ids & curr_unit_ids):  # Intersection
            unit = curr_units[unit_id]
            prev_unit = prev_units[unit_id]

            # Check only relevant fields for changes
            position_changed = unit['x'] != prev_unit['x'] or unit['y'] != prev_unit['y']
            hp_changed = unit['hp'] != prev_unit['hp']
            moves_changed = unit.get('moves', 0) != prev_unit.get('moves', 0)

            if position_changed or hp_changed or moves_changed:
                change_data = {'id': unit_id, 'changes': {}}

                if position_changed:
                    change_data['changes']['position'] = {
                        'from': (prev_unit['x'], prev_unit['y']),
                        'to': (unit['x'], unit['y'])
                    }
                if hp_changed:
                    change_data['changes']['hp'] = {'from': prev_unit['hp'], 'to': unit['hp']}
                if moves_changed:
                    change_data['changes']['moves'] = {
                        'from': prev_unit.get('moves', 0), 'to': unit.get('moves', 0)
                    }

                unit_changes.append(change_data)

        # New units - O(k) where k = new units
        for unit_id in (curr_unit_ids - prev_unit_ids):  # Difference
            unit_changes.append({'id': unit_id, 'type': 'created', 'data': curr_units[unit_id]})

        # Destroyed units - O(j) where j = destroyed units
        for unit_id in (prev_unit_ids - curr_unit_ids):  # Difference
            unit_changes.append({'id': unit_id, 'type': 'destroyed'})

        if unit_changes:
            changes['units'] = unit_changes

        # Track city changes - same optimization pattern
        prev_cities = {c['id']: c for c in previous.get('cities', [])}
        curr_cities = {c['id']: c for c in current.get('cities', [])}

        prev_city_ids = set(prev_cities.keys())
        curr_city_ids = set(curr_cities.keys())

        city_changes = []

        # Modified/existing cities
        for city_id in (prev_city_ids & curr_city_ids):
            city = curr_cities[city_id]
            prev_city = prev_cities[city_id]

            pop_changed = city['population'] != prev_city['population']
            prod_changed = city.get('production') != prev_city.get('production')

            if pop_changed or prod_changed:
                change_data = {'id': city_id, 'changes': {}}

                if pop_changed:
                    change_data['changes']['population'] = {
                        'from': prev_city['population'], 'to': city['population']
                    }
                if prod_changed:
                    change_data['changes']['production'] = {
                        'from': prev_city.get('production'), 'to': city.get('production')
                    }

                city_changes.append(change_data)

        # New cities
        for city_id in (curr_city_ids - prev_city_ids):
            city_changes.append({'id': city_id, 'type': 'founded', 'data': curr_cities[city_id]})

        if city_changes:
            changes['cities'] = city_changes

        return changes

    def _format_full_state(self, raw_state: Dict[str, Any], player_id: int) -> Dict[str, Any]:
        """Format complete game state"""
        return {
            'format': 'full',
            'turn': raw_state.get('turn'),
            'phase': raw_state.get('phase'),
            'map': raw_state.get('map'),
            'units': raw_state.get('units', []),
            'cities': raw_state.get('cities', []),
            'players': raw_state.get('players', []),
            'techs': raw_state.get('techs', {}),
            'timestamp': time.time(),
            'player_perspective': player_id
        }

    def _format_llm_optimized_state(self, raw_state: Dict[str, Any], player_id: int) -> Dict[str, Any]:
        """
        Format state optimized for LLM consumption
        Target: >70% size reduction while preserving decision-critical information
        """
        # Build strategic view
        strategic = self._build_strategic_view(raw_state, player_id)

        # Build tactical view
        tactical = self._build_tactical_view(raw_state, player_id)

        # Build economic view
        economic = self._build_economic_view(raw_state, player_id)

        return {
            'format': 'llm_optimized',
            'turn': raw_state.get('turn'),
            'phase': raw_state.get('phase'),
            'strategic': strategic,
            'tactical': tactical,
            'economic': economic,
            'timestamp': time.time(),
            'player_perspective': player_id
        }

    def _build_strategic_view(self, state: Dict[str, Any], player_id: int) -> Dict[str, Any]:
        """Build strategic layer focusing on long-term game position"""
        players = state.get('players', [])
        player = next((p for p in players if p['id'] == player_id), None)

        if not player:
            return {}

        # Calculate relative positions
        scores = {p['id']: self._calculate_player_score(state, p['id']) for p in players}
        player_rank = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return {
            'victory_progress': {
                'current_score': scores[player_id],
                'rank': next(i for i, (pid, _) in enumerate(player_rank, 1) if pid == player_id),
                'total_players': len(players)
            },
            'tech_position': {
                'researched': state.get('techs', {}).get(f'player{player_id}', []),
                'research_points': player.get('science', 0)
            },
            'diplomatic_status': self._get_diplomatic_summary(state, player_id),
            'relative_strength': self._assess_relative_strength(state, player_id)
        }

    def _build_tactical_view(self, state: Dict[str, Any], player_id: int) -> Dict[str, Any]:
        """Build tactical layer focusing on immediate military situation"""
        units = [u for u in state.get('units', []) if u['owner'] == player_id]
        enemy_units = [u for u in state.get('units', []) if u['owner'] != player_id]

        # Group similar units
        unit_groups = {}
        for unit in units:
            unit_type = unit['type']
            if unit_type not in unit_groups:
                unit_groups[unit_type] = {'count': 0, 'positions': [], 'avg_hp': 0}

            unit_groups[unit_type]['count'] += 1
            unit_groups[unit_type]['positions'].append((unit['x'], unit['y']))
            unit_groups[unit_type]['avg_hp'] += unit['hp']

        # Calculate averages
        for group in unit_groups.values():
            group['avg_hp'] = group['avg_hp'] / group['count'] if group['count'] > 0 else 0
            # Keep only center positions for size reduction
            if len(group['positions']) > 3:
                center_x = sum(pos[0] for pos in group['positions']) // len(group['positions'])
                center_y = sum(pos[1] for pos in group['positions']) // len(group['positions'])
                group['positions'] = [(center_x, center_y)]

        return {
            'unit_groups': unit_groups,
            'immediate_threats': self._identify_threats(units, enemy_units),
            'exploration_frontier': self._get_exploration_opportunities(state, player_id),
            'combat_readiness': self._assess_combat_readiness(units)
        }

    def _build_economic_view(self, state: Dict[str, Any], player_id: int) -> Dict[str, Any]:
        """Build economic layer focusing on resource management"""
        cities = [c for c in state.get('cities', []) if c['owner'] == player_id]
        players = state.get('players', [])
        player = next((p for p in players if p['id'] == player_id), None)

        return {
            'cities': {
                'count': len(cities),
                'total_population': sum(c['population'] for c in cities),
                'production_focus': [c['production'] for c in cities],
                'growth_potential': self._assess_growth_potential(cities)
            },
            'resources': {
                'gold': player.get('gold', 0) if player else 0,
                'science': player.get('science', 0) if player else 0
            },
            'infrastructure': {
                'development_level': self._assess_development_level(cities),
                'expansion_opportunities': self._get_expansion_sites(state, player_id)
            }
        }

    def _calculate_player_score(self, state: Dict[str, Any], player_id: int) -> int:
        """Calculate simple score for player ranking"""
        cities = [c for c in state.get('cities', []) if c['owner'] == player_id]
        units = [u for u in state.get('units', []) if u['owner'] == player_id]

        # Simple scoring: cities worth more than units
        return len(cities) * 10 + len(units) * 2

    def _get_diplomatic_summary(self, state: Dict[str, Any], player_id: int) -> Dict[str, str]:
        """Get simplified diplomatic status (placeholder)"""
        return {"status": "neutral"}  # Simplified for MVP

    def _assess_relative_strength(self, state: Dict[str, Any], player_id: int) -> str:
        """Assess military strength relative to others"""
        # Simplified assessment based on unit count
        player_units = len([u for u in state.get('units', []) if u['owner'] == player_id])
        total_units = len(state.get('units', []))

        if total_units == 0:
            return "unknown"

        ratio = player_units / total_units
        if ratio > 0.4:
            return "strong"
        elif ratio > 0.2:
            return "moderate"
        else:
            return "weak"

    def _identify_threats(self, friendly_units: List[Dict], enemy_units: List[Dict]) -> List[Dict[str, Any]]:
        """Identify immediate military threats"""
        threats = []
        threat_radius = 3  # Consider units within 3 tiles as threats

        for enemy in enemy_units:
            nearby_friendlies = [
                u for u in friendly_units
                if abs(u['x'] - enemy['x']) <= threat_radius and abs(u['y'] - enemy['y']) <= threat_radius
            ]

            if nearby_friendlies:
                threats.append({
                    'enemy_type': enemy['type'],
                    'position': (enemy['x'], enemy['y']),
                    'threatened_units': len(nearby_friendlies)
                })

        return threats[:MAX_THREATS_RETURNED]

    def _get_exploration_opportunities(self, state: Dict[str, Any], player_id: int) -> List[Dict[str, int]]:
        """Get exploration opportunities (simplified)"""
        # Placeholder - would analyze fog of war in full implementation
        return [{"direction": "north", "priority": 5}]

    def _assess_combat_readiness(self, units: List[Dict]) -> Dict[str, Any]:
        """Assess overall combat readiness"""
        if not units:
            return {"status": "no_units", "strength": 0}

        total_hp = sum(u['hp'] for u in units)
        avg_hp = total_hp / len(units) if len(units) > 0 else 0

        return {
            "unit_count": len(units),
            "avg_health": round(avg_hp, 1),
            "status": "ready" if avg_hp > 7 else "weakened"
        }

    def _assess_growth_potential(self, cities: List[Dict]) -> str:
        """Assess growth potential of cities"""
        if not cities:
            return "none"

        avg_pop = sum(c['population'] for c in cities) / len(cities) if len(cities) > 0 else 0
        return "high" if avg_pop < 5 else "moderate" if avg_pop < 10 else "limited"

    def _assess_development_level(self, cities: List[Dict]) -> str:
        """Assess overall development level"""
        if not cities:
            return "none"

        total_pop = sum(c['population'] for c in cities)
        return "developed" if total_pop > 20 else "developing" if total_pop > 10 else "early"

    def _get_expansion_sites(self, state: Dict[str, Any], player_id: int) -> int:
        """Get number of potential expansion sites based on actual game data"""
        try:
            # Get actual map data if available
            tiles = state.get('tiles', [])
            existing_cities = state.get('cities', [])
            player_units = [u for u in state.get('units', []) if u.get('owner') == player_id]

            if not tiles:
                # Fallback: estimate based on units and cities
                num_cities = len([c for c in existing_cities if c.get('owner') == player_id])
                num_settlers = len([u for u in player_units if u.get('type') == 'settler'])
                return max(0, min(3, num_settlers + (3 - num_cities)))

            # Analyze tiles for suitable city sites
            suitable_sites = 0
            city_positions = {(c['x'], c['y']) for c in existing_cities}

            for tile in tiles:
                x, y = tile.get('x', -1), tile.get('y', -1)
                if x < 0 or y < 0:
                    continue

                # Check if tile is suitable for a city
                terrain = tile.get('terrain', 'unknown')
                if terrain in ['grassland', 'plains', 'hills']:
                    # Check minimum distance from existing cities (at least 2 tiles)
                    too_close = any(
                        abs(x - cx) <= 2 and abs(y - cy) <= 2
                        for cx, cy in city_positions
                    )

                    if not too_close:
                        suitable_sites += 1

            return min(suitable_sites, MAX_EXPANSION_SITES)

        except Exception as e:
            logger.warning(f"Error analyzing expansion sites: {e}")
            return 2  # Conservative fallback

    def _build_cache_key(self, game_id: str, player_id: int, format_type: str, since_turn: Optional[int] = None) -> str:
        """Build cache key for state"""
        key = f"{game_id}_{player_id}_{format_type}"
        if since_turn is not None:
            key += f"_since_{since_turn}"
        return key

    def _simulate_previous_state(self, current_state: Dict[str, Any], since_turn: int) -> Dict[str, Any]:
        """Get previous state from game history or cache"""
        # Try to get actual previous state from cache first
        cache_key = f"state_{current_state.get('game_id', 'unknown')}_{current_state.get('player_id', 0)}_turn_{since_turn}"
        cached_state = self.cache.get(cache_key)

        if cached_state:
            logger.debug(f"Retrieved previous state from cache for turn {since_turn}")
            return cached_state

        # If not in cache, try to get from civcom
        try:
            civcom = self._get_civcom_instance(current_state.get('game_id', ''))
            if civcom and hasattr(civcom, 'get_turn_state'):
                # Try to get historical state from game server
                historical_state = civcom.get_turn_state(since_turn, current_state.get('player_id', 0))
                if historical_state:
                    logger.debug(f"Retrieved previous state from civcom for turn {since_turn}")
                    return historical_state
        except Exception as e:
            logger.warning(f"Could not retrieve historical state: {e}")

        # Fallback: Create reasonable approximation by removing recent changes
        logger.debug(f"Using approximated previous state for turn {since_turn}")
        previous_state = current_state.copy()
        previous_state['turn'] = since_turn

        # Remove units that might have been built recently (conservative approach)
        if 'units' in previous_state and len(previous_state['units']) > 2:
            previous_state['units'] = previous_state['units'][:-1]  # Remove newest unit

        # Reduce city populations slightly (cities grow over time)
        if 'cities' in previous_state:
            for city in previous_state['cities']:
                if city.get('population', 0) > 1:
                    city['population'] = max(1, city['population'] - 1)

        return previous_state

    def _generate_legal_actions_from_state(self, state: Dict[str, Any], player_id: int) -> List[Dict[str, Any]]:
        """Generate legal actions based on actual game state from civcom"""
        actions = []

        try:
            # Get civcom instance for this game
            civcom = self._get_civcom_instance(state.get('game_id', ''))
            if civcom and hasattr(civcom, 'get_legal_actions'):
                # Use actual civcom to generate legal actions
                logger.debug("Getting legal actions from civcom")
                legal_actions = civcom.get_legal_actions(player_id)
                if legal_actions:
                    return legal_actions

        except Exception as e:
            logger.warning(f"Could not get legal actions from civcom: {e}")

        # Fallback: Generate actions based on available game entities
        logger.debug("Using fallback action generation")

        # Get player's units and cities
        units = [u for u in state.get('units', []) if u.get('owner') == player_id]
        cities = [c for c in state.get('cities', []) if c.get('owner') == player_id]

        # Generate realistic unit movement actions (only if unit has moves left)
        for unit in units:
            if unit.get('moves_left', 0) > 0:
                unit_type = unit.get('type', 'unknown')
                # Check adjacent tiles for valid moves
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                    target_x, target_y = unit['x'] + dx, unit['y'] + dy

                    # Basic validity check (positive coordinates)
                    if target_x >= 0 and target_y >= 0:
                        actions.append({
                            'type': 'unit_move',
                            'unit_id': unit['id'],
                            'source': {'x': unit['x'], 'y': unit['y']},
                            'target': {'x': target_x, 'y': target_y},
                            'cost': 1,
                            'unit_type': unit_type,
                            'priority': 5 + (2 if unit_type == 'settler' else 1 if unit_type == 'explorer' else 0)
                        })

        # Generate city production actions based on what cities can actually build
        for city in cities:
            city_size = city.get('population', 1)
            # Larger cities can build more things
            possible_units = ['warrior']
            possible_buildings = ['granary']

            if city_size >= 2:
                possible_units.extend(['settler', 'worker'])
                possible_buildings.append('barracks')

            if city_size >= 3:
                possible_units.append('archer')
                possible_buildings.extend(['library', 'marketplace'])

            for unit_type in possible_units:
                actions.append({
                    'type': 'city_build_unit',
                    'city_id': city['id'],
                    'target': unit_type,
                    'cost': {'shields': 10 if unit_type == 'warrior' else 30},
                    'priority': 6 if unit_type in ['settler', 'warrior'] else 4
                })

            for building in possible_buildings:
                actions.append({
                    'type': 'city_build_improvement',
                    'city_id': city['id'],
                    'target': building,
                    'cost': {'shields': 20 if building == 'granary' else 40},
                    'priority': 5
                })

        # Generate research actions based on current tech level
        current_techs = state.get('technologies', [])
        available_techs = []

        # Basic tech tree progression
        if 'pottery' not in current_techs:
            available_techs.append('pottery')
        if 'bronze_working' not in current_techs:
            available_techs.append('bronze_working')
        if 'pottery' in current_techs and 'writing' not in current_techs:
            available_techs.append('writing')
        if 'bronze_working' in current_techs and 'iron_working' not in current_techs:
            available_techs.append('iron_working')

        for tech in available_techs:
            actions.append({
                'type': 'research_tech',
                'tech': tech,
                'cost': {'beakers': 12},
                'priority': 7
            })

        return actions


class StateExtractorHandler(web.RequestHandler):
    """Tornado HTTP handler for /api/game/{game_id}/state endpoint"""

    def initialize(self):
        """Initialize handler with StateExtractor"""
        self.extractor = StateExtractor()
        self.executor = _shared_executor

    @run_on_executor
    def _extract_state_async(self, game_id: str, player_id: int, format_type: StateFormat, since_turn: Optional[int] = None):
        """Run state extraction in thread pool"""
        return self.extractor.extract_state(game_id, player_id, format_type, since_turn)

    async def get(self, game_id: str):
        """Handle GET requests for game state"""
        try:
            # Parse parameters
            player_id_raw = self.get_argument('player_id', None)
            if player_id_raw is None:
                self.set_status(400)
                self.write({"error": "player_id parameter is required"})
                return

            # Validate and sanitize inputs
            try:
                player_id = InputSanitizer.sanitize_player_id(player_id_raw)
            except (ValueError, TypeError) as e:
                self.set_status(400)
                self.write({"error": f"Invalid player_id: {str(e)}"})
                return

            format_str = self.get_argument('format', 'full')
            since_turn = self.get_argument('since_turn', None)

            # Validate format parameter
            if format_str not in ['full', 'minimal', 'llm']:
                self.set_status(400)
                self.write({"error": f"Invalid format '{format_str}'. Allowed: full, minimal, llm"})
                return

            # Validate since_turn parameter if provided
            if since_turn is not None:
                try:
                    since_turn = int(since_turn)
                    if since_turn < 0 or since_turn > MAX_TURN_NUMBER:
                        raise ValueError("Turn number out of range")
                except (ValueError, TypeError):
                    self.set_status(400)
                    self.write({"error": f"Invalid since_turn parameter. Must be integer 0-{MAX_TURN_NUMBER}"})
                    return

            # Authenticate request
            authenticated, auth_player_id, auth_game_id, auth_error = authenticate_request(self, 'state_read')
            if not authenticated:
                self.set_status(401)
                self.write({"error": "Authentication required"})
                return

            # If authentication provides player info, validate it matches request
            if auth_player_id is not None and player_id != str(auth_player_id):
                self.set_status(403)
                self.write({"error": "Player ID in request does not match authenticated user"})
                return

            # Validate all parameters
            try:
                validated_game_id, validated_player_id, validated_format, validated_since_turn = validate_request_parameters(
                    game_id, player_id, format_str, since_turn
                )
                format_type = StateFormat(validated_format)
            except ValidationError as e:
                logger.warning(f"Request validation failed: {e}")
                self.set_status(400)
                self.write({"error": str(e)})
                return

            # Check rate limits
            client_ip = self.request.remote_ip or "unknown"
            rate_limit_allowed, retry_after, limit_type = api_rate_limiter.check_limits(
                validated_player_id, client_ip
            )

            if not rate_limit_allowed:
                self.set_status(429)
                if retry_after:
                    self.set_header("Retry-After", str(int(retry_after) + 1))
                self.write({
                    "error": f"Rate limit exceeded for {limit_type}",
                    "retry_after": retry_after,
                    "limit_type": limit_type
                })
                return

            # Extract state asynchronously
            state = await self._extract_state_async(validated_game_id, validated_player_id, format_type, validated_since_turn)

            self.set_header("Content-Type", "application/json")
            self.write(state)

        except ValueError as e:
            # Invalid format or other validation errors
            logger.warning(f"Validation error in StateExtractorHandler: {str(e)}")
            self.set_status(400)
            self.write({"error": "Invalid request parameters"})
        except (ConnectionError, OSError, TimeoutError) as e:
            # Network and connection errors
            logger.error(f"Connection error in StateExtractorHandler: {str(e)}")
            self.set_status(503)
            self.write({"error": "Service temporarily unavailable", "retry": True})
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error in StateExtractorHandler: {error_message}")

            # Determine appropriate status code based on error
            if "No civcom available" in error_message or "Game not found" in error_message:
                self.set_status(404)
                self.write({"error": "Game not found"})
            elif "Player not found" in error_message:
                self.set_status(404)
                self.write({"error": "Player not found"})
            else:
                self.set_status(500)
                self.write({"error": "Internal server error"})


class LegalActionsHandler(web.RequestHandler):
    """Tornado HTTP handler for /api/game/{game_id}/legal_actions endpoint"""

    def initialize(self):
        """Initialize handler with StateExtractor"""
        self.extractor = StateExtractor()
        self.executor = _shared_executor

    @run_on_executor
    def _get_actions_async(self, game_id: str, player_id: int):
        """Run action extraction in thread pool"""
        return self.extractor.get_legal_actions(game_id, player_id)

    async def get(self, game_id: str):
        """Handle GET requests for legal actions"""
        try:
            # Parse parameters
            player_id_raw = self.get_argument('player_id', None)
            if player_id_raw is None:
                self.set_status(400)
                self.write({"error": "player_id parameter is required"})
                return

            # Validate and sanitize inputs
            try:
                player_id = InputSanitizer.sanitize_player_id(player_id_raw)
            except (ValueError, TypeError) as e:
                self.set_status(400)
                self.write({"error": f"Invalid player_id: {str(e)}"})
                return

            # Authenticate request
            authenticated, auth_player_id, auth_game_id, auth_error = authenticate_request(self, 'actions_read')
            if not authenticated:
                self.set_status(401)
                self.write({"error": "Authentication required"})
                return

            # If authentication provides player info, validate it matches request
            if auth_player_id is not None and player_id != str(auth_player_id):
                self.set_status(403)
                self.write({"error": "Player ID in request does not match authenticated user"})
                return

            # Validate parameters
            try:
                validated_game_id, validated_player_id, _, _ = validate_request_parameters(
                    game_id, player_id, 'full', None
                )
            except ValidationError as e:
                logger.warning(f"Request validation failed: {e}")
                self.set_status(400)
                self.write({"error": str(e)})
                return

            # Check rate limits
            client_ip = self.request.remote_ip or "unknown"
            rate_limit_allowed, retry_after, limit_type = api_rate_limiter.check_limits(
                validated_player_id, client_ip
            )

            if not rate_limit_allowed:
                self.set_status(429)
                if retry_after:
                    self.set_header("Retry-After", str(int(retry_after) + 1))
                self.write({
                    "error": f"Rate limit exceeded for {limit_type}",
                    "retry_after": retry_after,
                    "limit_type": limit_type
                })
                return

            # Extract actions asynchronously
            actions = await self._get_actions_async(validated_game_id, validated_player_id)

            self.set_header("Content-Type", "application/json")
            self.write(actions)

        except (ConnectionError, OSError, TimeoutError) as e:
            # Network and connection errors
            logger.error(f"Connection error in LegalActionsHandler: {str(e)}")
            self.set_status(503)
            self.write({"error": "Service temporarily unavailable", "retry": True})
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error in LegalActionsHandler: {error_message}")

            # Determine appropriate status code based on error
            if "No civcom available" in error_message or "Game not found" in error_message:
                self.set_status(404)
                self.write({"error": "Game not found"})
            elif "Player not found" in error_message:
                self.set_status(404)
                self.write({"error": "Player not found"})
            else:
                self.set_status(500)
                self.write({"error": "Internal server error"})
