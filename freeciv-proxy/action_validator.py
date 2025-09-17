#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Action validation system for LLM agents in FreeCiv proxy
Validates actions before forwarding to the C server
"""

import logging
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger("freeciv-proxy")

class ValidationResult:
    """Result of action validation"""
    def __init__(self, is_valid: bool, error_code: Optional[str] = None, error_message: Optional[str] = None):
        self.is_valid = is_valid
        self.error_code = error_code
        self.error_message = error_message

class ActionType(Enum):
    """Supported action types for LLM agents"""
    UNIT_MOVE = "unit_move"
    UNIT_BUILD_CITY = "unit_build_city"
    UNIT_EXPLORE = "unit_explore"
    CITY_PRODUCTION = "city_production"
    CITY_BUY = "city_buy"
    TECH_RESEARCH = "tech_research"
    TRADE_ROUTE = "trade_route"

class LLMActionValidator:
    """
    Validates LLM actions before forwarding to FreeCiv server
    Implements capability-based permissions and game rule validation
    """

    # Default capabilities for LLM agents
    DEFAULT_CAPABILITIES = [
        ActionType.UNIT_MOVE,
        ActionType.UNIT_BUILD_CITY,
        ActionType.UNIT_EXPLORE,
        ActionType.CITY_PRODUCTION,
        ActionType.TECH_RESEARCH
    ]

    # Restricted actions that require special permissions
    RESTRICTED_ACTIONS = [
        ActionType.CITY_BUY,
        ActionType.TRADE_ROUTE
    ]

    def __init__(self, capabilities: Optional[List[ActionType]] = None):
        self.capabilities = capabilities or self.DEFAULT_CAPABILITIES.copy()
        self.validation_stats = {
            'total_actions': 0,
            'valid_actions': 0,
            'invalid_actions': 0,
            'errors_by_type': {}
        }

    def validate_action(self, action: Dict[str, Any], player_id: int, game_state: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate an LLM action before forwarding to server

        Args:
            action: Action dictionary with type and parameters
            player_id: ID of the player making the action
            game_state: Current game state for context validation

        Returns:
            ValidationResult indicating if action is valid
        """
        self.validation_stats['total_actions'] += 1

        # Basic structure validation
        if not isinstance(action, dict):
            return self._validation_error('E001', 'Action must be a dictionary')

        if 'type' not in action:
            return self._validation_error('E002', 'Action must specify a type')

        action_type_str = action['type']

        # Convert string to enum
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            return self._validation_error('E003', f'Unknown action type: {action_type_str}')

        # Capability check
        if action_type not in self.capabilities:
            return self._validation_error('E004', f'Action type {action_type_str} not permitted for this agent')

        # Validate required player_id
        action_player_id = action.get('player_id', player_id)
        if action_player_id != player_id:
            return self._validation_error('E005', 'Action player_id does not match authenticated player')

        # Type-specific validation
        if action_type == ActionType.UNIT_MOVE:
            result = self._validate_unit_move(action, player_id, game_state)
        elif action_type == ActionType.UNIT_BUILD_CITY:
            result = self._validate_unit_build_city(action, player_id, game_state)
        elif action_type == ActionType.CITY_PRODUCTION:
            result = self._validate_city_production(action, player_id, game_state)
        elif action_type == ActionType.TECH_RESEARCH:
            result = self._validate_tech_research(action, player_id, game_state)
        else:
            # Default validation for other action types
            result = self._validate_basic_action(action, player_id, game_state)

        # Update statistics
        if result.is_valid:
            self.validation_stats['valid_actions'] += 1
        else:
            self.validation_stats['invalid_actions'] += 1
            error_type = result.error_code or 'unknown'
            self.validation_stats['errors_by_type'][error_type] = (
                self.validation_stats['errors_by_type'].get(error_type, 0) + 1
            )

        return result

    def _validate_unit_move(self, action: Dict[str, Any], player_id: int, game_state: Optional[Dict[str, Any]]) -> ValidationResult:
        """Validate unit movement action"""
        required_fields = ['unit_id', 'dest_x', 'dest_y']
        for field in required_fields:
            if field not in action:
                return self._validation_error('E010', f'Unit move requires {field}')

        unit_id = action['unit_id']
        dest_x = action['dest_x']
        dest_y = action['dest_y']

        # Validate coordinates are integers
        try:
            dest_x = int(dest_x)
            dest_y = int(dest_y)
        except (ValueError, TypeError):
            return self._validation_error('E011', 'Destination coordinates must be integers')

        # Enhanced coordinate validation against actual game boundaries
        if not self._validate_coordinates(dest_x, dest_y, game_state):
            return self._validation_error('E012', 'Destination coordinates out of game bounds')

        # If game state is available, validate unit ownership
        if game_state and 'units' in game_state:
            unit_found = False
            for unit in game_state['units']:
                if isinstance(unit, dict) and unit.get('id') == unit_id:
                    unit_found = True
                    if unit.get('owner') != player_id:
                        return self._validation_error('E013', 'Player does not own this unit')
                    break

            if not unit_found:
                return self._validation_error('E014', 'Unit not found or not visible')

        return ValidationResult(True)

    def _validate_unit_build_city(self, action: Dict[str, Any], player_id: int, game_state: Optional[Dict[str, Any]]) -> ValidationResult:
        """Validate city building action"""
        if 'unit_id' not in action:
            return self._validation_error('E020', 'Build city requires unit_id')

        unit_id = action['unit_id']

        # If game state is available, verify unit can build city (settler)
        if game_state and 'units' in game_state:
            for unit in game_state['units']:
                if isinstance(unit, dict) and unit.get('id') == unit_id:
                    if unit.get('owner') != player_id:
                        return self._validation_error('E021', 'Player does not own this unit')

                    # Check if unit type can build cities (should be settler)
                    unit_type = unit.get('type', '').lower()
                    if 'settler' not in unit_type and 'colonist' not in unit_type:
                        return self._validation_error('E022', 'Unit cannot build cities')
                    break
            else:
                return self._validation_error('E023', 'Unit not found')

        return ValidationResult(True)

    def _validate_city_production(self, action: Dict[str, Any], player_id: int, game_state: Optional[Dict[str, Any]]) -> ValidationResult:
        """Validate city production change"""
        required_fields = ['city_id', 'production_type']
        for field in required_fields:
            if field not in action:
                return self._validation_error('E030', f'City production requires {field}')

        city_id = action['city_id']
        production_type = action['production_type']

        # Validate production type is reasonable
        valid_production_types = [
            'warrior', 'settler', 'worker', 'archer', 'spearman',
            'barracks', 'granary', 'library', 'marketplace',
            'temple', 'aqueduct', 'walls'
        ]

        if production_type not in valid_production_types:
            return self._validation_error('E031', f'Invalid production type: {production_type}')

        # If game state available, verify city ownership
        if game_state and 'cities' in game_state:
            for city in game_state['cities']:
                if isinstance(city, dict) and city.get('id') == city_id:
                    if city.get('owner') != player_id:
                        return self._validation_error('E032', 'Player does not own this city')
                    break
            else:
                return self._validation_error('E033', 'City not found')

        return ValidationResult(True)

    def _validate_tech_research(self, action: Dict[str, Any], player_id: int, game_state: Optional[Dict[str, Any]]) -> ValidationResult:
        """Validate technology research action"""
        if 'tech_name' not in action:
            return self._validation_error('E040', 'Tech research requires tech_name')

        tech_name = action['tech_name']

        # Basic tech name validation (simplified)
        valid_techs = [
            'pottery', 'animal_husbandry', 'agriculture', 'mining',
            'bronze_working', 'the_wheel', 'writing', 'alphabet',
            'iron_working', 'mathematics', 'construction', 'currency'
        ]

        if tech_name.lower() not in valid_techs:
            return self._validation_error('E041', f'Invalid technology: {tech_name}')

        return ValidationResult(True)

    def _validate_basic_action(self, action: Dict[str, Any], player_id: int, game_state: Optional[Dict[str, Any]]) -> ValidationResult:
        """Basic validation for other action types"""
        # Ensure action has reasonable structure
        if len(action) > 20:  # Prevent overly complex actions
            return self._validation_error('E050', 'Action has too many parameters')

        return ValidationResult(True)

    def _validation_error(self, code: str, message: str) -> ValidationResult:
        """Create validation error result"""
        logger.warning(f"Action validation failed: {code} - {message}")
        return ValidationResult(False, code, message)

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return self.validation_stats.copy()

    def add_capability(self, action_type: ActionType):
        """Add capability for this validator"""
        if action_type not in self.capabilities:
            self.capabilities.append(action_type)

    def remove_capability(self, action_type: ActionType):
        """Remove capability for this validator"""
        if action_type in self.capabilities:
            self.capabilities.remove(action_type)

    def _validate_coordinates(self, x: int, y: int, game_state: Optional[Dict[str, Any]] = None) -> bool:
        """Enhanced coordinate validation against actual game boundaries"""
        if game_state and 'map_info' in game_state:
            map_info = game_state['map_info']
            max_x = map_info.get('width', 100)  # Default to reasonable bounds
            max_y = map_info.get('height', 100)

            if not (0 <= x < max_x and 0 <= y < max_y):
                return False
        else:
            # Fallback to more reasonable coordinate bounds when no game state
            # Accept coordinates from 0 to 200 (reasonable for most game maps)
            if not (0 <= x <= 200 and 0 <= y <= 200):
                return False

        return True