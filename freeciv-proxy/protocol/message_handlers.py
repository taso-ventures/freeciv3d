#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Message handlers for LLM WebSocket Protocol
Handles routing and processing of LLM messages
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Callable, Optional, Awaitable
try:
    from .llm_protocol import MessageType, LLMMessage, create_error_response, create_success_response
except ImportError:
    from llm_protocol import MessageType, LLMMessage, create_error_response, create_success_response

# Import validation utilities
try:
    from pydantic import ValidationError as PydanticValidationError
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

logger = logging.getLogger("freeciv-proxy")


class UnknownMessageTypeError(Exception):
    """Raised when an unknown message type is encountered"""
    def __init__(self, message_type: MessageType):
        super().__init__(f"Unknown message type: {message_type}")
        self.message_type = message_type


class MessageHandlerRegistry:
    """Registry for LLM message handlers with routing capabilities"""

    def __init__(self):
        self.handlers: Dict[MessageType, Callable] = {}
        self._register_handlers()

    def _register_handlers(self):
        """Register all message type handlers"""
        self.handlers[MessageType.LLM_CONNECT] = self.handle_connect
        self.handlers[MessageType.STATE_QUERY] = self.handle_state_query
        self.handlers[MessageType.ACTION] = self.handle_action
        self.handlers[MessageType.TURN_START] = self.handle_turn_start
        self.handlers[MessageType.TURN_END] = self.handle_turn_end
        self.handlers[MessageType.SPECTATOR_UPDATE] = self.handle_spectator_update
        self.handlers[MessageType.LLM_DISCONNECT] = self.handle_disconnect
        self.handlers[MessageType.ERROR] = self.handle_error

    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register a custom handler for a message type"""
        self.handlers[message_type] = handler

    async def handle_message(self, message: LLMMessage, connection) -> Dict[str, Any]:
        """Route message to appropriate handler"""
        try:
            handler = self.handlers.get(message.type)
            if handler is None:
                raise UnknownMessageTypeError(message.type)

            # Call the handler and return the result
            result = await handler(message, connection)
            return result

        except UnknownMessageTypeError:
            logger.error(f"Unknown message type: {message.type}")
            return create_error_response(
                message,
                "E001",
                f"Unknown message type: {message.type}"
            ).data

        except Exception as e:
            logger.exception(f"Error handling message {message.type}: {e}")
            return create_error_response(
                message,
                "E500",
                "Internal error processing message"
            ).data

    async def handle_connect(self, message: LLMMessage, connection) -> Dict[str, Any]:
        """Handle LLM_CONNECT message for agent authentication"""
        try:
            # Validate message data structure
            required_fields = ['api_token', 'model', 'game_id']
            validation_error = self._validate_message_data(message, required_fields)
            if validation_error:
                return create_error_response(
                    message,
                    "E101",
                    validation_error
                ).data

            # Extract and sanitize connection data
            api_token = self._sanitize_string(message.data.get('api_token', ''), 256)
            model = self._sanitize_string(message.data.get('model', ''), 50)
            game_id = self._sanitize_string(message.data.get('game_id', ''), 64)

            # Validate API token (integrate with existing auth system)
            if not self._validate_token(api_token):
                return create_error_response(
                    message,
                    "E102",
                    "Invalid API token"
                ).data

            # Initialize agent session
            session_id = await self._create_agent_session(message.agent_id, api_token, connection)
            player_id = await self._assign_player_id(message.agent_id, game_id)

            # Set connection properties
            connection.is_llm_agent = True
            connection.agent_id = message.agent_id
            connection.player_id = player_id
            connection.session_id = session_id

            # Connect to civserver if needed
            await self._connect_to_civserver(connection, game_id)

            logger.info(f"LLM agent authenticated: {message.agent_id} (player {player_id})")

            return create_success_response(message, {
                "type": "auth_success",
                "agent_id": message.agent_id,
                "session_id": session_id,
                "player_id": player_id,
                "model": model,
                "game_id": game_id
            }).data

        except Exception as e:
            logger.exception(f"Error in handle_connect: {e}")
            return create_error_response(
                message,
                "E103",
                f"Connection failed: {str(e)}"
            ).data

    async def handle_state_query(self, message: LLMMessage, connection) -> Dict[str, Any]:
        """Handle STATE_QUERY message for game state requests"""
        try:
            # Check authentication
            if not getattr(connection, 'is_llm_agent', False):
                return create_error_response(
                    message,
                    "E120",
                    "Not authenticated as LLM agent"
                ).data

            # Validate query data structure
            validation_error = self._validate_message_data(message, [])
            if validation_error:
                return create_error_response(
                    message,
                    "E121",
                    validation_error
                ).data

            # Extract and validate query parameters
            format_type = self._sanitize_string(message.data.get('format', 'llm_optimized'), 20)
            include_actions = message.data.get('include_actions', True)
            since_turn = message.data.get('since_turn')

            # Get game state
            state_data = await self._get_game_state(
                connection,
                format_type,
                include_actions,
                since_turn
            )

            return create_success_response(message, {
                "type": "state_response",
                "format": format_type,
                "data": state_data,
                "timestamp": time.time()
            }).data

        except Exception as e:
            logger.exception(f"Error in handle_state_query: {e}")
            return create_error_response(
                message,
                "E121",
                f"State query failed: {str(e)}"
            ).data

    async def handle_action(self, message: LLMMessage, connection) -> Dict[str, Any]:
        """Handle ACTION message for game actions"""
        try:
            # Check authentication
            if not getattr(connection, 'is_llm_agent', False):
                return create_error_response(
                    message,
                    "E130",
                    "Not authenticated as LLM agent"
                ).data

            # Validate action data structure
            required_fields = ['action_type']
            validation_error = self._validate_message_data(message, required_fields)
            if validation_error:
                return create_error_response(
                    message,
                    "E131",
                    validation_error
                ).data

            # Extract and sanitize action data
            action_type = self._sanitize_string(message.data.get('action_type', ''), 50)
            actor_id = message.data.get('actor_id')
            target = message.data.get('target')
            parameters = message.data.get('parameters', {})

            # Validate action
            is_valid = await self._validate_action(connection, message.data)
            if not is_valid:
                return create_error_response(
                    message,
                    "E132",
                    "Invalid action for current game state"
                ).data

            # Execute action
            result = await self._execute_action(connection, message.data)

            return create_success_response(message, {
                "type": "action_result",
                "success": result.get('success', True),
                "action_type": action_type,
                "result": result
            }).data

        except Exception as e:
            logger.exception(f"Error in handle_action: {e}")
            return create_error_response(
                message,
                "E133",
                f"Action failed: {str(e)}"
            ).data

    async def handle_turn_start(self, message: LLMMessage, connection) -> Dict[str, Any]:
        """Handle TURN_START message"""
        # Implementation for turn start notifications
        return {"type": "turn_start_ack", "agent_id": message.agent_id}

    async def handle_turn_end(self, message: LLMMessage, connection) -> Dict[str, Any]:
        """Handle TURN_END message"""
        # Implementation for turn end notifications
        return {"type": "turn_end_ack", "agent_id": message.agent_id}

    async def handle_spectator_update(self, message: LLMMessage, connection) -> Dict[str, Any]:
        """Handle SPECTATOR_UPDATE message"""
        # Implementation for spectator updates
        return {"type": "spectator_ack", "agent_id": message.agent_id}

    async def handle_disconnect(self, message: LLMMessage, connection) -> Dict[str, Any]:
        """Handle LLM_DISCONNECT message"""
        try:
            # Clean up connection resources
            await self._cleanup_connection(connection)

            return create_success_response(message, {
                "type": "disconnect_ack",
                "agent_id": message.agent_id
            }).data

        except Exception as e:
            logger.exception(f"Error in handle_disconnect: {e}")
            return create_error_response(
                message,
                "E140",
                f"Disconnect failed: {str(e)}"
            ).data

    # Validation methods

    def _validate_message_data(self, message: LLMMessage, expected_fields: list) -> Optional[str]:
        """Validate message data contains required fields"""
        for field in expected_fields:
            if field not in message.data:
                return f"Missing required field: {field}"

        # Additional validation based on message type
        if message.type == MessageType.LLM_CONNECT:
            return self._validate_connect_data(message.data)
        elif message.type == MessageType.STATE_QUERY:
            return self._validate_state_query_data(message.data)
        elif message.type == MessageType.ACTION:
            return self._validate_action_data(message.data)

        return None

    def _validate_connect_data(self, data: Dict[str, Any]) -> Optional[str]:
        """Validate LLM_CONNECT message data"""
        api_token = data.get('api_token', '')
        if len(api_token) < 16:
            return "API token too short (minimum 16 characters)"

        import re
        if not re.match(r'^[a-zA-Z0-9\-_]+$', api_token):
            return "API token contains invalid characters"

        game_id = data.get('game_id', '')
        if not game_id or len(game_id) > 64:
            return "Invalid game_id length"

        if not re.match(r'^[a-zA-Z0-9\-_]+$', game_id):
            return "Game ID contains invalid characters"

        model = data.get('model', '')
        if not model or len(model) > 50:
            return "Invalid model name"

        return None

    def _validate_state_query_data(self, data: Dict[str, Any]) -> Optional[str]:
        """Validate STATE_QUERY message data"""
        format_type = data.get('format', 'llm_optimized')
        valid_formats = ['full', 'delta', 'llm_optimized']
        if format_type not in valid_formats:
            return f"Invalid format: {format_type}. Must be one of {valid_formats}"

        since_turn = data.get('since_turn')
        if since_turn is not None and (not isinstance(since_turn, int) or since_turn < 0):
            return "since_turn must be a non-negative integer"

        return None

    def _validate_action_data(self, data: Dict[str, Any]) -> Optional[str]:
        """Validate ACTION message data"""
        action_type = data.get('action_type')
        if not action_type:
            return "Missing action_type"

        valid_actions = [
            'unit_move', 'unit_attack', 'unit_build_city', 'unit_explore',
            'city_production', 'city_build_unit', 'city_build_improvement',
            'tech_research', 'diplomacy_message', 'end_turn'
        ]
        if action_type not in valid_actions:
            return f"Invalid action_type: {action_type}"

        # Validate required fields based on action type
        if action_type in ['unit_move', 'unit_attack']:
            if 'actor_id' not in data:
                return "Missing actor_id for unit action"
            if 'target' not in data:
                return "Missing target for unit action"

            target = data['target']
            if not isinstance(target, dict) or 'x' not in target or 'y' not in target:
                return "Invalid target coordinates"

            try:
                x, y = int(target['x']), int(target['y'])
                if x < 0 or y < 0 or x > 9999 or y > 9999:
                    return "Coordinates out of valid range"
            except (ValueError, TypeError):
                return "Invalid coordinate values"

        elif action_type in ['city_production', 'city_build_unit']:
            if 'actor_id' not in data:
                return "Missing actor_id for city action"
            if 'target' not in data:
                return "Missing target for city action"

        elif action_type == 'tech_research':
            if 'target' not in data:
                return "Missing target technology name"

        return None

    def _sanitize_string(self, value: str, max_length: int = 256) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            return ""

        # Remove control characters and null bytes
        import re
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)

        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        return sanitized.strip()

    # Helper methods for handler implementation

    def _validate_token(self, api_token: str) -> bool:
        """Validate API token - integrate with existing auth system"""
        if not api_token:
            return False

        try:
            # Import and use existing token validation
            from config_loader import llm_config
            return llm_config.validate_token(api_token)
        except ImportError as e:
            logger.warning(f"Token validation module not available: {e}")
            # Basic validation - minimum length and format check
            if len(api_token) < 16:
                return False
            # Check if it's a valid format (alphanumeric + hyphens)
            import re
            return bool(re.match(r'^[a-zA-Z0-9\-_]{16,}$', api_token))
        except Exception as e:
            logger.error(f"Error validating API token: {e}")
            return False

    async def _create_agent_session(self, agent_id: str, api_token: str, connection) -> str:
        """Create agent session - integrate with existing session management"""
        try:
            from session_manager import session_manager
            session_info = session_manager.create_session(agent_id, api_token, set())
            return session_info.session_id if session_info else f"session-{agent_id}"
        except ImportError as e:
            logger.warning(f"Session manager not available: {e}")
            # Generate a unique session ID with timestamp and random component
            import uuid
            session_id = f"session-{agent_id}-{int(time.time())}-{str(uuid.uuid4())[:8]}"
            logger.info(f"Created fallback session for agent {agent_id}: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Error creating session for {agent_id}: {e}")
            # Return a basic session ID as last resort
            return f"session-{agent_id}-{int(time.time())}"

    async def _assign_player_id(self, agent_id: str, game_id: str) -> int:
        """Assign player ID for agent"""
        # Simple assignment - in production this would integrate with game management
        return abs(hash(agent_id)) % 8 + 1  # Player IDs 1-8

    async def _connect_to_civserver(self, connection, game_id: str):
        """Connect to civserver - integrate with existing civcom"""
        try:
            if hasattr(connection, 'civcom') and connection.civcom:
                return  # Already connected

            # Create civcom connection (simplified)
            from civcom import CivCom
            civserver_port = 6001  # Default port

            connection.civcom = CivCom(
                connection.agent_id,
                civserver_port,
                f"{connection.agent_id}_{connection.id}",
                connection
            )
            connection.civcom.start()

        except Exception as e:
            logger.warning(f"Failed to connect to civserver: {e}")

    async def _get_game_state(self, connection, format_type: str, include_actions: bool, since_turn: Optional[int]) -> Dict[str, Any]:
        """Get game state in requested format"""
        try:
            # Use existing state extraction logic
            if hasattr(connection, 'civcom') and connection.civcom:
                raw_state = connection.civcom.get_full_state(connection.player_id)
            else:
                # Fallback state
                raw_state = {
                    "turn": 1,
                    "phase": "movement",
                    "units": [],
                    "cities": [],
                    "players": {}
                }

            # Format state based on requested type
            if format_type == "llm_optimized":
                return self._format_llm_optimized_state(raw_state, connection.player_id)
            elif format_type == "full":
                return self._format_full_state(raw_state, connection.player_id)
            else:
                return raw_state

        except Exception as e:
            logger.warning(f"Error getting game state: {e}")
            return {"error": str(e)}

    def _format_llm_optimized_state(self, raw_state: Dict[str, Any], player_id: int) -> Dict[str, Any]:
        """Format state for LLM consumption - integrate with existing logic"""
        return {
            "turn": raw_state.get("turn", 1),
            "phase": raw_state.get("phase", "movement"),
            "strategic_summary": {
                "cities_count": len(raw_state.get("cities", [])),
                "units_count": len(raw_state.get("units", [])),
                "tech_progress": "early"
            },
            "immediate_priorities": ["explore", "build_city", "research"],
            "threats": [],
            "opportunities": []
        }

    def _format_full_state(self, raw_state: Dict[str, Any], player_id: int) -> Dict[str, Any]:
        """Format full state"""
        return {
            **raw_state,
            "player_perspective": player_id,
            "timestamp": time.time()
        }

    async def _validate_action(self, connection, action_data: Dict[str, Any]) -> bool:
        """Validate action - integrate with existing validation"""
        if not action_data or 'action_type' not in action_data:
            return False

        try:
            from action_validator import LLMActionValidator
            if hasattr(connection, 'action_validator') and connection.action_validator:
                validation_result = connection.action_validator.validate_action(
                    action_data, connection.player_id, None
                )
                return validation_result.is_valid
        except ImportError as e:
            logger.warning(f"Action validator module not available: {e}")
        except Exception as e:
            logger.error(f"Error validating action: {e}")
            return False

        # Basic validation with more comprehensive checks
        action_type = action_data.get('action_type')
        valid_actions = ['unit_move', 'unit_attack', 'unit_build_city', 'city_production',
                        'city_build_unit', 'tech_research', 'end_turn']

        if action_type not in valid_actions:
            return False

        # Validate required fields based on action type
        if action_type in ['unit_move', 'unit_attack']:
            return 'actor_id' in action_data and 'target' in action_data
        elif action_type in ['city_production', 'city_build_unit']:
            return 'actor_id' in action_data and 'target' in action_data
        elif action_type == 'tech_research':
            return 'target' in action_data

        return True

    async def _execute_action(self, connection, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action - integrate with existing civcom"""
        try:
            if hasattr(connection, 'civcom') and connection.civcom:
                # Convert to FreeCiv packet format
                packet = self._convert_action_to_packet(action_data, connection.player_id)
                connection.civcom.queue_to_civserver(json.dumps(packet))
                return {"success": True, "executed": True}
        except Exception as e:
            logger.warning(f"Error executing action: {e}")

        return {"success": False, "error": "Action execution failed"}

    def _convert_action_to_packet(self, action_data: Dict[str, Any], player_id: int) -> Dict[str, Any]:
        """Convert action to FreeCiv packet format"""
        action_type = action_data.get('action_type')

        if action_type == 'unit_move':
            return {
                'pid': 31,  # PACKET_UNIT_ORDERS
                'unit_id': action_data.get('actor_id'),
                'dest_x': action_data.get('target', {}).get('x'),
                'dest_y': action_data.get('target', {}).get('y')
            }
        elif action_type == 'city_production':
            return {
                'pid': 45,  # City production packet
                'city_id': action_data.get('actor_id'),
                'production_type': action_data.get('target')
            }
        elif action_type == 'tech_research':
            return {
                'pid': 50,  # Research packet
                'tech_name': action_data.get('target'),
                'player_id': player_id
            }

        return action_data  # Fallback

    async def _cleanup_connection(self, connection):
        """Clean up connection resources"""
        try:
            if hasattr(connection, 'civcom') and connection.civcom:
                connection.civcom.stopped = True
                connection.civcom.close_connection()

            if hasattr(connection, 'session_id') and connection.session_id:
                try:
                    from session_manager import session_manager
                    session_manager.terminate_session(connection.session_id, "disconnect")
                    logger.info(f"Session {connection.session_id} terminated")
                except ImportError as e:
                    logger.warning(f"Session manager not available for cleanup: {e}")
                    logger.info(f"Session cleanup skipped for {connection.session_id}")
                except Exception as e:
                    logger.error(f"Error terminating session {connection.session_id}: {e}")

        except Exception as e:
            logger.warning(f"Error during connection cleanup: {e}")

    async def handle_error(self, message: LLMMessage, connection) -> Dict[str, Any]:
        """Handle error messages (mainly for logging and acknowledgment)"""
        try:
            logger.info(f"Error message received from {message.agent_id}: {message.data}")

            # Log the error details
            error_code = message.data.get("error_code", "UNKNOWN")
            error_message = message.data.get("error_message", "No details provided")
            logger.warning(f"Agent {message.agent_id} reported error {error_code}: {error_message}")

            # Acknowledge the error message
            return {
                "type": "error_acknowledged",
                "agent_id": message.agent_id,
                "timestamp": time.time(),
                "data": {
                    "success": True,
                    "message": "Error received and logged"
                },
                "correlation_id": message.correlation_id
            }

        except Exception as e:
            logger.error(f"Error handling error message from {message.agent_id}: {e}")
            return {
                "type": "error",
                "agent_id": message.agent_id,
                "timestamp": time.time(),
                "data": {
                    "success": False,
                    "error_code": "E500",
                    "error_message": "Internal error processing error message"
                },
                "correlation_id": message.correlation_id
            }