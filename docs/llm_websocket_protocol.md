# LLM WebSocket Protocol Specification

## Overview

This document specifies the WebSocket protocol for LLM agent communication between Game Arena and FreeCiv3D. The protocol enables reliable, bidirectional communication for LLM-driven gameplay.

**Version**: 1.0.0
**Date**: September 18, 2025

## Architecture

```
Game Arena → LLM Gateway (port 8003) → FreeCiv Proxy (port 8002) → FreeCiv Server
```

## Connection Flow

1. **Agent Connection**: LLM agent connects to `/ws/agent/{agent_id}` on LLM Gateway
2. **Authentication**: Agent sends `llm_connect` message with API token
3. **Gateway Registration**: Gateway registers agent and connects to FreeCiv proxy
4. **Game Communication**: Agent can send state queries and actions
5. **Disconnection**: Graceful cleanup when agent disconnects

## Message Format

All messages follow this structure:

```json
{
  "type": "message_type",
  "agent_id": "agent_identifier",
  "timestamp": 1234567890.123,
  "data": { /* message-specific data */ },
  "correlation_id": "optional_correlation_id"
}
```

### Required Fields
- `type`: Message type (see Message Types below)
- `agent_id`: Unique identifier for the LLM agent
- `timestamp`: Unix timestamp with millisecond precision
- `data`: Message payload (structure varies by type)

### Optional Fields
- `correlation_id`: For request/response correlation in async operations

## Message Types

### 1. Connection Management

#### LLM_CONNECT (Request)
```json
{
  "type": "llm_connect",
  "agent_id": "my-agent",
  "timestamp": 1234567890.123,
  "data": {
    "api_token": "secret_api_key",
    "model": "gpt-4",
    "game_id": "game-123",
    "capabilities": ["move", "build", "research"]
  }
}
```

#### AUTH_SUCCESS (Response)
```json
{
  "type": "llm_connect",
  "agent_id": "my-agent",
  "timestamp": 1234567890.124,
  "data": {
    "type": "auth_success",
    "success": true,
    "session_id": "session-456",
    "player_id": 1,
    "game_id": "game-123",
    "session_expires_in": 3600
  }
}
```

### 2. Game State Queries

#### STATE_QUERY (Request)
```json
{
  "type": "state_query",
  "agent_id": "my-agent",
  "timestamp": 1234567890.123,
  "correlation_id": "query-789",
  "data": {
    "format": "llm_optimized",
    "include_legal_actions": true,
    "since_turn": 10
  }
}
```

**Format Options:**
- `full`: Complete game state with all details
- `delta`: Changes since last query
- `llm_optimized`: Compressed state optimized for LLM consumption

#### STATE_UPDATE (Response)
```json
{
  "type": "state_update",
  "agent_id": "my-agent",
  "timestamp": 1234567890.125,
  "correlation_id": "query-789",
  "data": {
    "type": "state_response",
    "format": "llm_optimized",
    "data": {
      "turn": 15,
      "phase": "movement",
      "strategic_summary": {
        "cities_count": 3,
        "units_count": 8,
        "tech_progress": "developing"
      },
      "immediate_priorities": ["explore", "build_military"],
      "threats": [],
      "opportunities": [
        {
          "type": "expansion",
          "description": "Good settlement location at (25, 30)",
          "priority": "high"
        }
      ],
      "legal_actions": [
        {
          "type": "unit_move",
          "unit_id": 42,
          "target": {"x": 11, "y": 21},
          "priority": "medium"
        }
      ]
    }
  }
}
```

### 3. Action Submission

#### ACTION (Request)
```json
{
  "type": "action",
  "agent_id": "my-agent",
  "timestamp": 1234567890.123,
  "correlation_id": "action-456",
  "data": {
    "action_type": "unit_move",
    "actor_id": 42,
    "target": {"x": 11, "y": 21},
    "parameters": {"validate": true}
  }
}
```

**Action Types:**
- `unit_move`: Move unit to coordinates
- `unit_attack`: Attack target unit/city
- `unit_build_city`: Build city at current location
- `unit_explore`: Set unit to auto-explore
- `city_production`: Set city production
- `city_build_unit`: Build specific unit type
- `city_build_improvement`: Build city improvement
- `tech_research`: Research technology
- `diplomacy_message`: Send diplomatic message
- `end_turn`: End current turn

#### ACTION_RESULT (Response)
```json
{
  "type": "action_result",
  "agent_id": "my-agent",
  "timestamp": 1234567890.126,
  "correlation_id": "action-456",
  "data": {
    "type": "action_result",
    "success": true,
    "action_type": "unit_move",
    "result": {
      "action_id": "exec-789",
      "state_change": {
        "unit_moved": true,
        "new_position": {"x": 11, "y": 21}
      }
    }
  }
}
```

### 4. Turn Management

#### TURN_START
```json
{
  "type": "turn_start",
  "agent_id": "my-agent",
  "timestamp": 1234567890.123,
  "data": {
    "turn": 16,
    "phase": "movement",
    "time_limit": 300,
    "active_player": 1
  }
}
```

#### TURN_END
```json
{
  "type": "turn_end",
  "agent_id": "my-agent",
  "timestamp": 1234567890.123,
  "data": {
    "turn": 15,
    "next_player": 2,
    "turn_summary": {
      "actions_taken": 5,
      "units_moved": 3
    }
  }
}
```

### 5. Heartbeat

#### PING
```json
{
  "type": "ping",
  "agent_id": "my-agent",
  "timestamp": 1234567890.123,
  "data": {}
}
```

#### PONG (Response)
```json
{
  "type": "pong",
  "agent_id": "my-agent",
  "timestamp": 1234567890.124
}
```

### 6. Error Handling

#### ERROR (Response)
```json
{
  "type": "error",
  "agent_id": "my-agent",
  "timestamp": 1234567890.123,
  "correlation_id": "original-request-id",
  "data": {
    "type": "error",
    "success": false,
    "error_code": "E102",
    "error_message": "Invalid API token",
    "details": {
      "retry_after": 60
    }
  }
}
```

**Error Codes:**
- `E101`: Missing required field
- `E102`: Invalid API token
- `E103`: Unknown message type
- `E120`: Not authenticated
- `E121`: State query failed
- `E130`: Action validation failed
- `E131`: Action execution failed
- `E500`: Internal server error

## Connection Management

### Endpoints

#### LLM Gateway (port 8003)
- **Agent WebSocket**: `ws://localhost:8003/ws/agent/{agent_id}`
- **Spectator WebSocket**: `ws://localhost:8003/ws/spectator/{game_id}`
- **REST API**: `http://localhost:8003/api/`

#### FreeCiv Proxy (port 8002)
- **LLM WebSocket**: `ws://localhost:8002/llmsocket/8002`

### Authentication

1. **API Token**: Required in `llm_connect` message
2. **Session Management**: Gateway maintains sessions with expiration
3. **Player Assignment**: Gateway assigns player ID upon authentication

### Connection Limits

- **Max Connections per Agent**: 2
- **Max Concurrent Games**: 10 (configurable)
- **Session Timeout**: 120 seconds (configurable)
- **Heartbeat Interval**: 30 seconds

## State Formats

### LLM Optimized Format

Compressed state designed for LLM consumption:

```json
{
  "turn": 15,
  "phase": "movement",
  "strategic_summary": {
    "cities_count": 3,
    "units_count": 8,
    "tech_progress": "developing",
    "military_strength": "medium"
  },
  "immediate_priorities": [
    "explore_nearby_areas",
    "build_military_units",
    "research_bronze_working"
  ],
  "threats": [
    {
      "type": "military",
      "description": "Enemy units near Capital",
      "severity": "high",
      "location": {"x": 10, "y": 20}
    }
  ],
  "opportunities": [
    {
      "type": "expansion",
      "description": "Resource tiles available",
      "priority": "high",
      "locations": [{"x": 25, "y": 30, "resource": "wheat"}]
    }
  ]
}
```

### Full Format

Complete game state with all details:

```json
{
  "turn": 15,
  "phase": "movement",
  "player_id": 1,
  "units": [
    {
      "id": 42,
      "type": "warrior",
      "x": 10, "y": 20,
      "moves_left": 1,
      "hp": 10,
      "owner": 1
    }
  ],
  "cities": [
    {
      "id": 1,
      "name": "Capital",
      "x": 15, "y": 25,
      "population": 3,
      "production": "warrior",
      "owner": 1
    }
  ],
  "visible_tiles": [
    {
      "x": 10, "y": 20,
      "terrain": "grassland",
      "resource": "wheat"
    }
  ],
  "players": {
    "1": {"name": "Player 1", "score": 150},
    "2": {"name": "Player 2", "score": 120}
  },
  "technologies": ["pottery", "animal_husbandry"]
}
```

## Error Handling

### Reconnection Strategy

1. **Automatic Reconnection**: Gateway attempts reconnection on connection loss
2. **Exponential Backoff**: Increasing delays between reconnection attempts
3. **State Preservation**: Session state maintained during brief disconnections

### Validation

1. **Message Validation**: JSON schema validation for all messages
2. **Action Validation**: Game rule validation before execution
3. **Authentication**: Token and session validation for all requests

### Rate Limiting

- **Messages per minute**: 100 (configurable)
- **Burst limit**: 20 messages
- **Action rate limit**: 30 actions per minute

## Usage Examples

### Basic Agent Connection

```python
import asyncio
import websockets
import json

async def connect_agent():
    uri = "ws://localhost:8003/ws/agent/my-agent"

    async with websockets.connect(uri) as websocket:
        # Receive welcome message
        welcome = await websocket.recv()
        print(f"Welcome: {welcome}")

        # Authenticate
        auth_message = {
            "type": "llm_connect",
            "agent_id": "my-agent",
            "timestamp": time.time(),
            "data": {
                "api_token": "your_api_token_here",
                "model": "gpt-4",
                "game_id": "game-123"
            }
        }

        await websocket.send(json.dumps(auth_message))
        auth_response = await websocket.recv()
        print(f"Auth: {auth_response}")

        # Query game state
        state_query = {
            "type": "state_query",
            "agent_id": "my-agent",
            "timestamp": time.time(),
            "correlation_id": "query-1",
            "data": {
                "format": "llm_optimized",
                "include_legal_actions": True
            }
        }

        await websocket.send(json.dumps(state_query))
        state_response = await websocket.recv()
        print(f"State: {state_response}")

asyncio.run(connect_agent())
```

### Action Submission

```python
async def submit_action(websocket):
    action = {
        "type": "action",
        "agent_id": "my-agent",
        "timestamp": time.time(),
        "correlation_id": "action-1",
        "data": {
            "action_type": "unit_move",
            "actor_id": 42,
            "target": {"x": 11, "y": 21}
        }
    }

    await websocket.send(json.dumps(action))
    result = await websocket.recv()
    print(f"Action result: {result}")
```

## Security Considerations

1. **API Token Authentication**: Required for all agent connections
2. **Origin Validation**: WebSocket origin checking for browser clients
3. **Input Sanitization**: All inputs validated and sanitized
4. **Rate Limiting**: Protection against abuse and DoS attacks
5. **Session Management**: Secure session handling with expiration

## Performance Guidelines

1. **Message Size**: Keep messages under 1MB for optimal performance
2. **Query Frequency**: Limit state queries to essential updates
3. **Batch Actions**: Use batch endpoints for multiple actions
4. **Connection Reuse**: Maintain persistent connections when possible

## Monitoring and Debugging

### Health Check Endpoint
```http
GET http://localhost:8003/health
```

### Metrics Endpoint
```http
GET http://localhost:8003/api/metrics
```

### Log Levels
- `DEBUG`: Detailed protocol messages
- `INFO`: Connection events and state changes
- `WARN`: Validation failures and retries
- `ERROR`: Connection failures and system errors

## Version History

- **1.0.0** (Sept 2025): Initial protocol specification
  - Basic message types for connection, state, and actions
  - WebSocket transport with JSON messaging
  - Authentication and session management
  - Error handling and rate limiting