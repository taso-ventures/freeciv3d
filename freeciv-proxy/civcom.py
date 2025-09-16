# -*- coding: utf-8 -*-

'''
 Freeciv - Copyright (C) 2009-2014 - Andreas Røsdal   andrearo@pvv.ntnu.no
   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
'''

import socket
from struct import *
from threading import Thread
import logging
import time

HOST = '127.0.0.1'
logger = logging.getLogger("freeciv-proxy")

# The CivCom handles communication between freeciv-proxy and the Freeciv C
# server.


class CivCom(Thread):

    def __init__(self, username, civserverport, key, civwebserver):
        Thread.__init__(self)
        self.socket = None
        self.username = username
        self.civserverport = civserverport
        self.key = key
        self.send_buffer = []
        self.connect_time = time.time()
        self.civserver_messages = []
        self.stopped = False
        self.packet_size = -1
        self.net_buf = bytearray(0)
        self.header_buf = bytearray(0)
        self.daemon = True
        self.civwebserver = civwebserver

    def run(self):
        # setup connection to civserver
        if (logger.isEnabledFor(logging.INFO)):
            logger.info("Start connection to civserver for " + self.username)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setblocking(True)
        self.socket.settimeout(2)
        try:
            self.socket.connect((HOST, self.civserverport))
            self.socket.settimeout(0.01)
        except socket.error as reason:
            self.send_error_to_client(
                "Proxy unable to connect to civserver. Error: %s" %
                (reason))
            self.close_connection()
            return

        # send initial login packet to civserver
        self.civserver_messages = [self.civwebserver.loginpacket]
        self.send_packets_to_civserver()

        # receive packets from server
        while True:
            packet = self.read_from_connection()

            if (self.stopped):
                return

            if (packet is not None):
                self.net_buf += packet

                if (len(self.net_buf) == self.packet_size and self.net_buf[-1] == 0):
                    # valid packet received from freeciv server, send it to
                    # client.
                    self.send_buffer_append(self.net_buf[:-1])
                    self.packet_size = -1
                    self.net_buf = bytearray(0)
                    continue

            time.sleep(0.01)
            # prevent max CPU usage in case of error

    def read_from_connection(self):
        try:
            if (self.socket is not None and not self.stopped):
                if (self.packet_size == -1):
                    self.header_buf += self.socket.recv(2 -
                                                        len(self.header_buf))
                    if (len(self.header_buf) == 0):
                        self.close_connection()
                        return None
                    if (len(self.header_buf) == 2):
                        header_pck = unpack('>H', self.header_buf)
                        self.header_buf = bytearray(0)
                        self.packet_size = header_pck[0] - 2
                        if (self.packet_size <= 0 or self.packet_size > 32767):
                            logger.error("Invalid packet size " + str(self.packet_size))
                    else:
                        # complete header not read yet. return now, and read
                        # the rest next time.
                        return None

            if (self.socket is not None and self.net_buf is not None and self.packet_size > 0):
                data = self.socket.recv(self.packet_size - len(self.net_buf))
                if (len(data) == 0):
                    self.close_connection()
                    return None

                return data
        except socket.timeout:
            self.send_packets_to_client()
            self.send_packets_to_civserver()
            return None
        except OSError:
            return None

    def close_connection(self):
        if (logger.isEnabledFor(logging.INFO)):
            logger.info(
                "Server connection closed. Removing civcom thread for " +
                self.username)

        # Flush buffers
        self.send_packets_to_client()
        self.send_packets_to_civserver()

        if (hasattr(self.civwebserver, "civcoms") and self.key in list(self.civwebserver.civcoms.keys())):
            del self.civwebserver.civcoms[self.key]

        if (self.socket is not None):
            self.socket.close()
            self.socket = None
        self.civwebserver = None
        self.stopped = True

    # queue messages to be sent to client.
    def send_buffer_append(self, data):
        try:
            self.send_buffer.append(
                data.decode(
                    encoding="utf-8",
                    errors="ignore"))
        except UnicodeDecodeError:
            if (logger.isEnabledFor(logging.ERROR)):
                logger.error(
                    "Unable to decode string from civcom socket, for user: " +
                    self.username)
            return

    # sends packets to client (WebSockets client / browser)
    def send_packets_to_client(self):
        packet = self.get_client_result_string()
        if (packet is not None and self.civwebserver is not None):
            # Calls the write_message callback on the next Tornado I/O loop iteration (thread safely).
            conn = self.civwebserver
            conn.io_loop.add_callback(lambda: conn.write_message(packet))

    def get_client_result_string(self):
        result = ""
        try:
            if len(self.send_buffer) > 0:
                result = "[" + ",".join(self.send_buffer) + "]"
            else:
                result = None
        finally:
            del self.send_buffer[:]
        return result

    def send_error_to_client(self, message):
        if (logger.isEnabledFor(logging.ERROR)):
            logger.error(message)
        self.send_buffer_append(
            ("{\"pid\":25,\"event\":100,\"message\":\"" + message + "\"}").encode("utf-8"))

    # Send packets from freeciv-proxy to civserver
    def send_packets_to_civserver(self):
        if (self.civserver_messages is None or self.socket is None):
            return

        try:
            for net_message in self.civserver_messages:
                utf8_encoded = net_message.encode('utf-8')
                header = pack('>H', len(utf8_encoded) + 3)
                self.socket.sendall(
                    header +
                    utf8_encoded +
                    b'\0')
        except Exception:
            self.send_error_to_client(
                "Proxy unable to communicate with civserver on port " + str(self.civserverport))
        finally:
            self.civserver_messages = []

    # queue message for the civserver
    def queue_to_civserver(self, message):
        self.civserver_messages.append(message)

    # LLM-optimized state query methods
    def handle_state_query(self, player_id, format='full'):
        """Handle LLM state query requests with different formats"""
        if format == 'llm_optimized':
            return self.build_llm_optimized_state(player_id)
        elif format == 'delta':
            return self.get_state_delta(player_id)
        else:
            return self.get_full_state(player_id)

    def build_llm_optimized_state(self, player_id):
        """Build compressed state for LLMs (target < 4KB)"""
        # Basic game state structure for LLM consumption
        state = {
            'turn': getattr(self, 'game_turn', 1),
            'phase': getattr(self, 'game_phase', 'movement'),
            'player_id': player_id,
            'strategic': self._build_strategic_view(player_id),
            'tactical': self._build_tactical_view(player_id),
            'economic': self._build_economic_view(player_id),
            'legal_actions': self._get_legal_actions_optimized(player_id)
        }
        return state

    def _build_strategic_view(self, player_id):
        """Build strategic overview for LLM decision making"""
        return {
            'score': getattr(self, 'player_score', 0),
            'cities_count': len(getattr(self, 'player_cities', [])),
            'units_count': len(getattr(self, 'player_units', [])),
            'tech_level': getattr(self, 'tech_count', 0),
            'gold': getattr(self, 'player_gold', 0),
            'turn_progress': getattr(self, 'turn_progress', 'beginning')
        }

    def _build_tactical_view(self, player_id):
        """Build tactical view focusing on immediate unit/city actions"""
        tactical = {
            'active_units': [],
            'cities_needing_orders': [],
            'visible_threats': [],
            'exploration_targets': []
        }

        # Add simplified unit info (limit to 10 most important units)
        units = getattr(self, 'player_units', [])[:10]
        for unit in units:
            if isinstance(unit, dict):
                tactical['active_units'].append({
                    'id': unit.get('id'),
                    'type': unit.get('type'),
                    'x': unit.get('x'),
                    'y': unit.get('y'),
                    'moves_left': unit.get('moves_left', 0),
                    'can_act': unit.get('moves_left', 0) > 0
                })

        return tactical

    def _build_economic_view(self, player_id):
        """Build economic overview for resource management decisions"""
        return {
            'gold': getattr(self, 'player_gold', 0),
            'gold_per_turn': getattr(self, 'gold_income', 0),
            'research': getattr(self, 'research_progress', 0),
            'research_target': getattr(self, 'research_target', ''),
            'total_production': getattr(self, 'total_production', 0),
            'total_trade': getattr(self, 'total_trade', 0)
        }

    def _get_legal_actions_optimized(self, player_id):
        """Pre-compute and cache top legal actions for LLM"""
        # Simplified legal actions (would normally compute from game state)
        actions = []

        # Unit movement actions (most common)
        units = getattr(self, 'player_units', [])
        for unit in units[:5]:  # Limit to 5 units for size
            if isinstance(unit, dict) and unit.get('moves_left', 0) > 0:
                unit_id = unit.get('id')
                x, y = unit.get('x', 0), unit.get('y', 0)

                # Add movement options
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    actions.append({
                        'type': 'unit_move',
                        'unit_id': unit_id,
                        'dest_x': x + dx,
                        'dest_y': y + dy,
                        'priority': 'medium'
                    })

        # City production actions
        cities = getattr(self, 'player_cities', [])
        for city in cities[:3]:  # Limit to 3 cities
            if isinstance(city, dict):
                city_id = city.get('id')
                actions.append({
                    'type': 'city_production',
                    'city_id': city_id,
                    'production_type': 'warrior',
                    'priority': 'high'
                })

        # Research actions
        if not getattr(self, 'current_research', None):
            actions.append({
                'type': 'tech_research',
                'tech_name': 'pottery',
                'priority': 'high'
            })

        # Score and filter actions to top 20
        return self._score_and_filter_actions(actions, 20)

    def _score_and_filter_actions(self, actions, max_actions):
        """Score actions and return top N most important"""
        # Simple scoring based on priority
        priority_scores = {'high': 3, 'medium': 2, 'low': 1}

        scored_actions = []
        for action in actions:
            priority = action.get('priority', 'low')
            score = priority_scores.get(priority, 1)
            scored_actions.append((score, action))

        # Sort by score and return top actions
        scored_actions.sort(key=lambda x: x[0], reverse=True)
        return [action for score, action in scored_actions[:max_actions]]

    def get_full_state(self, player_id):
        """Get complete game state (larger format)"""
        return {
            'turn': getattr(self, 'game_turn', 1),
            'phase': getattr(self, 'game_phase', 'movement'),
            'player_id': player_id,
            'units': getattr(self, 'player_units', []),
            'cities': getattr(self, 'player_cities', []),
            'visible_tiles': getattr(self, 'visible_tiles', []),
            'players': getattr(self, 'all_players', {}),
            'techs': getattr(self, 'known_techs', []),
            'map_info': getattr(self, 'map_info', {})
        }

    def get_state_delta(self, player_id):
        """Get state changes since last query"""
        last_query_time = getattr(self, 'last_state_query_time', 0)
        current_time = time.time()

        delta = {
            'turn': getattr(self, 'game_turn', 1),
            'time_range': [last_query_time, current_time],
            'new_units': getattr(self, 'units_since_last_query', []),
            'moved_units': getattr(self, 'moved_units_since_last_query', []),
            'completed_production': getattr(self, 'completed_production_since_last_query', []),
            'tech_progress': getattr(self, 'tech_progress_since_last_query', {}),
            'gold_change': getattr(self, 'gold_change_since_last_query', 0)
        }

        # Update last query time
        self.last_state_query_time = current_time
        return delta
