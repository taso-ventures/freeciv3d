#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
State caching system for LLM agents in FreeCiv proxy
Provides TTL-based caching with size optimization
"""

import time
import json
import logging
import hmac
import hashlib
import os
import gzip
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("freeciv-proxy")

@dataclass
class CacheEntry:
    """Represents a cached state entry with metadata"""
    data: Dict[str, Any]
    timestamp: float
    size_bytes: int
    player_id: int
    cache_key: str = ""  # Original cache key
    signature: str = ""  # HMAC signature for integrity
    is_compressed: bool = False  # Whether data is compressed
    compressed_data: Optional[bytes] = None  # Compressed data if applicable

class StateCache:
    """
    In-memory state cache with TTL support for LLM agents
    Optimizes game state queries to meet < 4KB and < 50ms requirements
    """

    def __init__(self, ttl: int = 5, max_size_kb: int = 4, enable_compression: bool = True):
        self.ttl = ttl  # Time-to-live in seconds
        self.max_size_bytes = max_size_kb * 1024
        self.enable_compression = enable_compression
        self.cache: Dict[str, CacheEntry] = {}
        self.hit_count = 0
        self.miss_count = 0

        # Performance metrics
        self.compression_ratio_sum = 0.0
        self.compression_count = 0

        # HMAC secret for cache integrity
        self.hmac_secret = os.getenv('CACHE_HMAC_SECRET', 'default-secret-change-in-production')
        if self.hmac_secret == 'default-secret-change-in-production':
            logger.warning("Using default HMAC secret - set CACHE_HMAC_SECRET environment variable")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached state with TTL and integrity check"""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry.timestamp < self.ttl:
                # Verify cache integrity
                if self._verify_cache_integrity(entry):
                    self.hit_count += 1
                    logger.debug(f"Cache hit for key: {key}")
                    return entry.data
                else:
                    # Cache poisoning detected, remove entry
                    del self.cache[key]
                    logger.error(f"Cache integrity violation detected for key: {key}")
                    self.miss_count += 1
                    return None
            else:
                # TTL expired, remove entry
                del self.cache[key]
                logger.debug(f"Cache entry expired for key: {key}")

        self.miss_count += 1
        logger.debug(f"Cache miss for key: {key}")
        return None

    def set(self, key: str, data: Dict[str, Any], player_id: int) -> bool:
        """Set cache with size validation"""
        # Optimize data size first
        optimized = self.optimize_state_data(data)
        data_str = json.dumps(optimized, separators=(',', ':'))
        serialized_bytes = data_str.encode('utf-8')
        original_size = len(serialized_bytes)

        # Try compression if enabled and data is large enough
        compressed_data = None
        final_size = original_size

        if self.enable_compression and original_size > 1024:  # Only compress if > 1KB
            try:
                compressed_data = gzip.compress(serialized_bytes, compresslevel=6)
                compressed_size = len(compressed_data)

                # Use compression if it provides significant savings (>20%)
                if compressed_size < original_size * 0.8:
                    final_size = compressed_size
                    compression_ratio = original_size / compressed_size
                    self.compression_ratio_sum += compression_ratio
                    self.compression_count += 1
                    logger.debug(f"Compressed state: {original_size} -> {compressed_size} bytes (ratio: {compression_ratio:.2f})")
                else:
                    compressed_data = None  # Don't use compression
            except Exception as e:
                logger.warning(f"Compression failed: {e}")
                compressed_data = None

        if final_size > self.max_size_bytes:
            logger.warning(f"State too large for cache: {final_size} bytes (max: {self.max_size_bytes})")
            return False

        # Generate HMAC signature for integrity
        signature = self._generate_signature(optimized, player_id, key)

        # Store in cache with signature
        self.cache[key] = CacheEntry(
            data=optimized,
            timestamp=time.time(),
            size_bytes=size,
            player_id=player_id,
            cache_key=key,
            signature=signature
        )

        logger.debug(f"Cached state for key: {key}, size: {size} bytes")
        return True

    def invalidate(self, pattern: str = None, player_id: int = None):
        """Invalidate cache entries matching pattern or player"""
        keys_to_remove = []

        for key, entry in self.cache.items():
            should_remove = False

            if pattern and pattern in key:
                should_remove = True
            elif player_id and entry.player_id == player_id:
                should_remove = True

            if should_remove:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.cache[key]
            logger.debug(f"Invalidated cache entry: {key}")

    def optimize_state_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reduce state size for LLM consumption while preserving essential information
        Target: < 4KB optimized state
        """
        if not isinstance(data, dict):
            return data

        # Core game state for LLM decisions
        optimized = {
            'turn': data.get('turn', 0),
            'phase': data.get('phase', 'unknown'),
            'player_id': data.get('player_id'),
        }

        # Compress units - only essential info
        if 'units' in data and isinstance(data['units'], list):
            optimized['units'] = []
            for unit in data['units'][:10]:  # Limit to 10 most relevant units
                if isinstance(unit, dict):
                    optimized['units'].append({
                        'id': unit.get('id'),
                        'type': unit.get('type', '')[:8],  # Truncate type names
                        'x': unit.get('x'),
                        'y': unit.get('y'),
                        'owner': unit.get('owner'),
                        'moves': unit.get('moves_left', 0)
                    })

        # Compress cities - essential economic info
        if 'cities' in data and isinstance(data['cities'], list):
            optimized['cities'] = []
            for city in data['cities'][:5]:  # Limit to 5 cities
                if isinstance(city, dict):
                    optimized['cities'].append({
                        'id': city.get('id'),
                        'name': city.get('name', '')[:10],  # Truncate names
                        'x': city.get('x'),
                        'y': city.get('y'),
                        'owner': city.get('owner'),
                        'pop': city.get('population', 1)
                    })

        # Key player stats (limited)
        if 'players' in data:
            players_data = data['players']
            if isinstance(players_data, dict):
                optimized['players'] = {}
                # Only include up to 4 players to save space
                for i, (pid, pdata) in enumerate(players_data.items()):
                    if i >= 4:
                        break
                    if isinstance(pdata, dict):
                        optimized['players'][pid] = {
                            'name': pdata.get('name', '')[:8],  # Truncate names
                            'score': pdata.get('score', 0),
                            'gold': pdata.get('gold', 0)
                        }

        # Visible map tiles (highly compressed)
        if 'visible_tiles' in data and isinstance(data['visible_tiles'], list):
            # Only include tiles with strategic importance
            important_tiles = []
            for tile in data['visible_tiles']:
                if isinstance(tile, dict):
                    # Include only tiles with resources or cities (most strategic)
                    if tile.get('resource') or tile.get('city_id'):
                        important_tiles.append({
                            'x': tile.get('x'),
                            'y': tile.get('y'),
                            'terrain': tile.get('terrain', '')[:4],  # Truncate terrain
                            'resource': tile.get('resource', '')[:6] if tile.get('resource') else None
                        })
            optimized['visible_tiles'] = important_tiles[:20]  # Limit to 20 tiles

        return optimized

    def _generate_signature(self, data: Dict[str, Any], player_id: int, cache_key: str) -> str:
        """Generate HMAC signature for cache entry integrity"""
        # Create message to sign (data + metadata)
        message_parts = [
            json.dumps(data, sort_keys=True, separators=(',', ':')),
            str(player_id),
            cache_key,
            str(int(time.time() // 300))  # 5-minute time window for replay protection
        ]
        message = '|'.join(message_parts)

        # Generate HMAC signature
        signature = hmac.new(
            self.hmac_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return signature

    def _verify_cache_integrity(self, entry: CacheEntry) -> bool:
        """Verify cache entry integrity using HMAC"""
        if not entry.signature:
            # Allow entries without signatures (for backward compatibility)
            return True

        # Generate expected signature using the stored cache key
        expected_signature = self._generate_signature(entry.data, entry.player_id, entry.cache_key)

        # Compare signatures using constant-time comparison
        try:
            return hmac.compare_digest(entry.signature, expected_signature)
        except Exception as e:
            logger.error(f"Error verifying cache integrity: {e}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0

        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'total_size_bytes': sum(entry.size_bytes for entry in self.cache.values())
        }

    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        logger.info("Cache cleared")

# Global cache instance
state_cache = StateCache()