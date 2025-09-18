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
import math
from typing import Dict, Any, Optional, OrderedDict
from dataclasses import dataclass
from collections import OrderedDict

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
    last_accessed: float = 0.0  # For LRU tracking

class StateCache:
    """
    In-memory state cache with TTL support for LLM agents
    Optimizes game state queries to meet < 4KB and < 50ms requirements
    """

    def __init__(self, ttl: int = 5, max_size_kb: int = 4, enable_compression: bool = True,
                 max_cache_size_mb: int = 100, max_entries: int = 1000):
        self.ttl = ttl  # Time-to-live in seconds
        self.max_size_bytes = max_size_kb * 1024  # Max size per entry
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024  # Max total cache size
        self.max_entries = max_entries  # Maximum number of entries
        self.enable_compression = enable_compression

        # Configurable compression settings
        self.compression_threshold = int(os.getenv('CACHE_COMPRESSION_THRESHOLD', '1024'))  # bytes
        self.compression_level = int(os.getenv('CACHE_COMPRESSION_LEVEL', '6'))  # 1-9
        self.compression_ratio_threshold = float(os.getenv('CACHE_COMPRESSION_RATIO_THRESHOLD', '0.8'))  # 0.0-1.0

        # Use OrderedDict for efficient LRU operations
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()

        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0

        # Performance metrics
        self.compression_ratio_sum = 0.0
        self.compression_count = 0

        # HMAC secret for cache integrity - required for security
        self.hmac_secret = os.getenv('CACHE_HMAC_SECRET')
        if not self.hmac_secret:
            raise ValueError(
                "CACHE_HMAC_SECRET environment variable must be set for cache integrity. "
                "Generate with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        if len(self.hmac_secret) < 64:
            raise ValueError(
                "CACHE_HMAC_SECRET must be at least 64 characters long for security (512 bits). "
                "Generate with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )

        # Check for weak secrets using Shannon entropy
        entropy = self._calculate_shannon_entropy(self.hmac_secret)
        min_entropy = 3.5  # Minimum bits per character (hex = ~3.88, mixed case = ~5.95)
        if entropy < min_entropy:
            raise ValueError(
                f"CACHE_HMAC_SECRET has insufficient entropy: {entropy:.2f} bits/char (minimum: {min_entropy}). "
                "Use a cryptographically secure random string. "
                "Generate with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached state with TTL and integrity check, updates LRU order"""
        if key in self.cache:
            entry = self.cache[key]
            current_time = time.time()

            if current_time - entry.timestamp < self.ttl:
                # Verify cache integrity
                if self._verify_cache_integrity(entry):
                    self.hit_count += 1
                    entry.last_accessed = current_time

                    # Move to end for LRU (most recently used)
                    self.cache.move_to_end(key)

                    logger.debug(f"Cache hit for key: {key}")

                    # Handle decompression if needed
                    if entry.is_compressed and entry.compressed_data:
                        try:
                            decompressed_bytes = gzip.decompress(entry.compressed_data)
                            return json.loads(decompressed_bytes.decode('utf-8'))
                        except Exception as e:
                            logger.error(f"Decompression failed for key {key}: {e}")
                            del self.cache[key]
                            return None
                    
                    return entry.data
                
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

        if self.enable_compression and original_size > self.compression_threshold:
            try:
                compressed_data = gzip.compress(serialized_bytes, compresslevel=self.compression_level)
                compressed_size = len(compressed_data)

                # Use compression if it provides significant savings
                if compressed_size < original_size * self.compression_ratio_threshold:
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

        # Check if we need to evict entries before adding
        self._ensure_cache_capacity(final_size)

        # Generate HMAC signature for integrity
        signature = self._generate_signature(optimized, player_id, key)
        current_time = time.time()

        # Store in cache with signature
        entry = CacheEntry(
            data=optimized if compressed_data is None else None,  # Store original data only if not compressed
            timestamp=current_time,
            size_bytes=final_size,
            player_id=player_id,
            cache_key=key,
            signature=signature,
            is_compressed=compressed_data is not None,
            compressed_data=compressed_data,
            last_accessed=current_time
        )

        self.cache[key] = entry
        # Move to end (most recently used)
        self.cache.move_to_end(key)

        logger.debug(f"Cached state for key: {key}, size: {final_size} bytes")
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

    def _ensure_cache_capacity(self, new_entry_size: int):
        """
        Ensure cache has capacity for new entry, evicting LRU entries if needed
        """
        current_size = self._get_total_cache_size()

        # Check if we need to evict based on total size or entry count
        while (len(self.cache) >= self.max_entries or
               current_size + new_entry_size > self.max_cache_size_bytes):

            if not self.cache:
                break  # No entries to evict

            # Remove least recently used entry (first in OrderedDict)
            lru_key, lru_entry = self.cache.popitem(last=False)
            current_size -= lru_entry.size_bytes
            self.eviction_count += 1

            logger.debug(f"Evicted LRU cache entry: {lru_key} (size: {lru_entry.size_bytes} bytes)")

            # Safety check to prevent infinite loop
            if len(self.cache) == 0:
                break

    def _get_total_cache_size(self) -> int:
        """Calculate total cache size in bytes"""
        return sum(entry.size_bytes for entry in self.cache.values())

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

    def _calculate_shannon_entropy(self, data: str) -> float:
        """
        Calculate Shannon entropy of a string in bits per character

        Args:
            data: Input string to analyze

        Returns:
            float: Shannon entropy in bits per character (0.0 to ~8.0 for ASCII)
        """
        if not data:
            return 0.0

        # Count frequency of each character
        char_counts = {}
        for char in data:
            char_counts[char] = char_counts.get(char, 0) + 1

        # Calculate Shannon entropy
        entropy = 0.0
        data_len = len(data)

        for count in char_counts.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        avg_compression_ratio = (self.compression_ratio_sum / self.compression_count
                               if self.compression_count > 0 else 1.0)

        total_size = self._get_total_cache_size()
        cache_utilization = total_size / self.max_cache_size_bytes if self.max_cache_size_bytes > 0 else 0

        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'eviction_count': self.eviction_count,
            'cache_entries': len(self.cache),
            'total_size_bytes': total_size,
            'max_cache_size_bytes': self.max_cache_size_bytes,
            'cache_utilization_percent': cache_utilization * 100,
            'max_entries': self.max_entries,
            'average_compression_ratio': avg_compression_ratio,
            'compression_enabled': self.enable_compression,
            'compression_threshold_bytes': self.compression_threshold,
            'compression_level': self.compression_level,
            'compression_ratio_threshold': self.compression_ratio_threshold,
            'ttl_seconds': self.ttl
        }

    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.compression_ratio_sum = 0.0
        self.compression_count = 0
        logger.info("Cache cleared")

# Global cache instance
state_cache = StateCache()
