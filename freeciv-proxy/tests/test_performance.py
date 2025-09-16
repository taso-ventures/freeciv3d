#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performance tests for LLM agent support
Verifies performance requirements are met
"""

import unittest
import time
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state_cache import StateCache
from civcom import CivCom

class TestPerformanceRequirements(unittest.TestCase):
    """Performance tests to verify requirements are met"""

    def setUp(self):
        """Set up performance test fixtures"""
        self.cache = StateCache(ttl=5, max_size_kb=4)

    def test_state_cache_performance_50ms(self):
        """Test that cached state queries complete under 50ms"""
        # Create realistic game state
        test_state = {
            'turn': 25,
            'phase': 'movement',
            'strategic': {
                'score': 150,
                'cities_count': 3,
                'units_count': 8,
                'tech_level': 5
            },
            'tactical': {
                'active_units': [
                    {'id': i, 'type': 'warrior', 'x': i*2, 'y': i*3, 'moves_left': 1}
                    for i in range(10)
                ],
                'cities_needing_orders': [
                    {'id': i, 'name': f'City{i}', 'x': i*5, 'y': i*4}
                    for i in range(3)
                ]
            },
            'economic': {
                'gold': 75,
                'gold_per_turn': 8,
                'research': 45
            },
            'legal_actions': [
                {
                    'type': 'unit_move',
                    'unit_id': i,
                    'dest_x': i*2 + 1,
                    'dest_y': i*3 + 1,
                    'priority': 'medium'
                } for i in range(20)
            ]
        }

        # Cache the state
        cache_key = 'perf_test_state'
        self.assertTrue(self.cache.set(cache_key, test_state, player_id=1))

        # Measure performance of cache retrieval
        retrieval_times = []
        for _ in range(100):  # Test 100 retrievals
            start_time = time.perf_counter()
            cached_state = self.cache.get(cache_key)
            end_time = time.perf_counter()

            self.assertIsNotNone(cached_state)
            retrieval_times.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate statistics
        avg_time = sum(retrieval_times) / len(retrieval_times)
        max_time = max(retrieval_times)
        min_time = min(retrieval_times)

        print(f"Cache retrieval performance:")
        print(f"  Average: {avg_time:.3f}ms")
        print(f"  Max: {max_time:.3f}ms")
        print(f"  Min: {min_time:.3f}ms")

        # Verify performance requirements
        self.assertLess(avg_time, 5.0, f"Average cache retrieval time {avg_time:.3f}ms > 5ms")
        self.assertLess(max_time, 50.0, f"Maximum cache retrieval time {max_time:.3f}ms > 50ms")

    def test_state_optimization_size_limit(self):
        """Test that optimized states are under 4KB limit"""
        # Create large unoptimized state
        large_state = {
            'turn': 100,
            'units': [
                {
                    'id': i,
                    'type': f'unit_type_{i%5}',
                    'x': i % 50,
                    'y': i // 50,
                    'moves_left': i % 3,
                    'health': 100 - (i % 20),
                    'experience': i * 2,
                    'detailed_stats': {
                        'attack': 10 + (i % 5),
                        'defense': 8 + (i % 4),
                        'movement': 2 + (i % 2),
                        'extra_data': f'unit_{i}_data_' * 10
                    }
                } for i in range(100)  # 100 units
            ],
            'cities': [
                {
                    'id': i,
                    'name': f'City_{i}',
                    'x': i * 3,
                    'y': i * 4,
                    'population': 5 + (i % 10),
                    'buildings': [f'building_{j}' for j in range(i % 8)],
                    'production_queue': [f'item_{j}' for j in range(5)],
                    'detailed_info': {
                        'trade_routes': [f'route_{j}' for j in range(i % 4)],
                        'resources': [f'resource_{j}' for j in range(i % 6)],
                        'extra_metadata': f'city_{i}_metadata_' * 20
                    }
                } for i in range(20)  # 20 cities
            ],
            'visible_tiles': [
                {
                    'x': i,
                    'y': j,
                    'terrain': f'terrain_{(i+j) % 5}',
                    'resource': f'resource_{(i*j) % 10}' if (i*j) % 7 == 0 else None,
                    'improvements': [f'improvement_{k}' for k in range((i+j) % 3)],
                    'extra_tile_data': f'tile_{i}_{j}_data_' * 5
                } for i in range(50) for j in range(50)  # 2500 tiles
            ],
            'detailed_player_stats': {
                f'player_{i}': {
                    'score': 1000 + i * 100,
                    'gold': 500 + i * 50,
                    'technologies': [f'tech_{j}' for j in range(i * 5)],
                    'discovered_techs_history': [f'historical_tech_{j}' for j in range(i * 10)],
                    'diplomatic_relations': {f'player_{k}': f'status_{k}' for k in range(8)},
                    'extra_player_metadata': f'player_{i}_extra_data_' * 30
                } for i in range(8)  # 8 players
            }
        }

        # Measure original size
        original_json = json.dumps(large_state, separators=(',', ':'))
        original_size = len(original_json.encode('utf-8'))
        original_size_kb = original_size / 1024

        print(f"Original state size: {original_size_kb:.2f}KB")

        # Test optimization
        optimized_state = self.cache.optimize_state_data(large_state)
        optimized_json = json.dumps(optimized_state, separators=(',', ':'))
        optimized_size = len(optimized_json.encode('utf-8'))
        optimized_size_kb = optimized_size / 1024

        print(f"Optimized state size: {optimized_size_kb:.2f}KB")
        print(f"Compression ratio: {(1 - optimized_size/original_size)*100:.1f}%")

        # Verify size requirements
        self.assertLess(optimized_size_kb, 4.0, f"Optimized state {optimized_size_kb:.2f}KB exceeds 4KB limit")
        self.assertLess(optimized_size, original_size, "Optimization should reduce state size")

        # Verify essential data is preserved
        self.assertIn('turn', optimized_state)
        self.assertIn('phase', optimized_state)

    def test_cache_hit_rate_requirement(self):
        """Test that cache achieves >80% hit rate under typical usage"""
        cache = StateCache(ttl=10, max_size_kb=4)  # Longer TTL for this test

        # Simulate typical usage pattern
        states = []
        for i in range(5):  # 5 different game states
            state = {
                'turn': i + 1,
                'player_id': 1,
                'units': [{'id': j, 'x': j, 'y': j} for j in range(3)],
                'cities': [{'id': 1, 'name': 'Capital'}]
            }
            states.append((f'state_{i}', state))

        # Pre-populate cache
        for key, state in states:
            cache.set(key, state, player_id=1)

        # Simulate access pattern: mostly accessing cached states with some new ones
        hits = 0
        misses = 0
        total_requests = 200

        for request in range(total_requests):
            if request % 10 == 0:  # 10% new requests (cache miss)
                new_key = f'new_state_{request}'
                new_state = {'turn': 100 + request, 'data': f'new_{request}'}
                result = cache.get(new_key)
                if result is None:
                    misses += 1
                    cache.set(new_key, new_state, player_id=1)
                else:
                    hits += 1
            else:  # 90% requests to existing data (should be cache hits)
                existing_key = states[request % 5][0]
                result = cache.get(existing_key)
                if result is not None:
                    hits += 1
                else:
                    misses += 1

        hit_rate = hits / (hits + misses)
        print(f"Cache performance:")
        print(f"  Hits: {hits}")
        print(f"  Misses: {misses}")
        print(f"  Hit rate: {hit_rate*100:.1f}%")

        # Verify hit rate requirement
        self.assertGreater(hit_rate, 0.8, f"Cache hit rate {hit_rate*100:.1f}% < 80%")

    def test_civcom_state_generation_performance(self):
        """Test that CivCom state generation meets performance requirements"""
        # Create mock CivCom instance
        civcom = CivCom('test-agent', 6001, 'test-key', None)

        # Set up realistic game state
        civcom.game_turn = 50
        civcom.game_phase = 'movement'
        civcom.player_units = [
            {'id': i, 'type': 'warrior', 'x': i*2, 'y': i*3, 'moves_left': 1}
            for i in range(15)
        ]
        civcom.player_cities = [
            {'id': i, 'name': f'City{i}', 'x': i*5, 'y': i*4, 'population': 5+i}
            for i in range(5)
        ]

        # Measure state generation performance
        generation_times = []
        for _ in range(50):  # Test 50 generations
            start_time = time.perf_counter()
            state = civcom.build_llm_optimized_state(player_id=1)
            end_time = time.perf_counter()

            self.assertIsInstance(state, dict)
            self.assertIn('turn', state)
            self.assertIn('strategic', state)

            generation_times.append((end_time - start_time) * 1000)  # Convert to ms

        avg_generation_time = sum(generation_times) / len(generation_times)
        max_generation_time = max(generation_times)

        print(f"State generation performance:")
        print(f"  Average: {avg_generation_time:.3f}ms")
        print(f"  Max: {max_generation_time:.3f}ms")

        # Verify generation is fast enough that with caching, total query time < 50ms
        self.assertLess(avg_generation_time, 40.0, f"State generation {avg_generation_time:.3f}ms too slow")
        self.assertLess(max_generation_time, 45.0, f"Max state generation {max_generation_time:.3f}ms too slow")

if __name__ == '__main__':
    unittest.main(verbosity=2)