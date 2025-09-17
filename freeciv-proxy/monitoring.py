#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monitoring and metrics endpoints for FreeCiv proxy
Provides health checks, performance metrics, and Prometheus-compatible metrics
"""

import time
import json
import logging
from typing import Dict, Any
from tornado import web
from state_cache import state_cache
from api_rate_limiter import api_rate_limiter

logger = logging.getLogger("freeciv-proxy")


class HealthCheckHandler(web.RequestHandler):
    """Health check endpoint for monitoring systems"""

    async def get(self):
        """Return health status of the service"""
        try:
            start_time = time.time()

            # Basic service checks
            health_data = {
                "status": "healthy",
                "timestamp": start_time,
                "service": "freeciv-state-extractor",
                "version": "1.0.0",
                "checks": {}
            }

            # Check cache health
            try:
                cache_stats = state_cache.get_cache_stats()
                health_data["checks"]["cache"] = {
                    "status": "healthy",
                    "entries": cache_stats.get("cache_entries", 0),
                    "hit_rate": cache_stats.get("hit_rate", 0),
                    "utilization_percent": cache_stats.get("cache_utilization_percent", 0)
                }
            except Exception as e:
                health_data["checks"]["cache"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_data["status"] = "degraded"

            # Check rate limiter health
            try:
                rate_stats = api_rate_limiter.get_stats()
                health_data["checks"]["rate_limiter"] = {
                    "status": "healthy",
                    "active_buckets": rate_stats.get("active_player_buckets", 0) + rate_stats.get("active_ip_buckets", 0)
                }
            except Exception as e:
                health_data["checks"]["rate_limiter"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_data["status"] = "degraded"

            # Response time check
            response_time = (time.time() - start_time) * 1000
            health_data["response_time_ms"] = response_time

            if response_time > 100:  # Slow response
                health_data["status"] = "degraded"

            # Set appropriate HTTP status
            if health_data["status"] == "healthy":
                self.set_status(200)
            else:
                self.set_status(503)  # Service Unavailable

            self.set_header("Content-Type", "application/json")
            self.write(health_data)

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.set_status(503)
            self.write({
                "status": "unhealthy",
                "error": "Service unavailable",
                "timestamp": time.time()
            })


class MetricsHandler(web.RequestHandler):
    """Prometheus-compatible metrics endpoint"""

    async def get(self):
        """Return metrics in Prometheus format"""
        try:
            metrics = []
            timestamp = int(time.time() * 1000)  # Prometheus uses milliseconds

            # Cache metrics
            cache_stats = state_cache.get_cache_stats()

            metrics.extend([
                f"# HELP freeciv_cache_hits_total Total cache hits",
                f"# TYPE freeciv_cache_hits_total counter",
                f"freeciv_cache_hits_total {cache_stats.get('hit_count', 0)}",
                "",
                f"# HELP freeciv_cache_misses_total Total cache misses",
                f"# TYPE freeciv_cache_misses_total counter",
                f"freeciv_cache_misses_total {cache_stats.get('miss_count', 0)}",
                "",
                f"# HELP freeciv_cache_hit_rate Cache hit rate ratio",
                f"# TYPE freeciv_cache_hit_rate gauge",
                f"freeciv_cache_hit_rate {cache_stats.get('hit_rate', 0):.4f}",
                "",
                f"# HELP freeciv_cache_entries_current Current number of cache entries",
                f"# TYPE freeciv_cache_entries_current gauge",
                f"freeciv_cache_entries_current {cache_stats.get('cache_entries', 0)}",
                "",
                f"# HELP freeciv_cache_size_bytes_current Current cache size in bytes",
                f"# TYPE freeciv_cache_size_bytes_current gauge",
                f"freeciv_cache_size_bytes_current {cache_stats.get('total_size_bytes', 0)}",
                "",
                f"# HELP freeciv_cache_utilization_percent Cache utilization percentage",
                f"# TYPE freeciv_cache_utilization_percent gauge",
                f"freeciv_cache_utilization_percent {cache_stats.get('cache_utilization_percent', 0):.2f}",
                "",
                f"# HELP freeciv_cache_evictions_total Total cache evictions",
                f"# TYPE freeciv_cache_evictions_total counter",
                f"freeciv_cache_evictions_total {cache_stats.get('eviction_count', 0)}",
                "",
                f"# HELP freeciv_cache_compression_ratio_average Average compression ratio",
                f"# TYPE freeciv_cache_compression_ratio_average gauge",
                f"freeciv_cache_compression_ratio_average {cache_stats.get('average_compression_ratio', 1.0):.4f}",
                ""
            ])

            # Rate limiter metrics
            rate_stats = api_rate_limiter.get_stats()

            metrics.extend([
                f"# HELP freeciv_rate_limit_buckets_active_player Active player rate limit buckets",
                f"# TYPE freeciv_rate_limit_buckets_active_player gauge",
                f"freeciv_rate_limit_buckets_active_player {rate_stats.get('active_player_buckets', 0)}",
                "",
                f"# HELP freeciv_rate_limit_buckets_active_ip Active IP rate limit buckets",
                f"# TYPE freeciv_rate_limit_buckets_active_ip gauge",
                f"freeciv_rate_limit_buckets_active_ip {rate_stats.get('active_ip_buckets', 0)}",
                "",
                f"# HELP freeciv_rate_limit_rpm_player Player requests per minute limit",
                f"# TYPE freeciv_rate_limit_rpm_player gauge",
                f"freeciv_rate_limit_rpm_player {rate_stats.get('player_rpm_limit', 0)}",
                "",
                f"# HELP freeciv_rate_limit_rpm_ip IP requests per minute limit",
                f"# TYPE freeciv_rate_limit_rpm_ip gauge",
                f"freeciv_rate_limit_rpm_ip {rate_stats.get('ip_rpm_limit', 0)}",
                ""
            ])

            # Service metrics
            metrics.extend([
                f"# HELP freeciv_service_uptime_seconds Service uptime in seconds",
                f"# TYPE freeciv_service_uptime_seconds gauge",
                f"freeciv_service_uptime_seconds {time.time() - (rate_stats.get('last_cleanup', time.time()))}",
                ""
            ])

            self.set_header("Content-Type", "text/plain; version=0.0.4")
            self.write("\\n".join(metrics))

        except Exception as e:
            logger.error(f"Metrics generation failed: {e}")
            self.set_status(500)
            self.write("# Error generating metrics")


class StatsHandler(web.RequestHandler):
    """JSON stats endpoint for debugging and monitoring"""

    async def get(self):
        """Return detailed statistics in JSON format"""
        try:
            stats = {
                "timestamp": time.time(),
                "service": "freeciv-state-extractor",
                "version": "1.0.0"
            }

            # Cache statistics
            cache_stats = state_cache.get_cache_stats()
            stats["cache"] = cache_stats

            # Rate limiter statistics
            rate_stats = api_rate_limiter.get_stats()
            stats["rate_limiter"] = rate_stats

            # System statistics
            import psutil
            try:
                process = psutil.Process()
                stats["system"] = {
                    "memory_mb": process.memory_info().rss / (1024 * 1024),
                    "cpu_percent": process.cpu_percent(),
                    "threads": process.num_threads(),
                    "open_files": len(process.open_files()),
                    "connections": len(process.connections())
                }
            except ImportError:
                stats["system"] = {"note": "psutil not available"}
            except Exception as e:
                stats["system"] = {"error": str(e)}

            self.set_header("Content-Type", "application/json")
            self.write(json.dumps(stats, indent=2))

        except Exception as e:
            logger.error(f"Stats generation failed: {e}")
            self.set_status(500)
            self.write({"error": "Internal server error"})
