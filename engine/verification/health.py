"""
VØX Verification - Health Checks
--------------------------------

System health monitoring and readiness probes.

AXIØM Phase 10: Verify - "How do we know this works?"
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Awaitable

# Optional psutil for system metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

from .models import (
    HealthCheckResult,
    SystemHealthReport,
    HealthStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckConfig:
    """
    Configuration for health checks.

    Attributes:
        timeout: Check timeout in seconds
        memory_threshold_mb: Memory warning threshold
        cpu_threshold_percent: CPU warning threshold
        disk_threshold_percent: Disk warning threshold
    """
    timeout: float = 5.0
    memory_threshold_mb: float = 1024.0  # 1GB
    cpu_threshold_percent: float = 80.0
    disk_threshold_percent: float = 90.0
    include_system_metrics: bool = True


@dataclass
class HealthCheckDefinition:
    """
    Definition of a health check.

    Attributes:
        name: Check name
        check_fn: Async function that performs the check
        critical: Whether failure makes system unhealthy
        timeout: Check-specific timeout
    """
    name: str
    check_fn: Callable[..., Awaitable[HealthCheckResult]]
    critical: bool = False
    timeout: Optional[float] = None
    description: str = ""


class HealthChecker:
    """
    System health checker for VØX.

    Features:
        - Liveness probes
        - Readiness probes
        - Component health checks
        - System resource monitoring
        - Custom check registration
    """

    def __init__(
        self,
        config: Optional[HealthCheckConfig] = None,
    ):
        """
        Initialize health checker.

        Args:
            config: Health check configuration
        """
        self.config = config or HealthCheckConfig()
        self._checks: List[HealthCheckDefinition] = []
        self._start_time = time.time()
        self._last_report: Optional[SystemHealthReport] = None

    def register_check(
        self,
        name: str,
        check_fn: Callable[..., Awaitable[HealthCheckResult]],
        critical: bool = False,
        timeout: Optional[float] = None,
        description: str = "",
    ) -> HealthCheckDefinition:
        """
        Register a health check.

        Args:
            name: Check name
            check_fn: Check function
            critical: Whether failure is critical
            timeout: Check timeout
            description: Check description

        Returns:
            Health check definition
        """
        check = HealthCheckDefinition(
            name=name,
            check_fn=check_fn,
            critical=critical,
            timeout=timeout or self.config.timeout,
            description=description,
        )
        self._checks.append(check)
        return check

    def check(
        self,
        name: Optional[str] = None,
        critical: bool = False,
        timeout: Optional[float] = None,
    ):
        """
        Decorator to register a health check.

        Args:
            name: Check name
            critical: Whether failure is critical
            timeout: Check timeout

        Returns:
            Decorator function
        """
        def decorator(fn: Callable[..., Awaitable[HealthCheckResult]]):
            check_name = name or fn.__name__
            self.register_check(
                name=check_name,
                check_fn=fn,
                critical=critical,
                timeout=timeout,
                description=fn.__doc__ or "",
            )
            return fn
        return decorator

    async def check_health(self) -> SystemHealthReport:
        """
        Run all health checks.

        Returns:
            System health report
        """
        report = SystemHealthReport(
            uptime_seconds=time.time() - self._start_time,
            timestamp=time.time(),
        )

        # Run registered checks
        for check_def in self._checks:
            result = await self._run_check(check_def)
            report.checks.append(result)

        # Add system checks if enabled
        if self.config.include_system_metrics:
            system_checks = await self._run_system_checks()
            report.checks.extend(system_checks)

        # Calculate overall status
        report.calculate_overall()
        report.live = True  # If we got this far, system is alive

        self._last_report = report
        return report

    async def check_liveness(self) -> bool:
        """
        Simple liveness check.

        Returns:
            True if system is alive
        """
        return True  # If this runs, we're alive

    async def check_readiness(self) -> bool:
        """
        Check if system is ready to serve.

        Returns:
            True if system is ready
        """
        report = await self.check_health()
        return report.ready

    async def get_status(self) -> Dict[str, Any]:
        """
        Get system status.

        Returns:
            Status dictionary
        """
        report = await self.check_health()
        return report.to_dict()

    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return time.time() - self._start_time

    def get_last_report(self) -> Optional[SystemHealthReport]:
        """Get last health report."""
        return self._last_report

    async def _run_check(
        self,
        check_def: HealthCheckDefinition,
    ) -> HealthCheckResult:
        """Run a single health check."""
        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                check_def.check_fn(),
                timeout=check_def.timeout,
            )
            result.latency_ms = (time.time() - start_time) * 1000
            return result

        except asyncio.TimeoutError:
            return HealthCheckResult(
                name=check_def.name,
                status=HealthStatus.UNHEALTHY if check_def.critical else HealthStatus.DEGRADED,
                message=f"Check timed out after {check_def.timeout}s",
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                name=check_def.name,
                status=HealthStatus.UNHEALTHY if check_def.critical else HealthStatus.DEGRADED,
                message=f"Check failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)},
            )

    async def _run_system_checks(self) -> List[HealthCheckResult]:
        """Run system resource checks."""
        checks = []

        if not PSUTIL_AVAILABLE:
            checks.append(HealthCheckResult(
                name="system_metrics",
                status=HealthStatus.UNKNOWN,
                message="psutil not available for system metrics",
            ))
            return checks

        # Memory check
        try:
            memory = psutil.virtual_memory()
            memory_used_mb = memory.used / (1024 * 1024)
            memory_percent = memory.percent

            if memory_used_mb > self.config.memory_threshold_mb:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage: {memory_percent:.1f}%"

            checks.append(HealthCheckResult(
                name="memory",
                status=status,
                message=message,
                details={
                    "used_mb": memory_used_mb,
                    "total_mb": memory.total / (1024 * 1024),
                    "percent": memory_percent,
                },
            ))
        except Exception as e:
            logger.warning(f"Memory check failed: {e}")

        # CPU check
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)

            if cpu_percent > self.config.cpu_threshold_percent:
                status = HealthStatus.DEGRADED
                message = f"High CPU usage: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage: {cpu_percent:.1f}%"

            checks.append(HealthCheckResult(
                name="cpu",
                status=status,
                message=message,
                details={
                    "percent": cpu_percent,
                    "count": psutil.cpu_count(),
                },
            ))
        except Exception as e:
            logger.warning(f"CPU check failed: {e}")

        # Disk check
        try:
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent

            if disk_percent > self.config.disk_threshold_percent:
                status = HealthStatus.DEGRADED
                message = f"High disk usage: {disk_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage: {disk_percent:.1f}%"

            checks.append(HealthCheckResult(
                name="disk",
                status=status,
                message=message,
                details={
                    "used_gb": disk.used / (1024 ** 3),
                    "total_gb": disk.total / (1024 ** 3),
                    "percent": disk_percent,
                },
            ))
        except Exception as e:
            logger.warning(f"Disk check failed: {e}")

        return checks


def create_vox_health_checker() -> HealthChecker:
    """
    Create VØX health checker with standard checks.

    Returns:
        Configured health checker
    """
    checker = HealthChecker(HealthCheckConfig(
        timeout=5.0,
        include_system_metrics=True,
    ))

    # ========================================================================
    # Core Component Checks
    # ========================================================================

    @checker.check(name="engine", critical=True)
    async def check_engine():
        """Check VØX engine is operational."""
        try:
            from axiom_vox import VoxEngine

            engine = VoxEngine()
            await engine.start()
            await engine.stop()

            return HealthCheckResult(
                name="engine",
                status=HealthStatus.HEALTHY,
                message="Engine initialized successfully",
            )
        except Exception as e:
            return HealthCheckResult(
                name="engine",
                status=HealthStatus.UNHEALTHY,
                message=f"Engine failed: {str(e)}",
                details={"error": str(e)},
            )

    @checker.check(name="voices", critical=True)
    async def check_voices():
        """Check voices are available."""
        try:
            from axiom_vox import VoxEngine

            engine = VoxEngine()
            await engine.start()

            try:
                voices = await engine.list_voices()

                if not voices:
                    return HealthCheckResult(
                        name="voices",
                        status=HealthStatus.DEGRADED,
                        message="No voices available",
                    )

                return HealthCheckResult(
                    name="voices",
                    status=HealthStatus.HEALTHY,
                    message=f"{len(voices)} voices available",
                    details={"count": len(voices)},
                )
            finally:
                await engine.stop()

        except Exception as e:
            return HealthCheckResult(
                name="voices",
                status=HealthStatus.UNHEALTHY,
                message=f"Voice check failed: {str(e)}",
            )

    @checker.check(name="synthesis", critical=True)
    async def check_synthesis():
        """Check synthesis is working."""
        try:
            from axiom_vox import VoxEngine

            engine = VoxEngine()
            await engine.start()

            try:
                result = await engine.synthesize("Health check")

                if result and result.audio:
                    return HealthCheckResult(
                        name="synthesis",
                        status=HealthStatus.HEALTHY,
                        message="Synthesis operational",
                        details={"audio_bytes": len(result.audio)},
                    )
                else:
                    return HealthCheckResult(
                        name="synthesis",
                        status=HealthStatus.DEGRADED,
                        message="Synthesis returned empty audio",
                    )
            finally:
                await engine.stop()

        except Exception as e:
            return HealthCheckResult(
                name="synthesis",
                status=HealthStatus.UNHEALTHY,
                message=f"Synthesis failed: {str(e)}",
            )

    @checker.check(name="streaming")
    async def check_streaming():
        """Check streaming synthesis is working."""
        try:
            from axiom_vox import VoxEngine

            engine = VoxEngine()
            await engine.start()

            try:
                chunks = []
                async for chunk in engine.synthesize_stream("Stream check"):
                    chunks.append(chunk)
                    if len(chunks) >= 2:
                        break  # Just check first few chunks

                if chunks:
                    return HealthCheckResult(
                        name="streaming",
                        status=HealthStatus.HEALTHY,
                        message="Streaming operational",
                        details={"chunks_received": len(chunks)},
                    )
                else:
                    return HealthCheckResult(
                        name="streaming",
                        status=HealthStatus.DEGRADED,
                        message="No streaming chunks received",
                    )
            finally:
                await engine.stop()

        except Exception as e:
            return HealthCheckResult(
                name="streaming",
                status=HealthStatus.DEGRADED,
                message=f"Streaming check failed: {str(e)}",
            )

    # ========================================================================
    # Database Checks
    # ========================================================================

    @checker.check(name="database")
    async def check_database():
        """Check database connectivity."""
        try:
            from axiom_vox import get_persistence

            persistence = get_persistence()

            # Try a simple operation
            voices = await persistence.list_voices()

            return HealthCheckResult(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connected",
                details={"voice_count": len(voices)},
            )

        except Exception as e:
            return HealthCheckResult(
                name="database",
                status=HealthStatus.DEGRADED,
                message=f"Database check failed: {str(e)}",
            )

    # ========================================================================
    # Cache Checks
    # ========================================================================

    @checker.check(name="cache")
    async def check_cache():
        """Check cache system."""
        try:
            from axiom_vox.performance import get_cache_manager

            cache = get_cache_manager()
            stats = cache.get_combined_stats()

            return HealthCheckResult(
                name="cache",
                status=HealthStatus.HEALTHY,
                message="Cache operational",
                details=stats.get("combined", {}),
            )

        except Exception as e:
            return HealthCheckResult(
                name="cache",
                status=HealthStatus.DEGRADED,
                message=f"Cache check failed: {str(e)}",
            )

    return checker


class LivenessProbe:
    """
    Simple liveness probe for container orchestration.
    """

    def __init__(self):
        """Initialize liveness probe."""
        self._alive = True
        self._last_check = time.time()

    async def check(self) -> Dict[str, Any]:
        """
        Check liveness.

        Returns:
            Liveness status
        """
        self._last_check = time.time()
        return {
            "alive": self._alive,
            "timestamp": self._last_check,
        }

    def set_alive(self, alive: bool) -> None:
        """Set liveness status."""
        self._alive = alive


class ReadinessProbe:
    """
    Readiness probe for container orchestration.
    """

    def __init__(
        self,
        health_checker: Optional[HealthChecker] = None,
    ):
        """
        Initialize readiness probe.

        Args:
            health_checker: Health checker to use
        """
        self._health_checker = health_checker or create_vox_health_checker()
        self._ready = False
        self._last_check = time.time()

    async def check(self) -> Dict[str, Any]:
        """
        Check readiness.

        Returns:
            Readiness status
        """
        self._last_check = time.time()
        self._ready = await self._health_checker.check_readiness()

        return {
            "ready": self._ready,
            "timestamp": self._last_check,
        }

    async def wait_for_ready(
        self,
        timeout: float = 60.0,
        check_interval: float = 1.0,
    ) -> bool:
        """
        Wait for system to become ready.

        Args:
            timeout: Max wait time
            check_interval: Time between checks

        Returns:
            True if ready, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = await self.check()
            if status["ready"]:
                return True
            await asyncio.sleep(check_interval)

        return False


async def run_health_check() -> SystemHealthReport:
    """
    Run health check and return report.

    Returns:
        System health report
    """
    checker = create_vox_health_checker()
    return await checker.check_health()


async def check_system_ready() -> bool:
    """
    Check if system is ready.

    Returns:
        True if ready
    """
    checker = create_vox_health_checker()
    return await checker.check_readiness()
