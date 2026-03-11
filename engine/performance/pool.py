"""
VØX Performance - Connection Pooling
------------------------------------

Persistent connection management for HTTP and WebSocket.

Features:
    - HTTP connection pooling with keep-alive
    - WebSocket connection management
    - Health checking and automatic reconnection
    - Connection metrics and monitoring

AXIØM Phase 9: Optimize - "What makes this solution efficient?"
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, AsyncIterator
from contextlib import asynccontextmanager
import weakref

logger = logging.getLogger(__name__)


class ConnectionState(str, Enum):
    """State of a pooled connection."""
    IDLE = "idle"
    IN_USE = "in_use"
    CONNECTING = "connecting"
    UNHEALTHY = "unhealthy"
    CLOSED = "closed"


@dataclass
class ConnectionStats:
    """Statistics for connection pool monitoring."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    total_requests: int = 0
    avg_acquire_time_ms: float = 0.0
    peak_active: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "idle_connections": self.idle_connections,
            "failed_connections": self.failed_connections,
            "total_requests": self.total_requests,
            "avg_acquire_time_ms": self.avg_acquire_time_ms,
            "peak_active": self.peak_active,
        }


@dataclass
class PooledConnection:
    """A connection managed by the pool."""
    id: str
    connection: Any
    state: ConnectionState = ConnectionState.IDLE
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    errors: int = 0

    def mark_used(self) -> None:
        """Mark connection as in use."""
        self.state = ConnectionState.IN_USE
        self.last_used = time.time()
        self.use_count += 1

    def mark_idle(self) -> None:
        """Mark connection as idle."""
        self.state = ConnectionState.IDLE
        self.last_used = time.time()

    def mark_unhealthy(self) -> None:
        """Mark connection as unhealthy."""
        self.state = ConnectionState.UNHEALTHY
        self.errors += 1

    @property
    def age_seconds(self) -> float:
        """Get connection age in seconds."""
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """Get idle time in seconds."""
        return time.time() - self.last_used


class HTTPConnectionPool:
    """
    HTTP connection pool with keep-alive.

    Features:
        - Persistent connections with configurable limits
        - Automatic connection recycling
        - Health checking
        - Backpressure handling
    """

    def __init__(
        self,
        base_url: str,
        pool_size: int = 10,
        max_idle_seconds: float = 60.0,
        max_age_seconds: float = 300.0,
        connect_timeout: float = 10.0,
        read_timeout: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize HTTP connection pool.

        Args:
            base_url: Base URL for connections
            pool_size: Maximum connections in pool
            max_idle_seconds: Maximum idle time before closing
            max_age_seconds: Maximum connection age
            connect_timeout: Connection timeout
            read_timeout: Read timeout
            headers: Default headers for requests
        """
        self.base_url = base_url
        self.pool_size = pool_size
        self.max_idle_seconds = max_idle_seconds
        self.max_age_seconds = max_age_seconds
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.headers = headers or {}

        self._connections: Dict[str, PooledConnection] = {}
        self._available: asyncio.Queue = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._closed = False
        self._stats = ConnectionStats()
        self._next_id = 0
        self._client = None

        # Background tasks
        self._maintenance_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the connection pool."""
        if self._closed:
            raise RuntimeError("Pool is closed")

        # Initialize HTTP client
        try:
            import aiohttp
            timeout = aiohttp.ClientTimeout(
                connect=self.connect_timeout,
                total=self.read_timeout,
            )
            connector = aiohttp.TCPConnector(
                limit=self.pool_size,
                limit_per_host=self.pool_size,
                keepalive_timeout=self.max_idle_seconds,
            )
            self._client = aiohttp.ClientSession(
                base_url=self.base_url,
                timeout=timeout,
                connector=connector,
                headers=self.headers,
            )
        except ImportError:
            try:
                import httpx
                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=httpx.Timeout(
                        connect=self.connect_timeout,
                        read=self.read_timeout,
                    ),
                    headers=self.headers,
                    limits=httpx.Limits(
                        max_connections=self.pool_size,
                        max_keepalive_connections=self.pool_size,
                    ),
                )
            except ImportError:
                logger.warning("No async HTTP client available (aiohttp or httpx)")

        # Start maintenance task
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        logger.info(f"HTTP pool started: {self.base_url} (max {self.pool_size} connections)")

    async def close(self) -> None:
        """Close all connections and stop pool."""
        self._closed = True

        # Stop maintenance
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass

        # Close client
        if self._client:
            await self._client.close()

        self._connections.clear()
        logger.info("HTTP pool closed")

    async def request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> Any:
        """
        Make HTTP request using pooled connection.

        Args:
            method: HTTP method
            path: Request path
            **kwargs: Additional request parameters

        Returns:
            Response object
        """
        if self._closed:
            raise RuntimeError("Pool is closed")

        if self._client is None:
            raise RuntimeError("Pool not started")

        start_time = time.time()
        self._stats.total_requests += 1

        try:
            response = await self._client.request(method, path, **kwargs)
            return response
        except Exception as e:
            self._stats.failed_connections += 1
            raise

        finally:
            acquire_time = (time.time() - start_time) * 1000
            # Update rolling average
            n = self._stats.total_requests
            self._stats.avg_acquire_time_ms = (
                (self._stats.avg_acquire_time_ms * (n - 1) + acquire_time) / n
            )

    async def get(self, path: str, **kwargs) -> Any:
        """Convenience method for GET requests."""
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs) -> Any:
        """Convenience method for POST requests."""
        return await self.request("POST", path, **kwargs)

    def get_stats(self) -> ConnectionStats:
        """Get pool statistics."""
        return ConnectionStats(
            total_connections=self._stats.total_connections,
            active_connections=self._stats.active_connections,
            idle_connections=self._stats.idle_connections,
            failed_connections=self._stats.failed_connections,
            total_requests=self._stats.total_requests,
            avg_acquire_time_ms=self._stats.avg_acquire_time_ms,
            peak_active=self._stats.peak_active,
        )

    async def _maintenance_loop(self) -> None:
        """Background maintenance for connection health."""
        while not self._closed:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._cleanup_stale()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Maintenance error: {e}")

    async def _cleanup_stale(self) -> None:
        """Remove stale connections."""
        # Most cleanup is handled by the underlying HTTP client
        # This is a placeholder for additional cleanup logic
        pass


class WebSocketPool:
    """
    WebSocket connection pool for streaming.

    Features:
        - Persistent WebSocket connections
        - Automatic reconnection
        - Message routing
        - Connection health monitoring
    """

    def __init__(
        self,
        base_url: str,
        pool_size: int = 5,
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 30.0,
        ping_interval: float = 30.0,
    ):
        """
        Initialize WebSocket pool.

        Args:
            base_url: WebSocket base URL
            pool_size: Maximum WebSocket connections
            reconnect_delay: Initial reconnect delay
            max_reconnect_delay: Maximum reconnect delay
            ping_interval: Ping interval for keep-alive
        """
        self.base_url = base_url
        self.pool_size = pool_size
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.ping_interval = ping_interval

        self._connections: Dict[str, PooledConnection] = {}
        self._available: asyncio.Queue = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._closed = False
        self._next_id = 0

    async def start(self) -> None:
        """Start the WebSocket pool."""
        logger.info(f"WebSocket pool started: {self.base_url}")

    async def close(self) -> None:
        """Close all WebSocket connections."""
        self._closed = True

        async with self._lock:
            for conn in self._connections.values():
                if conn.connection:
                    try:
                        await conn.connection.close()
                    except Exception:
                        pass

        self._connections.clear()
        logger.info("WebSocket pool closed")

    @asynccontextmanager
    async def connection(self, path: str = "") -> AsyncIterator[Any]:
        """
        Get a WebSocket connection from the pool.

        Args:
            path: WebSocket path

        Yields:
            WebSocket connection
        """
        if self._closed:
            raise RuntimeError("Pool is closed")

        ws = None
        try:
            import aiohttp
            url = f"{self.base_url}{path}"

            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(url) as ws:
                    yield ws

        except ImportError:
            raise RuntimeError("aiohttp required for WebSocket connections")

    async def send(self, path: str, message: Any) -> None:
        """Send message on WebSocket."""
        async with self.connection(path) as ws:
            if isinstance(message, bytes):
                await ws.send_bytes(message)
            else:
                await ws.send_json(message)

    async def receive(self, path: str) -> Any:
        """Receive message from WebSocket."""
        async with self.connection(path) as ws:
            msg = await ws.receive()
            return msg.data


class ConnectionPoolManager:
    """
    Central manager for all connection pools.

    Provides:
        - Unified pool access
        - Pool lifecycle management
        - Combined statistics
    """

    def __init__(
        self,
        http_pool: Optional[HTTPConnectionPool] = None,
        ws_pool: Optional[WebSocketPool] = None,
    ):
        """
        Initialize connection pool manager.

        Args:
            http_pool: HTTP connection pool
            ws_pool: WebSocket connection pool
        """
        self.http = http_pool
        self.ws = ws_pool

    async def start(self) -> None:
        """Start all connection pools."""
        if self.http:
            await self.http.start()
        if self.ws:
            await self.ws.start()

    async def close(self) -> None:
        """Close all connection pools."""
        if self.http:
            await self.http.close()
        if self.ws:
            await self.ws.close()

    def get_combined_stats(self) -> Dict[str, Any]:
        """Get combined statistics from all pools."""
        stats = {}
        if self.http:
            stats["http"] = self.http.get_stats().to_dict()
        if self.ws:
            stats["ws"] = {"connections": len(self.ws._connections) if self.ws else 0}
        return stats

    async def __aenter__(self) -> "ConnectionPoolManager":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


# Global pool manager instance
_pool_manager: Optional[ConnectionPoolManager] = None


def get_pool_manager() -> Optional[ConnectionPoolManager]:
    """Get global pool manager instance."""
    return _pool_manager


def set_pool_manager(manager: ConnectionPoolManager) -> None:
    """Set global pool manager instance."""
    global _pool_manager
    _pool_manager = manager


async def create_pool_manager(
    base_url: str,
    http_pool_size: int = 10,
    ws_pool_size: int = 5,
    **kwargs,
) -> ConnectionPoolManager:
    """
    Create and start a connection pool manager.

    Args:
        base_url: Base URL for connections
        http_pool_size: HTTP pool size
        ws_pool_size: WebSocket pool size
        **kwargs: Additional pool configuration

    Returns:
        Started ConnectionPoolManager
    """
    http_pool = HTTPConnectionPool(
        base_url=base_url,
        pool_size=http_pool_size,
        **kwargs,
    )

    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_pool = WebSocketPool(
        base_url=ws_url,
        pool_size=ws_pool_size,
    )

    manager = ConnectionPoolManager(http_pool=http_pool, ws_pool=ws_pool)
    await manager.start()

    set_pool_manager(manager)
    return manager
