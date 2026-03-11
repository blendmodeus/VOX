"""
VØX SDK Session Management
--------------------------

Session management and context handling for VØX SDK.

Features:
    - Async context manager with automatic cleanup
    - Session-level state tracking
    - Automatic resource management
    - Request correlation

AXIØM Phase 8: Integrate - "How do the parts connect?"
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, AsyncIterator
from enum import Enum

from .config import VoxConfig
from .errors import VoxError, ErrorContext

logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    """Session lifecycle states."""
    CREATED = "created"
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class SessionMetrics:
    """Metrics collected during a session."""
    requests_made: int = 0
    requests_succeeded: int = 0
    requests_failed: int = 0
    bytes_synthesized: int = 0
    audio_seconds_generated: float = 0.0
    total_latency_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def success_rate(self) -> float:
        """Get request success rate."""
        total = self.requests_succeeded + self.requests_failed
        if total == 0:
            return 1.0
        return self.requests_succeeded / total

    @property
    def average_latency_ms(self) -> float:
        """Get average request latency."""
        if self.requests_made == 0:
            return 0.0
        return self.total_latency_ms / self.requests_made

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "requests_made": self.requests_made,
            "requests_succeeded": self.requests_succeeded,
            "requests_failed": self.requests_failed,
            "success_rate": self.success_rate,
            "bytes_synthesized": self.bytes_synthesized,
            "audio_seconds_generated": self.audio_seconds_generated,
            "average_latency_ms": self.average_latency_ms,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
        }


@dataclass
class RequestContext:
    """Context for a single request within a session."""
    request_id: str
    session_id: str
    operation: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    voice_id: Optional[str] = None
    text_length: Optional[int] = None
    success: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        """Get request latency in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def to_error_context(self) -> ErrorContext:
        """Convert to ErrorContext for error handling."""
        return ErrorContext(
            request_id=self.request_id,
            operation=self.operation,
            voice_id=self.voice_id,
            text_preview=None,
            details=self.metadata,
        )


class VoxSession:
    """
    Session manager for VØX SDK operations.

    Provides:
        - Automatic resource cleanup
        - Request tracking and correlation
        - Session-level metrics
        - Error context propagation

    Usage:
        async with VoxSession(config) as session:
            result = await session.synthesize("Hello world")
    """

    def __init__(
        self,
        config: Optional[VoxConfig] = None,
        session_id: Optional[str] = None,
    ):
        """
        Initialize VØX session.

        Args:
            config: SDK configuration
            session_id: Optional session ID (generated if not provided)
        """
        self.config = config or VoxConfig()
        self.session_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        self.state = SessionState.CREATED
        self.metrics = SessionMetrics()

        # Internal state
        self._pending_requests: Dict[str, RequestContext] = {}
        self._cleanup_tasks: List[asyncio.Task] = []
        self._http_client = None
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "VoxSession":
        """Enter async context - initialize session."""
        await self._initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context - cleanup session."""
        await self._cleanup()

    async def _initialize(self) -> None:
        """Initialize session resources."""
        logger.debug(f"Initializing session {self.session_id}")
        self.state = SessionState.ACTIVE
        self.metrics.start_time = time.time()

        # Initialize HTTP client if needed
        try:
            import aiohttp
            self._http_client = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    connect=self.config.timeout.connect_timeout_seconds,
                    total=self.config.timeout.read_timeout_seconds,
                ),
                headers=self._build_headers(),
            )
        except ImportError:
            logger.debug("aiohttp not available, using httpx fallback")
            try:
                import httpx
                self._http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        connect=self.config.timeout.connect_timeout_seconds,
                        read=self.config.timeout.read_timeout_seconds,
                    ),
                    headers=self._build_headers(),
                )
            except ImportError:
                logger.warning("No async HTTP client available (aiohttp or httpx)")

    async def _cleanup(self) -> None:
        """Cleanup session resources."""
        logger.debug(f"Cleaning up session {self.session_id}")
        self.state = SessionState.CLOSING
        self.metrics.end_time = time.time()

        # Cancel pending cleanup tasks
        for task in self._cleanup_tasks:
            if not task.done():
                task.cancel()

        # Close HTTP client
        if self._http_client:
            try:
                await self._http_client.close()
            except Exception as e:
                logger.warning(f"Error closing HTTP client: {e}")

        self.state = SessionState.CLOSED
        logger.info(
            f"Session {self.session_id} closed. "
            f"Requests: {self.metrics.requests_made}, "
            f"Success rate: {self.metrics.success_rate:.1%}, "
            f"Duration: {self.metrics.duration_seconds:.1f}s"
        )

    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers for requests."""
        headers = {
            "Content-Type": "application/json",
            "X-Session-ID": self.session_id,
            "User-Agent": "axiom-vox-sdk/0.13.0",
        }

        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        if self.config.user_id:
            headers["X-User-ID"] = self.config.user_id

        headers.update(self.config.custom_headers)

        return headers

    def create_request_context(
        self,
        operation: str,
        voice_id: Optional[str] = None,
        text_length: Optional[int] = None,
        **metadata,
    ) -> RequestContext:
        """
        Create a new request context.

        Args:
            operation: Operation name
            voice_id: Voice ID if applicable
            text_length: Text length if applicable
            **metadata: Additional metadata

        Returns:
            RequestContext
        """
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        context = RequestContext(
            request_id=request_id,
            session_id=self.session_id,
            operation=operation,
            voice_id=voice_id,
            text_length=text_length,
            metadata=metadata,
        )
        self._pending_requests[request_id] = context
        return context

    def complete_request(
        self,
        context: RequestContext,
        success: bool,
        error: Optional[str] = None,
        bytes_generated: int = 0,
        audio_seconds: float = 0.0,
    ) -> None:
        """
        Mark a request as complete and update metrics.

        Args:
            context: Request context
            success: Whether request succeeded
            error: Error message if failed
            bytes_generated: Bytes of audio generated
            audio_seconds: Seconds of audio generated
        """
        context.end_time = time.time()
        context.success = success
        context.error = error

        # Update metrics
        self.metrics.requests_made += 1
        self.metrics.total_latency_ms += context.latency_ms

        if success:
            self.metrics.requests_succeeded += 1
            self.metrics.bytes_synthesized += bytes_generated
            self.metrics.audio_seconds_generated += audio_seconds
        else:
            self.metrics.requests_failed += 1
            if error:
                self.metrics.errors.append(error)

        # Remove from pending
        self._pending_requests.pop(context.request_id, None)

    @asynccontextmanager
    async def request(
        self,
        operation: str,
        voice_id: Optional[str] = None,
        text_length: Optional[int] = None,
        **metadata,
    ) -> AsyncIterator[RequestContext]:
        """
        Context manager for a request within the session.

        Args:
            operation: Operation name
            voice_id: Voice ID if applicable
            text_length: Text length if applicable
            **metadata: Additional metadata

        Yields:
            RequestContext

        Example:
            async with session.request("synthesize", voice_id="warm") as ctx:
                audio = await do_synthesis()
                ctx.metadata["audio_bytes"] = len(audio)
        """
        context = self.create_request_context(
            operation=operation,
            voice_id=voice_id,
            text_length=text_length,
            **metadata,
        )

        try:
            yield context
            self.complete_request(context, success=True)
        except VoxError as e:
            self.complete_request(context, success=False, error=str(e))
            raise
        except Exception as e:
            self.complete_request(context, success=False, error=str(e))
            raise VoxError(
                f"Unexpected error in {operation}: {e}",
                context=context.to_error_context(),
                cause=e,
            )

    def get_metrics(self) -> SessionMetrics:
        """Get current session metrics."""
        return self.metrics

    def is_active(self) -> bool:
        """Check if session is active."""
        return self.state == SessionState.ACTIVE

    def get_pending_requests(self) -> List[RequestContext]:
        """Get list of pending requests."""
        return list(self._pending_requests.values())


class SessionPool:
    """
    Pool of VØX sessions for high-throughput scenarios.

    Manages multiple sessions with:
        - Automatic session creation and recycling
        - Load balancing across sessions
        - Session health monitoring
    """

    def __init__(
        self,
        config: Optional[VoxConfig] = None,
        max_sessions: int = 5,
        max_requests_per_session: int = 100,
    ):
        """
        Initialize session pool.

        Args:
            config: SDK configuration
            max_sessions: Maximum number of concurrent sessions
            max_requests_per_session: Max requests before session recycle
        """
        self.config = config or VoxConfig()
        self.max_sessions = max_sessions
        self.max_requests_per_session = max_requests_per_session

        self._sessions: List[VoxSession] = []
        self._lock = asyncio.Lock()

    async def get_session(self) -> VoxSession:
        """
        Get an available session from the pool.

        Returns:
            VoxSession
        """
        async with self._lock:
            # Find an active session with capacity
            for session in self._sessions:
                if session.is_active():
                    if session.metrics.requests_made < self.max_requests_per_session:
                        return session

            # Create new session if under limit
            if len(self._sessions) < self.max_sessions:
                session = VoxSession(config=self.config)
                await session._initialize()
                self._sessions.append(session)
                return session

            # Return least-used session
            active_sessions = [s for s in self._sessions if s.is_active()]
            if active_sessions:
                return min(active_sessions, key=lambda s: s.metrics.requests_made)

            # Create new session anyway (recycle old one)
            oldest = self._sessions.pop(0)
            await oldest._cleanup()

            session = VoxSession(config=self.config)
            await session._initialize()
            self._sessions.append(session)
            return session

    async def close_all(self) -> None:
        """Close all sessions in the pool."""
        async with self._lock:
            for session in self._sessions:
                await session._cleanup()
            self._sessions.clear()

    def get_pool_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics from all sessions."""
        return {
            "active_sessions": sum(1 for s in self._sessions if s.is_active()),
            "total_sessions": len(self._sessions),
            "total_requests": sum(s.metrics.requests_made for s in self._sessions),
            "total_succeeded": sum(s.metrics.requests_succeeded for s in self._sessions),
            "total_failed": sum(s.metrics.requests_failed for s in self._sessions),
            "total_audio_seconds": sum(s.metrics.audio_seconds_generated for s in self._sessions),
        }
