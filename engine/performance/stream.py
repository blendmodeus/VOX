"""
VØX Performance - Streaming Buffer
----------------------------------

Zero-copy streaming with backpressure support.

Features:
    - Efficient chunk buffering
    - Backpressure handling
    - Memory-limited buffers
    - Async iteration support

AXIØM Phase 9: Optimize - "What makes this solution efficient?"
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, AsyncIterator, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BufferState(str, Enum):
    """State of a stream buffer."""
    EMPTY = "empty"
    FILLING = "filling"
    READY = "ready"
    DRAINING = "draining"
    FULL = "full"
    CLOSED = "closed"


@dataclass
class BufferStats:
    """Statistics for buffer monitoring."""
    chunks_received: int = 0
    chunks_delivered: int = 0
    bytes_received: int = 0
    bytes_delivered: int = 0
    current_size: int = 0
    peak_size: int = 0
    backpressure_events: int = 0
    underflow_events: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunks_received": self.chunks_received,
            "chunks_delivered": self.chunks_delivered,
            "bytes_received": self.bytes_received,
            "bytes_delivered": self.bytes_delivered,
            "current_size": self.current_size,
            "peak_size": self.peak_size,
            "backpressure_events": self.backpressure_events,
            "underflow_events": self.underflow_events,
        }


@dataclass
class Chunk(Generic[T]):
    """A chunk of streaming data."""
    data: T
    size_bytes: int
    sequence: int
    timestamp: float = field(default_factory=time.time)
    is_final: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamBuffer(Generic[T]):
    """
    Efficient buffer for streaming data with backpressure.

    Features:
        - Configurable high/low watermarks
        - Async producer/consumer pattern
        - Memory limits
        - Backpressure signaling
    """

    def __init__(
        self,
        max_size: int = 100,
        max_bytes: Optional[int] = None,
        high_watermark: float = 0.8,
        low_watermark: float = 0.2,
    ):
        """
        Initialize stream buffer.

        Args:
            max_size: Maximum number of chunks
            max_bytes: Maximum bytes (None = unlimited)
            high_watermark: Trigger backpressure (fraction of max)
            low_watermark: Resume production (fraction of max)
        """
        self.max_size = max_size
        self.max_bytes = max_bytes
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark

        self._buffer: deque[Chunk[T]] = deque()
        self._current_bytes = 0
        self._sequence = 0
        self._state = BufferState.EMPTY
        self._stats = BufferStats()

        # Async primitives
        self._not_empty = asyncio.Event()
        self._not_full = asyncio.Event()
        self._not_full.set()
        self._closed = False
        self._lock = asyncio.Lock()

    async def put(
        self,
        data: T,
        size_bytes: int = 0,
        is_final: bool = False,
        timeout: Optional[float] = None,
        **metadata,
    ) -> bool:
        """
        Put data into the buffer.

        Args:
            data: Data chunk
            size_bytes: Size of data in bytes
            is_final: True if this is the last chunk
            timeout: Wait timeout (None = wait forever)
            **metadata: Additional metadata

        Returns:
            True if data was added

        Raises:
            asyncio.TimeoutError: If timeout expires
        """
        if self._closed:
            return False

        # Wait if buffer is full
        if self._is_full():
            self._stats.backpressure_events += 1
            try:
                await asyncio.wait_for(
                    self._not_full.wait(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                return False

        async with self._lock:
            if self._closed:
                return False

            chunk = Chunk(
                data=data,
                size_bytes=size_bytes,
                sequence=self._sequence,
                is_final=is_final,
                metadata=metadata,
            )
            self._sequence += 1

            self._buffer.append(chunk)
            self._current_bytes += size_bytes

            # Update stats
            self._stats.chunks_received += 1
            self._stats.bytes_received += size_bytes
            self._stats.current_size = len(self._buffer)
            if self._stats.current_size > self._stats.peak_size:
                self._stats.peak_size = self._stats.current_size

            # Update state
            self._update_state()
            self._not_empty.set()

            if self._is_full():
                self._not_full.clear()

        return True

    async def get(
        self,
        timeout: Optional[float] = None,
    ) -> Optional[Chunk[T]]:
        """
        Get data from the buffer.

        Args:
            timeout: Wait timeout (None = wait forever)

        Returns:
            Chunk or None if buffer is closed/empty
        """
        # Wait if buffer is empty
        if len(self._buffer) == 0 and not self._closed:
            self._stats.underflow_events += 1
            try:
                await asyncio.wait_for(
                    self._not_empty.wait(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                return None

        async with self._lock:
            if len(self._buffer) == 0:
                if self._closed:
                    return None
                self._not_empty.clear()
                return None

            chunk = self._buffer.popleft()
            self._current_bytes -= chunk.size_bytes

            # Update stats
            self._stats.chunks_delivered += 1
            self._stats.bytes_delivered += chunk.size_bytes
            self._stats.current_size = len(self._buffer)

            # Update state
            self._update_state()

            if len(self._buffer) == 0:
                self._not_empty.clear()

            # Signal producer if below low watermark
            if not self._is_full():
                self._not_full.set()

            return chunk

    async def close(self) -> None:
        """Close the buffer."""
        async with self._lock:
            self._closed = True
            self._state = BufferState.CLOSED
            self._not_empty.set()
            self._not_full.set()

    def is_closed(self) -> bool:
        """Check if buffer is closed."""
        return self._closed

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._buffer) == 0

    def size(self) -> int:
        """Get current number of chunks."""
        return len(self._buffer)

    def bytes_size(self) -> int:
        """Get current size in bytes."""
        return self._current_bytes

    def get_stats(self) -> BufferStats:
        """Get buffer statistics."""
        return BufferStats(
            chunks_received=self._stats.chunks_received,
            chunks_delivered=self._stats.chunks_delivered,
            bytes_received=self._stats.bytes_received,
            bytes_delivered=self._stats.bytes_delivered,
            current_size=len(self._buffer),
            peak_size=self._stats.peak_size,
            backpressure_events=self._stats.backpressure_events,
            underflow_events=self._stats.underflow_events,
        )

    def _is_full(self) -> bool:
        """Check if buffer is at high watermark."""
        if len(self._buffer) >= self.max_size:
            return True
        if self.max_bytes and self._current_bytes >= self.max_bytes * self.high_watermark:
            return True
        return False

    def _update_state(self) -> None:
        """Update buffer state."""
        size = len(self._buffer)

        if size == 0:
            self._state = BufferState.EMPTY
        elif size >= self.max_size:
            self._state = BufferState.FULL
        elif size >= self.max_size * self.high_watermark:
            self._state = BufferState.DRAINING
        elif size <= self.max_size * self.low_watermark:
            self._state = BufferState.FILLING
        else:
            self._state = BufferState.READY

    async def __aiter__(self) -> AsyncIterator[Chunk[T]]:
        """Async iteration over buffer contents."""
        while True:
            chunk = await self.get()
            if chunk is None:
                break
            yield chunk
            if chunk.is_final:
                break


class AudioStreamBuffer(StreamBuffer[bytes]):
    """
    Specialized buffer for audio streaming.

    Features:
        - Audio-specific defaults
        - Chunk duration tracking
        - Playback-ready iteration
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        bytes_per_sample: int = 2,
        buffer_seconds: float = 2.0,
        **kwargs,
    ):
        """
        Initialize audio stream buffer.

        Args:
            sample_rate: Audio sample rate
            bytes_per_sample: Bytes per audio sample
            buffer_seconds: Target buffer duration
            **kwargs: Additional StreamBuffer arguments
        """
        self.sample_rate = sample_rate
        self.bytes_per_sample = bytes_per_sample
        self.buffer_seconds = buffer_seconds

        # Calculate max bytes from buffer duration
        bytes_per_second = sample_rate * bytes_per_sample
        max_bytes = int(buffer_seconds * bytes_per_second)

        super().__init__(max_bytes=max_bytes, **kwargs)

    def buffered_seconds(self) -> float:
        """Get current buffer duration in seconds."""
        bytes_per_second = self.sample_rate * self.bytes_per_sample
        return self._current_bytes / bytes_per_second

    async def put_audio(
        self,
        audio: bytes,
        is_final: bool = False,
        **metadata,
    ) -> bool:
        """
        Put audio data into buffer.

        Args:
            audio: Audio bytes
            is_final: True if this is the last chunk
            **metadata: Additional metadata

        Returns:
            True if audio was added
        """
        return await self.put(
            data=audio,
            size_bytes=len(audio),
            is_final=is_final,
            **metadata,
        )

    async def get_audio(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """
        Get audio data from buffer.

        Args:
            timeout: Wait timeout

        Returns:
            Audio bytes or None
        """
        chunk = await self.get(timeout=timeout)
        return chunk.data if chunk else None


class StreamPipeline:
    """
    Pipeline for chaining stream processors.

    Features:
        - Compose multiple processors
        - Automatic buffer management
        - Error propagation
    """

    def __init__(self):
        """Initialize stream pipeline."""
        self._stages: List[tuple[str, callable]] = []
        self._buffers: Dict[str, StreamBuffer] = {}

    def add_stage(
        self,
        name: str,
        processor: callable,
        buffer_size: int = 10,
    ) -> "StreamPipeline":
        """
        Add a processing stage.

        Args:
            name: Stage name
            processor: Async function to process chunks
            buffer_size: Output buffer size

        Returns:
            Self for chaining
        """
        self._stages.append((name, processor))
        self._buffers[name] = StreamBuffer(max_size=buffer_size)
        return self

    async def process(
        self,
        source: AsyncIterator[bytes],
    ) -> AsyncIterator[bytes]:
        """
        Process data through the pipeline.

        Args:
            source: Source async iterator

        Yields:
            Processed data chunks
        """
        if not self._stages:
            async for chunk in source:
                yield chunk
            return

        # Start pipeline tasks
        tasks = []

        # First stage reads from source
        name, processor = self._stages[0]
        output_buffer = self._buffers[name]

        async def run_first_stage():
            async for chunk in source:
                result = await processor(chunk)
                await output_buffer.put(result, size_bytes=len(result))
            await output_buffer.close()

        tasks.append(asyncio.create_task(run_first_stage()))

        # Middle stages
        for i in range(1, len(self._stages)):
            prev_name = self._stages[i - 1][0]
            name, processor = self._stages[i]
            input_buffer = self._buffers[prev_name]
            output_buffer = self._buffers[name]

            async def run_stage(inp, proc, out):
                async for chunk in inp:
                    result = await proc(chunk.data)
                    await out.put(result, size_bytes=len(result))
                await out.close()

            tasks.append(asyncio.create_task(
                run_stage(input_buffer, processor, output_buffer)
            ))

        # Yield from final buffer
        final_buffer = self._buffers[self._stages[-1][0]]
        async for chunk in final_buffer:
            yield chunk.data

        # Wait for all tasks
        await asyncio.gather(*tasks, return_exceptions=True)


# Global audio buffer
_audio_buffer: Optional[AudioStreamBuffer] = None


def get_audio_buffer() -> AudioStreamBuffer:
    """Get global audio buffer instance."""
    global _audio_buffer
    if _audio_buffer is None:
        _audio_buffer = AudioStreamBuffer()
    return _audio_buffer


def set_audio_buffer(buffer: AudioStreamBuffer) -> None:
    """Set global audio buffer instance."""
    global _audio_buffer
    _audio_buffer = buffer
