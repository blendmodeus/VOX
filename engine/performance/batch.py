"""
VØX Performance - Batch Optimization
------------------------------------

Intelligent batching for efficient multi-item processing.

Features:
    - Automatic request batching
    - Parallel execution with concurrency limits
    - Voice-grouped batching for efficiency
    - Progress tracking and callbacks

AXIØM Phase 9: Optimize - "What makes this solution efficient?"
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, TypeVar, Generic, Awaitable
import heapq

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class BatchStrategy(str, Enum):
    """Strategy for batch execution."""
    PARALLEL = "parallel"          # All items in parallel
    SEQUENTIAL = "sequential"      # One at a time
    CHUNKED = "chunked"            # Fixed-size chunks
    VOICE_GROUPED = "voice_grouped"  # Group by voice ID
    PRIORITY = "priority"          # Priority queue ordering


@dataclass
class BatchStats:
    """Statistics for batch processing."""
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    total_batches: int = 0
    avg_batch_size: float = 0.0
    total_time_ms: float = 0.0
    avg_item_time_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.completed_items + self.failed_items
        return self.completed_items / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "failed_items": self.failed_items,
            "success_rate": self.success_rate,
            "total_batches": self.total_batches,
            "avg_batch_size": self.avg_batch_size,
            "total_time_ms": self.total_time_ms,
            "avg_item_time_ms": self.avg_item_time_ms,
        }


@dataclass
class BatchItem(Generic[T]):
    """A single item in a batch."""
    id: str
    data: T
    priority: int = 0
    voice_id: Optional[str] = None
    submitted_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None

    def __lt__(self, other: "BatchItem") -> bool:
        """Compare for priority queue (higher priority first)."""
        return self.priority > other.priority


@dataclass
class BatchResult(Generic[R]):
    """Result of a batch operation."""
    items: List[BatchItem[R]]
    stats: BatchStats
    success: bool = True
    errors: List[str] = field(default_factory=list)

    @property
    def results(self) -> List[R]:
        """Get successful results."""
        return [item.result for item in self.items if item.result is not None]

    @property
    def failed_ids(self) -> List[str]:
        """Get IDs of failed items."""
        return [item.id for item in self.items if item.error is not None]


class BatchOptimizer:
    """
    Optimizes batch processing for synthesis operations.

    Features:
        - Multiple batching strategies
        - Concurrency limiting
        - Voice-based grouping for model efficiency
        - Priority support
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        chunk_size: int = 5,
        strategy: BatchStrategy = BatchStrategy.PARALLEL,
        timeout_per_item: float = 30.0,
    ):
        """
        Initialize batch optimizer.

        Args:
            max_concurrent: Maximum concurrent operations
            chunk_size: Size of chunks for CHUNKED strategy
            strategy: Default batching strategy
            timeout_per_item: Timeout per item in seconds
        """
        self.max_concurrent = max_concurrent
        self.chunk_size = chunk_size
        self.strategy = strategy
        self.timeout_per_item = timeout_per_item

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._stats = BatchStats()

    async def process(
        self,
        items: List[T],
        processor: Callable[[T], Awaitable[R]],
        strategy: Optional[BatchStrategy] = None,
        voice_key: Optional[Callable[[T], str]] = None,
        priority_key: Optional[Callable[[T], int]] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult[R]:
        """
        Process a batch of items.

        Args:
            items: Items to process
            processor: Async function to process each item
            strategy: Override default strategy
            voice_key: Function to extract voice ID for grouping
            priority_key: Function to extract priority
            on_progress: Callback for progress updates (completed, total)

        Returns:
            BatchResult with all results
        """
        strategy = strategy or self.strategy
        start_time = time.time()

        # Create batch items
        batch_items = []
        for i, item in enumerate(items):
            voice_id = voice_key(item) if voice_key else None
            priority = priority_key(item) if priority_key else 0

            batch_items.append(BatchItem(
                id=f"item_{i}",
                data=item,
                priority=priority,
                voice_id=voice_id,
            ))

        # Process based on strategy
        if strategy == BatchStrategy.SEQUENTIAL:
            await self._process_sequential(batch_items, processor, on_progress)
        elif strategy == BatchStrategy.CHUNKED:
            await self._process_chunked(batch_items, processor, on_progress)
        elif strategy == BatchStrategy.VOICE_GROUPED:
            await self._process_voice_grouped(batch_items, processor, on_progress)
        elif strategy == BatchStrategy.PRIORITY:
            await self._process_priority(batch_items, processor, on_progress)
        else:  # PARALLEL
            await self._process_parallel(batch_items, processor, on_progress)

        # Compute stats
        total_time = (time.time() - start_time) * 1000
        completed = sum(1 for item in batch_items if item.result is not None)
        failed = sum(1 for item in batch_items if item.error is not None)

        stats = BatchStats(
            total_items=len(batch_items),
            completed_items=completed,
            failed_items=failed,
            total_batches=1,
            avg_batch_size=len(batch_items),
            total_time_ms=total_time,
            avg_item_time_ms=total_time / len(batch_items) if batch_items else 0,
        )

        errors = [item.error for item in batch_items if item.error]

        return BatchResult(
            items=batch_items,
            stats=stats,
            success=failed == 0,
            errors=errors,
        )

    async def _process_parallel(
        self,
        items: List[BatchItem[T]],
        processor: Callable[[T], Awaitable[R]],
        on_progress: Optional[Callable[[int, int], None]],
    ) -> None:
        """Process all items in parallel with semaphore."""
        completed = 0
        total = len(items)

        async def process_one(item: BatchItem[T]) -> None:
            nonlocal completed
            async with self._semaphore:
                try:
                    item.result = await asyncio.wait_for(
                        processor(item.data),
                        timeout=self.timeout_per_item,
                    )
                except asyncio.TimeoutError:
                    item.error = f"Timeout after {self.timeout_per_item}s"
                except Exception as e:
                    item.error = str(e)
                finally:
                    item.completed_at = time.time()
                    completed += 1
                    if on_progress:
                        on_progress(completed, total)

        await asyncio.gather(*[process_one(item) for item in items])

    async def _process_sequential(
        self,
        items: List[BatchItem[T]],
        processor: Callable[[T], Awaitable[R]],
        on_progress: Optional[Callable[[int, int], None]],
    ) -> None:
        """Process items one at a time."""
        total = len(items)

        for i, item in enumerate(items):
            try:
                item.result = await asyncio.wait_for(
                    processor(item.data),
                    timeout=self.timeout_per_item,
                )
            except asyncio.TimeoutError:
                item.error = f"Timeout after {self.timeout_per_item}s"
            except Exception as e:
                item.error = str(e)
            finally:
                item.completed_at = time.time()
                if on_progress:
                    on_progress(i + 1, total)

    async def _process_chunked(
        self,
        items: List[BatchItem[T]],
        processor: Callable[[T], Awaitable[R]],
        on_progress: Optional[Callable[[int, int], None]],
    ) -> None:
        """Process in fixed-size chunks."""
        total = len(items)
        completed = 0

        for i in range(0, len(items), self.chunk_size):
            chunk = items[i:i + self.chunk_size]

            async def process_one(item: BatchItem[T]) -> None:
                try:
                    item.result = await asyncio.wait_for(
                        processor(item.data),
                        timeout=self.timeout_per_item,
                    )
                except asyncio.TimeoutError:
                    item.error = f"Timeout after {self.timeout_per_item}s"
                except Exception as e:
                    item.error = str(e)
                finally:
                    item.completed_at = time.time()

            await asyncio.gather(*[process_one(item) for item in chunk])

            completed += len(chunk)
            if on_progress:
                on_progress(completed, total)

    async def _process_voice_grouped(
        self,
        items: List[BatchItem[T]],
        processor: Callable[[T], Awaitable[R]],
        on_progress: Optional[Callable[[int, int], None]],
    ) -> None:
        """Group items by voice and process groups sequentially."""
        total = len(items)
        completed = 0

        # Group by voice
        voice_groups: Dict[Optional[str], List[BatchItem[T]]] = defaultdict(list)
        for item in items:
            voice_groups[item.voice_id].append(item)

        # Process each voice group in parallel, groups sequentially
        for voice_id, group in voice_groups.items():
            async def process_one(item: BatchItem[T]) -> None:
                nonlocal completed
                async with self._semaphore:
                    try:
                        item.result = await asyncio.wait_for(
                            processor(item.data),
                            timeout=self.timeout_per_item,
                        )
                    except asyncio.TimeoutError:
                        item.error = f"Timeout after {self.timeout_per_item}s"
                    except Exception as e:
                        item.error = str(e)
                    finally:
                        item.completed_at = time.time()
                        completed += 1
                        if on_progress:
                            on_progress(completed, total)

            await asyncio.gather(*[process_one(item) for item in group])

    async def _process_priority(
        self,
        items: List[BatchItem[T]],
        processor: Callable[[T], Awaitable[R]],
        on_progress: Optional[Callable[[int, int], None]],
    ) -> None:
        """Process items by priority (highest first)."""
        total = len(items)
        completed = 0

        # Create priority queue
        pq = list(items)
        heapq.heapify(pq)

        while pq:
            # Get batch of highest priority items
            batch = []
            while pq and len(batch) < self.max_concurrent:
                batch.append(heapq.heappop(pq))

            async def process_one(item: BatchItem[T]) -> None:
                nonlocal completed
                try:
                    item.result = await asyncio.wait_for(
                        processor(item.data),
                        timeout=self.timeout_per_item,
                    )
                except asyncio.TimeoutError:
                    item.error = f"Timeout after {self.timeout_per_item}s"
                except Exception as e:
                    item.error = str(e)
                finally:
                    item.completed_at = time.time()
                    completed += 1
                    if on_progress:
                        on_progress(completed, total)

            await asyncio.gather(*[process_one(item) for item in batch])


class SynthesisBatchOptimizer(BatchOptimizer):
    """
    Specialized batch optimizer for synthesis operations.

    Features:
        - Automatic voice grouping
        - Cache integration
        - Quality-based retry
    """

    def __init__(
        self,
        synthesize_fn: Callable,
        cache: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize synthesis batch optimizer.

        Args:
            synthesize_fn: Function to synthesize audio
            cache: Optional AudioCache for caching results
            **kwargs: Additional BatchOptimizer arguments
        """
        super().__init__(strategy=BatchStrategy.VOICE_GROUPED, **kwargs)
        self.synthesize_fn = synthesize_fn
        self.cache = cache

    async def synthesize_batch(
        self,
        items: List[Dict[str, Any]],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult:
        """
        Synthesize a batch of items.

        Args:
            items: List of dicts with 'text', 'voice_id', and params
            on_progress: Progress callback

        Returns:
            BatchResult with audio bytes
        """
        async def process_item(item: Dict[str, Any]) -> bytes:
            text = item.get("text", "")
            voice_id = item.get("voice_id")
            params = {k: v for k, v in item.items() if k not in ("text", "voice_id")}

            # Check cache first
            if self.cache:
                cached = self.cache.get(text, voice_id, **params)
                if cached is not None:
                    return cached

            # Synthesize
            result = await self.synthesize_fn(text, voice_id=voice_id, **params)

            # Extract audio bytes
            if hasattr(result, "audio"):
                audio = result.audio
            elif isinstance(result, bytes):
                audio = result
            else:
                audio = bytes(result)

            # Cache result
            if self.cache:
                self.cache.put(text, voice_id, audio, **params)

            return audio

        return await self.process(
            items=items,
            processor=process_item,
            voice_key=lambda x: x.get("voice_id"),
            on_progress=on_progress,
        )


class RequestBatcher:
    """
    Automatic request batching with time-based flushing.

    Collects requests and processes them in batches based on
    time or count thresholds.
    """

    def __init__(
        self,
        processor: Callable[[List[T]], Awaitable[List[R]]],
        max_batch_size: int = 10,
        max_wait_ms: float = 50.0,
    ):
        """
        Initialize request batcher.

        Args:
            processor: Function to process a batch of items
            max_batch_size: Maximum items before auto-flush
            max_wait_ms: Maximum wait time before auto-flush
        """
        self.processor = processor
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        self._queue: List[tuple[T, asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._started = False

    async def start(self) -> None:
        """Start the batcher."""
        self._started = True
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        """Stop the batcher and flush remaining items."""
        self._started = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush remaining
        await self._flush()

    async def submit(self, item: T) -> R:
        """
        Submit an item for batched processing.

        Args:
            item: Item to process

        Returns:
            Processing result
        """
        future = asyncio.get_event_loop().create_future()

        async with self._lock:
            self._queue.append((item, future))

            # Auto-flush if batch is full
            if len(self._queue) >= self.max_batch_size:
                asyncio.create_task(self._flush())

        return await future

    async def _flush_loop(self) -> None:
        """Background loop for time-based flushing."""
        while self._started:
            await asyncio.sleep(self.max_wait_ms / 1000)
            if self._queue:
                await self._flush()

    async def _flush(self) -> None:
        """Flush current batch."""
        async with self._lock:
            if not self._queue:
                return

            batch = self._queue
            self._queue = []

        items = [item for item, _ in batch]
        futures = [future for _, future in batch]

        try:
            results = await self.processor(items)

            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)

        except Exception as e:
            for future in futures:
                if not future.done():
                    future.set_exception(e)


# Global batch optimizer
_batch_optimizer: Optional[BatchOptimizer] = None


def get_batch_optimizer() -> BatchOptimizer:
    """Get global batch optimizer instance."""
    global _batch_optimizer
    if _batch_optimizer is None:
        _batch_optimizer = BatchOptimizer()
    return _batch_optimizer


def set_batch_optimizer(optimizer: BatchOptimizer) -> None:
    """Set global batch optimizer instance."""
    global _batch_optimizer
    _batch_optimizer = optimizer
