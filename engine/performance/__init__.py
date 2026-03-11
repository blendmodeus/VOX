"""
VØX Performance Layer
---------------------

Performance optimization components for VØX.

Features:
    - LRU caching for audio and embeddings
    - HTTP/WebSocket connection pooling
    - Intelligent batch processing
    - Zero-copy streaming buffers
    - Lazy loading of heavy modules

Quick Start:
    >>> from axiom_vox.performance import (
    ...     CacheManager, AudioCache, EmbeddingCache,
    ...     HTTPConnectionPool, BatchOptimizer,
    ... )
    >>>
    >>> # Enable caching
    >>> cache = CacheManager(total_memory_mb=256)
    >>> audio = cache.audio.get("Hello", "warm")
    >>>
    >>> # Batch processing
    >>> optimizer = BatchOptimizer(max_concurrent=10)
    >>> results = await optimizer.process(items, processor)

AXIØM Phase 9: Optimize - "What makes this solution efficient?"
"""

from .cache import (
    # Stats
    CacheStats,
    CacheEntry,
    # LRU Cache
    LRUCache,
    # Specialized caches
    AudioCache,
    EmbeddingCache,
    # Manager
    CacheManager,
    get_cache_manager,
    set_cache_manager,
)

from .pool import (
    # States
    ConnectionState,
    ConnectionStats,
    PooledConnection,
    # Pools
    HTTPConnectionPool,
    WebSocketPool,
    # Manager
    ConnectionPoolManager,
    get_pool_manager,
    set_pool_manager,
    create_pool_manager,
)

from .batch import (
    # Enums
    BatchStrategy,
    # Stats
    BatchStats,
    BatchItem,
    BatchResult,
    # Optimizers
    BatchOptimizer,
    SynthesisBatchOptimizer,
    RequestBatcher,
    get_batch_optimizer,
    set_batch_optimizer,
)

from .stream import (
    # States
    BufferState,
    BufferStats,
    Chunk,
    # Buffers
    StreamBuffer,
    AudioStreamBuffer,
    StreamPipeline,
    get_audio_buffer,
    set_audio_buffer,
)

from .lazy import (
    # States
    LoadState,
    LoadStats,
    LazyResource,
    # Loaders
    LazyLoader,
    ModuleLazyLoader,
    LazyAttribute,
    lazy_import,
    get_resource_loader,
    get_module_loader,
    set_resource_loader,
    set_module_loader,
)


__all__ = [
    # Cache
    "CacheStats",
    "CacheEntry",
    "LRUCache",
    "AudioCache",
    "EmbeddingCache",
    "CacheManager",
    "get_cache_manager",
    "set_cache_manager",
    # Pool
    "ConnectionState",
    "ConnectionStats",
    "PooledConnection",
    "HTTPConnectionPool",
    "WebSocketPool",
    "ConnectionPoolManager",
    "get_pool_manager",
    "set_pool_manager",
    "create_pool_manager",
    # Batch
    "BatchStrategy",
    "BatchStats",
    "BatchItem",
    "BatchResult",
    "BatchOptimizer",
    "SynthesisBatchOptimizer",
    "RequestBatcher",
    "get_batch_optimizer",
    "set_batch_optimizer",
    # Stream
    "BufferState",
    "BufferStats",
    "Chunk",
    "StreamBuffer",
    "AudioStreamBuffer",
    "StreamPipeline",
    "get_audio_buffer",
    "set_audio_buffer",
    # Lazy
    "LoadState",
    "LoadStats",
    "LazyResource",
    "LazyLoader",
    "ModuleLazyLoader",
    "LazyAttribute",
    "lazy_import",
    "get_resource_loader",
    "get_module_loader",
    "set_resource_loader",
    "set_module_loader",
]
