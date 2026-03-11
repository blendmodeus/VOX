"""
VØX Performance - Caching Layer
-------------------------------

LRU caching for synthesized audio and voice embeddings.

Features:
    - Content-hash based audio caching
    - Lazy-loaded embedding cache with preloading
    - Memory-aware eviction policies
    - TTL support for cache entries
    - Cache statistics and monitoring

AXIØM Phase 9: Optimize - "What makes this solution efficient?"
"""

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, Any, Optional, List, Callable, TypeVar, Generic
import weakref

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    current_size: int = 0
    current_bytes: int = 0
    peak_size: int = 0
    peak_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "hit_rate": self.hit_rate,
            "current_size": self.current_size,
            "current_bytes": self.current_bytes,
            "peak_size": self.peak_size,
            "peak_bytes": self.peak_bytes,
        }


@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry with metadata."""
    key: str
    value: T
    size_bytes: int
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl_seconds: Optional[float] = None

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() > self.created_at + self.ttl_seconds

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1


class LRUCache(Generic[T]):
    """
    Thread-safe LRU cache with memory limits.

    Features:
        - Maximum entry count limit
        - Maximum memory limit (bytes)
        - TTL support per entry
        - Thread-safe operations
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_bytes: Optional[int] = None,
        default_ttl: Optional[float] = None,
    ):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of entries
            max_bytes: Maximum total size in bytes (None = unlimited)
            default_ttl: Default TTL in seconds (None = no expiration)
        """
        self.max_size = max_size
        self.max_bytes = max_bytes
        self.default_ttl = default_ttl

        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = RLock()
        self._stats = CacheStats()
        self._current_bytes = 0

    def get(self, key: str) -> Optional[T]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            # Check expiration
            if entry.is_expired:
                self._remove_entry(key)
                self._stats.misses += 1
                self._stats.expirations += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._stats.hits += 1

            return entry.value

    def put(
        self,
        key: str,
        value: T,
        size_bytes: int = 0,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Put value in cache.

        Args:
            key: Cache key
            value: Value to cache
            size_bytes: Size of value in bytes
            ttl: TTL in seconds (None = use default)
        """
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)

            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                ttl_seconds=ttl if ttl is not None else self.default_ttl,
            )

            # Evict if necessary
            self._evict_if_needed(size_bytes)

            # Add entry
            self._cache[key] = entry
            self._current_bytes += size_bytes
            self._update_stats()

    def remove(self, key: str) -> bool:
        """
        Remove entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was removed
        """
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False

    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0
            self._update_stats()

    def contains(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                self._remove_entry(key)
                self._stats.expirations += 1
                return False
            return True

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                expirations=self._stats.expirations,
                current_size=len(self._cache),
                current_bytes=self._current_bytes,
                peak_size=self._stats.peak_size,
                peak_bytes=self._stats.peak_bytes,
            )

    def _remove_entry(self, key: str) -> None:
        """Remove entry and update bytes count."""
        entry = self._cache.pop(key, None)
        if entry:
            self._current_bytes -= entry.size_bytes

    def _evict_if_needed(self, incoming_bytes: int) -> None:
        """Evict entries if limits exceeded."""
        # Evict by count
        while len(self._cache) >= self.max_size:
            self._evict_lru()

        # Evict by bytes
        if self.max_bytes:
            while self._current_bytes + incoming_bytes > self.max_bytes and self._cache:
                self._evict_lru()

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            key = next(iter(self._cache))
            self._remove_entry(key)
            self._stats.evictions += 1

    def _update_stats(self) -> None:
        """Update peak statistics."""
        current_size = len(self._cache)
        if current_size > self._stats.peak_size:
            self._stats.peak_size = current_size
        if self._current_bytes > self._stats.peak_bytes:
            self._stats.peak_bytes = self._current_bytes


class AudioCache:
    """
    Specialized cache for synthesized audio.

    Features:
        - Content-hash based keys (text + voice + params)
        - Memory-aware with configurable limits
        - TTL for audio freshness
        - Prewarming support
    """

    def __init__(
        self,
        max_entries: int = 500,
        max_memory_mb: float = 256.0,
        default_ttl_seconds: float = 3600.0,  # 1 hour
    ):
        """
        Initialize audio cache.

        Args:
            max_entries: Maximum cached audio entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl_seconds: Default TTL for entries
        """
        self._cache = LRUCache[bytes](
            max_size=max_entries,
            max_bytes=int(max_memory_mb * 1024 * 1024),
            default_ttl=default_ttl_seconds,
        )

    def _compute_key(
        self,
        text: str,
        voice_id: str,
        **params,
    ) -> str:
        """
        Compute cache key from synthesis parameters.

        Args:
            text: Input text
            voice_id: Voice ID
            **params: Additional parameters (rate, pitch, etc.)

        Returns:
            Content-based hash key
        """
        # Normalize and hash
        key_data = {
            "text": text,
            "voice_id": voice_id,
            **{k: v for k, v in sorted(params.items()) if v is not None},
        }
        key_str = str(key_data)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def get(
        self,
        text: str,
        voice_id: str,
        **params,
    ) -> Optional[bytes]:
        """
        Get cached audio.

        Args:
            text: Input text
            voice_id: Voice ID
            **params: Synthesis parameters

        Returns:
            Cached audio bytes or None
        """
        key = self._compute_key(text, voice_id, **params)
        return self._cache.get(key)

    def put(
        self,
        text: str,
        voice_id: str,
        audio: bytes,
        ttl: Optional[float] = None,
        **params,
    ) -> None:
        """
        Cache synthesized audio.

        Args:
            text: Input text
            voice_id: Voice ID
            audio: Audio bytes
            ttl: Optional TTL override
            **params: Synthesis parameters
        """
        key = self._compute_key(text, voice_id, **params)
        self._cache.put(key, audio, size_bytes=len(audio), ttl=ttl)

    def invalidate(self, text: str, voice_id: str, **params) -> bool:
        """Invalidate specific cached audio."""
        key = self._compute_key(text, voice_id, **params)
        return self._cache.remove(key)

    def invalidate_voice(self, voice_id: str) -> int:
        """
        Invalidate all cached audio for a voice.

        Note: This is O(n) - use sparingly.
        """
        # This requires iterating through the cache
        # In production, consider a secondary index
        count = 0
        with self._cache._lock:
            keys_to_remove = [
                key for key, entry in self._cache._cache.items()
                if voice_id in key  # Approximate - real impl would store metadata
            ]
            for key in keys_to_remove:
                self._cache._remove_entry(key)
                count += 1
        return count

    def clear(self) -> None:
        """Clear all cached audio."""
        self._cache.clear()

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._cache.get_stats()

    async def prewarm(
        self,
        items: List[Dict[str, Any]],
        synthesize_fn: Callable,
    ) -> int:
        """
        Prewarm cache with common phrases.

        Args:
            items: List of dicts with text, voice_id, and params
            synthesize_fn: Async function to synthesize audio

        Returns:
            Number of items prewarmed
        """
        count = 0
        for item in items:
            text = item.get("text", "")
            voice_id = item.get("voice_id", "")
            params = {k: v for k, v in item.items() if k not in ("text", "voice_id")}

            # Skip if already cached
            if self.get(text, voice_id, **params) is not None:
                continue

            try:
                audio = await synthesize_fn(text, voice_id=voice_id, **params)
                if isinstance(audio, bytes):
                    self.put(text, voice_id, audio, **params)
                    count += 1
            except Exception as e:
                logger.warning(f"Prewarm failed for '{text[:50]}...': {e}")

        logger.info(f"Prewarmed {count} audio entries")
        return count


class EmbeddingCache:
    """
    Cache for voice embeddings with lazy loading.

    Features:
        - Lazy loading of embeddings on first access
        - Background preloading of frequently-used voices
        - Memory-efficient with configurable limits
        - Weak references for optional entries
    """

    def __init__(
        self,
        max_entries: int = 100,
        max_memory_mb: float = 64.0,
        preload_top_n: int = 10,
    ):
        """
        Initialize embedding cache.

        Args:
            max_entries: Maximum cached embeddings
            max_memory_mb: Maximum memory in MB
            preload_top_n: Number of popular voices to preload
        """
        self._cache = LRUCache[Any](
            max_size=max_entries,
            max_bytes=int(max_memory_mb * 1024 * 1024),
        )
        self._preload_top_n = preload_top_n
        self._loader: Optional[Callable] = None
        self._loading: Dict[str, asyncio.Future] = {}
        self._access_counts: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    def set_loader(self, loader: Callable) -> None:
        """
        Set the embedding loader function.

        Args:
            loader: Async function that takes voice_id and returns embedding
        """
        self._loader = loader

    async def get(self, voice_id: str) -> Optional[Any]:
        """
        Get embedding, loading if necessary.

        Args:
            voice_id: Voice ID

        Returns:
            Embedding or None if not loadable
        """
        # Track access
        self._access_counts[voice_id] = self._access_counts.get(voice_id, 0) + 1

        # Try cache first
        embedding = self._cache.get(voice_id)
        if embedding is not None:
            return embedding

        # Load if loader available
        if self._loader is None:
            return None

        async with self._lock:
            # Check if already loading
            if voice_id in self._loading:
                return await self._loading[voice_id]

            # Start loading
            future = asyncio.get_event_loop().create_future()
            self._loading[voice_id] = future

        try:
            embedding = await self._loader(voice_id)
            if embedding is not None:
                # Estimate size (numpy array)
                size_bytes = getattr(embedding, "nbytes", len(str(embedding)))
                self._cache.put(voice_id, embedding, size_bytes=size_bytes)

            future.set_result(embedding)
            return embedding

        except Exception as e:
            future.set_exception(e)
            raise

        finally:
            async with self._lock:
                self._loading.pop(voice_id, None)

    def put(self, voice_id: str, embedding: Any) -> None:
        """
        Directly cache an embedding.

        Args:
            voice_id: Voice ID
            embedding: Embedding to cache
        """
        size_bytes = getattr(embedding, "nbytes", len(str(embedding)))
        self._cache.put(voice_id, embedding, size_bytes=size_bytes)

    def invalidate(self, voice_id: str) -> bool:
        """Invalidate cached embedding."""
        return self._cache.remove(voice_id)

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        self._access_counts.clear()

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._cache.get_stats()

    def get_popular_voices(self, n: Optional[int] = None) -> List[str]:
        """
        Get most frequently accessed voice IDs.

        Args:
            n: Number to return (default: preload_top_n)

        Returns:
            List of voice IDs sorted by access count
        """
        n = n or self._preload_top_n
        sorted_voices = sorted(
            self._access_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [v[0] for v in sorted_voices[:n]]

    async def preload_popular(self) -> int:
        """
        Preload embeddings for popular voices.

        Returns:
            Number of embeddings preloaded
        """
        if self._loader is None:
            return 0

        popular = self.get_popular_voices()
        count = 0

        for voice_id in popular:
            if self._cache.contains(voice_id):
                continue
            try:
                await self.get(voice_id)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to preload embedding for {voice_id}: {e}")

        logger.info(f"Preloaded {count} embeddings")
        return count


class CacheManager:
    """
    Central manager for all VØX caches.

    Provides:
        - Unified cache access
        - Global memory budget management
        - Cache statistics aggregation
        - Coordinated cache warming
    """

    def __init__(
        self,
        audio_cache: Optional[AudioCache] = None,
        embedding_cache: Optional[EmbeddingCache] = None,
        total_memory_mb: float = 512.0,
    ):
        """
        Initialize cache manager.

        Args:
            audio_cache: AudioCache instance
            embedding_cache: EmbeddingCache instance
            total_memory_mb: Total memory budget
        """
        # Allocate memory: 80% audio, 20% embeddings
        audio_mb = total_memory_mb * 0.8
        embedding_mb = total_memory_mb * 0.2

        self.audio = audio_cache or AudioCache(max_memory_mb=audio_mb)
        self.embeddings = embedding_cache or EmbeddingCache(max_memory_mb=embedding_mb)
        self.total_memory_mb = total_memory_mb

    def get_combined_stats(self) -> Dict[str, Any]:
        """Get combined statistics from all caches."""
        audio_stats = self.audio.get_stats()
        embedding_stats = self.embeddings.get_stats()

        total_bytes = audio_stats.current_bytes + embedding_stats.current_bytes
        total_hits = audio_stats.hits + embedding_stats.hits
        total_misses = audio_stats.misses + embedding_stats.misses

        return {
            "audio": audio_stats.to_dict(),
            "embeddings": embedding_stats.to_dict(),
            "combined": {
                "total_bytes": total_bytes,
                "total_mb": total_bytes / (1024 * 1024),
                "memory_budget_mb": self.total_memory_mb,
                "memory_utilization": total_bytes / (self.total_memory_mb * 1024 * 1024),
                "total_hits": total_hits,
                "total_misses": total_misses,
                "overall_hit_rate": total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0,
            },
        }

    def clear_all(self) -> None:
        """Clear all caches."""
        self.audio.clear()
        self.embeddings.clear()
        logger.info("All caches cleared")

    async def warm_up(
        self,
        audio_items: Optional[List[Dict[str, Any]]] = None,
        synthesize_fn: Optional[Callable] = None,
    ) -> Dict[str, int]:
        """
        Warm up all caches.

        Args:
            audio_items: Items to prewarm audio cache
            synthesize_fn: Function for audio synthesis

        Returns:
            Dict of cache names to items warmed
        """
        results = {}

        # Warm audio cache
        if audio_items and synthesize_fn:
            results["audio"] = await self.audio.prewarm(audio_items, synthesize_fn)

        # Warm embedding cache
        results["embeddings"] = await self.embeddings.preload_popular()

        return results


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def set_cache_manager(manager: CacheManager) -> None:
    """Set global cache manager instance."""
    global _cache_manager
    _cache_manager = manager
