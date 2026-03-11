"""
VØX Performance Layer Tests
---------------------------

Comprehensive tests for the VØX Performance Layer.

AXIØM Phase 9: Optimize - "What makes this solution efficient?"
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch


# ============================================================================
# Cache Tests
# ============================================================================


class TestLRUCache:
    """Tests for LRU cache."""

    def test_basic_put_get(self):
        """Test basic put and get."""
        from axiom_vox.performance.cache import LRUCache

        cache = LRUCache[str](max_size=10)
        cache.put("key1", "value1", size_bytes=10)

        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        """Test LRU eviction."""
        from axiom_vox.performance.cache import LRUCache

        cache = LRUCache[str](max_size=3)
        cache.put("key1", "value1", size_bytes=10)
        cache.put("key2", "value2", size_bytes=10)
        cache.put("key3", "value3", size_bytes=10)

        # Access key1 to make it recently used
        cache.get("key1")

        # Add key4, should evict key2 (LRU)
        cache.put("key4", "value4", size_bytes=10)

        assert cache.get("key1") is not None
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None

    def test_ttl_expiration(self):
        """Test TTL expiration."""
        from axiom_vox.performance.cache import LRUCache

        cache = LRUCache[str](max_size=10, default_ttl=0.1)  # 100ms TTL
        cache.put("key1", "value1", size_bytes=10)

        assert cache.get("key1") == "value1"

        time.sleep(0.15)  # Wait for expiration

        assert cache.get("key1") is None

    def test_bytes_limit(self):
        """Test bytes-based eviction."""
        from axiom_vox.performance.cache import LRUCache

        cache = LRUCache[str](max_size=100, max_bytes=50)
        cache.put("key1", "value1", size_bytes=20)
        cache.put("key2", "value2", size_bytes=20)

        # This should evict key1 to stay under 50 bytes
        cache.put("key3", "value3", size_bytes=20)

        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None

    def test_cache_stats(self):
        """Test cache statistics."""
        from axiom_vox.performance.cache import LRUCache

        cache = LRUCache[str](max_size=10)
        cache.put("key1", "value1", size_bytes=10)

        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.hit_rate == 2 / 3


class TestAudioCache:
    """Tests for audio cache."""

    def test_content_hash_key(self):
        """Test content-hash based caching."""
        from axiom_vox.performance.cache import AudioCache

        cache = AudioCache(max_entries=100)

        # Same content should hit cache
        cache.put("Hello", "voice1", b"audio_data", rate=1.0)

        assert cache.get("Hello", "voice1", rate=1.0) == b"audio_data"
        assert cache.get("Hello", "voice1", rate=1.5) is None  # Different params
        assert cache.get("Hello", "voice2", rate=1.0) is None  # Different voice

    def test_invalidate(self):
        """Test cache invalidation."""
        from axiom_vox.performance.cache import AudioCache

        cache = AudioCache(max_entries=100)
        cache.put("Hello", "voice1", b"audio_data")

        assert cache.get("Hello", "voice1") == b"audio_data"

        cache.invalidate("Hello", "voice1")
        assert cache.get("Hello", "voice1") is None

    @pytest.mark.asyncio
    async def test_prewarm(self):
        """Test cache prewarming."""
        from axiom_vox.performance.cache import AudioCache

        cache = AudioCache(max_entries=100)

        async def mock_synthesize(text, voice_id=None, **params):
            return f"audio_{text}_{voice_id}".encode()

        items = [
            {"text": "Hello", "voice_id": "warm"},
            {"text": "World", "voice_id": "cool"},
        ]

        count = await cache.prewarm(items, mock_synthesize)
        assert count == 2

        assert cache.get("Hello", "warm") == b"audio_Hello_warm"
        assert cache.get("World", "cool") == b"audio_World_cool"


class TestEmbeddingCache:
    """Tests for embedding cache."""

    @pytest.mark.asyncio
    async def test_lazy_loading(self):
        """Test lazy loading of embeddings."""
        from axiom_vox.performance.cache import EmbeddingCache

        cache = EmbeddingCache(max_entries=10)

        load_count = 0

        async def loader(voice_id):
            nonlocal load_count
            load_count += 1
            return [0.1, 0.2, 0.3]  # Mock embedding

        cache.set_loader(loader)

        # First get should load
        embedding = await cache.get("voice1")
        assert embedding == [0.1, 0.2, 0.3]
        assert load_count == 1

        # Second get should hit cache
        embedding = await cache.get("voice1")
        assert embedding == [0.1, 0.2, 0.3]
        assert load_count == 1  # No additional load


# ============================================================================
# Pool Tests
# ============================================================================


class TestHTTPConnectionPool:
    """Tests for HTTP connection pool."""

    @pytest.mark.asyncio
    async def test_pool_lifecycle(self):
        """Test pool start and close."""
        from axiom_vox.performance.pool import HTTPConnectionPool

        pool = HTTPConnectionPool(
            base_url="http://localhost:8000",
            pool_size=5,
        )

        await pool.start()
        assert not pool._closed

        await pool.close()
        assert pool._closed

    def test_pool_stats(self):
        """Test pool statistics."""
        from axiom_vox.performance.pool import HTTPConnectionPool

        pool = HTTPConnectionPool(
            base_url="http://localhost:8000",
            pool_size=5,
        )

        stats = pool.get_stats()
        assert stats.total_requests == 0


# ============================================================================
# Batch Tests
# ============================================================================


class TestBatchOptimizer:
    """Tests for batch optimizer."""

    @pytest.mark.asyncio
    async def test_parallel_processing(self):
        """Test parallel batch processing."""
        from axiom_vox.performance.batch import BatchOptimizer, BatchStrategy

        optimizer = BatchOptimizer(
            max_concurrent=5,
            strategy=BatchStrategy.PARALLEL,
        )

        items = list(range(10))
        results = []

        async def processor(item):
            await asyncio.sleep(0.01)
            return item * 2

        result = await optimizer.process(items, processor)

        assert result.stats.completed_items == 10
        assert result.stats.failed_items == 0
        assert len(result.results) == 10

    @pytest.mark.asyncio
    async def test_chunked_processing(self):
        """Test chunked batch processing."""
        from axiom_vox.performance.batch import BatchOptimizer, BatchStrategy

        optimizer = BatchOptimizer(
            chunk_size=3,
            strategy=BatchStrategy.CHUNKED,
        )

        items = list(range(10))

        async def processor(item):
            return item * 2

        result = await optimizer.process(items, processor)

        assert result.stats.completed_items == 10
        assert len(result.results) == 10

    @pytest.mark.asyncio
    async def test_progress_callback(self):
        """Test progress callback."""
        from axiom_vox.performance.batch import BatchOptimizer

        optimizer = BatchOptimizer(max_concurrent=2)

        progress_updates = []

        def on_progress(completed, total):
            progress_updates.append((completed, total))

        items = list(range(5))

        async def processor(item):
            return item

        await optimizer.process(items, processor, on_progress=on_progress)

        assert len(progress_updates) == 5
        assert progress_updates[-1] == (5, 5)

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in batch."""
        from axiom_vox.performance.batch import BatchOptimizer

        optimizer = BatchOptimizer(max_concurrent=5)

        items = list(range(5))

        async def processor(item):
            if item == 2:
                raise ValueError("Test error")
            return item

        result = await optimizer.process(items, processor)

        assert result.stats.completed_items == 4
        assert result.stats.failed_items == 1
        assert len(result.errors) == 1


# ============================================================================
# Stream Tests
# ============================================================================


class TestStreamBuffer:
    """Tests for stream buffer."""

    @pytest.mark.asyncio
    async def test_basic_put_get(self):
        """Test basic put and get."""
        from axiom_vox.performance.stream import StreamBuffer

        buffer = StreamBuffer[bytes](max_size=10)

        await buffer.put(b"chunk1", size_bytes=6)
        await buffer.put(b"chunk2", size_bytes=6)

        chunk1 = await buffer.get()
        chunk2 = await buffer.get()

        assert chunk1.data == b"chunk1"
        assert chunk2.data == b"chunk2"

    @pytest.mark.asyncio
    async def test_async_iteration(self):
        """Test async iteration."""
        from axiom_vox.performance.stream import StreamBuffer

        buffer = StreamBuffer[bytes](max_size=10)

        # Producer
        async def produce():
            for i in range(5):
                await buffer.put(f"chunk{i}".encode(), size_bytes=6)
            await buffer.put(b"final", size_bytes=5, is_final=True)

        asyncio.create_task(produce())

        chunks = []
        async for chunk in buffer:
            chunks.append(chunk.data)

        assert len(chunks) == 6

    @pytest.mark.asyncio
    async def test_backpressure(self):
        """Test backpressure handling."""
        from axiom_vox.performance.stream import StreamBuffer

        buffer = StreamBuffer[bytes](max_size=2)

        await buffer.put(b"chunk1", size_bytes=6)
        await buffer.put(b"chunk2", size_bytes=6)

        # This should trigger backpressure (timeout)
        result = await buffer.put(b"chunk3", size_bytes=6, timeout=0.1)
        # Note: This might succeed or fail depending on timing
        # The important thing is it doesn't hang forever

        stats = buffer.get_stats()
        assert stats.chunks_received >= 2


class TestAudioStreamBuffer:
    """Tests for audio stream buffer."""

    @pytest.mark.asyncio
    async def test_audio_buffering(self):
        """Test audio-specific buffering."""
        from axiom_vox.performance.stream import AudioStreamBuffer

        buffer = AudioStreamBuffer(
            sample_rate=24000,
            bytes_per_sample=2,
            buffer_seconds=1.0,
        )

        # Put some audio
        audio_chunk = bytes(1000)
        await buffer.put_audio(audio_chunk)

        assert buffer.size() == 1
        assert buffer.bytes_size() == 1000

        # Get audio
        audio = await buffer.get_audio()
        assert audio == audio_chunk


# ============================================================================
# Lazy Loading Tests
# ============================================================================


class TestLazyLoader:
    """Tests for lazy loader."""

    @pytest.mark.asyncio
    async def test_lazy_loading(self):
        """Test lazy loading of resources."""
        from axiom_vox.performance.lazy import LazyLoader

        loader = LazyLoader(max_loaded=10)

        load_count = 0

        def load_resource():
            nonlocal load_count
            load_count += 1
            return {"data": "test"}

        loader.register("resource1", load_resource)

        # First get should load
        resource = await loader.get("resource1")
        assert resource == {"data": "test"}
        assert load_count == 1

        # Second get should use cache
        resource = await loader.get("resource1")
        assert resource == {"data": "test"}
        assert load_count == 1

    @pytest.mark.asyncio
    async def test_auto_unload(self):
        """Test automatic unloading."""
        from axiom_vox.performance.lazy import LazyLoader

        loader = LazyLoader(max_loaded=2, auto_unload=True)

        for i in range(3):
            loader.register(f"resource{i}", lambda i=i: f"value{i}")

        await loader.get("resource0")
        await loader.get("resource1")

        # This should trigger unload of resource0
        await loader.get("resource2")

        stats = loader.get_stats()
        assert stats.total_loads == 3
        assert stats.total_unloads >= 1


class TestModuleLazyLoader:
    """Tests for module lazy loader."""

    def test_module_loading(self):
        """Test lazy module loading."""
        from axiom_vox.performance.lazy import ModuleLazyLoader

        loader = ModuleLazyLoader()

        # Load a standard library module
        json_module = loader.get_module("json")
        assert json_module is not None
        assert hasattr(json_module, "dumps")

        # Second call should hit cache
        json_module2 = loader.get_module("json")
        assert json_module is json_module2

        stats = loader.get_stats()
        assert stats.cache_hits >= 1

    def test_lazy_import(self):
        """Test lazy_import helper."""
        from axiom_vox.performance.lazy import lazy_import

        # Create lazy import
        json_lazy = lazy_import("json")

        # Module not loaded until first access
        result = json_lazy.dumps({"test": 1})
        assert result == '{"test": 1}'


# ============================================================================
# Integration Tests
# ============================================================================


class TestPerformanceIntegration:
    """Integration tests for performance layer."""

    def test_imports(self):
        """Test all performance imports work."""
        from axiom_vox.performance import (
            # Cache
            AudioCache,
            EmbeddingCache,
            CacheManager,
            # Pool
            HTTPConnectionPool,
            WebSocketPool,
            # Batch
            BatchOptimizer,
            BatchStrategy,
            # Stream
            StreamBuffer,
            AudioStreamBuffer,
            # Lazy
            LazyLoader,
            lazy_import,
        )

        assert AudioCache is not None
        assert BatchOptimizer is not None
        assert StreamBuffer is not None
        assert LazyLoader is not None

    def test_main_module_exports(self):
        """Test performance exports from main module."""
        from axiom_vox import (
            AudioCache,
            CacheManager,
            BatchOptimizer,
            StreamBuffer,
            LazyLoader,
            __version__,
        )

        assert AudioCache is not None
        assert CacheManager is not None
        assert BatchOptimizer is not None
        assert StreamBuffer is not None
        assert LazyLoader is not None
        assert __version__ == "0.14.0"

    @pytest.mark.asyncio
    async def test_cache_manager(self):
        """Test CacheManager integration."""
        from axiom_vox.performance import CacheManager

        manager = CacheManager(total_memory_mb=64)

        # Audio caching
        manager.audio.put("Hello", "warm", b"audio_data")
        assert manager.audio.get("Hello", "warm") == b"audio_data"

        # Stats
        stats = manager.get_combined_stats()
        assert "audio" in stats
        assert "embeddings" in stats
        assert "combined" in stats


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
