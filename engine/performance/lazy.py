"""
VØX Performance - Lazy Loading
------------------------------

Deferred loading of heavy modules and resources.

Features:
    - On-demand module loading
    - Background preloading
    - Memory-aware unloading
    - Load time tracking

AXIØM Phase 9: Optimize - "What makes this solution efficient?"
"""

import asyncio
import importlib
import logging
import sys
import time
import weakref
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, TypeVar, Generic, Type

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LoadState(str, Enum):
    """State of a lazy-loaded resource."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    UNLOADING = "unloading"


@dataclass
class LoadStats:
    """Statistics for lazy loading."""
    total_loads: int = 0
    total_unloads: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_load_time_ms: float = 0.0
    peak_memory_bytes: int = 0
    current_loaded: int = 0

    @property
    def avg_load_time_ms(self) -> float:
        """Average load time."""
        return self.total_load_time_ms / self.total_loads if self.total_loads > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_loads": self.total_loads,
            "total_unloads": self.total_unloads,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "avg_load_time_ms": self.avg_load_time_ms,
            "current_loaded": self.current_loaded,
        }


@dataclass
class LazyResource(Generic[T]):
    """A lazily-loaded resource."""
    name: str
    loader: Callable[[], T]
    state: LoadState = LoadState.UNLOADED
    value: Optional[T] = None
    load_time_ms: float = 0.0
    memory_bytes: int = 0
    loaded_at: Optional[float] = None
    last_accessed: Optional[float] = None
    access_count: int = 0
    error: Optional[str] = None

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1


class LazyLoader:
    """
    Manager for lazy-loaded resources.

    Features:
        - On-demand loading
        - LRU-based unloading
        - Memory budget management
        - Background preloading
    """

    def __init__(
        self,
        max_loaded: int = 10,
        max_memory_mb: Optional[float] = None,
        auto_unload: bool = True,
    ):
        """
        Initialize lazy loader.

        Args:
            max_loaded: Maximum resources to keep loaded
            max_memory_mb: Maximum memory in MB
            auto_unload: Automatically unload LRU resources
        """
        self.max_loaded = max_loaded
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024) if max_memory_mb else None
        self.auto_unload = auto_unload

        self._resources: Dict[str, LazyResource] = {}
        self._stats = LoadStats()
        self._lock = asyncio.Lock()
        self._current_memory = 0

    def register(
        self,
        name: str,
        loader: Callable[[], T],
        memory_estimate_bytes: int = 0,
    ) -> None:
        """
        Register a resource for lazy loading.

        Args:
            name: Resource name
            loader: Function to load the resource
            memory_estimate_bytes: Estimated memory usage
        """
        self._resources[name] = LazyResource(
            name=name,
            loader=loader,
            memory_bytes=memory_estimate_bytes,
        )

    async def get(self, name: str) -> Optional[T]:
        """
        Get a resource, loading if necessary.

        Args:
            name: Resource name

        Returns:
            Resource value or None if not found
        """
        resource = self._resources.get(name)
        if resource is None:
            return None

        async with self._lock:
            if resource.state == LoadState.LOADED:
                self._stats.cache_hits += 1
                resource.touch()
                return resource.value

            if resource.state == LoadState.LOADING:
                # Wait for loading to complete
                pass

            self._stats.cache_misses += 1

            # Check if we need to unload something
            if self.auto_unload:
                await self._maybe_unload()

            # Load resource
            return await self._load(resource)

    async def preload(self, names: List[str]) -> int:
        """
        Preload resources in the background.

        Args:
            names: Resource names to preload

        Returns:
            Number of resources loaded
        """
        loaded = 0

        for name in names:
            resource = self._resources.get(name)
            if resource and resource.state == LoadState.UNLOADED:
                try:
                    await self.get(name)
                    loaded += 1
                except Exception as e:
                    logger.warning(f"Failed to preload {name}: {e}")

        return loaded

    async def unload(self, name: str) -> bool:
        """
        Unload a resource.

        Args:
            name: Resource name

        Returns:
            True if resource was unloaded
        """
        resource = self._resources.get(name)
        if resource is None or resource.state != LoadState.LOADED:
            return False

        async with self._lock:
            return await self._unload(resource)

    async def unload_all(self) -> int:
        """
        Unload all resources.

        Returns:
            Number of resources unloaded
        """
        count = 0
        async with self._lock:
            for resource in self._resources.values():
                if resource.state == LoadState.LOADED:
                    if await self._unload(resource):
                        count += 1
        return count

    def is_loaded(self, name: str) -> bool:
        """Check if resource is loaded."""
        resource = self._resources.get(name)
        return resource is not None and resource.state == LoadState.LOADED

    def get_stats(self) -> LoadStats:
        """Get loading statistics."""
        self._stats.current_loaded = sum(
            1 for r in self._resources.values() if r.state == LoadState.LOADED
        )
        return LoadStats(
            total_loads=self._stats.total_loads,
            total_unloads=self._stats.total_unloads,
            cache_hits=self._stats.cache_hits,
            cache_misses=self._stats.cache_misses,
            total_load_time_ms=self._stats.total_load_time_ms,
            current_loaded=self._stats.current_loaded,
        )

    async def _load(self, resource: LazyResource[T]) -> T:
        """Load a resource."""
        resource.state = LoadState.LOADING
        start_time = time.time()

        try:
            # Load synchronously (most loaders are sync)
            value = resource.loader()

            load_time = (time.time() - start_time) * 1000
            resource.value = value
            resource.state = LoadState.LOADED
            resource.load_time_ms = load_time
            resource.loaded_at = time.time()
            resource.touch()

            self._stats.total_loads += 1
            self._stats.total_load_time_ms += load_time
            self._current_memory += resource.memory_bytes

            logger.debug(f"Loaded {resource.name} in {load_time:.1f}ms")
            return value

        except Exception as e:
            resource.state = LoadState.FAILED
            resource.error = str(e)
            logger.error(f"Failed to load {resource.name}: {e}")
            raise

    async def _unload(self, resource: LazyResource) -> bool:
        """Unload a resource."""
        resource.state = LoadState.UNLOADING

        try:
            # Clear reference
            resource.value = None
            resource.state = LoadState.UNLOADED

            self._stats.total_unloads += 1
            self._current_memory -= resource.memory_bytes

            logger.debug(f"Unloaded {resource.name}")
            return True

        except Exception as e:
            logger.warning(f"Failed to unload {resource.name}: {e}")
            return False

    async def _maybe_unload(self) -> None:
        """Unload LRU resources if at capacity."""
        loaded = [
            r for r in self._resources.values()
            if r.state == LoadState.LOADED
        ]

        # Check count limit
        while len(loaded) >= self.max_loaded:
            lru = min(loaded, key=lambda r: r.last_accessed or 0)
            await self._unload(lru)
            loaded.remove(lru)

        # Check memory limit
        if self.max_memory_bytes:
            while self._current_memory >= self.max_memory_bytes and loaded:
                lru = min(loaded, key=lambda r: r.last_accessed or 0)
                await self._unload(lru)
                loaded.remove(lru)


class ModuleLazyLoader:
    """
    Lazy loader specifically for Python modules.

    Features:
        - Import on demand
        - Module caching
        - Dependency tracking
    """

    def __init__(self):
        """Initialize module lazy loader."""
        self._modules: Dict[str, Any] = {}
        self._loading: set = set()
        self._stats = LoadStats()

    def get_module(self, name: str, package: Optional[str] = None) -> Any:
        """
        Get a module, importing if necessary.

        Args:
            name: Module name
            package: Package for relative imports

        Returns:
            Module object
        """
        full_name = f"{package}.{name}" if package else name

        # Check cache
        if full_name in self._modules:
            self._stats.cache_hits += 1
            return self._modules[full_name]

        # Check if already loaded in sys.modules
        if full_name in sys.modules:
            self._stats.cache_hits += 1
            self._modules[full_name] = sys.modules[full_name]
            return sys.modules[full_name]

        # Prevent circular loading
        if full_name in self._loading:
            raise ImportError(f"Circular import detected: {full_name}")

        self._stats.cache_misses += 1
        self._loading.add(full_name)

        try:
            start_time = time.time()
            module = importlib.import_module(name, package)
            load_time = (time.time() - start_time) * 1000

            self._modules[full_name] = module
            self._stats.total_loads += 1
            self._stats.total_load_time_ms += load_time

            logger.debug(f"Imported {full_name} in {load_time:.1f}ms")
            return module

        finally:
            self._loading.discard(full_name)

    def unload_module(self, name: str) -> bool:
        """
        Unload a module.

        Args:
            name: Module name

        Returns:
            True if module was unloaded
        """
        if name in self._modules:
            del self._modules[name]
            if name in sys.modules:
                del sys.modules[name]
            self._stats.total_unloads += 1
            return True
        return False

    def is_loaded(self, name: str) -> bool:
        """Check if module is loaded."""
        return name in self._modules or name in sys.modules

    def get_stats(self) -> LoadStats:
        """Get loading statistics."""
        self._stats.current_loaded = len(self._modules)
        return self._stats


class LazyAttribute:
    """
    Descriptor for lazy attribute loading.

    Usage:
        class MyClass:
            heavy_attr = LazyAttribute(lambda self: load_heavy_thing())
    """

    def __init__(self, loader: Callable):
        """
        Initialize lazy attribute.

        Args:
            loader: Function to compute the attribute value
        """
        self.loader = loader
        self.attr_name: Optional[str] = None

    def __set_name__(self, owner: Type, name: str) -> None:
        """Store the attribute name."""
        self.attr_name = f"_lazy_{name}"

    def __get__(self, obj: Any, objtype: Optional[Type] = None) -> Any:
        """Get the attribute, computing if necessary."""
        if obj is None:
            return self

        if not hasattr(obj, self.attr_name):
            value = self.loader(obj)
            setattr(obj, self.attr_name, value)

        return getattr(obj, self.attr_name)


def lazy_import(module_name: str, package: Optional[str] = None) -> Any:
    """
    Create a lazy import proxy.

    Args:
        module_name: Module to import
        package: Package for relative imports

    Returns:
        Lazy module proxy

    Usage:
        torch = lazy_import('torch')
        # torch is not imported until first use
        x = torch.tensor([1, 2, 3])  # Imported here
    """

    class LazyModule:
        _module = None

        def __getattr__(self, name: str) -> Any:
            if LazyModule._module is None:
                LazyModule._module = importlib.import_module(module_name, package)
            return getattr(LazyModule._module, name)

    return LazyModule()


# Global lazy loader instances
_resource_loader: Optional[LazyLoader] = None
_module_loader: Optional[ModuleLazyLoader] = None


def get_resource_loader() -> LazyLoader:
    """Get global resource lazy loader."""
    global _resource_loader
    if _resource_loader is None:
        _resource_loader = LazyLoader()
    return _resource_loader


def get_module_loader() -> ModuleLazyLoader:
    """Get global module lazy loader."""
    global _module_loader
    if _module_loader is None:
        _module_loader = ModuleLazyLoader()
    return _module_loader


def set_resource_loader(loader: LazyLoader) -> None:
    """Set global resource lazy loader."""
    global _resource_loader
    _resource_loader = loader


def set_module_loader(loader: ModuleLazyLoader) -> None:
    """Set global module lazy loader."""
    global _module_loader
    _module_loader = loader
