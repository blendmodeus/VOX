"""
Resource Manager
----------------

GPU, memory, and queue management for voice operations.

AXIØM Phase 7: Constrain - "What limits clarify the solution?"
"""

import logging
import os
import time
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Callable, Any
from enum import Enum
import heapq

from .models import (
    ResourceConfig,
    ResourceStatus,
    ResourceAllocation,
    ResourceType,
    QueueConfig,
    QueueStatus,
    RESOURCE_LIMITS,
)

logger = logging.getLogger(__name__)


class AllocationError(Exception):
    """Error during resource allocation."""
    pass


class QueueFullError(Exception):
    """Queue is at maximum capacity."""
    pass


@dataclass
class QueuedRequest:
    """A request waiting in the queue."""
    request_id: str
    priority: int  # Lower = higher priority
    enqueued_at: float
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        # For heapq: compare by priority, then by time
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.enqueued_at < other.enqueued_at


class ResourceManager:
    """
    Manages computational resources for voice operations.

    Tracks GPU memory, CPU usage, and general memory,
    providing allocation and release mechanisms.
    """

    def __init__(
        self,
        config: Optional[ResourceConfig] = None,
    ):
        """
        Initialize resource manager.

        Args:
            config: Resource configuration
        """
        self.config = config or ResourceConfig()
        self._allocations: Dict[str, ResourceAllocation] = {}
        self._lock = threading.RLock()

        # Track usage by type
        self._usage: Dict[ResourceType, float] = {
            ResourceType.GPU: 0.0,
            ResourceType.CPU: 0.0,
            ResourceType.MEMORY: 0.0,
        }

    def get_status(self, resource_type: ResourceType) -> ResourceStatus:
        """
        Get current status of a resource type.

        Args:
            resource_type: Type of resource

        Returns:
            ResourceStatus
        """
        with self._lock:
            if resource_type == ResourceType.GPU:
                total = self.config.max_gpu_memory_mb
                used = self._usage[ResourceType.GPU]
            elif resource_type == ResourceType.MEMORY:
                total = self.config.max_memory_mb
                used = self._usage[ResourceType.MEMORY]
            elif resource_type == ResourceType.CPU:
                total = 100.0  # Percentage
                used = self._usage[ResourceType.CPU]
            else:
                total = 0.0
                used = 0.0

            available = max(0, total - used)
            utilization = (used / total * 100) if total > 0 else 0

            return ResourceStatus(
                resource_type=resource_type,
                total=total,
                used=used,
                available=available,
                utilization_percent=utilization,
                healthy=utilization < 90,
            )

    def allocate(
        self,
        request_id: str,
        resource_type: ResourceType,
        amount: float,
        timeout_seconds: Optional[float] = None,
    ) -> ResourceAllocation:
        """
        Allocate resources for a request.

        Args:
            request_id: Request identifier
            resource_type: Type of resource
            amount: Amount to allocate
            timeout_seconds: Optional timeout for allocation

        Returns:
            ResourceAllocation

        Raises:
            AllocationError: If resources not available
        """
        with self._lock:
            status = self.get_status(resource_type)

            if amount > status.available:
                raise AllocationError(
                    f"Insufficient {resource_type.value}: "
                    f"requested {amount}, available {status.available}"
                )

            # Create allocation
            allocation_id = f"alloc_{uuid.uuid4().hex[:12]}"
            expires_at = None
            if timeout_seconds:
                expires_at = time.time() + timeout_seconds

            allocation = ResourceAllocation(
                allocation_id=allocation_id,
                request_id=request_id,
                resource_type=resource_type,
                amount=amount,
                expires_at=expires_at,
            )

            # Update usage
            self._usage[resource_type] += amount
            self._allocations[allocation_id] = allocation

            logger.debug(
                f"Allocated {amount} {resource_type.value} for {request_id}: "
                f"{allocation_id}"
            )

            return allocation

    def release(self, allocation_id: str) -> bool:
        """
        Release an allocation.

        Args:
            allocation_id: Allocation to release

        Returns:
            True if released
        """
        with self._lock:
            allocation = self._allocations.get(allocation_id)
            if not allocation or allocation.released:
                return False

            self._usage[allocation.resource_type] -= allocation.amount
            allocation.released = True

            logger.debug(
                f"Released {allocation.amount} {allocation.resource_type.value}: "
                f"{allocation_id}"
            )

            return True

    def release_for_request(self, request_id: str) -> int:
        """
        Release all allocations for a request.

        Args:
            request_id: Request identifier

        Returns:
            Number of allocations released
        """
        with self._lock:
            released = 0
            for allocation in list(self._allocations.values()):
                if allocation.request_id == request_id and not allocation.released:
                    self.release(allocation.allocation_id)
                    released += 1
            return released

    def cleanup_expired(self) -> int:
        """
        Clean up expired allocations.

        Returns:
            Number of allocations cleaned up
        """
        with self._lock:
            now = time.time()
            expired = []

            for alloc_id, allocation in self._allocations.items():
                if allocation.expires_at and allocation.expires_at < now:
                    if not allocation.released:
                        expired.append(alloc_id)

            for alloc_id in expired:
                self.release(alloc_id)
                logger.warning(f"Cleaned up expired allocation: {alloc_id}")

            return len(expired)

    def get_allocations(
        self,
        request_id: Optional[str] = None,
    ) -> List[ResourceAllocation]:
        """Get active allocations."""
        with self._lock:
            allocations = [
                a for a in self._allocations.values()
                if not a.released
            ]
            if request_id:
                allocations = [a for a in allocations if a.request_id == request_id]
            return allocations

    def can_allocate(
        self,
        resource_type: ResourceType,
        amount: float,
    ) -> bool:
        """Check if allocation is possible."""
        status = self.get_status(resource_type)
        return amount <= status.available


class QueueManager:
    """
    Priority queue manager for request scheduling.

    Supports multiple priority levels and fair scheduling.
    """

    def __init__(
        self,
        config: Optional[QueueConfig] = None,
    ):
        """
        Initialize queue manager.

        Args:
            config: Queue configuration
        """
        self.config = config or QueueConfig()
        self._queue: List[QueuedRequest] = []
        self._processing: Dict[str, QueuedRequest] = {}
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

    def enqueue(
        self,
        request_id: str,
        priority: int = 1,
        callback: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QueuedRequest:
        """
        Add request to queue.

        Args:
            request_id: Request identifier
            priority: Priority level (0 = highest)
            callback: Optional callback when dequeued
            metadata: Optional metadata

        Returns:
            QueuedRequest

        Raises:
            QueueFullError: If queue is at capacity
        """
        with self._condition:
            if len(self._queue) >= self.config.max_depth:
                raise QueueFullError(
                    f"Queue at maximum capacity: {self.config.max_depth}"
                )

            request = QueuedRequest(
                request_id=request_id,
                priority=priority,
                enqueued_at=time.time(),
                callback=callback,
                metadata=metadata or {},
            )

            heapq.heappush(self._queue, request)
            self._condition.notify()

            logger.debug(
                f"Enqueued request {request_id} with priority {priority}, "
                f"queue depth: {len(self._queue)}"
            )

            return request

    def dequeue(
        self,
        timeout: Optional[float] = None,
    ) -> Optional[QueuedRequest]:
        """
        Get next request from queue.

        Args:
            timeout: Maximum wait time

        Returns:
            QueuedRequest or None if timeout
        """
        with self._condition:
            deadline = time.time() + (timeout or self.config.max_wait_seconds)

            while not self._queue:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

            # Apply priority boost for waiting requests
            if self.config.priority_boost_per_second > 0:
                now = time.time()
                for req in self._queue:
                    wait_time = now - req.enqueued_at
                    boost = int(wait_time * self.config.priority_boost_per_second)
                    req.priority = max(0, req.priority - boost)
                heapq.heapify(self._queue)

            request = heapq.heappop(self._queue)
            self._processing[request.request_id] = request

            logger.debug(
                f"Dequeued request {request.request_id}, "
                f"wait time: {time.time() - request.enqueued_at:.2f}s"
            )

            return request

    def complete(self, request_id: str) -> bool:
        """
        Mark request as completed.

        Args:
            request_id: Request identifier

        Returns:
            True if request was processing
        """
        with self._lock:
            request = self._processing.pop(request_id, None)
            if request:
                logger.debug(f"Completed request {request_id}")
                return True
            return False

    def cancel(self, request_id: str) -> bool:
        """
        Cancel a queued or processing request.

        Args:
            request_id: Request identifier

        Returns:
            True if request was cancelled
        """
        with self._lock:
            # Check processing
            if request_id in self._processing:
                del self._processing[request_id]
                return True

            # Check queue
            for i, req in enumerate(self._queue):
                if req.request_id == request_id:
                    self._queue.pop(i)
                    heapq.heapify(self._queue)
                    return True

            return False

    def get_status(self) -> QueueStatus:
        """Get current queue status."""
        with self._lock:
            now = time.time()

            depth = len(self._queue)
            oldest_age = 0.0
            total_age = 0.0

            if self._queue:
                ages = [now - req.enqueued_at for req in self._queue]
                oldest_age = max(ages)
                total_age = sum(ages)

            avg_wait = total_age / depth if depth > 0 else 0.0

            return QueueStatus(
                depth=depth,
                oldest_request_age_seconds=oldest_age,
                average_wait_seconds=avg_wait,
                processing_rate_per_second=len(self._processing) / 60.0,
                blocked=depth >= self.config.max_depth,
            )

    def get_position(self, request_id: str) -> int:
        """
        Get queue position for a request.

        Args:
            request_id: Request identifier

        Returns:
            Position (1-indexed) or -1 if not in queue
        """
        with self._lock:
            for i, req in enumerate(sorted(self._queue)):
                if req.request_id == request_id:
                    return i + 1
            return -1


class ResourceGovernor:
    """
    Combined resource and queue governance.

    Coordinates resource allocation with queue management
    to ensure fair and efficient resource usage.
    """

    def __init__(
        self,
        resource_config: Optional[ResourceConfig] = None,
        queue_config: Optional[QueueConfig] = None,
    ):
        """
        Initialize resource governor.

        Args:
            resource_config: Resource configuration
            queue_config: Queue configuration
        """
        self.resources = ResourceManager(resource_config)
        self.queue = QueueManager(queue_config)
        self._lock = threading.RLock()

    def request_resources(
        self,
        request_id: str,
        requirements: Dict[ResourceType, float],
        priority: int = 1,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, ResourceAllocation]:
        """
        Request resources, queuing if not immediately available.

        Args:
            request_id: Request identifier
            requirements: Resource requirements by type
            priority: Queue priority
            timeout_seconds: Timeout for allocation

        Returns:
            Dict of allocations by resource type
        """
        with self._lock:
            # Check if resources are available
            available = all(
                self.resources.can_allocate(rtype, amount)
                for rtype, amount in requirements.items()
            )

            if not available:
                # Queue the request
                self.queue.enqueue(
                    request_id=request_id,
                    priority=priority,
                    metadata={"requirements": requirements},
                )
                raise AllocationError(
                    f"Resources not immediately available, request queued"
                )

            # Allocate resources
            allocations = {}
            try:
                for rtype, amount in requirements.items():
                    allocation = self.resources.allocate(
                        request_id=request_id,
                        resource_type=rtype,
                        amount=amount,
                        timeout_seconds=timeout_seconds,
                    )
                    allocations[rtype.value] = allocation
            except AllocationError:
                # Rollback partial allocations
                for allocation in allocations.values():
                    self.resources.release(allocation.allocation_id)
                raise

            return allocations

    def release_resources(self, request_id: str) -> None:
        """Release all resources for a request."""
        self.resources.release_for_request(request_id)
        self.queue.complete(request_id)

    def get_status(self) -> Dict[str, Any]:
        """Get combined status."""
        return {
            "resources": {
                rtype.value: self.resources.get_status(rtype).__dict__
                for rtype in [ResourceType.GPU, ResourceType.MEMORY, ResourceType.CPU]
            },
            "queue": self.queue.get_status().__dict__,
        }


# Singleton instance
_manager_instance: Optional[ResourceManager] = None


def get_resource_manager(
    config: Optional[ResourceConfig] = None,
) -> ResourceManager:
    """Get or create resource manager singleton."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ResourceManager(config=config)
    return _manager_instance


def set_resource_manager(manager: ResourceManager) -> None:
    """Set the resource manager singleton."""
    global _manager_instance
    _manager_instance = manager
