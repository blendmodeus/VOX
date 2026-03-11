"""
VØX Fine-Tuning Job Manager
---------------------------

Manages fine-tuning job queue and status tracking.

Features:
- Job creation and tracking
- Background job execution
- Status updates and progress reporting
- Cancellation support

Usage:
    from axiom_vox.finetuning import FineTuningJobManager, JobStatus

    manager = FineTuningJobManager(db)

    # Create job
    job_id = await manager.create_job(
        voice_id="my_voice",
        audio_files=["voice1.wav", "voice2.wav"],
        consent_verified=True,
    )

    # Check status
    status = await manager.get_status(job_id)
    print(f"Status: {status.status}, Progress: {status.progress}")

    # Cancel if needed
    await manager.cancel_job(job_id)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from enum import Enum
from pathlib import Path

if TYPE_CHECKING:
    from axiom_vox.persistence import VoxDatabase

logger = logging.getLogger(__name__)


# ============================================================================
# JOB STATUS
# ============================================================================

class JobStatus(str, Enum):
    """Fine-tuning job status values."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    VERIFIED = "verified"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobInfo:
    """Information about a fine-tuning job."""

    job_id: str
    voice_id: str
    status: JobStatus
    created_at: datetime

    # Progress
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_remaining_seconds: Optional[float] = None

    # Input
    sample_count: int = 0
    total_duration_seconds: float = 0.0
    consent_verified: bool = False

    # Output
    adapter_path: Optional[str] = None
    run_id: Optional[str] = None

    # Metrics
    final_loss: Optional[float] = None
    similarity_score: Optional[float] = None
    quality_score: Optional[float] = None

    # Verification
    verification_passed: Optional[bool] = None

    # Error
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "voice_id": self.voice_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "progress": self.progress,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
            "sample_count": self.sample_count,
            "total_duration_seconds": self.total_duration_seconds,
            "consent_verified": self.consent_verified,
            "adapter_path": self.adapter_path,
            "run_id": self.run_id,
            "final_loss": self.final_loss,
            "similarity_score": self.similarity_score,
            "quality_score": self.quality_score,
            "verification_passed": self.verification_passed,
            "error_message": self.error_message,
        }


# ============================================================================
# JOB MANAGER
# ============================================================================

class FineTuningJobManager:
    """
    Manages fine-tuning job queue and execution.

    Features:
    - Create and track jobs
    - Execute jobs in background
    - Update progress
    - Handle cancellation
    """

    def __init__(
        self,
        db: Optional["VoxDatabase"] = None,
        max_concurrent_jobs: int = 1,
    ):
        """
        Initialize job manager.

        Args:
            db: Database for persistence (optional)
            max_concurrent_jobs: Maximum concurrent training jobs
        """
        self.db = db
        self.max_concurrent_jobs = max_concurrent_jobs

        # In-memory tracking
        self._jobs: Dict[str, JobInfo] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._cancel_events: Dict[str, asyncio.Event] = {}

    async def create_job(
        self,
        voice_id: str,
        audio_files: List[str],
        consent_verified: bool,
        total_duration: float = 0.0,
        requestor_id: Optional[str] = None,
        epochs: int = 50,
        fast_mode: bool = False,
    ) -> str:
        """
        Create a new fine-tuning job.

        Args:
            voice_id: Identifier for the voice being cloned
            audio_files: List of audio file paths
            consent_verified: Whether consent is verified
            total_duration: Total audio duration in seconds
            requestor_id: Optional requestor identifier
            epochs: Number of training epochs
            fast_mode: Use fast training configuration

        Returns:
            Job ID
        """
        job_id = f"ft_{uuid.uuid4().hex[:12]}"

        job = JobInfo(
            job_id=job_id,
            voice_id=voice_id,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            sample_count=len(audio_files),
            total_duration_seconds=total_duration,
            consent_verified=consent_verified,
            total_epochs=epochs,
        )

        self._jobs[job_id] = job
        self._cancel_events[job_id] = asyncio.Event()

        # Persist to database
        if self.db:
            self.db.create_finetune_job(
                voice_id=voice_id,
                sample_count=len(audio_files),
                total_duration=total_duration,
                consent_verified=consent_verified,
                requestor_id=requestor_id,
            )

        logger.info(f"Created job {job_id} for voice {voice_id}")
        return job_id

    async def start_job(
        self,
        job_id: str,
        audio_files: List[str],
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Start a pending job in the background.

        Args:
            job_id: Job to start
            audio_files: Audio file paths
            config: Optional training configuration

        Returns:
            True if job started successfully
        """
        if job_id not in self._jobs:
            logger.error(f"Job not found: {job_id}")
            return False

        job = self._jobs[job_id]

        if job.status != JobStatus.PENDING:
            logger.error(f"Job {job_id} is not pending: {job.status}")
            return False

        # Check concurrent job limit
        running_count = sum(
            1 for j in self._jobs.values()
            if j.status == JobStatus.PROCESSING
        )
        if running_count >= self.max_concurrent_jobs:
            logger.warning(f"Max concurrent jobs ({self.max_concurrent_jobs}) reached")
            return False

        # Start background task
        task = asyncio.create_task(
            self._run_job(job_id, audio_files, config or {})
        )
        self._running_tasks[job_id] = task

        # Update status
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.now()

        if self.db:
            self.db.update_job_status(job_id, "processing")

        logger.info(f"Started job {job_id}")
        return True

    async def _run_job(
        self,
        job_id: str,
        audio_files: List[str],
        config: Dict[str, Any],
    ) -> None:
        """Execute a fine-tuning job."""
        job = self._jobs[job_id]
        cancel_event = self._cancel_events[job_id]

        try:
            from axiom_vox.finetuning.training_pipeline import (
                VoxFineTuningPipeline,
                FineTuningConfig,
            )

            # Create pipeline
            ft_config = FineTuningConfig(
                epochs=job.total_epochs,
                fast=config.get("fast", False),
            )

            pipeline = VoxFineTuningPipeline(
                config=ft_config,
                voice_id=job.voice_id,
                db=self.db,
            )

            job.run_id = pipeline.run_id

            # Progress callback
            async def update_progress(epoch: int, loss: float):
                if cancel_event.is_set():
                    raise asyncio.CancelledError("Job cancelled")

                job.current_epoch = epoch
                job.progress = epoch / job.total_epochs
                job.final_loss = loss

                # Estimate remaining time
                if job.started_at:
                    elapsed = (datetime.now() - job.started_at).total_seconds()
                    if epoch > 0:
                        per_epoch = elapsed / epoch
                        remaining_epochs = job.total_epochs - epoch
                        job.estimated_remaining_seconds = per_epoch * remaining_epochs

            # Run training
            result = await pipeline.train(
                audio_samples=audio_files,
                consent_verified=job.consent_verified,
            )

            # Update job with results
            if result.success:
                job.status = JobStatus.VERIFIED if result.verification_passed else JobStatus.COMPLETED
                job.adapter_path = result.adapter_path
                job.final_loss = result.final_loss
                job.similarity_score = result.similarity_score
                job.quality_score = result.quality_score
                job.verification_passed = result.verification_passed
            else:
                job.status = JobStatus.FAILED
                job.error_message = result.error

            job.completed_at = datetime.now()
            job.progress = 1.0

            # Update database
            if self.db:
                self.db.update_job_status(
                    job_id=job_id,
                    status=job.status.value,
                    final_loss=job.final_loss,
                    epochs_completed=job.total_epochs,
                    verification_passed=job.verification_passed,
                )

            logger.info(f"Job {job_id} completed: {job.status.value}")

        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            logger.info(f"Job {job_id} cancelled")

            if self.db:
                self.db.update_job_status(job_id, "cancelled")

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            logger.exception(f"Job {job_id} failed: {e}")

            if self.db:
                self.db.update_job_status(job_id, "failed")

        finally:
            # Cleanup
            if job_id in self._running_tasks:
                del self._running_tasks[job_id]
            if job_id in self._cancel_events:
                del self._cancel_events[job_id]

    async def get_status(self, job_id: str) -> Optional[JobInfo]:
        """
        Get status of a job.

        Args:
            job_id: Job identifier

        Returns:
            JobInfo or None if not found
        """
        # Check in-memory first
        if job_id in self._jobs:
            return self._jobs[job_id]

        # Check database
        if self.db:
            data = self.db.get_job(job_id)
            if data:
                return self._job_from_db(data)

        return None

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running or pending job.

        Args:
            job_id: Job to cancel

        Returns:
            True if cancellation was successful
        """
        if job_id not in self._jobs:
            return False

        job = self._jobs[job_id]

        if job.status not in (JobStatus.PENDING, JobStatus.PROCESSING):
            logger.warning(f"Cannot cancel job {job_id} in status {job.status}")
            return False

        # Signal cancellation
        if job_id in self._cancel_events:
            self._cancel_events[job_id].set()

        # Cancel task if running
        if job_id in self._running_tasks:
            self._running_tasks[job_id].cancel()
            try:
                await self._running_tasks[job_id]
            except asyncio.CancelledError:
                pass

        # Update status
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now()

        if self.db:
            self.db.update_job_status(job_id, "cancelled")

        logger.info(f"Cancelled job {job_id}")
        return True

    async def list_jobs(
        self,
        voice_id: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 100,
    ) -> List[JobInfo]:
        """
        List jobs with optional filtering.

        Args:
            voice_id: Filter by voice ID
            status: Filter by status
            limit: Maximum number of jobs to return

        Returns:
            List of JobInfo objects
        """
        jobs = list(self._jobs.values())

        if voice_id:
            jobs = [j for j in jobs if j.voice_id == voice_id]

        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by creation time (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    def _job_from_db(self, data: Dict[str, Any]) -> JobInfo:
        """Convert database row to JobInfo."""
        return JobInfo(
            job_id=data["job_id"],
            voice_id=data["voice_id"],
            status=JobStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            sample_count=data.get("sample_count", 0),
            total_duration_seconds=data.get("total_duration_seconds", 0.0),
            consent_verified=bool(data.get("consent_verified")),
            adapter_path=data.get("adapter_path"),
            run_id=data.get("run_id"),
            final_loss=data.get("final_loss"),
            verification_passed=data.get("verification_passed"),
        )


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_manager_instance: Optional[FineTuningJobManager] = None


def get_job_manager(db: Optional["VoxDatabase"] = None) -> FineTuningJobManager:
    """Get or create the singleton job manager."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = FineTuningJobManager(db)
    return _manager_instance


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("=" * 70)
        print("  VØX Fine-Tuning Job Manager Demo")
        print("=" * 70)

        print("\n1. Creating job manager...")
        manager = FineTuningJobManager(db=None, max_concurrent_jobs=2)
        print(f"   Max concurrent jobs: {manager.max_concurrent_jobs}")

        print("\n2. Creating a job...")
        job_id = await manager.create_job(
            voice_id="demo_voice",
            audio_files=["voice1.wav", "voice2.wav"],
            consent_verified=True,
            total_duration=120.0,
            epochs=10,
        )
        print(f"   Job ID: {job_id}")

        print("\n3. Getting job status...")
        status = await manager.get_status(job_id)
        if status:
            print(f"   Status: {status.status.value}")
            print(f"   Voice ID: {status.voice_id}")
            print(f"   Created: {status.created_at}")

        print("\n4. Listing jobs...")
        jobs = await manager.list_jobs()
        print(f"   Found {len(jobs)} job(s)")
        for job in jobs:
            print(f"   - {job.job_id}: {job.status.value}")

        print("\n5. Cancelling job...")
        cancelled = await manager.cancel_job(job_id)
        print(f"   Cancelled: {cancelled}")

        status = await manager.get_status(job_id)
        if status:
            print(f"   New status: {status.status.value}")

        print("\n" + "=" * 70)
        print("  Demo complete!")
        print("=" * 70)

    asyncio.run(demo())
