"""
VØX Fine-Tuning Module Tests
-----------------------------

Tests for the voice cloning pipeline components.

Run with: pytest axiom_vox/tests/test_finetuning.py -v
"""

import asyncio
import pytest
import tempfile
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_db():
    """Create a mock database."""
    db = Mock()
    db.create_finetune_job.return_value = "ft_test123"
    db.get_job.return_value = {
        "job_id": "ft_test123",
        "voice_id": "clone_abc",
        "status": "pending",
        "created_at": datetime.now().isoformat(),
    }
    db.get_adapter.return_value = {
        "adapter_id": "lora_clone_abc",
        "voice_id": "clone_abc",
        "adapter_path": "/tmp/adapter.pt",
        "lora_rank": 8,
        "lora_alpha": 16.0,
    }
    return db


# ============================================================================
# LORA CONFIG TESTS
# ============================================================================

class TestLoRAConfig:
    """Tests for LoRA configuration."""

    def test_default_config(self):
        """Test default LoRA configuration values."""
        from axiom_vox.finetuning.lora_adapter import LoRAConfig

        config = LoRAConfig()

        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.dropout == 0.05
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules

    def test_scaling_calculation(self):
        """Test LoRA scaling factor calculation."""
        from axiom_vox.finetuning.lora_adapter import LoRAConfig

        # Scaling = alpha / rank
        config = LoRAConfig(rank=8, alpha=16.0)
        expected_scaling = config.alpha / config.rank
        assert expected_scaling == 2.0

        config = LoRAConfig(rank=4, alpha=8.0)
        expected_scaling = config.alpha / config.rank
        assert expected_scaling == 2.0

    def test_custom_config(self):
        """Test custom LoRA configuration."""
        from axiom_vox.finetuning.lora_adapter import LoRAConfig

        config = LoRAConfig(
            rank=16,
            alpha=32.0,
            dropout=0.1,
            target_modules=["q_proj", "k_proj"],
        )

        assert config.rank == 16
        assert config.alpha == 32.0
        assert config.dropout == 0.1
        assert config.target_modules == ["q_proj", "k_proj"]


# ============================================================================
# JOB STATUS TESTS
# ============================================================================

class TestJobStatus:
    """Tests for job status enumeration."""

    def test_job_status_values(self):
        """Test all job status values exist."""
        from axiom_vox.finetuning.job_manager import JobStatus

        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.PROCESSING.value == "processing"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.VERIFIED.value == "verified"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_job_status_from_string(self):
        """Test creating job status from string."""
        from axiom_vox.finetuning.job_manager import JobStatus

        assert JobStatus("pending") == JobStatus.PENDING
        assert JobStatus("completed") == JobStatus.COMPLETED


# ============================================================================
# JOB INFO TESTS
# ============================================================================

class TestJobInfo:
    """Tests for JobInfo dataclass."""

    def test_job_info_creation(self):
        """Test creating a JobInfo instance."""
        from axiom_vox.finetuning.job_manager import JobInfo, JobStatus

        job = JobInfo(
            job_id="ft_test123",
            voice_id="clone_abc",
            status=JobStatus.PENDING,
            created_at=datetime.now(),
        )

        assert job.job_id == "ft_test123"
        assert job.voice_id == "clone_abc"
        assert job.status == JobStatus.PENDING
        assert job.progress == 0.0
        assert job.consent_verified == False

    def test_job_info_to_dict(self):
        """Test converting JobInfo to dictionary."""
        from axiom_vox.finetuning.job_manager import JobInfo, JobStatus

        now = datetime.now()
        job = JobInfo(
            job_id="ft_test123",
            voice_id="clone_abc",
            status=JobStatus.COMPLETED,
            created_at=now,
            progress=1.0,
            final_loss=0.05,
        )

        data = job.to_dict()

        assert data["job_id"] == "ft_test123"
        assert data["status"] == "completed"
        assert data["progress"] == 1.0
        assert data["final_loss"] == 0.05


# ============================================================================
# JOB MANAGER TESTS
# ============================================================================

class TestFineTuningJobManager:
    """Tests for the fine-tuning job manager."""

    def test_create_job(self):
        """Test creating a new job."""
        from axiom_vox.finetuning.job_manager import FineTuningJobManager

        async def _test():
            manager = FineTuningJobManager(db=None, max_concurrent_jobs=2)

            job_id = await manager.create_job(
                voice_id="test_voice",
                audio_files=["file1.wav", "file2.wav"],
                consent_verified=True,
                total_duration=120.0,
            )

            assert job_id.startswith("ft_")
            assert len(job_id) == 15  # "ft_" + 12 hex chars

        asyncio.run(_test())

    def test_get_status(self):
        """Test getting job status."""
        from axiom_vox.finetuning.job_manager import FineTuningJobManager, JobStatus

        async def _test():
            manager = FineTuningJobManager(db=None)

            job_id = await manager.create_job(
                voice_id="test_voice",
                audio_files=["file1.wav"],
                consent_verified=True,
            )

            status = await manager.get_status(job_id)

            assert status is not None
            assert status.job_id == job_id
            assert status.status == JobStatus.PENDING

        asyncio.run(_test())

    def test_cancel_job(self):
        """Test cancelling a job."""
        from axiom_vox.finetuning.job_manager import FineTuningJobManager, JobStatus

        async def _test():
            manager = FineTuningJobManager(db=None)

            job_id = await manager.create_job(
                voice_id="test_voice",
                audio_files=["file1.wav"],
                consent_verified=True,
            )

            cancelled = await manager.cancel_job(job_id)

            assert cancelled == True

            status = await manager.get_status(job_id)
            assert status.status == JobStatus.CANCELLED

        asyncio.run(_test())

    def test_list_jobs(self):
        """Test listing jobs."""
        from axiom_vox.finetuning.job_manager import FineTuningJobManager

        async def _test():
            manager = FineTuningJobManager(db=None)

            # Create multiple jobs
            await manager.create_job(
                voice_id="voice_1",
                audio_files=["file1.wav"],
                consent_verified=True,
            )
            await manager.create_job(
                voice_id="voice_2",
                audio_files=["file2.wav"],
                consent_verified=True,
            )

            jobs = await manager.list_jobs()

            assert len(jobs) == 2

        asyncio.run(_test())

    def test_list_jobs_with_filter(self):
        """Test listing jobs with voice_id filter."""
        from axiom_vox.finetuning.job_manager import FineTuningJobManager

        async def _test():
            manager = FineTuningJobManager(db=None)

            await manager.create_job(
                voice_id="voice_1",
                audio_files=["file1.wav"],
                consent_verified=True,
            )
            await manager.create_job(
                voice_id="voice_2",
                audio_files=["file2.wav"],
                consent_verified=True,
            )

            jobs = await manager.list_jobs(voice_id="voice_1")

            assert len(jobs) == 1
            assert jobs[0].voice_id == "voice_1"

        asyncio.run(_test())


# ============================================================================
# VERIFICATION RESULT TESTS
# ============================================================================

class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_verification_result_passed(self):
        """Test a passed verification result."""
        from axiom_vox.finetuning.verification import VerificationResult

        result = VerificationResult(
            passed=True,
            similarity_score=0.85,
            quality_score=0.75,
            naturalness_score=0.70,
            governance_compliant=True,
        )

        assert result.passed == True
        assert result.similarity_score == 0.85
        assert result.governance_compliant == True
        assert len(result.warnings) == 0

    def test_verification_result_failed(self):
        """Test a failed verification result."""
        from axiom_vox.finetuning.verification import VerificationResult

        result = VerificationResult(
            passed=False,
            similarity_score=0.3,
            quality_score=0.5,
            naturalness_score=0.5,
            governance_compliant=True,
            warnings=["Low voice similarity: 0.30 < 0.50"],
        )

        assert result.passed == False
        assert len(result.warnings) == 1

    def test_verification_result_to_dict(self):
        """Test converting VerificationResult to dictionary."""
        from axiom_vox.finetuning.verification import VerificationResult

        result = VerificationResult(
            passed=True,
            similarity_score=0.8,
            quality_score=0.7,
            naturalness_score=0.6,
            consistency_score=0.9,
        )

        data = result.to_dict()

        assert data["passed"] == True
        assert data["similarity_score"] == 0.8
        assert data["quality_score"] == 0.7


# ============================================================================
# VOICE VERIFIER TESTS
# ============================================================================

class TestVoiceVerifier:
    """Tests for the voice verifier."""

    def test_verifier_initialization(self):
        """Test verifier initialization with custom thresholds."""
        from axiom_vox.finetuning.verification import VoiceVerifier

        verifier = VoiceVerifier(
            min_similarity=0.6,
            min_quality=0.5,
            min_naturalness=0.5,
        )

        assert verifier.min_similarity == 0.6
        assert verifier.min_quality == 0.5
        assert verifier.min_naturalness == 0.5

    def test_test_prompts_exist(self):
        """Test that verification prompts are defined."""
        from axiom_vox.finetuning.verification import VoiceVerifier

        assert len(VoiceVerifier.TEST_PROMPTS) >= 5
        assert all(isinstance(p, str) for p in VoiceVerifier.TEST_PROMPTS)


# ============================================================================
# FINE-TUNING CONFIG TESTS
# ============================================================================

class TestFineTuningConfig:
    """Tests for fine-tuning configuration."""

    def test_default_config(self):
        """Test default fine-tuning configuration."""
        from axiom_vox.finetuning.training_pipeline import FineTuningConfig

        config = FineTuningConfig()

        assert config.epochs == 50
        assert config.batch_size == 4
        assert config.learning_rate == 1e-4
        assert config.lora_rank == 8
        assert config.fast == False

    def test_fast_mode_config(self):
        """Test fast mode configuration."""
        from axiom_vox.finetuning.training_pipeline import FineTuningConfig

        config = FineTuningConfig(fast=True)

        # Fast mode should have fewer epochs
        assert config.fast == True


# ============================================================================
# FINE-TUNING RESULT TESTS
# ============================================================================

class TestFineTuningResult:
    """Tests for fine-tuning results."""

    def test_successful_result(self):
        """Test a successful fine-tuning result."""
        from axiom_vox.finetuning.training_pipeline import FineTuningResult

        result = FineTuningResult(
            success=True,
            voice_id="clone_abc123",
            run_id="run_test123",
            adapter_path="/tmp/adapters/clone_abc123.pt",
            final_loss=0.05,
        )

        assert result.success == True
        assert result.voice_id == "clone_abc123"
        assert result.run_id == "run_test123"

    def test_failed_result(self):
        """Test a failed fine-tuning result."""
        from axiom_vox.finetuning.training_pipeline import FineTuningResult

        result = FineTuningResult(
            success=False,
            voice_id="clone_failed",
            run_id="run_failed123",
            error="Insufficient audio samples",
        )

        assert result.success == False
        assert result.error == "Insufficient audio samples"


# ============================================================================
# CHECKPOINT MANAGER TESTS
# ============================================================================

class TestCheckpointManager:
    """Tests for checkpoint management (without PyTorch)."""

    def test_checkpoint_manager_creation(self, temp_dir):
        """Test that checkpoint manager is created properly."""
        from axiom_vox.finetuning.checkpoint import VoxCheckpointManager

        manager = VoxCheckpointManager(
            trace_dir=temp_dir,
            max_checkpoints=3,
        )

        assert manager.max_checkpoints == 3
        assert str(manager.trace_dir) == temp_dir


class TestTraceWriter:
    """Tests for trace writing."""

    def test_trace_writer(self, temp_dir):
        """Test writing trace events."""
        from axiom_vox.finetuning.checkpoint import TraceWriter

        run_id = "test_run_123"
        writer = TraceWriter(temp_dir, run_id)

        writer.write({"event": "test_event", "key": "value"})
        writer.close()

        trace_file = os.path.join(temp_dir, f"trace_{run_id}.jsonl")
        assert os.path.exists(trace_file)

        # Read and verify
        with open(trace_file) as f:
            content = f.read()
            assert "test_event" in content
            assert "key" in content


# ============================================================================
# AUDIO SAMPLE TESTS
# ============================================================================

class TestAudioSample:
    """Tests for AudioSample dataclass."""

    def test_audio_sample_creation(self):
        """Test creating an audio sample."""
        from axiom_vox.finetuning.audio_processor import AudioSample

        # AudioSample requires waveform, sample_rate, duration_seconds, mel_spectrogram
        sample = AudioSample(
            waveform=None,  # Would be tensor in production
            sample_rate=24000,
            duration_seconds=5.0,
            mel_spectrogram=None,  # Would be tensor in production
            source_path="/tmp/voice.wav",
        )

        assert sample.source_path == "/tmp/voice.wav"
        assert sample.duration_seconds == 5.0
        assert sample.sample_rate == 24000


# ============================================================================
# PERSISTENCE INTEGRATION TESTS
# ============================================================================

class TestPersistenceIntegration:
    """Tests for database integration."""

    def test_create_finetune_job(self, temp_dir):
        """Test creating a fine-tuning job in the database."""
        from axiom_vox.persistence import VoxDatabase

        db_path = os.path.join(temp_dir, "test.db")
        db = VoxDatabase(db_path)

        job_id = db.create_finetune_job(
            voice_id="test_voice",
            sample_count=5,
            total_duration=120.0,
            consent_verified=True,
        )

        assert job_id.startswith("ft_")

        # Retrieve the job
        job = db.get_job(job_id)
        assert job is not None
        assert job["voice_id"] == "test_voice"
        assert job["status"] == "pending"
        assert job["consent_verified"] == True

    def test_update_job_status(self, temp_dir):
        """Test updating job status."""
        from axiom_vox.persistence import VoxDatabase

        db_path = os.path.join(temp_dir, "test.db")
        db = VoxDatabase(db_path)

        job_id = db.create_finetune_job(
            voice_id="test_voice",
            sample_count=5,
            total_duration=120.0,
            consent_verified=True,
        )

        db.update_job_status(
            job_id=job_id,
            status="completed",
            final_loss=0.05,
            verification_passed=True,
        )

        job = db.get_job(job_id)
        assert job["status"] == "completed"
        assert job["final_loss"] == 0.05
        assert job["verification_passed"] == True

    def test_register_adapter(self, temp_dir):
        """Test registering a LoRA adapter."""
        from axiom_vox.persistence import VoxDatabase

        db_path = os.path.join(temp_dir, "test.db")
        db = VoxDatabase(db_path)

        # First create a job
        job_id = db.create_finetune_job(
            voice_id="clone_test123",
            sample_count=5,
            total_duration=120.0,
            consent_verified=True,
        )

        # Register adapter
        adapter_id = db.register_adapter(
            voice_id="clone_test123",
            job_id=job_id,
            adapter_path="/tmp/adapters/clone_test123.pt",
            lora_rank=8,
            lora_alpha=16.0,
            parameter_count=50000,
        )

        assert adapter_id == "lora_clone_test123"

        # Retrieve adapter
        adapter = db.get_adapter("clone_test123")
        assert adapter is not None
        assert adapter["lora_rank"] == 8
        assert adapter["lora_alpha"] == 16.0

    def test_list_adapters(self, temp_dir):
        """Test listing adapters."""
        from axiom_vox.persistence import VoxDatabase

        db_path = os.path.join(temp_dir, "test.db")
        db = VoxDatabase(db_path)

        # Create jobs and adapters
        for i in range(3):
            job_id = db.create_finetune_job(
                voice_id=f"clone_test{i}",
                sample_count=5,
                total_duration=120.0,
                consent_verified=True,
            )
            db.register_adapter(
                voice_id=f"clone_test{i}",
                job_id=job_id,
                adapter_path=f"/tmp/adapters/clone_test{i}.pt",
                lora_rank=8,
                lora_alpha=16.0,
                parameter_count=50000,
            )

        adapters = db.list_adapters()
        assert len(adapters) == 3

    def test_deactivate_adapter(self, temp_dir):
        """Test deactivating an adapter."""
        from axiom_vox.persistence import VoxDatabase

        db_path = os.path.join(temp_dir, "test.db")
        db = VoxDatabase(db_path)

        job_id = db.create_finetune_job(
            voice_id="clone_test",
            sample_count=5,
            total_duration=120.0,
            consent_verified=True,
        )

        db.register_adapter(
            voice_id="clone_test",
            job_id=job_id,
            adapter_path="/tmp/adapters/clone_test.pt",
            lora_rank=8,
            lora_alpha=16.0,
            parameter_count=50000,
        )

        # Deactivate
        db.deactivate_adapter("clone_test")

        # Should not be found when active_only=True
        adapter = db.get_adapter("clone_test")
        assert adapter is None


# ============================================================================
# EXPORTS TEST
# ============================================================================

class TestModuleExports:
    """Tests for module exports."""

    def test_finetuning_submodule_exports(self):
        """Test that fine-tuning submodule exports work."""
        from axiom_vox.finetuning import (
            FineTuningConfig,
            LoRAConfig,
            VoiceVerifier,
            VerificationResult,
            FineTuningJobManager,
            JobStatus,
            JobInfo,
        )

        # Just verify they're importable
        assert FineTuningConfig is not None
        assert LoRAConfig is not None
        assert JobStatus is not None

    def test_job_status_values(self):
        """Test that JobStatus enum has correct values."""
        from axiom_vox.finetuning import JobStatus

        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.COMPLETED.value == "completed"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
