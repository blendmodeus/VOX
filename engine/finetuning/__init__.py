"""
AXIOM VOX Fine-Tuning Module
-----------------------------

Voice cloning via LoRA adapters for Qwen3-TTS.

Components:
    - LoRAConfig, LoRALayer, VoxLoRAAdapter: Low-rank adaptation
    - AudioProcessor: Audio preprocessing pipeline
    - VoxFineTuningPipeline: Training orchestrator
    - VoxCheckpointManager: Checkpoint management
    - VoiceVerifier: Quality and governance verification
    - FineTuningJobManager: Job queue management

Usage:
    from axiom_vox.finetuning import VoxFineTuningPipeline, FineTuningConfig

    pipeline = VoxFineTuningPipeline(
        config=FineTuningConfig(epochs=50),
        voice_id="clone_my_voice",
    )

    result = await pipeline.train(
        audio_samples=["path/to/audio1.wav", "path/to/audio2.wav"],
        consent_verified=True,
    )

    if result.success:
        print(f"Voice cloned: {result.voice_id}")
        print(f"Adapter saved: {result.adapter_path}")
"""

from axiom_vox.finetuning.lora_adapter import (
    LoRAConfig,
    LoRALayer,
    VoxLoRAAdapter,
)

from axiom_vox.finetuning.checkpoint import (
    VoxCheckpointManager,
)

from axiom_vox.finetuning.audio_processor import (
    AudioProcessor,
    AudioSample,
)

from axiom_vox.finetuning.training_pipeline import (
    VoxFineTuningPipeline,
    FineTuningConfig,
    FineTuningResult,
)

from axiom_vox.finetuning.verification import (
    VoiceVerifier,
    VerificationResult,
)

from axiom_vox.finetuning.job_manager import (
    FineTuningJobManager,
    JobStatus,
    JobInfo,
)

__all__ = [
    # LoRA
    "LoRAConfig",
    "LoRALayer",
    "VoxLoRAAdapter",
    # Checkpoint
    "VoxCheckpointManager",
    # Audio
    "AudioProcessor",
    "AudioSample",
    # Training
    "VoxFineTuningPipeline",
    "FineTuningConfig",
    "FineTuningResult",
    # Verification
    "VoiceVerifier",
    "VerificationResult",
    # Jobs
    "FineTuningJobManager",
    "JobStatus",
    "JobInfo",
]

__version__ = "0.1.0"
