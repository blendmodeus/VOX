"""
VØX Fine-Tuning Pipeline
------------------------

Main training orchestrator for voice cloning via LoRA adapters.

Stages:
1. Data Preparation: Process audio samples
2. Training: Fine-tune LoRA adapters
3. Verification: Check voice quality and governance compliance

Pattern from: axiom_organism/experiments/run_train.py

Usage:
    from axiom_vox.finetuning import VoxFineTuningPipeline, FineTuningConfig

    config = FineTuningConfig(epochs=50, fast=False)
    pipeline = VoxFineTuningPipeline(config, voice_id="my_voice")

    result = await pipeline.train(
        audio_samples=["voice1.wav", "voice2.wav"],
        consent_verified=True,
    )

    if result.success:
        print(f"Adapter saved: {result.adapter_path}")
"""

from __future__ import annotations

import uuid
import logging
import asyncio
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

from axiom_vox.finetuning.lora_adapter import LoRAConfig, VoxLoRAAdapter
from axiom_vox.finetuning.audio_processor import AudioProcessor, AudioSample
from axiom_vox.finetuning.checkpoint import (
    VoxCheckpointManager,
    TraceWriter,
    write_run_manifest,
)

if TYPE_CHECKING:
    from axiom_vox.persistence import VoxDatabase

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class FineTuningConfig:
    """
    Configuration for voice fine-tuning.

    Pattern from: axiom_organism/configs/thresholds.yaml
    """

    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    gradient_accumulation: int = 4
    max_grad_norm: float = 1.0

    # LoRA configuration
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05

    # Checkpointing
    checkpoint_every: int = 10
    max_checkpoints: int = 3

    # Device
    device: str = "auto"  # auto/cuda/mps/cpu

    # Fast mode for iteration
    fast: bool = False

    # Audio processing
    min_audio_duration: float = 60.0   # Minimum total duration
    max_audio_duration: float = 600.0  # Maximum total duration

    # Verification
    run_verification: bool = True
    min_similarity_score: float = 0.5
    min_quality_score: float = 0.4

    def __post_init__(self):
        """Apply fast mode adjustments."""
        if self.fast:
            self.epochs = min(self.epochs, 20)
            self.batch_size = min(self.batch_size, 2)
            self.warmup_steps = min(self.warmup_steps, 50)
            self.lora_rank = min(self.lora_rank, 4)
            self.lora_alpha = min(self.lora_alpha, 8.0)
            self.checkpoint_every = min(self.checkpoint_every, 5)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FineTuningConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str) -> "FineTuningConfig":
        """Load from YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML required: pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


@dataclass
class FineTuningResult:
    """Result from fine-tuning pipeline."""

    success: bool
    voice_id: str
    run_id: str
    adapter_path: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    # Training details
    epochs_completed: int = 0
    final_loss: Optional[float] = None
    training_time_seconds: float = 0.0

    # Verification
    verification_passed: Optional[bool] = None
    similarity_score: Optional[float] = None
    quality_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

class VoxFineTuningPipeline:
    """
    Main voice cloning fine-tuning pipeline.

    Stages:
    1. Data Preparation: Process audio samples
    2. Training: Fine-tune LoRA adapters
    3. Verification: Check voice quality and governance compliance
    """

    def __init__(
        self,
        config: FineTuningConfig,
        voice_id: str,
        db: Optional["VoxDatabase"] = None,
        trace_dir: Optional[str] = None,
    ):
        """
        Initialize pipeline.

        Args:
            config: Training configuration
            voice_id: Identifier for the cloned voice
            db: Optional database for persistence
            trace_dir: Optional directory for traces (default: ~/.axiom_vox/runs)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for fine-tuning")

        self.config = config
        self.voice_id = voice_id
        self.db = db

        # Generate run ID (8-char hex, following organism pattern)
        self.run_id = uuid.uuid4().hex[:8]

        # Set up trace directory
        if trace_dir:
            self.trace_dir = Path(trace_dir)
        else:
            self.trace_dir = Path.home() / ".axiom_vox" / "runs" / f"finetune_{self.run_id}"
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        # Initialize checkpoint manager
        self.ckpt_mgr = VoxCheckpointManager(
            trace_dir=str(self.trace_dir),
            max_checkpoints=config.max_checkpoints,
        )

        # Detect device
        self.device = self._detect_device(config.device)
        logger.info(f"Using device: {self.device}")

        # Will be initialized during training
        self._model = None
        self._adapter = None
        self._optimizer = None
        self._scheduler = None
        self._trace_writer = None

    def _detect_device(self, preference: str) -> str:
        """Detect best available device (following organism pattern)."""
        if preference != "auto":
            return preference
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    async def train(
        self,
        audio_samples: List[str],
        consent_verified: bool,
        transcripts: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FineTuningResult:
        """
        Run the full fine-tuning pipeline.

        Args:
            audio_samples: Paths to audio files
            consent_verified: Whether voice owner consent is verified
            transcripts: Optional dict mapping file path to transcript
            metadata: Additional metadata

        Returns:
            FineTuningResult with adapter path and metrics
        """
        start_time = datetime.now()
        warnings = []

        try:
            # Initialize trace writer
            self._trace_writer = TraceWriter(
                trace_dir=str(self.trace_dir),
                run_id=self.run_id,
            )

            self._trace_writer.write({
                "event": "pipeline_start",
                "voice_id": self.voice_id,
                "run_id": self.run_id,
                "config": self.config.to_dict(),
            })

            # ================================================================
            # STAGE 1: DATA PREPARATION
            # ================================================================
            self._trace_writer.write({"event": "stage_start", "stage": "data_prep"})
            logger.info("Stage 1: Data Preparation")

            processor = AudioProcessor(
                min_duration=1.0,
                max_duration=30.0,
                enable_augmentation=True,
            )

            samples = processor.process_files(
                audio_paths=audio_samples,
                voice_id=self.voice_id,
                transcripts=transcripts,
            )

            # Validate samples
            is_valid, validation_warnings = processor.validate_samples(
                samples,
                min_total_duration=self.config.min_audio_duration,
                max_total_duration=self.config.max_audio_duration,
            )
            warnings.extend(validation_warnings)

            if not is_valid:
                return FineTuningResult(
                    success=False,
                    voice_id=self.voice_id,
                    run_id=self.run_id,
                    error="Audio validation failed",
                    warnings=warnings,
                )

            total_duration = processor.get_total_duration(samples)
            logger.info(f"Processed {len(samples)} samples ({total_duration:.1f}s total)")

            self._trace_writer.write({
                "event": "data_prep_complete",
                "sample_count": len(samples),
                "total_duration": total_duration,
            })

            # Write run manifest
            write_run_manifest(
                trace_dir=str(self.trace_dir),
                run_id=self.run_id,
                voice_id=self.voice_id,
                config=self.config.to_dict(),
                sample_info={
                    "count": len(samples),
                    "total_duration": total_duration,
                    "source_files": list(set(s.source_path for s in samples)),
                },
                extra=metadata,
            )

            # ================================================================
            # STAGE 2: TRAINING
            # ================================================================
            self._trace_writer.write({"event": "stage_start", "stage": "training"})
            logger.info("Stage 2: Training")

            adapter, final_loss, epochs_completed = await self._train_adapter(samples)

            adapter_path = self.trace_dir / f"adapter_{self.voice_id}.pt"
            adapter.save_adapter(str(adapter_path))

            logger.info(f"Training complete: loss={final_loss:.4f}, epochs={epochs_completed}")

            self._trace_writer.write({
                "event": "training_complete",
                "final_loss": final_loss,
                "epochs_completed": epochs_completed,
                "adapter_path": str(adapter_path),
            })

            # ================================================================
            # STAGE 3: VERIFICATION
            # ================================================================
            verification_passed = None
            similarity_score = None
            quality_score = None

            if self.config.run_verification:
                self._trace_writer.write({"event": "stage_start", "stage": "verification"})
                logger.info("Stage 3: Verification")

                try:
                    from axiom_vox.finetuning.verification import VoiceVerifier

                    verifier = VoiceVerifier()
                    verification = await verifier.verify(
                        adapter=adapter,
                        reference_samples=samples[:3],  # Use first 3 samples
                        consent_verified=consent_verified,
                    )

                    verification_passed = verification.passed
                    similarity_score = verification.similarity_score
                    quality_score = verification.quality_score
                    warnings.extend(verification.warnings)

                    logger.info(
                        f"Verification: passed={verification_passed}, "
                        f"similarity={similarity_score:.2f}, quality={quality_score:.2f}"
                    )

                    self._trace_writer.write({
                        "event": "verification_complete",
                        "passed": verification_passed,
                        "similarity_score": similarity_score,
                        "quality_score": quality_score,
                    })

                except ImportError:
                    logger.warning("Verification module not available, skipping")
                    warnings.append("Verification skipped (module not available)")

            # ================================================================
            # COMPLETE
            # ================================================================
            training_time = (datetime.now() - start_time).total_seconds()

            self._trace_writer.write({
                "event": "pipeline_complete",
                "success": True,
                "training_time_seconds": training_time,
            })
            self._trace_writer.close()

            # Update database if available
            if self.db:
                self._update_database(
                    adapter_path=str(adapter_path),
                    final_loss=final_loss,
                    epochs_completed=epochs_completed,
                    verification_passed=verification_passed,
                    metadata=metadata,
                )

            return FineTuningResult(
                success=True,
                voice_id=self.voice_id,
                run_id=self.run_id,
                adapter_path=str(adapter_path),
                metrics={
                    "final_loss": final_loss,
                    "total_duration": total_duration,
                    "sample_count": len(samples),
                },
                warnings=warnings,
                epochs_completed=epochs_completed,
                final_loss=final_loss,
                training_time_seconds=training_time,
                verification_passed=verification_passed,
                similarity_score=similarity_score,
                quality_score=quality_score,
            )

        except Exception as e:
            logger.exception(f"Fine-tuning failed: {e}")

            if self._trace_writer:
                self._trace_writer.write({
                    "event": "pipeline_error",
                    "error": str(e),
                })
                self._trace_writer.close()

            return FineTuningResult(
                success=False,
                voice_id=self.voice_id,
                run_id=self.run_id,
                error=str(e),
                warnings=warnings,
            )

    async def _train_adapter(
        self,
        samples: List[AudioSample],
    ) -> tuple:
        """
        Train LoRA adapter on samples.

        Returns:
            (adapter, final_loss, epochs_completed)
        """
        # Load base model
        from axiom_vox.synthesis import get_synthesizer

        synthesizer = get_synthesizer(model_size="large")
        synthesizer._ensure_loaded()
        base_model = synthesizer._model

        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False

        # Create LoRA adapter
        lora_config = LoRAConfig(
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
        )
        adapter = VoxLoRAAdapter(base_model, lora_config, self.voice_id)
        adapter.inject_adapters()

        # Move to device
        for lora in adapter.lora_layers.values():
            lora.to(self.device)

        # Optimizer (only LoRA params)
        trainable_params = adapter.get_trainable_parameters()
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        # Scheduler
        total_steps = self.config.epochs * len(samples) // self.config.batch_size
        scheduler = self._get_scheduler(optimizer, total_steps)

        self._adapter = adapter
        self._optimizer = optimizer
        self._scheduler = scheduler

        # Training loop
        best_loss = float('inf')
        final_loss = 0.0

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Shuffle samples
            indices = torch.randperm(len(samples)).tolist()

            for i in range(0, len(indices), self.config.batch_size):
                batch_indices = indices[i:i + self.config.batch_size]
                batch_samples = [samples[j] for j in batch_indices]

                # Compute loss
                loss = self._compute_batch_loss(batch_samples)
                loss = loss / self.config.gradient_accumulation
                loss.backward()

                epoch_loss += loss.item() * self.config.gradient_accumulation
                num_batches += 1

                # Gradient accumulation step
                if num_batches % self.config.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        self.config.max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # Epoch complete
            avg_loss = epoch_loss / max(num_batches, 1)
            final_loss = avg_loss

            # Log
            self._trace_writer.write({
                "event": "epoch_end",
                "epoch": epoch,
                "loss": avg_loss,
                "lr": scheduler.get_last_lr()[0],
            })

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.config.epochs}: loss={avg_loss:.4f}")

            # Checkpoint
            if (epoch + 1) % self.config.checkpoint_every == 0:
                self.ckpt_mgr.save(
                    epoch=epoch,
                    adapter=adapter,
                    optimizer=optimizer,
                    metrics={"loss": avg_loss, "lr": scheduler.get_last_lr()[0]},
                    scheduler=scheduler,
                )

            # Track best
            if avg_loss < best_loss:
                best_loss = avg_loss

            # Allow async cancellation
            await asyncio.sleep(0)

        return adapter, final_loss, self.config.epochs

    def _compute_batch_loss(self, samples: List[AudioSample]) -> torch.Tensor:
        """
        Compute loss for a batch of samples.

        This is a simplified loss computation. In a full implementation,
        this would involve:
        1. Forward pass through base model with LoRA
        2. Compute mel reconstruction loss
        3. Optionally add speaker consistency loss
        """
        # Stack mel spectrograms
        mels = torch.stack([s.mel_spectrogram for s in samples])
        mels = mels.to(self.device)

        # For now, compute a simple reconstruction-style loss
        # In production, this would be the actual TTS loss
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for name, lora in self._adapter.lora_layers.items():
            # Simple regularization loss on LoRA weights
            # This encourages small adaptations
            a_norm = lora.lora_A.norm()
            b_norm = lora.lora_B.norm()
            loss = loss + 0.01 * (a_norm + b_norm)

        # Add a dummy term to ensure gradient flow
        # In production, replace with actual mel reconstruction loss
        dummy_pred = mels.mean() * 0.0
        loss = loss + dummy_pred

        return loss

    def _get_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
    ) -> Any:
        """Create learning rate scheduler with warmup."""
        try:
            from transformers import get_cosine_schedule_with_warmup
            return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps,
            )
        except ImportError:
            # Fallback to simple linear warmup + constant
            from torch.optim.lr_scheduler import LambdaLR

            def lr_lambda(step):
                if step < self.config.warmup_steps:
                    return step / max(self.config.warmup_steps, 1)
                return 1.0

            return LambdaLR(optimizer, lr_lambda)

    def _update_database(
        self,
        adapter_path: str,
        final_loss: float,
        epochs_completed: int,
        verification_passed: Optional[bool],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Update database with training results."""
        if not self.db:
            return

        try:
            # Register adapter
            self.db.register_adapter(
                voice_id=self.voice_id,
                job_id=self.run_id,
                adapter_path=adapter_path,
                lora_rank=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                parameter_count=self._adapter._parameter_count if self._adapter else 0,
            )

            # Update job status
            self.db.update_job_status(
                job_id=self.run_id,
                status="verified" if verification_passed else "completed",
                final_loss=final_loss,
                epochs_completed=epochs_completed,
                verification_passed=verification_passed,
            )
        except Exception as e:
            logger.error(f"Failed to update database: {e}")

    def resume_from_checkpoint(self) -> Optional[int]:
        """
        Resume training from latest checkpoint.

        Returns:
            Starting epoch or None if no checkpoint
        """
        checkpoint = self.ckpt_mgr.load_latest()
        if checkpoint is None:
            return None

        if not self._adapter:
            raise RuntimeError("Adapter not initialized. Call train() first.")

        epoch = self.ckpt_mgr.load_into_adapter(
            checkpoint,
            self._adapter,
            self._optimizer,
            self._scheduler,
        )

        logger.info(f"Resumed from epoch {epoch}")
        return epoch


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def finetune_voice(
    audio_files: List[str],
    voice_name: str,
    consent_verified: bool = False,
    fast: bool = False,
    epochs: int = 50,
) -> FineTuningResult:
    """
    Convenience function to fine-tune a voice.

    Args:
        audio_files: Paths to audio files
        voice_name: Name for the cloned voice
        consent_verified: Whether consent is verified
        fast: Use fast training configuration
        epochs: Number of training epochs

    Returns:
        FineTuningResult with adapter path
    """
    import hashlib

    # Generate voice ID from name
    voice_id = f"clone_{hashlib.sha256(voice_name.encode()).hexdigest()[:12]}"

    config = FineTuningConfig(epochs=epochs, fast=fast)
    pipeline = VoxFineTuningPipeline(config, voice_id)

    return await pipeline.train(
        audio_samples=audio_files,
        consent_verified=consent_verified,
    )


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  VØX Fine-Tuning Pipeline Demo")
    print("=" * 70)

    if not HAS_TORCH:
        print("\nPyTorch not available. Install with: pip install torch")
        exit(1)

    print("\n1. Creating configuration...")
    config = FineTuningConfig(
        epochs=5,
        batch_size=2,
        fast=True,
    )
    print(f"   Epochs: {config.epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   LoRA rank: {config.lora_rank}")
    print(f"   Fast mode: {config.fast}")

    print("\n2. Creating pipeline...")
    pipeline = VoxFineTuningPipeline(
        config=config,
        voice_id="demo_voice",
    )
    print(f"   Run ID: {pipeline.run_id}")
    print(f"   Device: {pipeline.device}")
    print(f"   Trace dir: {pipeline.trace_dir}")

    print("\n3. Pipeline ready for training")
    print("   (Requires audio files to actually train)")
    print("   Usage:")
    print("     result = await pipeline.train(")
    print('         audio_samples=["voice1.wav", "voice2.wav"],')
    print("         consent_verified=True,")
    print("     )")

    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)
