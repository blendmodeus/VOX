"""
VØX Checkpoint Manager
----------------------

Checkpoint management for voice fine-tuning, following the pattern
from axiom_organism/experiments/checkpoint.py.

Features:
- Save/load LoRA adapter states
- Track training progress
- Keep N most recent checkpoints
- Resume from latest or specific checkpoint
- Metadata files for quick inspection

Usage:
    from axiom_vox.finetuning import VoxCheckpointManager

    ckpt_mgr = VoxCheckpointManager(trace_dir, max_checkpoints=3)

    # Save
    path = ckpt_mgr.save(epoch, adapter, optimizer, metrics)

    # Load
    state = ckpt_mgr.load_latest()
    if state:
        start_epoch = state["epoch"]
        # ... resume training
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from axiom_vox.finetuning.lora_adapter import VoxLoRAAdapter

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

logger = logging.getLogger(__name__)


class VoxCheckpointManager:
    """
    Checkpoint manager for voice fine-tuning.

    Pattern from: axiom_organism/experiments/checkpoint.py

    Features:
    - Save/load LoRA adapter states
    - Track training progress
    - Keep N most recent checkpoints
    - Resume from latest or specific checkpoint
    """

    def __init__(
        self,
        trace_dir: str,
        max_checkpoints: int = 3,
        checkpoint_prefix: str = "checkpoint",
    ):
        """
        Initialize checkpoint manager.

        Args:
            trace_dir: Directory for traces and checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            checkpoint_prefix: Prefix for checkpoint files
        """
        self.trace_dir = Path(trace_dir)
        self.checkpoint_dir = self.trace_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.checkpoint_prefix = checkpoint_prefix

    def save(
        self,
        epoch: int,
        adapter: "VoxLoRAAdapter",
        optimizer: Any,
        metrics: Dict[str, Any],
        scheduler: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a checkpoint.

        Args:
            epoch: Current epoch number
            adapter: VoxLoRAAdapter with LoRA weights
            optimizer: Optimizer state
            metrics: Training metrics (loss, etc.)
            scheduler: Optional learning rate scheduler
            extra: Additional data to save

        Returns:
            Path to saved checkpoint
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required to save checkpoints")

        checkpoint = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "voice_id": adapter.voice_id,
            "lora_config": adapter.config.to_dict(),
            "lora_state": {
                name: layer.state_dict()
                for name, layer in adapter.lora_layers.items()
            },
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "extra": extra or {},
        }

        # Save scheduler if provided
        if scheduler is not None:
            checkpoint["scheduler"] = scheduler.state_dict()

        # Save metadata about adapter
        checkpoint["adapter_metadata"] = {
            "adapted_modules": adapter._adapted_modules,
            "parameter_count": adapter._parameter_count,
        }

        # Generate filename
        filename = f"{self.checkpoint_prefix}_ep{epoch:05d}.pt"
        filepath = self.checkpoint_dir / filename

        # Save checkpoint
        torch.save(checkpoint, filepath)

        # Also save metadata JSON for quick inspection
        meta_file = self.checkpoint_dir / f"{self.checkpoint_prefix}_ep{epoch:05d}_meta.json"
        meta = {
            "epoch": epoch,
            "timestamp": checkpoint["timestamp"],
            "voice_id": adapter.voice_id,
            "metrics": self._serialize_metrics(metrics),
            "filepath": str(filepath),
            "parameter_count": adapter._parameter_count,
        }
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        logger.info(f"Saved checkpoint epoch {epoch} to {filepath}")
        return str(filepath)

    def _serialize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize metrics for JSON (handle tensors, etc.)."""
        result = {}
        for k, v in metrics.items():
            if HAS_TORCH and isinstance(v, torch.Tensor):
                result[k] = v.item() if v.numel() == 1 else v.tolist()
            elif isinstance(v, (int, float, str, bool, list, dict)):
                result[k] = v
            else:
                result[k] = str(v)
        return result

    def load(self, filepath: str) -> Dict[str, Any]:
        """
        Load a specific checkpoint.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Checkpoint dict with epoch, lora_state, optimizer, metrics
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required to load checkpoints")

        return torch.load(filepath, weights_only=False)

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.

        Returns:
            Checkpoint dict or None if no checkpoints exist
        """
        checkpoints = self._list_checkpoints()
        if not checkpoints:
            return None

        latest = checkpoints[-1]  # Sorted by epoch
        return self.load(latest["filepath"])

    def load_into_adapter(
        self,
        checkpoint: Dict[str, Any],
        adapter: "VoxLoRAAdapter",
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> int:
        """
        Load checkpoint state into adapter and optimizer.

        Args:
            checkpoint: Loaded checkpoint dict
            adapter: VoxLoRAAdapter to load into
            optimizer: Optional optimizer to restore state
            scheduler: Optional scheduler to restore state

        Returns:
            Epoch number from checkpoint
        """
        # Validate voice ID
        if checkpoint["voice_id"] != adapter.voice_id:
            logger.warning(
                f"Voice ID mismatch: checkpoint has {checkpoint['voice_id']}, "
                f"adapter has {adapter.voice_id}"
            )

        # Load LoRA states
        for name, layer_state in checkpoint["lora_state"].items():
            if name in adapter.lora_layers:
                adapter.lora_layers[name].load_state_dict(layer_state)
            else:
                logger.warning(f"Skipping unknown module in checkpoint: {name}")

        # Load optimizer state
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        # Load scheduler state
        if scheduler is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])

        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]

    def _list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints sorted by epoch."""
        checkpoints = []

        for meta_file in self.checkpoint_dir.glob(f"{self.checkpoint_prefix}_*_meta.json"):
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                    checkpoints.append(meta)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read checkpoint meta: {meta_file}: {e}")

        return sorted(checkpoints, key=lambda x: x["epoch"])

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints."""
        checkpoints = self._list_checkpoints()

        if len(checkpoints) <= self.max_checkpoints:
            return

        # Remove oldest checkpoints
        to_remove = checkpoints[:-self.max_checkpoints]

        for ckpt in to_remove:
            filepath = Path(ckpt["filepath"])
            meta_path = self.checkpoint_dir / f"{filepath.stem}_meta.json"

            if filepath.exists():
                filepath.unlink()
                logger.debug(f"Removed old checkpoint: {filepath}")
            if meta_path.exists():
                meta_path.unlink()

    def get_resume_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the latest checkpoint for resuming.

        Returns:
            Dict with epoch, metrics, filepath or None
        """
        checkpoints = self._list_checkpoints()
        if not checkpoints:
            return None
        return checkpoints[-1]

    def has_checkpoint(self) -> bool:
        """Check if any checkpoints exist."""
        return len(self._list_checkpoints()) > 0

    def get_best_checkpoint(self, metric: str = "loss", lower_is_better: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get the checkpoint with the best metric value.

        Args:
            metric: Metric name to compare
            lower_is_better: If True, lower values are better (e.g., loss)

        Returns:
            Best checkpoint metadata or None
        """
        checkpoints = self._list_checkpoints()
        if not checkpoints:
            return None

        best = None
        best_value = float('inf') if lower_is_better else float('-inf')

        for ckpt in checkpoints:
            value = ckpt.get("metrics", {}).get(metric)
            if value is None:
                continue

            if lower_is_better and value < best_value:
                best_value = value
                best = ckpt
            elif not lower_is_better and value > best_value:
                best_value = value
                best = ckpt

        return best


# ============================================================================
# TRACE WRITER (Following organism/logging.py pattern)
# ============================================================================

class TraceWriter:
    """
    Writes training trace events to JSONL file.

    Pattern from: axiom_organism/organism/logging.py

    Usage:
        writer = TraceWriter(trace_dir, run_id)
        writer.write({"event": "epoch_start", "epoch": 1})
        writer.write({"event": "loss", "value": 0.5})
        writer.close()
    """

    def __init__(
        self,
        trace_dir: str,
        run_id: str,
        flush_every: int = 10,
    ):
        """
        Initialize trace writer.

        Args:
            trace_dir: Directory for trace files
            run_id: Unique run identifier
            flush_every: Flush to disk every N events
        """
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = run_id
        self.flush_every = flush_every

        self.trace_path = self.trace_dir / f"trace_{run_id}.jsonl"
        self._file = open(self.trace_path, "a")
        self._event_count = 0

    def write(self, event: Dict[str, Any]) -> None:
        """Write an event to the trace file."""
        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = datetime.now().isoformat()

        self._file.write(json.dumps(event) + "\n")
        self._event_count += 1

        if self._event_count % self.flush_every == 0:
            self._file.flush()

    def flush(self) -> None:
        """Flush buffer to disk."""
        self._file.flush()

    def close(self) -> None:
        """Close the trace file."""
        self._file.flush()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def write_run_manifest(
    trace_dir: str,
    run_id: str,
    voice_id: str,
    config: Dict[str, Any],
    sample_info: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Write a run manifest file for reproducibility.

    Pattern from: axiom_organism/organism/repro.py

    Args:
        trace_dir: Directory for manifest
        run_id: Unique run identifier
        voice_id: Voice being fine-tuned
        config: Training configuration
        sample_info: Information about training samples
        extra: Additional metadata

    Returns:
        Path to manifest file
    """
    import sys
    import platform

    manifest = {
        "run_id": run_id,
        "voice_id": voice_id,
        "created_at": datetime.now().isoformat(),
        "config": config,
        "sample_info": sample_info,
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__ if HAS_TORCH else "not installed",
        },
        "extra": extra or {},
    }

    # Try to get git info
    try:
        import subprocess
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=trace_dir,
        ).decode().strip()
        manifest["git_commit"] = git_commit
    except Exception:
        pass

    manifest_path = Path(trace_dir) / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Wrote run manifest to {manifest_path}")
    return str(manifest_path)


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 70)
    print("  VØX Checkpoint Manager Demo")
    print("=" * 70)

    if not HAS_TORCH:
        print("\nPyTorch not available. Install with: pip install torch")
        exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n1. Creating checkpoint manager in {tmpdir}")
        ckpt_mgr = VoxCheckpointManager(tmpdir, max_checkpoints=2)

        print("\n2. Creating dummy adapter...")
        from axiom_vox.finetuning.lora_adapter import LoRAConfig, VoxLoRAAdapter
        import torch.nn as nn

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)

        model = DummyModel()
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["q_proj", "v_proj"])
        adapter = VoxLoRAAdapter(model, config, voice_id="test_voice")
        adapter.inject_adapters()

        optimizer = torch.optim.AdamW(adapter.get_trainable_parameters(), lr=1e-4)

        print("\n3. Saving checkpoints...")
        for epoch in range(5):
            metrics = {"loss": 1.0 - epoch * 0.1, "lr": 1e-4}
            path = ckpt_mgr.save(epoch, adapter, optimizer, metrics)
            print(f"   Saved epoch {epoch}: {path}")

        print("\n4. Listing checkpoints...")
        checkpoints = ckpt_mgr._list_checkpoints()
        for ckpt in checkpoints:
            print(f"   Epoch {ckpt['epoch']}: loss={ckpt['metrics'].get('loss', 'N/A')}")

        print(f"\n   (max_checkpoints=2, so only {len(checkpoints)} remain)")

        print("\n5. Loading latest checkpoint...")
        latest = ckpt_mgr.load_latest()
        if latest:
            print(f"   Loaded epoch {latest['epoch']}")
            print(f"   Metrics: {latest['metrics']}")

        print("\n6. Getting best checkpoint...")
        best = ckpt_mgr.get_best_checkpoint(metric="loss", lower_is_better=True)
        if best:
            print(f"   Best epoch {best['epoch']} with loss={best['metrics']['loss']}")

        print("\n7. Testing trace writer...")
        with TraceWriter(tmpdir, "test_run") as writer:
            writer.write({"event": "start", "voice_id": "test"})
            writer.write({"event": "epoch", "epoch": 0, "loss": 1.0})
            writer.write({"event": "epoch", "epoch": 1, "loss": 0.5})
            writer.write({"event": "end"})

        trace_path = Path(tmpdir) / "trace_test_run.jsonl"
        with open(trace_path) as f:
            lines = f.readlines()
            print(f"   Wrote {len(lines)} events to trace")

        print("\n8. Writing run manifest...")
        manifest_path = write_run_manifest(
            tmpdir,
            run_id="test_run",
            voice_id="test_voice",
            config={"epochs": 50, "lr": 1e-4},
            sample_info={"count": 10, "duration": 180.0},
        )
        print(f"   Manifest: {manifest_path}")

    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)
