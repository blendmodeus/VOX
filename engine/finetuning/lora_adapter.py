"""
LoRA Adapter for VØX Voice Cloning
----------------------------------

Low-Rank Adaptation (LoRA) layers for efficient voice cloning.
Keeps base Qwen3-TTS model frozen, trains lightweight adapters.

Reference: https://arxiv.org/abs/2106.09685

Architecture:
    W' = W + BA where B in R^(d x r), A in R^(r x k)
    r << min(d, k) keeps parameters minimal (~2-4MB per voice)

Usage:
    from axiom_vox.finetuning import LoRAConfig, VoxLoRAAdapter

    config = LoRAConfig(rank=8, alpha=16.0)
    adapter = VoxLoRAAdapter(base_model, config, voice_id="my_voice")
    adapter.inject_adapters()

    # Train adapter...

    adapter.save_adapter("path/to/adapter.pt")
"""

from __future__ import annotations

import math
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class LoRAConfig:
    """
    Configuration for LoRA adapters.

    Attributes:
        rank: LoRA rank (4-16 typical). Higher = more capacity, more params.
        alpha: Scaling factor. Typically alpha = 2 * rank.
        dropout: Dropout rate for regularization.
        target_modules: List of module name patterns to adapt.
        bias: Bias handling: "none", "all", or "lora_only".
    """

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",      # Query projection
        "v_proj",      # Value projection
        "k_proj",      # Key projection
        "o_proj",      # Output projection
        "gate_proj",   # MLP gate
        "up_proj",     # MLP up projection
    ])
    bias: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoRAConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def fast(cls) -> "LoRAConfig":
        """Fast configuration with fewer parameters."""
        return cls(
            rank=4,
            alpha=8.0,
            dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        )

    @classmethod
    def high_quality(cls) -> "LoRAConfig":
        """High quality configuration with more capacity."""
        return cls(
            rank=16,
            alpha=32.0,
            dropout=0.1,
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )


# ============================================================================
# LORA LAYER
# ============================================================================

if HAS_TORCH:

    class LoRALayer(nn.Module):
        """
        Low-Rank Adaptation layer.

        Implements: output = base_output + (x @ A.T @ B.T) * scaling

        The base model weights stay frozen. Only A and B are trainable.
        A is initialized with Kaiming uniform, B with zeros (starts at identity).
        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: int = 8,
            alpha: float = 16.0,
            dropout: float = 0.05,
        ):
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank

            # Low-rank decomposition matrices
            # A: (rank, in_features) - down projection
            # B: (out_features, rank) - up projection
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

            # Dropout for regularization
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

            # Initialize
            self.reset_parameters()

        def reset_parameters(self):
            """Initialize A with Kaiming, B with zeros."""
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        def forward(
            self,
            x: torch.Tensor,
            base_output: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Compute LoRA output.

            If base_output is provided, adds LoRA delta to it.
            Otherwise returns just the LoRA contribution.
            """
            # Apply dropout to input
            x_dropped = self.dropout(x)

            # Compute low-rank output: x @ A.T @ B.T
            lora_out = x_dropped @ self.lora_A.T @ self.lora_B.T

            # Scale
            lora_out = lora_out * self.scaling

            if base_output is not None:
                return base_output + lora_out
            return lora_out

        def extra_repr(self) -> str:
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"rank={self.rank}, alpha={self.alpha}"
            )

else:
    # Placeholder when torch not available
    class LoRALayer:
        """Placeholder LoRA layer (torch not available)."""

        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LoRA layers")


# ============================================================================
# ADAPTER MANAGER
# ============================================================================

class VoxLoRAAdapter:
    """
    Manages LoRA adapters for VØX voice cloning.

    Key design:
    - Base model stays frozen
    - Each cloned voice gets a separate adapter (~2-4MB)
    - Adapters can be hot-swapped at inference time

    Usage:
        adapter = VoxLoRAAdapter(base_model, config, voice_id="my_voice")
        adapter.inject_adapters()

        # Train...

        adapter.save_adapter("adapter.pt")
        adapter.load_adapter("adapter.pt")
    """

    def __init__(
        self,
        base_model: Any,
        config: LoRAConfig,
        voice_id: str,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for VoxLoRAAdapter")

        self.base_model = base_model
        self.config = config
        self.voice_id = voice_id

        # LoRA layers keyed by original module name
        self.lora_layers: Dict[str, LoRALayer] = {}

        # Track which modules have been adapted
        self._adapted_modules: List[str] = []

        # Metadata
        self._injected = False
        self._parameter_count = 0

    def inject_adapters(self) -> int:
        """
        Inject LoRA layers into target modules.

        Returns the total number of trainable parameters added.
        """
        if self._injected:
            logger.warning("Adapters already injected")
            return self._parameter_count

        total_params = 0

        for name, module in self.base_model.named_modules():
            # Check if this module should be adapted
            if not self._should_adapt(name, module):
                continue

            if isinstance(module, nn.Linear):
                # Create LoRA layer for this linear module
                lora = LoRALayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=self.config.rank,
                    alpha=self.config.alpha,
                    dropout=self.config.dropout,
                )

                # Move to same device as original module
                device = next(module.parameters()).device
                lora = lora.to(device)

                self.lora_layers[name] = lora
                self._adapted_modules.append(name)

                # Count parameters
                params = sum(p.numel() for p in lora.parameters())
                total_params += params

                logger.debug(
                    f"Injected LoRA into {name}: "
                    f"{module.in_features}x{module.out_features} -> rank {self.config.rank}"
                )

        self._injected = True
        self._parameter_count = total_params

        logger.info(
            f"Injected {len(self.lora_layers)} LoRA layers "
            f"({total_params:,} trainable parameters)"
        )

        return total_params

    def _should_adapt(self, name: str, module: Any) -> bool:
        """Check if a module should be adapted."""
        return any(target in name for target in self.config.target_modules)

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get all trainable LoRA parameters."""
        params = []
        for lora in self.lora_layers.values():
            params.extend(lora.parameters())
        return params

    def get_lora_output(
        self,
        module_name: str,
        x: torch.Tensor,
        base_output: torch.Tensor,
    ) -> torch.Tensor:
        """Get LoRA-adapted output for a specific module."""
        if module_name in self.lora_layers:
            return self.lora_layers[module_name](x, base_output)
        return base_output

    def save_adapter(self, path: str) -> None:
        """
        Save only the LoRA weights (small file).

        File structure:
        {
            "voice_id": str,
            "config": dict,
            "lora_state": {module_name: state_dict, ...},
            "metadata": {...}
        }
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required to save adapter")

        state = {
            "voice_id": self.voice_id,
            "config": self.config.to_dict(),
            "lora_state": {
                name: layer.state_dict()
                for name, layer in self.lora_layers.items()
            },
            "metadata": {
                "adapted_modules": self._adapted_modules,
                "parameter_count": self._parameter_count,
            },
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

        logger.info(f"Saved adapter to {path} ({self._parameter_count:,} params)")

    def load_adapter(self, path: str) -> None:
        """Load LoRA weights from file."""
        if not HAS_TORCH:
            raise ImportError("PyTorch required to load adapter")

        state = torch.load(path, weights_only=True)

        # Validate voice ID matches
        if state["voice_id"] != self.voice_id:
            logger.warning(
                f"Voice ID mismatch: expected {self.voice_id}, "
                f"got {state['voice_id']}"
            )

        # Validate config compatibility
        saved_config = LoRAConfig.from_dict(state["config"])
        if saved_config.rank != self.config.rank:
            raise ValueError(
                f"LoRA rank mismatch: adapter has rank {saved_config.rank}, "
                f"expected {self.config.rank}"
            )

        # Load weights
        for name, layer_state in state["lora_state"].items():
            if name in self.lora_layers:
                self.lora_layers[name].load_state_dict(layer_state)
            else:
                logger.warning(f"Skipping unknown module: {name}")

        self._parameter_count = state["metadata"]["parameter_count"]

        logger.info(f"Loaded adapter from {path}")

    @classmethod
    def from_file(
        cls,
        path: str,
        base_model: Any,
    ) -> "VoxLoRAAdapter":
        """
        Create adapter from saved file.

        Loads config and weights, then injects into base model.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        state = torch.load(path, weights_only=True)
        config = LoRAConfig.from_dict(state["config"])

        adapter = cls(base_model, config, state["voice_id"])
        adapter.inject_adapters()
        adapter.load_adapter(path)

        return adapter

    def merge_into_base(self) -> None:
        """
        Merge LoRA weights into base model (permanent).

        Warning: This modifies the base model weights permanently.
        Use only for deployment when you want a single model without adapter overhead.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        for name, lora in self.lora_layers.items():
            # Find the original module
            parts = name.split(".")
            module = self.base_model
            for part in parts:
                module = getattr(module, part)

            if isinstance(module, nn.Linear):
                # Compute merged weight: W' = W + B @ A * scaling
                delta = lora.lora_B @ lora.lora_A * lora.scaling
                module.weight.data += delta

        logger.warning(
            f"Merged LoRA weights into base model. "
            f"This is permanent and cannot be undone."
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert adapter info to dictionary."""
        return {
            "voice_id": self.voice_id,
            "config": self.config.to_dict(),
            "injected": self._injected,
            "parameter_count": self._parameter_count,
            "adapted_modules": self._adapted_modules,
        }


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  VØX LoRA Adapter Demo")
    print("=" * 70)

    if not HAS_TORCH:
        print("\nPyTorch not available. Install with: pip install torch")
        exit(1)

    # Create a dummy model for testing
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(256, 256)
            self.v_proj = nn.Linear(256, 256)
            self.k_proj = nn.Linear(256, 256)
            self.o_proj = nn.Linear(256, 256)
            self.gate_proj = nn.Linear(256, 512)
            self.up_proj = nn.Linear(256, 512)

    print("\n1. Creating dummy model...")
    model = DummyModel()
    base_params = sum(p.numel() for p in model.parameters())
    print(f"   Base model parameters: {base_params:,}")

    print("\n2. Creating LoRA config...")
    config = LoRAConfig(rank=8, alpha=16.0)
    print(f"   Rank: {config.rank}")
    print(f"   Alpha: {config.alpha}")
    print(f"   Target modules: {config.target_modules}")

    print("\n3. Creating adapter...")
    adapter = VoxLoRAAdapter(model, config, voice_id="test_voice")

    print("\n4. Injecting adapters...")
    lora_params = adapter.inject_adapters()
    print(f"   LoRA parameters added: {lora_params:,}")
    print(f"   Overhead: {lora_params / base_params * 100:.1f}%")

    print("\n5. Testing forward pass...")
    x = torch.randn(1, 10, 256)

    # Get base output
    base_q = model.q_proj(x)

    # Get LoRA-adapted output
    adapted_q = adapter.get_lora_output("q_proj", x, base_q)

    print(f"   Input shape: {x.shape}")
    print(f"   Base output shape: {base_q.shape}")
    print(f"   Adapted output shape: {adapted_q.shape}")
    print(f"   Difference norm: {(adapted_q - base_q).norm().item():.6f}")

    print("\n6. Saving adapter...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        adapter.save_adapter(f.name)
        print(f"   Saved to: {f.name}")

        # Check file size
        import os
        size_kb = os.path.getsize(f.name) / 1024
        print(f"   File size: {size_kb:.1f} KB")

    print("\n7. Loading adapter...")
    adapter2 = VoxLoRAAdapter(model, config, voice_id="test_voice")
    adapter2.inject_adapters()
    adapter2.load_adapter(f.name)
    print("   Loaded successfully!")

    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)
