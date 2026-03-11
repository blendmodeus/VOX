"""
VØX Release - Packaging
-----------------------

Package building and validation.

AXIØM Phase 12: Release/Reflect - "How do we ship this and learn from it?"
"""

import hashlib
import importlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Set

from .models import (
    PackageMetadata,
    PackageFormat,
)

logger = logging.getLogger(__name__)


@dataclass
class PackageConfig:
    """
    Configuration for package builder.

    Attributes:
        project_root: Root directory of project
        dist_dir: Output directory for packages
        clean_dist: Clean dist directory before build
        include_tests: Include test files
        validate_imports: Validate package imports
    """
    project_root: str = "."
    dist_dir: str = "dist"
    clean_dist: bool = True
    include_tests: bool = False
    validate_imports: bool = True


@dataclass
class ValidationResult:
    """Result of package validation."""
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        """Add error message."""
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        """Add warning message."""
        self.warnings.append(msg)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
        }


@dataclass
class PackageArtifact:
    """Built package artifact."""
    name: str
    path: str
    format: PackageFormat
    size: int = 0
    checksum: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.path and os.path.exists(self.path):
            self.size = os.path.getsize(self.path)
            if not self.checksum:
                self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute SHA256 checksum."""
        sha256 = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "format": self.format.value,
            "size": self.size,
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat(),
        }


class PackageBuilder:
    """
    Builder for Python packages.

    Features:
        - Build wheel and sdist
        - Validate package structure
        - Check imports
        - Generate checksums
    """

    def __init__(
        self,
        config: Optional[PackageConfig] = None,
    ):
        """
        Initialize package builder.

        Args:
            config: Build configuration
        """
        self.config = config or PackageConfig()
        self._artifacts: List[PackageArtifact] = []

    def build_wheel(self) -> Optional[PackageArtifact]:
        """
        Build wheel distribution.

        Returns:
            Built artifact or None on failure
        """
        dist_dir = Path(self.config.project_root) / self.config.dist_dir

        if self.config.clean_dist and dist_dir.exists():
            for f in dist_dir.glob("*.whl"):
                f.unlink()

        dist_dir.mkdir(exist_ok=True)

        try:
            result = subprocess.run(
                [sys.executable, "-m", "build", "--wheel", "-o", str(dist_dir)],
                cwd=self.config.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.error(f"Wheel build failed: {result.stderr}")
                return None

            # Find built wheel
            wheels = list(dist_dir.glob("*.whl"))
            if not wheels:
                logger.error("No wheel file found after build")
                return None

            wheel = wheels[0]
            artifact = PackageArtifact(
                name=wheel.name,
                path=str(wheel),
                format=PackageFormat.WHEEL,
            )

            self._artifacts.append(artifact)
            logger.info(f"Built wheel: {wheel.name}")
            return artifact

        except Exception as e:
            logger.error(f"Wheel build failed: {e}")
            return None

    def build_sdist(self) -> Optional[PackageArtifact]:
        """
        Build source distribution.

        Returns:
            Built artifact or None on failure
        """
        dist_dir = Path(self.config.project_root) / self.config.dist_dir

        if self.config.clean_dist and dist_dir.exists():
            for f in dist_dir.glob("*.tar.gz"):
                f.unlink()

        dist_dir.mkdir(exist_ok=True)

        try:
            result = subprocess.run(
                [sys.executable, "-m", "build", "--sdist", "-o", str(dist_dir)],
                cwd=self.config.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.error(f"Sdist build failed: {result.stderr}")
                return None

            # Find built sdist
            sdists = list(dist_dir.glob("*.tar.gz"))
            if not sdists:
                logger.error("No sdist file found after build")
                return None

            sdist = sdists[0]
            artifact = PackageArtifact(
                name=sdist.name,
                path=str(sdist),
                format=PackageFormat.SDIST,
            )

            self._artifacts.append(artifact)
            logger.info(f"Built sdist: {sdist.name}")
            return artifact

        except Exception as e:
            logger.error(f"Sdist build failed: {e}")
            return None

    def build_all(self) -> List[PackageArtifact]:
        """
        Build all distribution formats.

        Returns:
            List of built artifacts
        """
        artifacts = []

        wheel = self.build_wheel()
        if wheel:
            artifacts.append(wheel)

        sdist = self.build_sdist()
        if sdist:
            artifacts.append(sdist)

        return artifacts

    def validate_structure(self) -> ValidationResult:
        """
        Validate package structure.

        Returns:
            Validation result
        """
        result = ValidationResult()
        root = Path(self.config.project_root)

        # Check required files
        required_files = [
            "pyproject.toml",
            "README.md",
        ]

        for file in required_files:
            if not (root / file).exists():
                result.add_error(f"Missing required file: {file}")

        # Check package directory
        package_dirs = ["axiom_vox"]
        for pkg in package_dirs:
            pkg_path = root / pkg
            if not pkg_path.exists():
                result.add_error(f"Missing package directory: {pkg}")
            elif not (pkg_path / "__init__.py").exists():
                result.add_error(f"Missing __init__.py in {pkg}")

        # Check for common issues
        if (root / "setup.py").exists():
            result.info["has_setup_py"] = True
        if (root / "setup.cfg").exists():
            result.info["has_setup_cfg"] = True
        if (root / "pyproject.toml").exists():
            result.info["has_pyproject"] = True

        # Check LICENSE
        license_files = ["LICENSE", "LICENSE.txt", "LICENSE.md"]
        has_license = any((root / f).exists() for f in license_files)
        if not has_license:
            result.add_warning("No LICENSE file found")

        return result

    def validate_imports(
        self,
        package: str = "axiom_vox",
    ) -> ValidationResult:
        """
        Validate package imports.

        Args:
            package: Package name to validate

        Returns:
            Validation result
        """
        result = ValidationResult()

        try:
            # Try to import the package
            module = importlib.import_module(package)
            result.info["version"] = getattr(module, "__version__", "unknown")
            result.info["path"] = getattr(module, "__file__", "unknown")

            # Check __all__ exports
            all_exports = getattr(module, "__all__", [])
            result.info["export_count"] = len(all_exports)

            # Validate exports are accessible
            missing = []
            for name in all_exports[:50]:  # Check first 50
                if not hasattr(module, name):
                    missing.append(name)

            if missing:
                result.add_error(f"Missing exports: {', '.join(missing)}")

            logger.info(f"Package {package} imports successfully")

        except ImportError as e:
            result.add_error(f"Import failed: {e}")
        except Exception as e:
            result.add_error(f"Validation failed: {e}")

        return result

    def validate_wheel(
        self,
        wheel_path: str,
    ) -> ValidationResult:
        """
        Validate a wheel file.

        Args:
            wheel_path: Path to wheel file

        Returns:
            Validation result
        """
        result = ValidationResult()

        if not os.path.exists(wheel_path):
            result.add_error(f"Wheel not found: {wheel_path}")
            return result

        # Use wheel tool if available
        try:
            import wheel
            from wheel.wheelfile import WheelFile

            with WheelFile(wheel_path) as wf:
                result.info["wheel_version"] = wf.wheel_version
                result.info["name"] = wf.parsed_filename.project
                result.info["version"] = wf.parsed_filename.version

        except ImportError:
            result.add_warning("wheel package not installed, limited validation")
        except Exception as e:
            result.add_error(f"Wheel validation failed: {e}")

        # Check file size
        size = os.path.getsize(wheel_path)
        result.info["size_bytes"] = size
        result.info["size_mb"] = round(size / 1024 / 1024, 2)

        if size < 1000:
            result.add_warning("Wheel is suspiciously small")
        if size > 100 * 1024 * 1024:
            result.add_warning("Wheel is very large (>100MB)")

        return result

    def test_install(
        self,
        wheel_path: Optional[str] = None,
    ) -> ValidationResult:
        """
        Test package installation in isolated environment.

        Args:
            wheel_path: Path to wheel to test

        Returns:
            Validation result
        """
        result = ValidationResult()

        # Find wheel if not provided
        if not wheel_path:
            dist_dir = Path(self.config.project_root) / self.config.dist_dir
            wheels = list(dist_dir.glob("*.whl"))
            if not wheels:
                result.add_error("No wheel found to test")
                return result
            wheel_path = str(wheels[0])

        # Create temp venv
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_dir = Path(tmpdir) / "venv"

            try:
                # Create venv
                subprocess.run(
                    [sys.executable, "-m", "venv", str(venv_dir)],
                    check=True,
                    capture_output=True,
                )

                # Get pip path
                if sys.platform == "win32":
                    pip = venv_dir / "Scripts" / "pip"
                    python = venv_dir / "Scripts" / "python"
                else:
                    pip = venv_dir / "bin" / "pip"
                    python = venv_dir / "bin" / "python"

                # Install wheel
                install_result = subprocess.run(
                    [str(pip), "install", wheel_path],
                    capture_output=True,
                    text=True,
                )

                if install_result.returncode != 0:
                    result.add_error(f"Install failed: {install_result.stderr}")
                    return result

                # Test import
                import_result = subprocess.run(
                    [str(python), "-c", "import axiom_vox; print(axiom_vox.__version__)"],
                    capture_output=True,
                    text=True,
                )

                if import_result.returncode != 0:
                    result.add_error(f"Import test failed: {import_result.stderr}")
                else:
                    result.info["installed_version"] = import_result.stdout.strip()
                    logger.info(f"Test install successful: {result.info['installed_version']}")

            except Exception as e:
                result.add_error(f"Test install failed: {e}")

        return result

    def generate_checksums(self) -> Dict[str, str]:
        """
        Generate checksums for all artifacts.

        Returns:
            Dict mapping filename to SHA256 checksum
        """
        checksums = {}

        dist_dir = Path(self.config.project_root) / self.config.dist_dir
        if not dist_dir.exists():
            return checksums

        for f in dist_dir.iterdir():
            if f.is_file() and not f.name.endswith(".sha256"):
                sha256 = hashlib.sha256()
                with open(f, "rb") as fh:
                    for chunk in iter(lambda: fh.read(8192), b""):
                        sha256.update(chunk)
                checksums[f.name] = sha256.hexdigest()

                # Write checksum file
                checksum_file = f.with_suffix(f.suffix + ".sha256")
                checksum_file.write_text(f"{sha256.hexdigest()}  {f.name}\n")

        return checksums

    def get_artifacts(self) -> List[PackageArtifact]:
        """
        Get all built artifacts.

        Returns:
            List of artifacts
        """
        return self._artifacts


def build_package(
    wheel: bool = True,
    sdist: bool = True,
    validate: bool = True,
) -> Dict[str, Any]:
    """
    Build VØX package.

    Args:
        wheel: Build wheel
        sdist: Build sdist
        validate: Validate after build

    Returns:
        Build results
    """
    builder = PackageBuilder()
    results = {
        "artifacts": [],
        "validation": None,
    }

    if wheel:
        artifact = builder.build_wheel()
        if artifact:
            results["artifacts"].append(artifact.to_dict())

    if sdist:
        artifact = builder.build_sdist()
        if artifact:
            results["artifacts"].append(artifact.to_dict())

    if validate:
        validation = builder.validate_structure()
        results["validation"] = validation.to_dict()

    return results


def validate_package() -> ValidationResult:
    """
    Validate VØX package.

    Returns:
        Validation result
    """
    builder = PackageBuilder()

    # Validate structure
    struct_result = builder.validate_structure()
    if not struct_result.valid:
        return struct_result

    # Validate imports
    import_result = builder.validate_imports()

    # Combine results
    combined = ValidationResult(
        valid=struct_result.valid and import_result.valid,
        errors=struct_result.errors + import_result.errors,
        warnings=struct_result.warnings + import_result.warnings,
        info={**struct_result.info, **import_result.info},
    )

    return combined
