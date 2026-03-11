"""
VØX Release - Introspect
------------------------

System introspection and capability reporting.

AXIØM Phase 12: Release/Reflect - "How do we ship this and learn from it?"
"""

import importlib
import inspect
import logging
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Type

from .models import CapabilityReport

logger = logging.getLogger(__name__)


@dataclass
class IntrospectionConfig:
    """
    Configuration for system introspection.

    Attributes:
        package_name: Package to introspect
        include_private: Include private modules
        check_optional_deps: Check optional dependencies
        deep_analysis: Perform deep analysis of components
    """
    package_name: str = "axiom_vox"
    include_private: bool = False
    check_optional_deps: bool = True
    deep_analysis: bool = False


@dataclass
class ModuleInfo:
    """Information about a module."""
    name: str
    path: str = ""
    doc: str = ""
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    submodules: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "doc": self.doc[:200] + "..." if len(self.doc) > 200 else self.doc,
            "classes": self.classes,
            "functions": self.functions[:20],
            "exports": len(self.exports),
            "submodules": self.submodules,
        }


@dataclass
class DependencyInfo:
    """Information about a dependency."""
    name: str
    required_version: str = ""
    installed_version: str = ""
    is_installed: bool = False
    is_optional: bool = False
    used_by: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "required_version": self.required_version,
            "installed_version": self.installed_version,
            "is_installed": self.is_installed,
            "is_optional": self.is_optional,
        }


class SystemIntrospector:
    """
    Introspector for VØX system.

    Features:
        - Module structure analysis
        - Dependency graph
        - Capability inventory
        - API endpoint discovery
    """

    def __init__(
        self,
        config: Optional[IntrospectionConfig] = None,
    ):
        """
        Initialize introspector.

        Args:
            config: Introspection configuration
        """
        self.config = config or IntrospectionConfig()
        self._modules: Dict[str, ModuleInfo] = {}
        self._dependencies: Dict[str, DependencyInfo] = {}

    def generate_capability_report(self) -> CapabilityReport:
        """
        Generate comprehensive capability report.

        Returns:
            Capability report
        """
        # Import main package
        try:
            package = importlib.import_module(self.config.package_name)
            version = getattr(package, "__version__", "unknown")
        except ImportError as e:
            logger.error(f"Failed to import {self.config.package_name}: {e}")
            return CapabilityReport(
                name=self.config.package_name,
                version="error",
            )

        report = CapabilityReport(
            name=self.config.package_name,
            version=version,
            python_version=platform.python_version(),
            platform=platform.platform(),
        )

        # Check components
        report.components = self._check_components()

        # Check features
        report.features = self._check_features()

        # Get dependencies
        report.dependencies = self._get_installed_dependencies()

        # Check optional dependencies
        report.optional_deps = self._check_optional_dependencies()

        # Discover API endpoints
        report.api_endpoints = self._discover_endpoints()

        return report

    def analyze_module(
        self,
        module_name: str,
    ) -> ModuleInfo:
        """
        Analyze a module.

        Args:
            module_name: Full module name

        Returns:
            Module information
        """
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            logger.warning(f"Failed to import {module_name}: {e}")
            return ModuleInfo(name=module_name)

        info = ModuleInfo(
            name=module_name,
            path=getattr(module, "__file__", ""),
            doc=inspect.getdoc(module) or "",
            exports=list(getattr(module, "__all__", [])),
        )

        # Get classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if not name.startswith("_") or self.config.include_private:
                if obj.__module__ == module_name:
                    info.classes.append(name)

        # Get functions
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("_") or self.config.include_private:
                if obj.__module__ == module_name:
                    info.functions.append(name)

        # Get submodules
        if hasattr(module, "__path__"):
            for item in Path(module.__path__[0]).iterdir():
                if item.is_dir() and (item / "__init__.py").exists():
                    if not item.name.startswith("_") or self.config.include_private:
                        info.submodules.append(item.name)
                elif item.suffix == ".py" and item.stem != "__init__":
                    if not item.stem.startswith("_") or self.config.include_private:
                        info.submodules.append(item.stem)

        self._modules[module_name] = info
        return info

    def analyze_package(self) -> Dict[str, ModuleInfo]:
        """
        Analyze entire package structure.

        Returns:
            Dict mapping module name to info
        """
        modules = {}

        # Analyze main package
        main_info = self.analyze_module(self.config.package_name)
        modules[self.config.package_name] = main_info

        # Analyze submodules recursively
        def analyze_submodules(parent_name: str, parent_info: ModuleInfo):
            for submod in parent_info.submodules:
                full_name = f"{parent_name}.{submod}"
                try:
                    info = self.analyze_module(full_name)
                    modules[full_name] = info
                    if self.config.deep_analysis:
                        analyze_submodules(full_name, info)
                except Exception as e:
                    logger.debug(f"Failed to analyze {full_name}: {e}")

        analyze_submodules(self.config.package_name, main_info)

        return modules

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Get module dependency graph.

        Returns:
            Dict mapping module to its imports
        """
        graph = {}

        for module_name in self._modules:
            try:
                module = importlib.import_module(module_name)
                path = getattr(module, "__file__", "")

                if path and os.path.exists(path):
                    imports = self._extract_imports(path)
                    # Filter to only internal imports
                    internal = [
                        i for i in imports
                        if i.startswith(self.config.package_name)
                    ]
                    graph[module_name] = internal

            except Exception as e:
                logger.debug(f"Failed to get imports for {module_name}: {e}")

        return graph

    def _check_components(self) -> Dict[str, bool]:
        """Check available components."""
        components = {
            # Core
            "VoiceSpaceDirector": False,
            "GovernedTTS": False,
            "VoiceCloneManager": False,
            "StreamingPipeline": False,
            # Analytics
            "AnalyticsPipeline": False,
            "StreamingAnalyzer": False,
            # Multi-voice
            "MultiVoiceSynthesizer": False,
            "DialogueScript": False,
            # Biometrics
            "VoiceBiometricService": False,
            "SpectralFingerprint": False,
            # Unified
            "VoxUnifiedPipeline": False,
            # Governance
            "RateLimiter": False,
            "QuotaManager": False,
            "PolicyEngine": False,
            # SDK
            "VoxClient": False,
            "VoxSession": False,
            # Performance
            "AudioCache": False,
            "BatchOptimizer": False,
            # Verification
            "E2ETestRunner": False,
            "BenchmarkSuite": False,
            "HealthChecker": False,
            # Documentation
            "DocGenerator": False,
            "ExampleRunner": False,
            "TutorialRunner": False,
            # Release
            "ReleaseManager": False,
            "PackageBuilder": False,
        }

        try:
            package = importlib.import_module(self.config.package_name)

            for comp in components:
                if hasattr(package, comp):
                    components[comp] = True

        except ImportError:
            pass

        return components

    def _check_features(self) -> Dict[str, bool]:
        """Check enabled features."""
        features = {
            "voice_matching": False,
            "voice_cloning": False,
            "streaming_synthesis": False,
            "real_time_analytics": False,
            "multi_voice_dialogue": False,
            "biometric_verification": False,
            "liveness_detection": False,
            "rate_limiting": False,
            "quota_management": False,
            "policy_engine": False,
            "caching": False,
            "connection_pooling": False,
            "batch_optimization": False,
            "e2e_testing": False,
            "benchmarking": False,
            "health_checks": False,
            "api_documentation": False,
            "runnable_examples": False,
            "tutorials": False,
        }

        try:
            package = importlib.import_module(self.config.package_name)

            # Check based on available components
            if hasattr(package, "VoiceSpaceDirector"):
                features["voice_matching"] = True

            if hasattr(package, "VoiceCloneManager"):
                features["voice_cloning"] = True

            if hasattr(package, "StreamingPipeline"):
                features["streaming_synthesis"] = True

            if hasattr(package, "StreamingAnalyzer"):
                features["real_time_analytics"] = True

            if hasattr(package, "MultiVoiceSynthesizer"):
                features["multi_voice_dialogue"] = True

            if hasattr(package, "VoiceBiometricService"):
                features["biometric_verification"] = True
                features["liveness_detection"] = True

            if hasattr(package, "RateLimiter"):
                features["rate_limiting"] = True

            if hasattr(package, "QuotaManager"):
                features["quota_management"] = True

            if hasattr(package, "PolicyEngine"):
                features["policy_engine"] = True

            if hasattr(package, "AudioCache"):
                features["caching"] = True

            if hasattr(package, "HTTPConnectionPool"):
                features["connection_pooling"] = True

            if hasattr(package, "BatchOptimizer"):
                features["batch_optimization"] = True

            if hasattr(package, "E2ETestRunner"):
                features["e2e_testing"] = True

            if hasattr(package, "BenchmarkSuite"):
                features["benchmarking"] = True

            if hasattr(package, "HealthChecker"):
                features["health_checks"] = True

            if hasattr(package, "DocGenerator"):
                features["api_documentation"] = True

            if hasattr(package, "ExampleRunner"):
                features["runnable_examples"] = True

            if hasattr(package, "TutorialRunner"):
                features["tutorials"] = True

        except ImportError:
            pass

        return features

    def _get_installed_dependencies(self) -> Dict[str, str]:
        """Get installed dependency versions."""
        deps = {}

        core_deps = [
            "numpy",
            "scipy",
            "aiohttp",
            "pydantic",
            "fastapi",
            "uvicorn",
        ]

        for dep in core_deps:
            try:
                module = importlib.import_module(dep)
                version = getattr(module, "__version__", "installed")
                deps[dep] = version
            except ImportError:
                deps[dep] = "not installed"

        return deps

    def _check_optional_dependencies(self) -> Dict[str, bool]:
        """Check optional dependencies."""
        optional = {
            "torch": False,
            "torchaudio": False,
            "transformers": False,
            "speechbrain": False,
            "librosa": False,
            "soundfile": False,
            "psutil": False,
            "prometheus_client": False,
        }

        for dep in optional:
            try:
                importlib.import_module(dep)
                optional[dep] = True
            except ImportError:
                pass

        return optional

    def _discover_endpoints(self) -> List[str]:
        """Discover API endpoints."""
        endpoints = []

        try:
            # Try to import the API module
            api_module = importlib.import_module(
                f"{self.config.package_name}.governed_tts_api"
            )

            # Look for FastAPI app
            if hasattr(api_module, "app"):
                app = api_module.app
                for route in getattr(app, "routes", []):
                    path = getattr(route, "path", "")
                    methods = getattr(route, "methods", set())
                    if path and methods:
                        for method in methods:
                            endpoints.append(f"{method} {path}")

        except Exception as e:
            logger.debug(f"Failed to discover endpoints: {e}")

        # Add known endpoints as fallback
        if not endpoints:
            endpoints = [
                "POST /synthesize",
                "POST /stream",
                "GET /voices",
                "GET /health",
                "GET /ready",
                "POST /biometrics/enroll",
                "POST /biometrics/verify",
            ]

        return sorted(endpoints)

    def _extract_imports(self, file_path: str) -> List[str]:
        """Extract imports from a Python file."""
        imports = []

        try:
            with open(file_path) as f:
                content = f.read()

            # Simple regex-based import extraction
            import re

            # Match "import x" and "from x import y"
            patterns = [
                r"^import\s+(\S+)",
                r"^from\s+(\S+)\s+import",
            ]

            for line in content.split("\n"):
                line = line.strip()
                for pattern in patterns:
                    match = re.match(pattern, line)
                    if match:
                        imports.append(match.group(1).split(".")[0])

        except Exception:
            pass

        return list(set(imports))


def generate_capability_report() -> CapabilityReport:
    """
    Generate VØX capability report.

    Returns:
        Capability report
    """
    introspector = SystemIntrospector()
    return introspector.generate_capability_report()


def analyze_vox_structure() -> Dict[str, ModuleInfo]:
    """
    Analyze VØX module structure.

    Returns:
        Module structure
    """
    config = IntrospectionConfig(deep_analysis=True)
    introspector = SystemIntrospector(config)
    return introspector.analyze_package()


def print_capability_report() -> None:
    """Print capability report to stdout."""
    report = generate_capability_report()
    print(report.to_markdown())
