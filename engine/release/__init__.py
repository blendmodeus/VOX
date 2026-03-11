"""
VØX Release Layer
-----------------

Release management, packaging, deployment, and reflection.

Features:
    - Semantic version management
    - Package building and validation
    - Deployment configuration generation
    - System introspection
    - Reflection and improvement analysis

Quick Start:
    >>> from axiom_vox.release import (
    ...     ReleaseManager, PackageBuilder,
    ...     DeploymentHelper, SystemIntrospector,
    ...     ReflectionEngine,
    ... )
    >>>
    >>> # Create a release
    >>> manager = ReleaseManager()
    >>> release = manager.create_release(ReleaseType.MINOR)
    >>>
    >>> # Build package
    >>> builder = PackageBuilder()
    >>> artifacts = builder.build_all()
    >>>
    >>> # Generate deployment configs
    >>> deployer = DeploymentHelper()
    >>> deployer.generate_all_configs("deploy/")
    >>>
    >>> # Introspect system
    >>> introspector = SystemIntrospector()
    >>> report = introspector.generate_capability_report()
    >>>
    >>> # Reflect on performance
    >>> engine = ReflectionEngine()
    >>> reflection = engine.generate_report("0.17.0")

AXIØM Phase 12: Release/Reflect - "How do we ship this and learn from it?"
"""

from .models import (
    # Enums
    ReleaseType,
    ReleaseStatus,
    DeploymentTarget,
    PackageFormat,
    ReflectionCategory,
    # Version
    SemanticVersion,
    # Release models
    ReleaseInfo,
    # Package models
    PackageMetadata,
    # Deployment models
    DeploymentConfig,
    # Introspection models
    CapabilityReport,
    # Reflection models
    ReflectionInsight,
    ReflectionReport,
)

from .manager import (
    ReleaseConfig,
    ReleaseManager,
    create_release,
    get_current_version,
)

from .packaging import (
    PackageConfig,
    ValidationResult,
    PackageArtifact,
    PackageBuilder,
    build_package,
    validate_package,
)

from .deploy import (
    DeploymentResult,
    DeploymentHelper,
    generate_deployment_configs,
    deploy_docker,
)

from .introspect import (
    IntrospectionConfig,
    ModuleInfo,
    DependencyInfo,
    SystemIntrospector,
    generate_capability_report,
    analyze_vox_structure,
    print_capability_report,
)

from .reflect import (
    ReflectionConfig,
    UsageMetrics,
    PerformanceMetrics,
    ErrorMetrics,
    ReflectionEngine,
    generate_reflection_report,
    print_reflection_report,
)


__all__ = [
    # Enums
    "ReleaseType",
    "ReleaseStatus",
    "DeploymentTarget",
    "PackageFormat",
    "ReflectionCategory",
    # Version
    "SemanticVersion",
    # Release models
    "ReleaseInfo",
    # Package models
    "PackageMetadata",
    # Deployment models
    "DeploymentConfig",
    # Introspection models
    "CapabilityReport",
    # Reflection models
    "ReflectionInsight",
    "ReflectionReport",
    # Manager
    "ReleaseConfig",
    "ReleaseManager",
    "create_release",
    "get_current_version",
    # Packaging
    "PackageConfig",
    "ValidationResult",
    "PackageArtifact",
    "PackageBuilder",
    "build_package",
    "validate_package",
    # Deploy
    "DeploymentResult",
    "DeploymentHelper",
    "generate_deployment_configs",
    "deploy_docker",
    # Introspect
    "IntrospectionConfig",
    "ModuleInfo",
    "DependencyInfo",
    "SystemIntrospector",
    "generate_capability_report",
    "analyze_vox_structure",
    "print_capability_report",
    # Reflect
    "ReflectionConfig",
    "UsageMetrics",
    "PerformanceMetrics",
    "ErrorMetrics",
    "ReflectionEngine",
    "generate_reflection_report",
    "print_reflection_report",
]
