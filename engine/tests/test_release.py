"""
VØX Release Layer Tests
-----------------------

Comprehensive tests for the VØX Release Layer.

AXIØM Phase 12: Release/Reflect - "How do we ship this and learn from it?"
"""

import pytest
from datetime import datetime, timedelta


# ============================================================================
# Model Tests
# ============================================================================


class TestSemanticVersion:
    """Tests for SemanticVersion model."""

    def test_parse_simple(self):
        """Test parsing simple version."""
        from axiom_vox.release import SemanticVersion

        version = SemanticVersion.parse("1.2.3")

        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.prerelease is None

    def test_parse_prerelease(self):
        """Test parsing version with prerelease."""
        from axiom_vox.release import SemanticVersion

        version = SemanticVersion.parse("1.0.0-alpha.1")

        assert version.major == 1
        assert version.minor == 0
        assert version.patch == 0
        assert version.prerelease == "alpha.1"

    def test_parse_build(self):
        """Test parsing version with build metadata."""
        from axiom_vox.release import SemanticVersion

        version = SemanticVersion.parse("1.0.0+build.123")

        assert version.major == 1
        assert version.build == "build.123"

    def test_bump_major(self):
        """Test major version bump."""
        from axiom_vox.release import SemanticVersion, ReleaseType

        version = SemanticVersion(1, 2, 3)
        bumped = version.bump(ReleaseType.MAJOR)

        assert bumped.major == 2
        assert bumped.minor == 0
        assert bumped.patch == 0

    def test_bump_minor(self):
        """Test minor version bump."""
        from axiom_vox.release import SemanticVersion, ReleaseType

        version = SemanticVersion(1, 2, 3)
        bumped = version.bump(ReleaseType.MINOR)

        assert bumped.major == 1
        assert bumped.minor == 3
        assert bumped.patch == 0

    def test_bump_patch(self):
        """Test patch version bump."""
        from axiom_vox.release import SemanticVersion, ReleaseType

        version = SemanticVersion(1, 2, 3)
        bumped = version.bump(ReleaseType.PATCH)

        assert bumped.major == 1
        assert bumped.minor == 2
        assert bumped.patch == 4

    def test_comparison(self):
        """Test version comparison."""
        from axiom_vox.release import SemanticVersion

        v1 = SemanticVersion.parse("1.0.0")
        v2 = SemanticVersion.parse("1.0.1")
        v3 = SemanticVersion.parse("1.0.0-alpha")

        assert v1 < v2
        assert v3 < v1  # Prerelease < release

    def test_str(self):
        """Test version string representation."""
        from axiom_vox.release import SemanticVersion

        version = SemanticVersion(1, 2, 3, "beta.1", "build.456")

        assert str(version) == "1.2.3-beta.1+build.456"


class TestReleaseInfo:
    """Tests for ReleaseInfo model."""

    def test_creation(self):
        """Test release info creation."""
        from axiom_vox.release import ReleaseInfo, ReleaseType

        release = ReleaseInfo(
            version="1.0.0",
            release_type=ReleaseType.MAJOR,
            summary="Major release",
        )

        assert release.version == "1.0.0"
        assert release.tag_name == "v1.0.0"
        assert release.title == "VØX 1.0.0"

    def test_to_markdown(self):
        """Test markdown generation."""
        from axiom_vox.release import ReleaseInfo, ReleaseType, ReleaseStatus

        release = ReleaseInfo(
            version="1.0.0",
            release_type=ReleaseType.MAJOR,
            summary="First stable release",
            release_notes="- Feature A\n- Feature B",
        )

        md = release.to_markdown()

        assert "VØX 1.0.0" in md
        assert "First stable release" in md
        assert "Feature A" in md


class TestPackageMetadata:
    """Tests for PackageMetadata model."""

    def test_defaults(self):
        """Test default metadata."""
        from axiom_vox.release import PackageMetadata

        meta = PackageMetadata()

        assert meta.name == "axiom-vox"
        assert "tts" in meta.keywords

    def test_to_pyproject(self):
        """Test pyproject.toml generation."""
        from axiom_vox.release import PackageMetadata

        meta = PackageMetadata(
            name="test-pkg",
            version="1.0.0",
        )

        pyproject = meta.to_pyproject()

        assert pyproject["project"]["name"] == "test-pkg"
        assert pyproject["project"]["version"] == "1.0.0"


class TestDeploymentConfig:
    """Tests for DeploymentConfig model."""

    def test_dockerfile_generation(self):
        """Test Dockerfile generation."""
        from axiom_vox.release import DeploymentConfig

        config = DeploymentConfig(
            name="test-app",
            port=8080,
        )

        dockerfile = config.generate_dockerfile()

        assert "EXPOSE 8080" in dockerfile
        assert "HEALTHCHECK" in dockerfile

    def test_docker_compose_generation(self):
        """Test docker-compose generation."""
        from axiom_vox.release import DeploymentConfig

        config = DeploymentConfig(
            name="test-app",
            replicas=3,
        )

        compose = config.generate_docker_compose()

        assert "test-app" in compose
        assert "replicas: 3" in compose

    def test_kubernetes_generation(self):
        """Test Kubernetes manifest generation."""
        from axiom_vox.release import DeploymentConfig

        config = DeploymentConfig(
            name="test-app",
            memory_limit="1Gi",
        )

        k8s = config.generate_kubernetes_manifest()

        assert "Deployment" in k8s
        assert "Service" in k8s
        assert "memory: 1Gi" in k8s


class TestCapabilityReport:
    """Tests for CapabilityReport model."""

    def test_to_markdown(self):
        """Test markdown generation."""
        from axiom_vox.release import CapabilityReport

        report = CapabilityReport(
            name="VØX",
            version="0.17.0",
            python_version="3.11.0",
            platform="Linux",
            components={"VoiceSpaceDirector": True, "Missing": False},
            features={"streaming": True},
        )

        md = report.to_markdown()

        assert "VØX" in md
        assert "0.17.0" in md
        assert "VoiceSpaceDirector" in md


class TestReflectionReport:
    """Tests for ReflectionReport model."""

    def test_to_markdown(self):
        """Test markdown generation."""
        from axiom_vox.release import (
            ReflectionReport,
            ReflectionInsight,
            ReflectionCategory,
        )

        report = ReflectionReport(
            version="0.17.0",
            period_start=datetime.now() - timedelta(days=30),
            period_end=datetime.now(),
            insights=[
                ReflectionInsight(
                    category=ReflectionCategory.PERFORMANCE,
                    title="Good Latency",
                    description="P95 latency is under 200ms",
                    impact="low",
                )
            ],
            recommendations=["Keep monitoring"],
        )

        md = report.to_markdown()

        assert "0.17.0" in md
        assert "Good Latency" in md
        assert "Keep monitoring" in md


# ============================================================================
# Manager Tests
# ============================================================================


class TestReleaseManager:
    """Tests for ReleaseManager."""

    def test_get_current_version(self):
        """Test getting current version."""
        from axiom_vox.release import ReleaseManager, ReleaseConfig

        config = ReleaseConfig(project_root=".")
        manager = ReleaseManager(config)

        version = manager.get_current_version()

        # Should return a version (may be default or actual)
        assert version is not None
        assert version.major >= 0

    def test_bump_version(self):
        """Test version bumping."""
        from axiom_vox.release import ReleaseManager, ReleaseType

        manager = ReleaseManager()
        manager._current_version = None  # Reset

        # Set a known version
        from axiom_vox.release import SemanticVersion
        manager._current_version = SemanticVersion(0, 16, 0)

        new_version = manager.bump_version(ReleaseType.MINOR)

        assert new_version.major == 0
        assert new_version.minor == 17
        assert new_version.patch == 0

    def test_create_release(self):
        """Test release creation."""
        from axiom_vox.release import ReleaseManager, ReleaseType, ReleaseStatus

        manager = ReleaseManager()

        release = manager.create_release(
            release_type=ReleaseType.PATCH,
            summary="Bug fixes",
        )

        assert release.status == ReleaseStatus.DRAFT
        assert "Bug fixes" in release.summary


# ============================================================================
# Packaging Tests
# ============================================================================


class TestPackageBuilder:
    """Tests for PackageBuilder."""

    def test_validate_structure(self):
        """Test structure validation."""
        from axiom_vox.release import PackageBuilder

        builder = PackageBuilder()

        result = builder.validate_structure()

        # Should return a result (may have errors if not in package root)
        assert result is not None
        assert hasattr(result, "valid")
        assert hasattr(result, "errors")

    def test_validate_imports(self):
        """Test import validation."""
        from axiom_vox.release import PackageBuilder

        builder = PackageBuilder()

        result = builder.validate_imports("axiom_vox")

        assert result is not None
        if result.valid:
            assert "version" in result.info


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_add_error(self):
        """Test adding errors."""
        from axiom_vox.release import ValidationResult

        result = ValidationResult()
        assert result.valid

        result.add_error("Test error")

        assert not result.valid
        assert "Test error" in result.errors

    def test_add_warning(self):
        """Test adding warnings."""
        from axiom_vox.release import ValidationResult

        result = ValidationResult()

        result.add_warning("Test warning")

        assert result.valid  # Warnings don't invalidate
        assert "Test warning" in result.warnings


# ============================================================================
# Deployment Tests
# ============================================================================


class TestDeploymentHelper:
    """Tests for DeploymentHelper."""

    def test_generate_dockerfile(self):
        """Test Dockerfile generation."""
        from axiom_vox.release import DeploymentHelper

        helper = DeploymentHelper()

        dockerfile = helper.generate_dockerfile()

        assert "FROM python" in dockerfile
        assert "EXPOSE" in dockerfile

    def test_generate_lambda_config(self):
        """Test Lambda config generation."""
        from axiom_vox.release import DeploymentHelper

        helper = DeploymentHelper()

        template = helper.generate_lambda_config()

        assert "AWS::Serverless" in template
        assert "VoxFunction" in template

    def test_generate_cloud_run_config(self):
        """Test Cloud Run config generation."""
        from axiom_vox.release import DeploymentHelper

        helper = DeploymentHelper()

        config = helper.generate_cloud_run_config()

        assert "knative.dev" in config
        assert "containerPort" in config

    def test_check_prerequisites(self):
        """Test prerequisite checking."""
        from axiom_vox.release import DeploymentHelper, DeploymentTarget

        helper = DeploymentHelper()

        checks = helper.check_prerequisites(DeploymentTarget.DOCKER)

        assert "docker" in checks


# ============================================================================
# Introspection Tests
# ============================================================================


class TestSystemIntrospector:
    """Tests for SystemIntrospector."""

    def test_generate_capability_report(self):
        """Test capability report generation."""
        from axiom_vox.release import SystemIntrospector

        introspector = SystemIntrospector()

        report = introspector.generate_capability_report()

        assert report.name == "axiom_vox"
        assert report.version is not None
        assert len(report.components) > 0

    def test_analyze_module(self):
        """Test module analysis."""
        from axiom_vox.release import SystemIntrospector

        introspector = SystemIntrospector()

        info = introspector.analyze_module("axiom_vox")

        assert info.name == "axiom_vox"
        assert len(info.exports) > 0


# ============================================================================
# Reflection Tests
# ============================================================================


class TestReflectionEngine:
    """Tests for ReflectionEngine."""

    def test_generate_report(self):
        """Test report generation."""
        from axiom_vox.release import ReflectionEngine

        engine = ReflectionEngine()

        report = engine.generate_report("0.17.0")

        assert report.version == "0.17.0"
        assert len(report.insights) > 0
        assert report.usage_summary is not None

    def test_analyze_usage_patterns(self):
        """Test usage pattern analysis."""
        from axiom_vox.release import ReflectionEngine

        engine = ReflectionEngine()

        # Generate report first to load metrics
        engine.generate_report("0.17.0")

        insights = engine.analyze_usage_patterns()

        assert isinstance(insights, list)

    def test_analyze_performance_trends(self):
        """Test performance trend analysis."""
        from axiom_vox.release import ReflectionEngine

        engine = ReflectionEngine()

        # Generate report first to load metrics
        engine.generate_report("0.17.0")

        insights = engine.analyze_performance_trends()

        assert isinstance(insights, list)


class TestUsageMetrics:
    """Tests for UsageMetrics."""

    def test_success_rate(self):
        """Test success rate calculation."""
        from axiom_vox.release.reflect import UsageMetrics

        metrics = UsageMetrics(
            total_requests=100,
            successful_requests=95,
        )

        assert metrics.success_rate() == 0.95

    def test_success_rate_zero_requests(self):
        """Test success rate with zero requests."""
        from axiom_vox.release.reflect import UsageMetrics

        metrics = UsageMetrics(total_requests=0)

        assert metrics.success_rate() == 1.0


# ============================================================================
# Integration Tests
# ============================================================================


class TestReleaseIntegration:
    """Integration tests for release layer."""

    def test_imports(self):
        """Test all release imports work."""
        from axiom_vox.release import (
            # Enums
            ReleaseType,
            ReleaseStatus,
            DeploymentTarget,
            PackageFormat,
            ReflectionCategory,
            # Models
            SemanticVersion,
            ReleaseInfo,
            PackageMetadata,
            DeploymentConfig,
            CapabilityReport,
            ReflectionInsight,
            ReflectionReport,
            # Components
            ReleaseManager,
            PackageBuilder,
            DeploymentHelper,
            SystemIntrospector,
            ReflectionEngine,
        )

        assert ReleaseType is not None
        assert ReleaseManager is not None
        assert PackageBuilder is not None
        assert DeploymentHelper is not None
        assert SystemIntrospector is not None
        assert ReflectionEngine is not None

    def test_main_module_exports(self):
        """Test release exports from main module."""
        from axiom_vox import (
            ReleaseType,
            ReleaseStatus,
            SemanticVersion,
            ReleaseManager,
            PackageBuilder,
            DeploymentHelper,
            SystemIntrospector,
            ReflectionEngine,
            __version__,
        )

        assert ReleaseType is not None
        assert ReleaseManager is not None
        assert __version__ == "0.17.0"

    def test_full_release_workflow(self):
        """Test complete release workflow."""
        from axiom_vox.release import (
            ReleaseManager,
            ReleaseType,
            PackageBuilder,
            SystemIntrospector,
            ReflectionEngine,
        )

        # 1. Get current version
        manager = ReleaseManager()
        current = manager.get_current_version()
        assert current is not None

        # 2. Create release (dry run)
        from axiom_vox.release import ReleaseConfig
        config = ReleaseConfig(dry_run=True)
        manager = ReleaseManager(config)
        release = manager.create_release(ReleaseType.MINOR)
        assert release is not None

        # 3. Validate package
        builder = PackageBuilder()
        validation = builder.validate_imports("axiom_vox")
        assert validation is not None

        # 4. Generate capability report
        introspector = SystemIntrospector()
        capability = introspector.generate_capability_report()
        assert capability.version is not None

        # 5. Generate reflection
        engine = ReflectionEngine()
        reflection = engine.generate_report(str(current))
        assert len(reflection.insights) > 0


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_current_version(self):
        """Test get_current_version function."""
        from axiom_vox.release import get_current_version

        version = get_current_version()

        assert version is not None
        assert "." in version

    def test_generate_capability_report(self):
        """Test generate_capability_report function."""
        from axiom_vox.release import generate_capability_report

        report = generate_capability_report()

        assert report.name == "axiom_vox"
        assert report.version is not None

    def test_generate_reflection_report(self):
        """Test generate_reflection_report function."""
        from axiom_vox.release import generate_reflection_report

        report = generate_reflection_report("0.17.0")

        assert report.version == "0.17.0"


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
