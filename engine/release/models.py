"""
VØX Release - Models
--------------------

Data models for release management.

AXIØM Phase 12: Release/Reflect - "How do we ship this and learn from it?"
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List


class ReleaseType(Enum):
    """Release type classification."""
    MAJOR = "major"      # Breaking changes
    MINOR = "minor"      # New features
    PATCH = "patch"      # Bug fixes
    PRERELEASE = "prerelease"  # Alpha/beta/rc
    DEV = "dev"          # Development


class ReleaseStatus(Enum):
    """Release status."""
    DRAFT = "draft"
    CANDIDATE = "candidate"
    RELEASED = "released"
    YANKED = "yanked"


class DeploymentTarget(Enum):
    """Deployment target platforms."""
    PYPI = "pypi"
    DOCKER = "docker"
    LAMBDA = "lambda"
    CLOUD_RUN = "cloud_run"
    KUBERNETES = "kubernetes"
    LOCAL = "local"


class PackageFormat(Enum):
    """Package distribution formats."""
    WHEEL = "wheel"
    SDIST = "sdist"
    DOCKER_IMAGE = "docker_image"
    ZIP = "zip"


class ReflectionCategory(Enum):
    """Categories for reflection insights."""
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    USABILITY = "usability"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"


@dataclass
class SemanticVersion:
    """Semantic version representation."""
    major: int = 0
    minor: int = 0
    patch: int = 0
    prerelease: Optional[str] = None  # e.g., "alpha.1", "beta.2", "rc.1"
    build: Optional[str] = None       # e.g., "build.123"

    def __str__(self) -> str:
        """Format as version string."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    @classmethod
    def parse(cls, version_str: str) -> "SemanticVersion":
        """Parse version string."""
        # Handle build metadata
        build = None
        if "+" in version_str:
            version_str, build = version_str.split("+", 1)

        # Handle prerelease
        prerelease = None
        if "-" in version_str:
            version_str, prerelease = version_str.split("-", 1)

        # Parse major.minor.patch
        parts = version_str.split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0

        return cls(
            major=major,
            minor=minor,
            patch=patch,
            prerelease=prerelease,
            build=build,
        )

    def bump(self, release_type: ReleaseType) -> "SemanticVersion":
        """Create bumped version."""
        if release_type == ReleaseType.MAJOR:
            return SemanticVersion(self.major + 1, 0, 0)
        elif release_type == ReleaseType.MINOR:
            return SemanticVersion(self.major, self.minor + 1, 0)
        elif release_type == ReleaseType.PATCH:
            return SemanticVersion(self.major, self.minor, self.patch + 1)
        elif release_type == ReleaseType.PRERELEASE:
            # Increment prerelease
            if self.prerelease:
                parts = self.prerelease.rsplit(".", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    new_pre = f"{parts[0]}.{int(parts[1]) + 1}"
                else:
                    new_pre = f"{self.prerelease}.1"
            else:
                new_pre = "alpha.1"
            return SemanticVersion(self.major, self.minor, self.patch, new_pre)
        else:
            return SemanticVersion(self.major, self.minor, self.patch)

    def __lt__(self, other: "SemanticVersion") -> bool:
        """Compare versions."""
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        # Prerelease versions are less than release
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False
        return (self.prerelease or "") < (other.prerelease or "")


@dataclass
class ReleaseInfo:
    """Information about a release."""
    version: str
    release_type: ReleaseType = ReleaseType.MINOR
    status: ReleaseStatus = ReleaseStatus.DRAFT
    title: str = ""
    summary: str = ""
    release_notes: str = ""
    changelog_excerpt: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    released_at: Optional[datetime] = None
    author: str = ""
    commit_sha: str = ""
    tag_name: str = ""
    artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.tag_name:
            self.tag_name = f"v{self.version}"
        if not self.title:
            self.title = f"VØX {self.version}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "release_type": self.release_type.value,
            "status": self.status.value,
            "title": self.title,
            "summary": self.summary,
            "release_notes": self.release_notes,
            "created_at": self.created_at.isoformat(),
            "released_at": self.released_at.isoformat() if self.released_at else None,
            "author": self.author,
            "commit_sha": self.commit_sha,
            "tag_name": self.tag_name,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        """Generate release notes markdown."""
        lines = [
            f"# {self.title}",
            "",
            f"**Version:** {self.version}",
            f"**Type:** {self.release_type.value.title()}",
            f"**Status:** {self.status.value.title()}",
        ]

        if self.released_at:
            lines.append(f"**Released:** {self.released_at.strftime('%Y-%m-%d')}")

        if self.summary:
            lines.extend(["", "## Summary", "", self.summary])

        if self.release_notes:
            lines.extend(["", "## Release Notes", "", self.release_notes])

        if self.changelog_excerpt:
            lines.extend(["", "## Changes", "", self.changelog_excerpt])

        if self.artifacts:
            lines.extend(["", "## Artifacts", ""])
            for artifact in self.artifacts:
                lines.append(f"- `{artifact}`")

        return "\n".join(lines)


@dataclass
class PackageMetadata:
    """Package metadata for distribution."""
    name: str = "axiom-vox"
    version: str = ""
    description: str = "AXIØM VØX - Governed text-to-speech with AXIØM intelligence"
    author: str = ""
    author_email: str = ""
    license: str = "MIT"
    url: str = ""
    python_requires: str = ">=3.9"
    keywords: List[str] = field(default_factory=lambda: [
        "tts", "text-to-speech", "voice", "synthesis", "axiom"
    ])
    classifiers: List[str] = field(default_factory=lambda: [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ])
    dependencies: List[str] = field(default_factory=list)
    optional_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    entry_points: Dict[str, List[str]] = field(default_factory=dict)
    package_data: Dict[str, List[str]] = field(default_factory=dict)

    def to_pyproject(self) -> Dict[str, Any]:
        """Generate pyproject.toml content."""
        return {
            "project": {
                "name": self.name,
                "version": self.version,
                "description": self.description,
                "authors": [{"name": self.author, "email": self.author_email}] if self.author else [],
                "license": {"text": self.license},
                "readme": "README.md",
                "requires-python": self.python_requires,
                "keywords": self.keywords,
                "classifiers": self.classifiers,
                "dependencies": self.dependencies,
                "optional-dependencies": self.optional_dependencies,
            },
            "project.urls": {
                "Homepage": self.url,
            } if self.url else {},
            "project.scripts": self.entry_points.get("console_scripts", {}),
        }

    def to_setup_cfg(self) -> str:
        """Generate setup.cfg content."""
        lines = [
            "[metadata]",
            f"name = {self.name}",
            f"version = {self.version}",
            f"description = {self.description}",
            f"author = {self.author}",
            f"author_email = {self.author_email}",
            f"license = {self.license}",
            f"url = {self.url}",
            "",
            "[options]",
            f"python_requires = {self.python_requires}",
            "packages = find:",
        ]

        if self.dependencies:
            lines.extend([
                "install_requires =",
                *[f"    {dep}" for dep in self.dependencies],
            ])

        return "\n".join(lines)


@dataclass
class DeploymentConfig:
    """Configuration for deployment."""
    target: DeploymentTarget = DeploymentTarget.LOCAL
    name: str = "axiom-vox"
    version: str = ""
    image_name: str = "axiom-vox"
    image_tag: str = "latest"
    port: int = 8000
    replicas: int = 1
    memory_limit: str = "512Mi"
    cpu_limit: str = "500m"
    environment: Dict[str, str] = field(default_factory=dict)
    secrets: List[str] = field(default_factory=list)
    health_check_path: str = "/health"
    readiness_path: str = "/ready"

    def __post_init__(self):
        if self.version and not self.image_tag:
            self.image_tag = self.version

    def generate_dockerfile(self) -> str:
        """Generate Dockerfile content."""
        return f'''# VØX Docker Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install package
RUN pip install --no-cache-dir -e .

# Expose port
EXPOSE {self.port}

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \\
    CMD curl -f http://localhost:{self.port}{self.health_check_path} || exit 1

# Run application
CMD ["python", "-m", "axiom_vox.governed_tts_api"]
'''

    def generate_docker_compose(self) -> str:
        """Generate docker-compose.yml content."""
        env_lines = "\n".join(f"      - {k}={v}" for k, v in self.environment.items())

        return f'''version: "3.9"

services:
  {self.name}:
    build: .
    image: {self.image_name}:{self.image_tag}
    ports:
      - "{self.port}:{self.port}"
    environment:
{env_lines if env_lines else "      - VOX_ENV=production"}
    deploy:
      replicas: {self.replicas}
      resources:
        limits:
          memory: {self.memory_limit}
          cpus: "{self.cpu_limit}"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{self.port}{self.health_check_path}"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
'''

    def generate_kubernetes_manifest(self) -> str:
        """Generate Kubernetes deployment manifest."""
        return f'''apiVersion: apps/v1
kind: Deployment
metadata:
  name: {self.name}
  labels:
    app: {self.name}
spec:
  replicas: {self.replicas}
  selector:
    matchLabels:
      app: {self.name}
  template:
    metadata:
      labels:
        app: {self.name}
    spec:
      containers:
      - name: {self.name}
        image: {self.image_name}:{self.image_tag}
        ports:
        - containerPort: {self.port}
        resources:
          limits:
            memory: {self.memory_limit}
            cpu: {self.cpu_limit}
        livenessProbe:
          httpGet:
            path: {self.health_check_path}
            port: {self.port}
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: {self.readiness_path}
            port: {self.port}
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: {self.name}
spec:
  selector:
    app: {self.name}
  ports:
  - port: {self.port}
    targetPort: {self.port}
  type: ClusterIP
'''


@dataclass
class CapabilityReport:
    """Report of system capabilities."""
    name: str
    version: str
    python_version: str = ""
    platform: str = ""
    components: Dict[str, bool] = field(default_factory=dict)
    features: Dict[str, bool] = field(default_factory=dict)
    dependencies: Dict[str, str] = field(default_factory=dict)
    optional_deps: Dict[str, bool] = field(default_factory=dict)
    api_endpoints: List[str] = field(default_factory=list)
    models_available: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "python_version": self.python_version,
            "platform": self.platform,
            "components": self.components,
            "features": self.features,
            "dependencies": self.dependencies,
            "optional_deps": self.optional_deps,
            "api_endpoints": self.api_endpoints,
            "models_available": self.models_available,
            "generated_at": self.generated_at.isoformat(),
        }

    def to_markdown(self) -> str:
        """Generate capability report markdown."""
        lines = [
            f"# {self.name} Capability Report",
            "",
            f"**Version:** {self.version}",
            f"**Python:** {self.python_version}",
            f"**Platform:** {self.platform}",
            f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Components",
            "",
        ]

        for comp, available in sorted(self.components.items()):
            status = "✓" if available else "✗"
            lines.append(f"- {status} {comp}")

        lines.extend(["", "## Features", ""])
        for feat, enabled in sorted(self.features.items()):
            status = "✓" if enabled else "✗"
            lines.append(f"- {status} {feat}")

        if self.dependencies:
            lines.extend(["", "## Dependencies", ""])
            for dep, ver in sorted(self.dependencies.items()):
                lines.append(f"- {dep}: {ver}")

        if self.optional_deps:
            lines.extend(["", "## Optional Dependencies", ""])
            for dep, available in sorted(self.optional_deps.items()):
                status = "✓ installed" if available else "✗ not installed"
                lines.append(f"- {dep}: {status}")

        if self.api_endpoints:
            lines.extend(["", "## API Endpoints", ""])
            for endpoint in sorted(self.api_endpoints):
                lines.append(f"- `{endpoint}`")

        return "\n".join(lines)


@dataclass
class ReflectionInsight:
    """Individual insight from reflection analysis."""
    category: ReflectionCategory
    title: str
    description: str
    impact: str = "medium"  # low, medium, high
    recommendation: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "impact": self.impact,
            "recommendation": self.recommendation,
            "metrics": self.metrics,
        }


@dataclass
class ReflectionReport:
    """Comprehensive reflection report."""
    version: str
    period_start: datetime
    period_end: datetime
    insights: List[ReflectionInsight] = field(default_factory=list)
    usage_summary: Dict[str, Any] = field(default_factory=dict)
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    error_summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "insights": [i.to_dict() for i in self.insights],
            "usage_summary": self.usage_summary,
            "performance_summary": self.performance_summary,
            "error_summary": self.error_summary,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat(),
        }

    def to_markdown(self) -> str:
        """Generate reflection report markdown."""
        lines = [
            "# VØX Reflection Report",
            "",
            f"**Version:** {self.version}",
            f"**Period:** {self.period_start.strftime('%Y-%m-%d')} to {self.period_end.strftime('%Y-%m-%d')}",
            f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # Group insights by category
        by_category: Dict[ReflectionCategory, List[ReflectionInsight]] = {}
        for insight in self.insights:
            by_category.setdefault(insight.category, []).append(insight)

        for category in ReflectionCategory:
            if category in by_category:
                lines.extend([
                    f"## {category.value.title()}",
                    "",
                ])
                for insight in by_category[category]:
                    lines.extend([
                        f"### {insight.title}",
                        "",
                        insight.description,
                        "",
                        f"**Impact:** {insight.impact}",
                        "",
                    ])
                    if insight.recommendation:
                        lines.append(f"**Recommendation:** {insight.recommendation}")
                        lines.append("")

        if self.recommendations:
            lines.extend(["## Top Recommendations", ""])
            for i, rec in enumerate(self.recommendations[:5], 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        return "\n".join(lines)
