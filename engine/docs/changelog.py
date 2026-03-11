"""
VØX Documentation - Changelog
-----------------------------

Changelog and migration guide management.

AXIØM Phase 11: Document - "How do we teach this to others?"
"""

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from .models import (
    ChangeType,
    ChangelogEntry,
    VersionChangelog,
    Changelog,
    MigrationStep,
    MigrationGuide,
)

logger = logging.getLogger(__name__)


# Conventional commit patterns
COMMIT_PATTERNS = {
    "feat": ChangeType.ADDED,
    "fix": ChangeType.FIXED,
    "deprecate": ChangeType.DEPRECATED,
    "remove": ChangeType.REMOVED,
    "security": ChangeType.SECURITY,
    "perf": ChangeType.CHANGED,
    "refactor": ChangeType.CHANGED,
    "docs": ChangeType.CHANGED,
    "style": ChangeType.CHANGED,
    "test": ChangeType.CHANGED,
    "chore": ChangeType.CHANGED,
}


@dataclass
class ChangelogConfig:
    """
    Configuration for changelog manager.

    Attributes:
        repo_path: Path to git repository
        changelog_path: Path to CHANGELOG.md
        unreleased_section: Include unreleased section
        group_by_type: Group entries by type
        include_commits: Include commit hashes
        include_authors: Include commit authors
    """
    repo_path: str = "."
    changelog_path: str = "CHANGELOG.md"
    unreleased_section: bool = True
    group_by_type: bool = True
    include_commits: bool = False
    include_authors: bool = False


class ChangelogManager:
    """
    Manager for project changelogs.

    Features:
        - Parse existing changelogs
        - Generate from git commits
        - Semantic versioning support
        - Migration guide generation
    """

    def __init__(
        self,
        config: Optional[ChangelogConfig] = None,
    ):
        """
        Initialize changelog manager.

        Args:
            config: Manager configuration
        """
        self.config = config or ChangelogConfig()
        self._changelog: Optional[Changelog] = None

    def parse(self, content: str) -> Changelog:
        """
        Parse a changelog from markdown content.

        Args:
            content: Markdown changelog content

        Returns:
            Parsed changelog
        """
        changelog = Changelog(project="VØX")
        lines = content.split("\n")

        current_version: Optional[VersionChangelog] = None
        current_type: Optional[ChangeType] = None

        for line in lines:
            # Check for version header
            version_match = re.match(
                r"^##\s*\[([^\]]+)\](?:\s*-\s*(.+))?$",
                line,
            )
            if version_match:
                # Save previous version
                if current_version:
                    changelog.versions.append(current_version)

                version = version_match.group(1)
                date = version_match.group(2) or ""

                current_version = VersionChangelog(
                    version=version,
                    date=date.strip(),
                )
                current_type = None
                continue

            # Check for type header
            type_match = re.match(r"^###\s*(.+)$", line)
            if type_match:
                type_name = type_match.group(1).lower().strip()
                for ct in ChangeType:
                    if ct.value == type_name:
                        current_type = ct
                        break
                continue

            # Check for entry
            entry_match = re.match(r"^-\s*(.+)$", line)
            if entry_match and current_version:
                entry_text = entry_match.group(1).strip()

                # Check for breaking change
                breaking = entry_text.startswith("**BREAKING:**")
                if breaking:
                    entry_text = entry_text.replace("**BREAKING:**", "").strip()

                # Extract issue/PR references
                issue = None
                pr = None

                ref_match = re.search(r"\(#(\d+)\)", entry_text)
                if ref_match:
                    issue = ref_match.group(1)
                    entry_text = entry_text.replace(ref_match.group(0), "").strip()

                pr_match = re.search(r"\(PR #(\d+)\)", entry_text)
                if pr_match:
                    pr = pr_match.group(1)
                    entry_text = entry_text.replace(pr_match.group(0), "").strip()

                entry = ChangelogEntry(
                    change_type=current_type or ChangeType.CHANGED,
                    description=entry_text,
                    issue=issue,
                    pr=pr,
                    breaking=breaking,
                )
                current_version.entries.append(entry)

        # Save last version
        if current_version:
            changelog.versions.append(current_version)

        self._changelog = changelog
        return changelog

    def parse_file(self, path: Optional[str] = None) -> Changelog:
        """
        Parse changelog from file.

        Args:
            path: Path to changelog file

        Returns:
            Parsed changelog
        """
        path = path or self.config.changelog_path

        if not os.path.exists(path):
            return Changelog(project="VØX")

        with open(path) as f:
            content = f.read()

        return self.parse(content)

    def generate_from_git(
        self,
        since_tag: Optional[str] = None,
        until_tag: Optional[str] = None,
    ) -> VersionChangelog:
        """
        Generate changelog from git commits.

        Args:
            since_tag: Start from this tag
            until_tag: End at this tag

        Returns:
            Version changelog
        """
        # Build git log command
        cmd = ["git", "log", "--pretty=format:%H|%s|%an|%ad", "--date=short"]

        if since_tag and until_tag:
            cmd.append(f"{since_tag}..{until_tag}")
        elif since_tag:
            cmd.append(f"{since_tag}..HEAD")
        elif until_tag:
            cmd.append(until_tag)

        try:
            result = subprocess.run(
                cmd,
                cwd=self.config.repo_path,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.warning(f"Git log failed: {result.stderr}")
                return VersionChangelog(version="Unreleased")

            commits = result.stdout.strip().split("\n")

        except Exception as e:
            logger.warning(f"Failed to get git log: {e}")
            return VersionChangelog(version="Unreleased")

        version = VersionChangelog(
            version=until_tag or "Unreleased",
            date=datetime.now().strftime("%Y-%m-%d"),
        )

        for commit in commits:
            if not commit:
                continue

            parts = commit.split("|")
            if len(parts) < 4:
                continue

            commit_hash, message, author, date = parts[0], parts[1], parts[2], parts[3]

            # Parse conventional commit
            entry = self._parse_commit(message)
            if entry:
                version.entries.append(entry)

        return version

    def add_entry(
        self,
        version: str,
        change_type: ChangeType,
        description: str,
        breaking: bool = False,
        issue: Optional[str] = None,
        pr: Optional[str] = None,
    ) -> ChangelogEntry:
        """
        Add an entry to the changelog.

        Args:
            version: Version to add to
            change_type: Type of change
            description: Change description
            breaking: Is breaking change
            issue: Related issue
            pr: Related PR

        Returns:
            Created entry
        """
        if not self._changelog:
            self._changelog = Changelog(project="VØX")

        # Find or create version
        ver = self._changelog.get_version(version)
        if not ver:
            ver = VersionChangelog(version=version)
            self._changelog.versions.insert(0, ver)

        entry = ChangelogEntry(
            change_type=change_type,
            description=description,
            breaking=breaking,
            issue=issue,
            pr=pr,
        )
        ver.entries.append(entry)

        return entry

    def release(
        self,
        version: str,
        date: Optional[str] = None,
    ) -> VersionChangelog:
        """
        Release a version (move unreleased to version).

        Args:
            version: New version number
            date: Release date

        Returns:
            Released version changelog
        """
        if not self._changelog:
            self._changelog = Changelog(project="VØX")

        # Find unreleased
        unreleased = self._changelog.get_version("Unreleased")
        if unreleased:
            unreleased.version = version
            unreleased.date = date or datetime.now().strftime("%Y-%m-%d")
            return unreleased

        # Create new version
        ver = VersionChangelog(
            version=version,
            date=date or datetime.now().strftime("%Y-%m-%d"),
        )
        self._changelog.versions.insert(0, ver)
        return ver

    def generate_migration_guide(
        self,
        from_version: str,
        to_version: str,
    ) -> MigrationGuide:
        """
        Generate migration guide between versions.

        Args:
            from_version: Source version
            to_version: Target version

        Returns:
            Migration guide
        """
        guide = MigrationGuide(
            from_version=from_version,
            to_version=to_version,
        )

        if not self._changelog:
            return guide

        # Find versions in range
        from_idx = None
        to_idx = None

        for i, ver in enumerate(self._changelog.versions):
            if ver.version == to_version:
                to_idx = i
            if ver.version == from_version:
                from_idx = i
                break

        if from_idx is None or to_idx is None:
            return guide

        # Collect changes
        for i in range(to_idx, from_idx):
            ver = self._changelog.versions[i]

            for entry in ver.entries:
                if entry.breaking:
                    guide.breaking_changes.append(
                        f"[{ver.version}] {entry.description}"
                    )
                    # Create migration step for breaking changes
                    guide.steps.append(MigrationStep(
                        description=entry.description,
                        notes=f"Changed in version {ver.version}",
                    ))

                if entry.change_type == ChangeType.DEPRECATED:
                    guide.deprecations.append(
                        f"[{ver.version}] {entry.description}"
                    )

                if entry.change_type == ChangeType.ADDED:
                    guide.new_features.append(
                        f"[{ver.version}] {entry.description}"
                    )

        return guide

    def render(self, format: str = "markdown") -> str:
        """
        Render changelog to string.

        Args:
            format: Output format ("markdown", "json")

        Returns:
            Rendered changelog
        """
        if not self._changelog:
            return ""

        if format == "json":
            import json
            return json.dumps(
                {"versions": [v.to_dict() for v in self._changelog.versions]},
                indent=2,
            )

        return self._changelog.to_markdown()

    def save(self, path: Optional[str] = None) -> None:
        """
        Save changelog to file.

        Args:
            path: Output path
        """
        path = path or self.config.changelog_path
        content = self.render()

        with open(path, "w") as f:
            f.write(content)

    def _parse_commit(self, message: str) -> Optional[ChangelogEntry]:
        """Parse a conventional commit message."""
        # Match conventional commit format: type(scope): description
        match = re.match(
            r"^(\w+)(?:\(([^)]+)\))?(!)?:\s*(.+)$",
            message,
        )

        if not match:
            return None

        commit_type = match.group(1).lower()
        scope = match.group(2)
        breaking = match.group(3) == "!"
        description = match.group(4)

        change_type = COMMIT_PATTERNS.get(commit_type, ChangeType.CHANGED)

        return ChangelogEntry(
            change_type=change_type,
            description=description,
            breaking=breaking,
            component=scope or "",
        )


def create_vox_changelog() -> Changelog:
    """
    Create VØX changelog with version history.

    Returns:
        VØX changelog
    """
    changelog = Changelog(
        project="VØX",
        description="All notable changes to AXIØM VØX.",
    )

    # v0.16.0
    v016 = VersionChangelog(
        version="0.16.0",
        date=datetime.now().strftime("%Y-%m-%d"),
        summary="Documentation & Examples Layer",
    )
    v016.entries = [
        ChangelogEntry(ChangeType.ADDED, "DocGenerator for automatic API documentation"),
        ChangelogEntry(ChangeType.ADDED, "ExampleRunner with validation"),
        ChangelogEntry(ChangeType.ADDED, "TutorialBuilder framework"),
        ChangelogEntry(ChangeType.ADDED, "ChangelogManager with git integration"),
        ChangelogEntry(ChangeType.ADDED, "MigrationGuide generator"),
    ]
    changelog.versions.append(v016)

    # v0.15.0
    v015 = VersionChangelog(
        version="0.15.0",
        summary="Verification Suite",
    )
    v015.entries = [
        ChangelogEntry(ChangeType.ADDED, "E2ETestRunner for end-to-end testing"),
        ChangelogEntry(ChangeType.ADDED, "BenchmarkSuite for performance testing"),
        ChangelogEntry(ChangeType.ADDED, "QualityValidator for audio validation"),
        ChangelogEntry(ChangeType.ADDED, "HealthChecker with liveness/readiness probes"),
    ]
    changelog.versions.append(v015)

    # v0.14.0
    v014 = VersionChangelog(
        version="0.14.0",
        summary="Performance Layer",
    )
    v014.entries = [
        ChangelogEntry(ChangeType.ADDED, "AudioCache with LRU eviction"),
        ChangelogEntry(ChangeType.ADDED, "EmbeddingCache with lazy loading"),
        ChangelogEntry(ChangeType.ADDED, "HTTPConnectionPool and WebSocketPool"),
        ChangelogEntry(ChangeType.ADDED, "BatchOptimizer with multiple strategies"),
        ChangelogEntry(ChangeType.ADDED, "StreamBuffer with backpressure"),
        ChangelogEntry(ChangeType.ADDED, "LazyLoader for heavy modules"),
    ]
    changelog.versions.append(v014)

    # v0.13.0
    v013 = VersionChangelog(
        version="0.13.0",
        summary="VØX Client SDK",
    )
    v013.entries = [
        ChangelogEntry(ChangeType.ADDED, "VoxClient high-level SDK"),
        ChangelogEntry(ChangeType.ADDED, "RetryPolicy with exponential backoff"),
        ChangelogEntry(ChangeType.ADDED, "VoxSession for request tracking"),
        ChangelogEntry(ChangeType.ADDED, "Workflow helpers for common patterns"),
    ]
    changelog.versions.append(v013)

    # v0.12.0
    v012 = VersionChangelog(
        version="0.12.0",
        summary="Resource Governance Layer",
    )
    v012.entries = [
        ChangelogEntry(ChangeType.ADDED, "Sliding window rate limiting"),
        ChangelogEntry(ChangeType.ADDED, "Token bucket rate limiting"),
        ChangelogEntry(ChangeType.ADDED, "Tiered usage quotas"),
        ChangelogEntry(ChangeType.ADDED, "Policy engine for content/usage policies"),
        ChangelogEntry(ChangeType.ADDED, "SecurityManager with RBAC"),
    ]
    changelog.versions.append(v012)

    # v0.11.0
    v011 = VersionChangelog(
        version="0.11.0",
        summary="Unified Voice Pipeline",
    )
    v011.entries = [
        ChangelogEntry(ChangeType.ADDED, "VoxUnifiedPipeline single entry point"),
        ChangelogEntry(ChangeType.ADDED, "BiometricVoiceRouter for intelligent routing"),
        ChangelogEntry(ChangeType.ADDED, "RealTimeQualityMonitor"),
        ChangelogEntry(ChangeType.ADDED, "UnifiedConsentRegistry"),
    ]
    changelog.versions.append(v011)

    # v0.10.0
    v010 = VersionChangelog(
        version="0.10.0",
        summary="Voice Biometric Verification",
    )
    v010.entries = [
        ChangelogEntry(ChangeType.ADDED, "VoiceBiometricService"),
        ChangelogEntry(ChangeType.ADDED, "SpectralFingerprint 256-dim embeddings"),
        ChangelogEntry(ChangeType.ADDED, "LivenessDetector for replay/deepfake detection"),
        ChangelogEntry(ChangeType.ADDED, "DriftMonitor for voice changes"),
    ]
    changelog.versions.append(v010)

    # v0.9.0
    v09 = VersionChangelog(
        version="0.9.0",
        summary="Multi-voice Synthesis",
    )
    v09.entries = [
        ChangelogEntry(ChangeType.ADDED, "DialogueScript for multi-voice content"),
        ChangelogEntry(ChangeType.ADDED, "CharacterRegistry for voice mapping"),
        ChangelogEntry(ChangeType.ADDED, "TransitionProcessor for voice transitions"),
        ChangelogEntry(ChangeType.ADDED, "MultiVoiceSynthesizer"),
    ]
    changelog.versions.append(v09)

    return changelog
