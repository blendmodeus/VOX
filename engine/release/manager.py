"""
VØX Release - Manager
---------------------

Release management: versioning, changelog, release notes.

AXIØM Phase 12: Release/Reflect - "How do we ship this and learn from it?"
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
    ReleaseInfo,
    ReleaseType,
    ReleaseStatus,
    SemanticVersion,
)

logger = logging.getLogger(__name__)


@dataclass
class ReleaseConfig:
    """
    Configuration for release manager.

    Attributes:
        project_root: Root directory of project
        version_files: Files containing version strings to update
        changelog_path: Path to CHANGELOG.md
        tag_prefix: Prefix for git tags
        sign_tags: GPG sign git tags
        sign_commits: GPG sign release commits
        dry_run: Don't make actual changes
    """
    project_root: str = "."
    version_files: List[str] = field(default_factory=lambda: [
        "axiom_vox/__init__.py",
        "pyproject.toml",
        "setup.py",
    ])
    changelog_path: str = "CHANGELOG.md"
    tag_prefix: str = "v"
    sign_tags: bool = False
    sign_commits: bool = False
    dry_run: bool = False


class ReleaseManager:
    """
    Manager for software releases.

    Features:
        - Semantic version management
        - Changelog generation
        - Release notes creation
        - Git tag management
        - Version file updates
    """

    def __init__(
        self,
        config: Optional[ReleaseConfig] = None,
    ):
        """
        Initialize release manager.

        Args:
            config: Release configuration
        """
        self.config = config or ReleaseConfig()
        self._current_version: Optional[SemanticVersion] = None
        self._releases: List[ReleaseInfo] = []

    def get_current_version(self) -> SemanticVersion:
        """
        Get current version from source.

        Returns:
            Current semantic version
        """
        if self._current_version:
            return self._current_version

        # Try to read from __init__.py
        for version_file in self.config.version_files:
            path = Path(self.config.project_root) / version_file
            if path.exists():
                content = path.read_text()

                # Match __version__ = "x.y.z"
                match = re.search(
                    r'__version__\s*=\s*["\']([^"\']+)["\']',
                    content,
                )
                if match:
                    self._current_version = SemanticVersion.parse(match.group(1))
                    return self._current_version

                # Match version = "x.y.z" in pyproject.toml
                match = re.search(
                    r'version\s*=\s*["\']([^"\']+)["\']',
                    content,
                )
                if match:
                    self._current_version = SemanticVersion.parse(match.group(1))
                    return self._current_version

        # Default version
        self._current_version = SemanticVersion(0, 1, 0)
        return self._current_version

    def bump_version(
        self,
        release_type: ReleaseType,
        prerelease: Optional[str] = None,
    ) -> SemanticVersion:
        """
        Bump version number.

        Args:
            release_type: Type of release
            prerelease: Optional prerelease tag

        Returns:
            New version
        """
        current = self.get_current_version()
        new_version = current.bump(release_type)

        if prerelease:
            new_version.prerelease = prerelease

        return new_version

    def update_version_files(
        self,
        new_version: SemanticVersion,
    ) -> List[str]:
        """
        Update version in all version files.

        Args:
            new_version: New version to set

        Returns:
            List of updated files
        """
        updated = []
        version_str = str(new_version)

        for version_file in self.config.version_files:
            path = Path(self.config.project_root) / version_file
            if not path.exists():
                continue

            content = path.read_text()
            original = content

            # Update __version__ = "x.y.z"
            content = re.sub(
                r'(__version__\s*=\s*["\'])([^"\']+)(["\'])',
                rf'\g<1>{version_str}\g<3>',
                content,
            )

            # Update version = "x.y.z" in pyproject.toml
            content = re.sub(
                r'(version\s*=\s*["\'])([^"\']+)(["\'])',
                rf'\g<1>{version_str}\g<3>',
                content,
            )

            if content != original:
                if not self.config.dry_run:
                    path.write_text(content)
                updated.append(version_file)
                logger.info(f"Updated version in {version_file}")

        self._current_version = new_version
        return updated

    def generate_release_notes(
        self,
        version: SemanticVersion,
        since_tag: Optional[str] = None,
    ) -> str:
        """
        Generate release notes from git history.

        Args:
            version: Version being released
            since_tag: Generate notes since this tag

        Returns:
            Markdown release notes
        """
        notes = []

        # Get commits since last tag
        commits = self._get_commits_since(since_tag)

        # Categorize commits
        features = []
        fixes = []
        changes = []
        breaking = []

        for commit in commits:
            msg = commit.get("message", "")
            sha = commit.get("sha", "")[:7]

            if msg.startswith("feat"):
                features.append(f"- {self._clean_commit_msg(msg)} ({sha})")
            elif msg.startswith("fix"):
                fixes.append(f"- {self._clean_commit_msg(msg)} ({sha})")
            elif msg.startswith("!") or "BREAKING" in msg:
                breaking.append(f"- {self._clean_commit_msg(msg)} ({sha})")
            else:
                changes.append(f"- {self._clean_commit_msg(msg)} ({sha})")

        # Build notes
        notes.append(f"# VØX {version}")
        notes.append("")
        notes.append(f"Released: {datetime.now().strftime('%Y-%m-%d')}")
        notes.append("")

        if breaking:
            notes.append("## ⚠️ Breaking Changes")
            notes.append("")
            notes.extend(breaking)
            notes.append("")

        if features:
            notes.append("## ✨ Features")
            notes.append("")
            notes.extend(features)
            notes.append("")

        if fixes:
            notes.append("## 🐛 Bug Fixes")
            notes.append("")
            notes.extend(fixes)
            notes.append("")

        if changes:
            notes.append("## 🔄 Changes")
            notes.append("")
            notes.extend(changes)
            notes.append("")

        return "\n".join(notes)

    def create_release(
        self,
        release_type: ReleaseType,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        prerelease: Optional[str] = None,
    ) -> ReleaseInfo:
        """
        Create a new release.

        Args:
            release_type: Type of release
            title: Optional title
            summary: Optional summary
            prerelease: Optional prerelease tag

        Returns:
            Release information
        """
        current = self.get_current_version()
        new_version = self.bump_version(release_type, prerelease)

        # Get previous tag for release notes
        previous_tag = f"{self.config.tag_prefix}{current}"

        # Generate release notes
        release_notes = self.generate_release_notes(new_version, previous_tag)

        release = ReleaseInfo(
            version=str(new_version),
            release_type=release_type,
            status=ReleaseStatus.DRAFT,
            title=title or f"VØX {new_version}",
            summary=summary or "",
            release_notes=release_notes,
            created_at=datetime.now(),
            tag_name=f"{self.config.tag_prefix}{new_version}",
        )

        self._releases.append(release)
        return release

    def finalize_release(
        self,
        release: ReleaseInfo,
        update_files: bool = True,
        create_tag: bool = True,
        create_commit: bool = True,
    ) -> ReleaseInfo:
        """
        Finalize a release (update files, create tag).

        Args:
            release: Release to finalize
            update_files: Update version files
            create_tag: Create git tag
            create_commit: Create release commit

        Returns:
            Finalized release
        """
        version = SemanticVersion.parse(release.version)

        # Update version files
        if update_files:
            updated = self.update_version_files(version)
            logger.info(f"Updated {len(updated)} version files")

        # Create commit
        if create_commit and not self.config.dry_run:
            commit_sha = self._create_release_commit(release)
            release.commit_sha = commit_sha

        # Create tag
        if create_tag and not self.config.dry_run:
            self._create_tag(release)

        release.status = ReleaseStatus.RELEASED
        release.released_at = datetime.now()

        return release

    def get_changelog_entry(
        self,
        version: SemanticVersion,
    ) -> str:
        """
        Generate changelog entry for version.

        Args:
            version: Version to generate entry for

        Returns:
            Changelog markdown entry
        """
        # Get previous tag
        tags = self._get_tags()
        previous_tag = tags[0] if tags else None

        commits = self._get_commits_since(previous_tag)

        lines = [
            f"## [{version}] - {datetime.now().strftime('%Y-%m-%d')}",
            "",
        ]

        # Categorize commits
        categories = {
            "Added": [],
            "Changed": [],
            "Fixed": [],
            "Removed": [],
            "Deprecated": [],
            "Security": [],
        }

        for commit in commits:
            msg = commit.get("message", "")
            cleaned = self._clean_commit_msg(msg)

            if msg.startswith("feat"):
                categories["Added"].append(cleaned)
            elif msg.startswith("fix"):
                categories["Fixed"].append(cleaned)
            elif msg.startswith("deprecate"):
                categories["Deprecated"].append(cleaned)
            elif msg.startswith("remove"):
                categories["Removed"].append(cleaned)
            elif msg.startswith("security"):
                categories["Security"].append(cleaned)
            else:
                categories["Changed"].append(cleaned)

        for category, items in categories.items():
            if items:
                lines.append(f"### {category}")
                lines.append("")
                for item in items:
                    lines.append(f"- {item}")
                lines.append("")

        return "\n".join(lines)

    def update_changelog(
        self,
        version: SemanticVersion,
        entry: Optional[str] = None,
    ) -> None:
        """
        Update CHANGELOG.md with new version entry.

        Args:
            version: Version to add
            entry: Optional pre-generated entry
        """
        if not entry:
            entry = self.get_changelog_entry(version)

        changelog_path = Path(self.config.project_root) / self.config.changelog_path

        if changelog_path.exists():
            content = changelog_path.read_text()

            # Insert after header
            if "# Changelog" in content:
                parts = content.split("# Changelog", 1)
                new_content = parts[0] + "# Changelog\n\n" + entry + parts[1].lstrip()
            else:
                new_content = f"# Changelog\n\n{entry}\n{content}"
        else:
            new_content = f"# Changelog\n\n{entry}"

        if not self.config.dry_run:
            changelog_path.write_text(new_content)
            logger.info(f"Updated {self.config.changelog_path}")

    def list_releases(self) -> List[ReleaseInfo]:
        """
        List all releases.

        Returns:
            List of release info
        """
        # Get releases from git tags
        tags = self._get_tags()
        releases = []

        for tag in tags:
            # Parse version from tag
            version_str = tag.lstrip(self.config.tag_prefix)
            try:
                version = SemanticVersion.parse(version_str)

                # Get tag info
                info = self._get_tag_info(tag)

                release = ReleaseInfo(
                    version=str(version),
                    status=ReleaseStatus.RELEASED,
                    tag_name=tag,
                    commit_sha=info.get("sha", ""),
                    released_at=info.get("date"),
                    author=info.get("author", ""),
                )
                releases.append(release)
            except Exception:
                continue

        return releases

    def _get_commits_since(
        self,
        since_tag: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get commits since tag."""
        cmd = ["git", "log", "--pretty=format:%H|%s|%an"]

        if since_tag:
            cmd.append(f"{since_tag}..HEAD")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.config.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return []

            commits = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 3:
                    commits.append({
                        "sha": parts[0],
                        "message": parts[1],
                        "author": parts[2],
                    })

            return commits

        except Exception as e:
            logger.warning(f"Failed to get commits: {e}")
            return []

    def _get_tags(self) -> List[str]:
        """Get git tags sorted by version."""
        try:
            result = subprocess.run(
                ["git", "tag", "--sort=-v:refname"],
                cwd=self.config.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return []

            return [t.strip() for t in result.stdout.strip().split("\n") if t.strip()]

        except Exception:
            return []

    def _get_tag_info(self, tag: str) -> Dict[str, Any]:
        """Get information about a tag."""
        try:
            result = subprocess.run(
                ["git", "show", tag, "--pretty=format:%H|%an|%ai", "-s"],
                cwd=self.config.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return {}

            parts = result.stdout.strip().split("|")
            if len(parts) >= 3:
                return {
                    "sha": parts[0],
                    "author": parts[1],
                    "date": datetime.fromisoformat(parts[2].replace(" ", "T")),
                }

            return {}

        except Exception:
            return {}

    def _create_release_commit(self, release: ReleaseInfo) -> str:
        """Create release commit."""
        message = f"chore(release): {release.version}\n\n{release.summary or 'Release ' + release.version}"

        cmd = ["git", "commit", "-am", message]
        if self.config.sign_commits:
            cmd.append("-S")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.config.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.warning(f"Failed to create commit: {result.stderr}")
                return ""

            # Get commit SHA
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.config.project_root,
                capture_output=True,
                text=True,
            )

            return result.stdout.strip()

        except Exception as e:
            logger.warning(f"Failed to create commit: {e}")
            return ""

    def _create_tag(self, release: ReleaseInfo) -> bool:
        """Create git tag for release."""
        cmd = ["git", "tag", "-a", release.tag_name, "-m", f"Release {release.version}"]
        if self.config.sign_tags:
            cmd.append("-s")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.config.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.warning(f"Failed to create tag: {result.stderr}")
                return False

            logger.info(f"Created tag {release.tag_name}")
            return True

        except Exception as e:
            logger.warning(f"Failed to create tag: {e}")
            return False

    def _clean_commit_msg(self, msg: str) -> str:
        """Clean commit message for changelog."""
        # Remove conventional commit prefix
        match = re.match(r"^(\w+)(?:\([^)]+\))?:\s*(.+)$", msg)
        if match:
            return match.group(2)
        return msg


def create_release(
    release_type: ReleaseType = ReleaseType.MINOR,
    dry_run: bool = True,
) -> ReleaseInfo:
    """
    Create a new VØX release.

    Args:
        release_type: Type of release
        dry_run: Don't make actual changes

    Returns:
        Release information
    """
    config = ReleaseConfig(dry_run=dry_run)
    manager = ReleaseManager(config)
    return manager.create_release(release_type)


def get_current_version() -> str:
    """
    Get current VØX version.

    Returns:
        Version string
    """
    manager = ReleaseManager()
    return str(manager.get_current_version())
