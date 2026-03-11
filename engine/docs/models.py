"""
VØX Documentation - Models
--------------------------

Data models for documentation generation.

AXIØM Phase 11: Document - "How do we teach this to others?"
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List


class DocType(str, Enum):
    """Type of documentation."""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    PROPERTY = "property"
    CONSTANT = "constant"
    ENUM = "enum"
    EXAMPLE = "example"
    TUTORIAL = "tutorial"


class DocFormat(str, Enum):
    """Output format for documentation."""
    MARKDOWN = "markdown"
    HTML = "html"
    RST = "rst"
    JSON = "json"


class ExampleStatus(str, Enum):
    """Status of an example."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ChangeType(str, Enum):
    """Type of changelog entry."""
    ADDED = "added"
    CHANGED = "changed"
    DEPRECATED = "deprecated"
    REMOVED = "removed"
    FIXED = "fixed"
    SECURITY = "security"


class TutorialLevel(str, Enum):
    """Difficulty level of tutorial."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class Parameter:
    """
    Documentation for a function/method parameter.

    Attributes:
        name: Parameter name
        type_hint: Type annotation
        description: Parameter description
        default: Default value if any
        required: Whether parameter is required
    """
    name: str
    type_hint: str = "Any"
    description: str = ""
    default: Optional[str] = None
    required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type_hint,
            "description": self.description,
            "default": self.default,
            "required": self.required,
        }


@dataclass
class ReturnDoc:
    """
    Documentation for a return value.

    Attributes:
        type_hint: Return type annotation
        description: Description of return value
    """
    type_hint: str = "None"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type_hint,
            "description": self.description,
        }


@dataclass
class ExceptionDoc:
    """
    Documentation for an exception.

    Attributes:
        exception_type: Exception class name
        description: When this exception is raised
    """
    exception_type: str
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.exception_type,
            "description": self.description,
        }


@dataclass
class DocEntry:
    """
    A documentation entry for any code element.

    Attributes:
        name: Element name
        doc_type: Type of element
        description: Full description
        signature: Function/method signature
        parameters: Parameter documentation
        returns: Return value documentation
        raises: Exception documentation
        examples: Usage examples
        see_also: Related elements
        deprecated: Deprecation notice if any
        version_added: Version when added
    """
    name: str
    doc_type: DocType
    description: str = ""
    module: str = ""
    signature: str = ""
    parameters: List[Parameter] = field(default_factory=list)
    returns: Optional[ReturnDoc] = None
    raises: List[ExceptionDoc] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    see_also: List[str] = field(default_factory=list)
    deprecated: Optional[str] = None
    version_added: Optional[str] = None
    source_file: str = ""
    line_number: int = 0

    @property
    def qualified_name(self) -> str:
        """Get fully qualified name."""
        if self.module:
            return f"{self.module}.{self.name}"
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "type": self.doc_type.value,
            "description": self.description,
            "module": self.module,
            "signature": self.signature,
            "parameters": [p.to_dict() for p in self.parameters],
            "returns": self.returns.to_dict() if self.returns else None,
            "raises": [e.to_dict() for e in self.raises],
            "examples": self.examples,
            "see_also": self.see_also,
            "deprecated": self.deprecated,
            "version_added": self.version_added,
            "source_file": self.source_file,
            "line_number": self.line_number,
        }

    def to_markdown(self) -> str:
        """Generate markdown documentation."""
        lines = []

        # Header
        if self.doc_type == DocType.MODULE:
            lines.append(f"# {self.name}")
        elif self.doc_type == DocType.CLASS:
            lines.append(f"## class {self.name}")
        else:
            lines.append(f"### {self.name}")

        # Deprecation warning
        if self.deprecated:
            lines.append(f"\n> **Deprecated:** {self.deprecated}\n")

        # Description
        if self.description:
            lines.append(f"\n{self.description}\n")

        # Signature
        if self.signature:
            lines.append(f"\n```python\n{self.signature}\n```\n")

        # Parameters
        if self.parameters:
            lines.append("\n**Parameters:**\n")
            for param in self.parameters:
                req = "" if param.required else " (optional)"
                default = f" = `{param.default}`" if param.default else ""
                lines.append(f"- `{param.name}` ({param.type_hint}){req}{default}: {param.description}")

        # Returns
        if self.returns and self.returns.type_hint != "None":
            lines.append(f"\n**Returns:** `{self.returns.type_hint}`")
            if self.returns.description:
                lines.append(f"  {self.returns.description}")

        # Raises
        if self.raises:
            lines.append("\n**Raises:**\n")
            for exc in self.raises:
                lines.append(f"- `{exc.exception_type}`: {exc.description}")

        # Examples
        if self.examples:
            lines.append("\n**Examples:**\n")
            for example in self.examples:
                lines.append(f"```python\n{example}\n```\n")

        # See Also
        if self.see_also:
            lines.append("\n**See Also:** " + ", ".join(f"`{s}`" for s in self.see_also))

        # Version
        if self.version_added:
            lines.append(f"\n*Added in version {self.version_added}*")

        return "\n".join(lines)


@dataclass
class ModuleDoc:
    """
    Documentation for a complete module.

    Attributes:
        name: Module name
        description: Module description
        entries: All documentation entries
        submodules: Child modules
    """
    name: str
    description: str = ""
    entries: List[DocEntry] = field(default_factory=list)
    submodules: List[str] = field(default_factory=list)
    version: str = ""
    author: str = ""

    @property
    def classes(self) -> List[DocEntry]:
        """Get all class entries."""
        return [e for e in self.entries if e.doc_type == DocType.CLASS]

    @property
    def functions(self) -> List[DocEntry]:
        """Get all function entries."""
        return [e for e in self.entries if e.doc_type == DocType.FUNCTION]

    @property
    def constants(self) -> List[DocEntry]:
        """Get all constant entries."""
        return [e for e in self.entries if e.doc_type == DocType.CONSTANT]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "submodules": self.submodules,
            "entries": [e.to_dict() for e in self.entries],
        }

    def to_markdown(self) -> str:
        """Generate markdown documentation."""
        lines = [
            f"# {self.name}",
            "",
            self.description,
            "",
        ]

        if self.version:
            lines.append(f"**Version:** {self.version}\n")

        # Table of contents
        if self.classes or self.functions:
            lines.append("## Contents\n")

            if self.classes:
                lines.append("### Classes\n")
                for cls in self.classes:
                    lines.append(f"- [{cls.name}](#{cls.name.lower()})")

            if self.functions:
                lines.append("\n### Functions\n")
                for func in self.functions:
                    lines.append(f"- [{func.name}](#{func.name.lower()})")

            lines.append("")

        # Entries
        for entry in self.entries:
            lines.append(entry.to_markdown())
            lines.append("\n---\n")

        return "\n".join(lines)


@dataclass
class Example:
    """
    A runnable code example.

    Attributes:
        name: Example name
        description: What the example demonstrates
        code: Python code to run
        expected_output: Expected output (if any)
        tags: Categorization tags
        requirements: Required modules/setup
    """
    name: str
    description: str
    code: str
    expected_output: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    setup_code: str = ""
    teardown_code: str = ""
    timeout: float = 30.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "code": self.code,
            "expected_output": self.expected_output,
            "tags": self.tags,
            "requirements": self.requirements,
        }


@dataclass
class ExampleResult:
    """
    Result of running an example.

    Attributes:
        example: The example that was run
        status: Pass/fail status
        output: Actual output
        error: Error message if failed
        duration_ms: Execution time
    """
    example: Example
    status: ExampleStatus
    output: str = ""
    error: Optional[str] = None
    duration_ms: float = 0.0

    @property
    def passed(self) -> bool:
        """Check if example passed."""
        return self.status == ExampleStatus.PASSED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.example.name,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "passed": self.passed,
        }


@dataclass
class TutorialStep:
    """
    A single step in a tutorial.

    Attributes:
        title: Step title
        content: Explanation content
        code: Code to demonstrate
        expected_result: What should happen
        checkpoint: Validation checkpoint
    """
    title: str
    content: str
    code: str = ""
    expected_result: str = ""
    checkpoint: Optional[str] = None
    hints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "code": self.code,
            "expected_result": self.expected_result,
            "checkpoint": self.checkpoint,
            "hints": self.hints,
        }


@dataclass
class Tutorial:
    """
    A complete tutorial.

    Attributes:
        title: Tutorial title
        description: What the tutorial teaches
        level: Difficulty level
        steps: Tutorial steps
        prerequisites: Required knowledge
        estimated_time_minutes: Time to complete
    """
    title: str
    description: str
    level: TutorialLevel = TutorialLevel.BEGINNER
    steps: List[TutorialStep] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    estimated_time_minutes: int = 30
    tags: List[str] = field(default_factory=list)
    author: str = ""
    version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "level": self.level.value,
            "steps": [s.to_dict() for s in self.steps],
            "prerequisites": self.prerequisites,
            "estimated_time_minutes": self.estimated_time_minutes,
            "tags": self.tags,
        }

    def to_markdown(self) -> str:
        """Generate markdown tutorial."""
        lines = [
            f"# {self.title}",
            "",
            f"**Level:** {self.level.value.title()}",
            f"**Time:** ~{self.estimated_time_minutes} minutes",
            "",
            self.description,
            "",
        ]

        if self.prerequisites:
            lines.append("## Prerequisites\n")
            for prereq in self.prerequisites:
                lines.append(f"- {prereq}")
            lines.append("")

        lines.append("## Steps\n")
        for i, step in enumerate(self.steps, 1):
            lines.append(f"### Step {i}: {step.title}\n")
            lines.append(step.content)
            lines.append("")

            if step.code:
                lines.append("```python")
                lines.append(step.code)
                lines.append("```\n")

            if step.expected_result:
                lines.append(f"**Expected Result:** {step.expected_result}\n")

            if step.hints:
                lines.append("<details>")
                lines.append("<summary>Hints</summary>\n")
                for hint in step.hints:
                    lines.append(f"- {hint}")
                lines.append("</details>\n")

        return "\n".join(lines)


@dataclass
class ChangelogEntry:
    """
    A changelog entry.

    Attributes:
        change_type: Type of change
        description: What changed
        issue: Related issue number
        pr: Related PR number
        breaking: Whether this is a breaking change
    """
    change_type: ChangeType
    description: str
    issue: Optional[str] = None
    pr: Optional[str] = None
    breaking: bool = False
    component: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.change_type.value,
            "description": self.description,
            "issue": self.issue,
            "pr": self.pr,
            "breaking": self.breaking,
            "component": self.component,
        }

    def to_markdown(self) -> str:
        """Generate markdown entry."""
        prefix = "**BREAKING:** " if self.breaking else ""
        refs = []
        if self.issue:
            refs.append(f"#{self.issue}")
        if self.pr:
            refs.append(f"PR #{self.pr}")
        ref_str = f" ({', '.join(refs)})" if refs else ""
        return f"- {prefix}{self.description}{ref_str}"


@dataclass
class VersionChangelog:
    """
    Changelog for a specific version.

    Attributes:
        version: Version string
        date: Release date
        entries: All changes in this version
        summary: Brief summary
    """
    version: str
    date: str = ""
    entries: List[ChangelogEntry] = field(default_factory=list)
    summary: str = ""

    @property
    def added(self) -> List[ChangelogEntry]:
        """Get added entries."""
        return [e for e in self.entries if e.change_type == ChangeType.ADDED]

    @property
    def changed(self) -> List[ChangelogEntry]:
        """Get changed entries."""
        return [e for e in self.entries if e.change_type == ChangeType.CHANGED]

    @property
    def fixed(self) -> List[ChangelogEntry]:
        """Get fixed entries."""
        return [e for e in self.entries if e.change_type == ChangeType.FIXED]

    @property
    def has_breaking_changes(self) -> bool:
        """Check if version has breaking changes."""
        return any(e.breaking for e in self.entries)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "date": self.date,
            "summary": self.summary,
            "has_breaking_changes": self.has_breaking_changes,
            "entries": [e.to_dict() for e in self.entries],
        }

    def to_markdown(self) -> str:
        """Generate markdown changelog."""
        lines = [f"## [{self.version}] - {self.date or 'Unreleased'}"]

        if self.summary:
            lines.append(f"\n{self.summary}\n")

        if self.has_breaking_changes:
            lines.append("\n> **Warning:** This version contains breaking changes.\n")

        for change_type in ChangeType:
            type_entries = [e for e in self.entries if e.change_type == change_type]
            if type_entries:
                lines.append(f"\n### {change_type.value.title()}\n")
                for entry in type_entries:
                    lines.append(entry.to_markdown())

        return "\n".join(lines)


@dataclass
class Changelog:
    """
    Complete project changelog.

    Attributes:
        project: Project name
        versions: All version changelogs
    """
    project: str
    versions: List[VersionChangelog] = field(default_factory=list)
    description: str = ""

    def get_version(self, version: str) -> Optional[VersionChangelog]:
        """Get changelog for specific version."""
        return next((v for v in self.versions if v.version == version), None)

    def to_markdown(self) -> str:
        """Generate full markdown changelog."""
        lines = [
            "# Changelog",
            "",
            self.description or f"All notable changes to {self.project}.",
            "",
            "The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),",
            "and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).",
            "",
        ]

        for version in self.versions:
            lines.append(version.to_markdown())
            lines.append("")

        return "\n".join(lines)


@dataclass
class MigrationStep:
    """
    A single migration step.

    Attributes:
        description: What to change
        before: Code before migration
        after: Code after migration
        automated: Whether this can be auto-migrated
    """
    description: str
    before: str = ""
    after: str = ""
    automated: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "description": self.description,
            "before": self.before,
            "after": self.after,
            "automated": self.automated,
            "notes": self.notes,
        }

    def to_markdown(self) -> str:
        """Generate markdown step."""
        lines = [f"#### {self.description}"]

        if self.before:
            lines.append("\n**Before:**")
            lines.append(f"```python\n{self.before}\n```")

        if self.after:
            lines.append("\n**After:**")
            lines.append(f"```python\n{self.after}\n```")

        if self.notes:
            lines.append(f"\n> {self.notes}")

        if self.automated:
            lines.append("\n*This change can be automated.*")

        return "\n".join(lines)


@dataclass
class MigrationGuide:
    """
    Guide for migrating between versions.

    Attributes:
        from_version: Source version
        to_version: Target version
        steps: Migration steps
        breaking_changes: List of breaking changes
    """
    from_version: str
    to_version: str
    steps: List[MigrationStep] = field(default_factory=list)
    breaking_changes: List[str] = field(default_factory=list)
    deprecations: List[str] = field(default_factory=list)
    new_features: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "steps": [s.to_dict() for s in self.steps],
            "breaking_changes": self.breaking_changes,
            "deprecations": self.deprecations,
            "new_features": self.new_features,
        }

    def to_markdown(self) -> str:
        """Generate markdown guide."""
        lines = [
            f"# Migration Guide: {self.from_version} → {self.to_version}",
            "",
        ]

        if self.breaking_changes:
            lines.append("## Breaking Changes\n")
            for change in self.breaking_changes:
                lines.append(f"- {change}")
            lines.append("")

        if self.deprecations:
            lines.append("## Deprecations\n")
            for dep in self.deprecations:
                lines.append(f"- {dep}")
            lines.append("")

        if self.new_features:
            lines.append("## New Features\n")
            for feature in self.new_features:
                lines.append(f"- {feature}")
            lines.append("")

        if self.steps:
            lines.append("## Migration Steps\n")
            for i, step in enumerate(self.steps, 1):
                lines.append(f"### {i}. {step.description}\n")
                lines.append(step.to_markdown())
                lines.append("")

        return "\n".join(lines)
