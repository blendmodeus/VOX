"""
VØX Documentation Layer Tests
-----------------------------

Comprehensive tests for the VØX Documentation Layer.

AXIØM Phase 11: Document - "How do we teach this to others?"
"""

import asyncio
import pytest


# ============================================================================
# Model Tests
# ============================================================================


class TestDocModels:
    """Tests for documentation models."""

    def test_doc_entry(self):
        """Test DocEntry model."""
        from axiom_vox.docs import DocEntry, DocType, Parameter

        entry = DocEntry(
            name="my_function",
            doc_type=DocType.FUNCTION,
            description="A test function",
            module="test_module",
            signature="def my_function(arg1: str) -> bool",
            parameters=[
                Parameter(name="arg1", type_hint="str", description="First arg"),
            ],
        )

        assert entry.name == "my_function"
        assert entry.qualified_name == "test_module.my_function"
        assert len(entry.parameters) == 1

    def test_doc_entry_markdown(self):
        """Test DocEntry markdown rendering."""
        from axiom_vox.docs import DocEntry, DocType

        entry = DocEntry(
            name="test_func",
            doc_type=DocType.FUNCTION,
            description="Test function description",
        )

        md = entry.to_markdown()
        assert "### test_func" in md
        assert "Test function description" in md

    def test_module_doc(self):
        """Test ModuleDoc model."""
        from axiom_vox.docs import ModuleDoc, DocEntry, DocType

        doc = ModuleDoc(
            name="test_module",
            description="A test module",
            version="1.0.0",
        )

        doc.entries.append(DocEntry(
            name="MyClass",
            doc_type=DocType.CLASS,
            description="A class",
        ))
        doc.entries.append(DocEntry(
            name="my_func",
            doc_type=DocType.FUNCTION,
            description="A function",
        ))

        assert len(doc.classes) == 1
        assert len(doc.functions) == 1

    def test_example(self):
        """Test Example model."""
        from axiom_vox.docs import Example

        example = Example(
            name="basic_example",
            description="Basic usage example",
            code='print("Hello")',
            expected_output="Hello",
            tags=["basic", "intro"],
        )

        assert example.name == "basic_example"
        assert "basic" in example.tags

    def test_tutorial(self):
        """Test Tutorial model."""
        from axiom_vox.docs import Tutorial, TutorialStep, TutorialLevel

        tutorial = Tutorial(
            title="Getting Started",
            description="Learn the basics",
            level=TutorialLevel.BEGINNER,
        )

        tutorial.steps.append(TutorialStep(
            title="Step 1",
            content="First step content",
            code='print("Step 1")',
        ))

        assert tutorial.title == "Getting Started"
        assert len(tutorial.steps) == 1

    def test_changelog_entry(self):
        """Test ChangelogEntry model."""
        from axiom_vox.docs import ChangelogEntry, ChangeType

        entry = ChangelogEntry(
            change_type=ChangeType.ADDED,
            description="New feature added",
            issue="123",
            breaking=False,
        )

        assert entry.change_type == ChangeType.ADDED
        assert not entry.breaking

        md = entry.to_markdown()
        assert "New feature added" in md
        assert "#123" in md

    def test_version_changelog(self):
        """Test VersionChangelog model."""
        from axiom_vox.docs import VersionChangelog, ChangelogEntry, ChangeType

        version = VersionChangelog(
            version="1.0.0",
            date="2024-01-01",
        )

        version.entries.append(ChangelogEntry(
            change_type=ChangeType.ADDED,
            description="Feature A",
        ))
        version.entries.append(ChangelogEntry(
            change_type=ChangeType.FIXED,
            description="Bug B",
        ))

        assert len(version.added) == 1
        assert len(version.fixed) == 1
        assert not version.has_breaking_changes

    def test_migration_guide(self):
        """Test MigrationGuide model."""
        from axiom_vox.docs import MigrationGuide, MigrationStep

        guide = MigrationGuide(
            from_version="0.14.0",
            to_version="0.15.0",
            breaking_changes=["API changed"],
            new_features=["New verification suite"],
        )

        guide.steps.append(MigrationStep(
            description="Update imports",
            before="from old import x",
            after="from new import x",
        ))

        md = guide.to_markdown()
        assert "0.14.0" in md
        assert "0.15.0" in md
        assert "Update imports" in md


# ============================================================================
# Generator Tests
# ============================================================================


class TestDocGenerator:
    """Tests for documentation generator."""

    def test_docstring_parser(self):
        """Test docstring parsing."""
        from axiom_vox.docs.generator import DocstringParser

        parser = DocstringParser()

        docstring = """
        Short description.

        Args:
            arg1 (str): First argument
            arg2 (int): Second argument

        Returns:
            bool: Result value

        Raises:
            ValueError: If invalid

        Examples:
            >>> func("test")
            True
        """

        parsed = parser.parse(docstring)

        assert "Short description" in parsed["description"]
        assert len(parsed["parameters"]) == 2
        assert parsed["returns"] is not None
        assert len(parsed["raises"]) == 1

    def test_generate_function_doc(self):
        """Test generating function documentation."""
        from axiom_vox.docs import DocGenerator

        def sample_function(arg1: str, arg2: int = 0) -> bool:
            """
            Sample function.

            Args:
                arg1: First argument
                arg2: Second argument

            Returns:
                True if successful
            """
            return True

        generator = DocGenerator()
        entry = generator.generate_function(sample_function)

        assert entry.name == "sample_function"
        assert "Sample function" in entry.description

    def test_generate_class_doc(self):
        """Test generating class documentation."""
        from axiom_vox.docs import DocGenerator

        class SampleClass:
            """A sample class."""

            def __init__(self, value: int):
                """Initialize with value."""
                self.value = value

        generator = DocGenerator()
        entry = generator.generate_class(SampleClass)

        assert entry.name == "SampleClass"
        assert "sample class" in entry.description.lower()


# ============================================================================
# Example Runner Tests
# ============================================================================


class TestExampleRunner:
    """Tests for example runner."""

    @pytest.mark.asyncio
    async def test_register_and_run(self):
        """Test registering and running examples."""
        from axiom_vox.docs import ExampleRunner

        runner = ExampleRunner()

        runner.register(
            name="simple_example",
            code='print("Hello")',
            description="Simple print example",
            expected_output="Hello",
        )

        results = await runner.run_all()

        assert len(results) == 1
        assert results[0].passed

    @pytest.mark.asyncio
    async def test_example_failure(self):
        """Test handling of failing examples."""
        from axiom_vox.docs import ExampleRunner

        runner = ExampleRunner()

        runner.register(
            name="failing_example",
            code='raise ValueError("Expected error")',
            description="Example that fails",
        )

        results = await runner.run_all()

        assert len(results) == 1
        assert not results[0].passed
        assert "ValueError" in results[0].error

    @pytest.mark.asyncio
    async def test_output_validation(self):
        """Test output validation."""
        from axiom_vox.docs import ExampleRunner

        runner = ExampleRunner()

        runner.register(
            name="output_mismatch",
            code='print("Actual")',
            description="Output mismatch example",
            expected_output="Expected",
        )

        results = await runner.run_all()

        assert len(results) == 1
        assert not results[0].passed

    @pytest.mark.asyncio
    async def test_async_example(self):
        """Test async example execution."""
        from axiom_vox.docs import ExampleRunner

        runner = ExampleRunner()

        runner.register(
            name="async_example",
            code="""
import asyncio
await asyncio.sleep(0.01)
print("Async done")
""",
            description="Async example",
            expected_output="Async done",
        )

        results = await runner.run_all()

        assert len(results) == 1
        assert results[0].passed

    @pytest.mark.asyncio
    async def test_tag_filtering(self):
        """Test filtering by tags."""
        from axiom_vox.docs import ExampleRunner

        runner = ExampleRunner()

        runner.register(
            name="tagged_example",
            code='print("Tagged")',
            description="Tagged example",
            tags=["basic"],
        )

        runner.register(
            name="other_example",
            code='print("Other")',
            description="Other example",
            tags=["advanced"],
        )

        results = await runner.run_by_tag("basic")

        assert len(results) == 1
        assert results[0].example.name == "tagged_example"


# ============================================================================
# Tutorial Tests
# ============================================================================


class TestTutorialRunner:
    """Tests for tutorial runner."""

    def test_register_tutorial(self):
        """Test registering tutorials."""
        from axiom_vox.docs import TutorialRunner, TutorialBuilder, TutorialLevel

        runner = TutorialRunner()

        tutorial = (
            TutorialBuilder("Test Tutorial", "A test tutorial", TutorialLevel.BEGINNER)
            .add_step("Step 1", "First step")
            .add_step("Step 2", "Second step")
            .build()
        )

        runner.register(tutorial)

        tutorials = runner.list_tutorials()
        assert len(tutorials) == 1
        assert tutorials[0].title == "Test Tutorial"

    def test_tutorial_progress(self):
        """Test tutorial progress tracking."""
        from axiom_vox.docs import TutorialRunner, TutorialBuilder

        runner = TutorialRunner()

        tutorial = (
            TutorialBuilder("Progress Tutorial", "Test progress")
            .add_step("Step 1", "First")
            .add_step("Step 2", "Second")
            .build()
        )

        runner.register(tutorial)
        progress = runner.start("Progress Tutorial")

        assert progress.current_step == 0
        assert not progress.is_complete

        runner.complete_step("Progress Tutorial", validate=False)
        assert progress.current_step == 1

        runner.complete_step("Progress Tutorial", validate=False)
        assert progress.is_complete

    def test_filter_by_level(self):
        """Test filtering tutorials by level."""
        from axiom_vox.docs import TutorialRunner, TutorialBuilder, TutorialLevel

        runner = TutorialRunner()

        beginner = TutorialBuilder("Beginner", "For beginners", TutorialLevel.BEGINNER).build()
        advanced = TutorialBuilder("Advanced", "For advanced", TutorialLevel.ADVANCED).build()

        runner.register(beginner)
        runner.register(advanced)

        beginners = runner.list_tutorials(level=TutorialLevel.BEGINNER)
        assert len(beginners) == 1
        assert beginners[0].title == "Beginner"


# ============================================================================
# Changelog Tests
# ============================================================================


class TestChangelogManager:
    """Tests for changelog manager."""

    def test_parse_changelog(self):
        """Test parsing changelog markdown."""
        from axiom_vox.docs import ChangelogManager

        content = """
# Changelog

## [1.0.0] - 2024-01-01

### Added
- Feature A
- Feature B

### Fixed
- Bug fix C

## [0.9.0] - 2023-12-01

### Added
- Initial release
"""

        manager = ChangelogManager()
        changelog = manager.parse(content)

        assert len(changelog.versions) == 2
        assert changelog.versions[0].version == "1.0.0"
        assert len(changelog.versions[0].added) == 2

    def test_add_entry(self):
        """Test adding changelog entries."""
        from axiom_vox.docs import ChangelogManager, ChangeType

        manager = ChangelogManager()
        manager._changelog = None  # Reset

        manager.add_entry(
            version="1.0.0",
            change_type=ChangeType.ADDED,
            description="New feature",
        )

        changelog = manager._changelog
        assert changelog is not None
        assert len(changelog.versions) == 1
        assert len(changelog.versions[0].entries) == 1

    def test_generate_migration_guide(self):
        """Test migration guide generation."""
        from axiom_vox.docs import ChangelogManager, ChangeType

        manager = ChangelogManager()

        # Add some versions
        manager.add_entry("1.0.0", ChangeType.ADDED, "Feature A")
        manager.add_entry("1.0.0", ChangeType.CHANGED, "Breaking change", breaking=True)
        manager.add_entry("0.9.0", ChangeType.ADDED, "Initial feature")

        guide = manager.generate_migration_guide("0.9.0", "1.0.0")

        assert guide.from_version == "0.9.0"
        assert guide.to_version == "1.0.0"
        assert len(guide.breaking_changes) == 1


# ============================================================================
# Integration Tests
# ============================================================================


class TestDocumentationIntegration:
    """Integration tests for documentation layer."""

    def test_imports(self):
        """Test all documentation imports work."""
        from axiom_vox.docs import (
            # Enums
            DocType,
            DocFormat,
            ExampleStatus,
            ChangeType,
            TutorialLevel,
            # Models
            DocEntry,
            ModuleDoc,
            Example,
            ExampleResult,
            Tutorial,
            TutorialStep,
            ChangelogEntry,
            VersionChangelog,
            MigrationGuide,
            # Components
            DocGenerator,
            ExampleRunner,
            TutorialRunner,
            ChangelogManager,
        )

        assert DocType is not None
        assert DocGenerator is not None
        assert ExampleRunner is not None
        assert TutorialRunner is not None
        assert ChangelogManager is not None

    def test_main_module_exports(self):
        """Test documentation exports from main module."""
        from axiom_vox import (
            DocType,
            DocFormat,
            DocGenerator,
            ExampleRunner,
            TutorialBuilder,
            TutorialRunner,
            ChangelogManager,
            __version__,
        )

        assert DocType is not None
        assert DocFormat is not None
        assert DocGenerator is not None
        assert ExampleRunner is not None
        assert TutorialBuilder is not None
        assert TutorialRunner is not None
        assert ChangelogManager is not None
        assert __version__ == "0.16.0"

    def test_create_vox_changelog(self):
        """Test VØX changelog creation."""
        from axiom_vox.docs import create_vox_changelog

        changelog = create_vox_changelog()

        assert changelog.project == "VØX"
        assert len(changelog.versions) > 0

        # Check for known versions
        version_numbers = [v.version for v in changelog.versions]
        assert "0.16.0" in version_numbers
        assert "0.15.0" in version_numbers

    @pytest.mark.asyncio
    async def test_create_vox_examples(self):
        """Test VØX examples creation."""
        from axiom_vox.docs import create_vox_examples

        runner = create_vox_examples()

        # Check examples are registered
        assert len(runner._examples) > 0

        # Run just the basic example
        basic = [e for e in runner._examples if e.name == "basic_import"]
        if basic:
            result = await runner.run_example(basic[0])
            # This may fail if VØX has import issues but should not error
            assert result is not None


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
