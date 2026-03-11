"""
VØX Documentation Layer
-----------------------

Documentation generation, examples, and tutorials.

Features:
    - Auto-generate API documentation
    - Runnable code examples
    - Interactive tutorials
    - Changelog management
    - Migration guides

Quick Start:
    >>> from axiom_vox.docs import (
    ...     DocGenerator, ExampleRunner,
    ...     TutorialRunner, ChangelogManager,
    ... )
    >>>
    >>> # Generate API docs
    >>> generator = DocGenerator()
    >>> doc = generator.generate_module(my_module)
    >>> print(doc.to_markdown())
    >>>
    >>> # Run examples
    >>> runner = ExampleRunner()
    >>> results = await runner.run_all()
    >>>
    >>> # Follow tutorials
    >>> tutorials = TutorialRunner()
    >>> tutorials.start("Getting Started")

AXIØM Phase 11: Document - "How do we teach this to others?"
"""

from .models import (
    # Enums
    DocType,
    DocFormat,
    ExampleStatus,
    ChangeType,
    TutorialLevel,
    # Documentation models
    Parameter,
    ReturnDoc,
    ExceptionDoc,
    DocEntry,
    ModuleDoc,
    # Example models
    Example,
    ExampleResult,
    # Tutorial models
    TutorialStep,
    Tutorial,
    # Changelog models
    ChangelogEntry,
    VersionChangelog,
    Changelog,
    # Migration models
    MigrationStep,
    MigrationGuide,
)

from .generator import (
    GeneratorConfig,
    DocstringParser,
    DocGenerator,
    generate_api_docs,
)

from .examples import (
    ExampleConfig,
    ExampleRunner,
    create_vox_examples,
    run_examples,
)

from .tutorials import (
    TutorialProgress,
    TutorialBuilder,
    TutorialRunner,
    create_vox_tutorials,
)

from .changelog import (
    ChangelogConfig,
    ChangelogManager,
    create_vox_changelog,
)


__all__ = [
    # Enums
    "DocType",
    "DocFormat",
    "ExampleStatus",
    "ChangeType",
    "TutorialLevel",
    # Documentation models
    "Parameter",
    "ReturnDoc",
    "ExceptionDoc",
    "DocEntry",
    "ModuleDoc",
    # Example models
    "Example",
    "ExampleResult",
    # Tutorial models
    "TutorialStep",
    "Tutorial",
    "TutorialProgress",
    # Changelog models
    "ChangelogEntry",
    "VersionChangelog",
    "Changelog",
    # Migration models
    "MigrationStep",
    "MigrationGuide",
    # Generator
    "GeneratorConfig",
    "DocstringParser",
    "DocGenerator",
    "generate_api_docs",
    # Examples
    "ExampleConfig",
    "ExampleRunner",
    "create_vox_examples",
    "run_examples",
    # Tutorials
    "TutorialBuilder",
    "TutorialRunner",
    "create_vox_tutorials",
    # Changelog
    "ChangelogConfig",
    "ChangelogManager",
    "create_vox_changelog",
]
