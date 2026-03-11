"""
VØX Documentation - Generator
-----------------------------

Auto-generate API documentation from code.

AXIØM Phase 11: Document - "How do we teach this to others?"
"""

import ast
import inspect
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Type, Callable, get_type_hints

from .models import (
    DocEntry,
    DocType,
    DocFormat,
    ModuleDoc,
    Parameter,
    ReturnDoc,
    ExceptionDoc,
)

logger = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    """
    Configuration for documentation generator.

    Attributes:
        include_private: Include private members (_name)
        include_dunder: Include dunder methods (__name__)
        include_source_links: Add source file links
        parse_google_style: Parse Google-style docstrings
        parse_numpy_style: Parse NumPy-style docstrings
        max_signature_length: Max length before wrapping
    """
    include_private: bool = False
    include_dunder: bool = False
    include_source_links: bool = True
    parse_google_style: bool = True
    parse_numpy_style: bool = True
    max_signature_length: int = 80
    output_format: DocFormat = DocFormat.MARKDOWN


class DocstringParser:
    """
    Parse docstrings into structured documentation.

    Supports:
        - Google-style docstrings
        - NumPy-style docstrings
        - Sphinx-style docstrings
    """

    # Regex patterns for Google-style
    GOOGLE_ARGS_PATTERN = re.compile(
        r"^\s*Args:\s*$",
        re.MULTILINE,
    )
    GOOGLE_RETURNS_PATTERN = re.compile(
        r"^\s*Returns:\s*$",
        re.MULTILINE,
    )
    GOOGLE_RAISES_PATTERN = re.compile(
        r"^\s*Raises:\s*$",
        re.MULTILINE,
    )
    GOOGLE_EXAMPLE_PATTERN = re.compile(
        r"^\s*(?:Example|Examples):\s*$",
        re.MULTILINE,
    )

    # Regex for parameter parsing
    PARAM_PATTERN = re.compile(
        r"^\s*(\w+)(?:\s*\(([^)]+)\))?\s*:\s*(.+?)(?=^\s*\w+(?:\s*\([^)]+\))?\s*:|$)",
        re.MULTILINE | re.DOTALL,
    )

    def parse(self, docstring: Optional[str]) -> Dict[str, Any]:
        """
        Parse a docstring into components.

        Args:
            docstring: Raw docstring text

        Returns:
            Dictionary with description, params, returns, etc.
        """
        if not docstring:
            return {
                "description": "",
                "parameters": [],
                "returns": None,
                "raises": [],
                "examples": [],
            }

        # Clean up docstring
        docstring = inspect.cleandoc(docstring)

        result = {
            "description": "",
            "parameters": [],
            "returns": None,
            "raises": [],
            "examples": [],
        }

        # Split into sections
        sections = self._split_sections(docstring)

        # Parse description (text before first section)
        result["description"] = sections.get("description", "").strip()

        # Parse parameters
        if "args" in sections:
            result["parameters"] = self._parse_params(sections["args"])

        # Parse returns
        if "returns" in sections:
            result["returns"] = self._parse_returns(sections["returns"])

        # Parse raises
        if "raises" in sections:
            result["raises"] = self._parse_raises(sections["raises"])

        # Parse examples
        if "examples" in sections:
            result["examples"] = self._parse_examples(sections["examples"])

        return result

    def _split_sections(self, docstring: str) -> Dict[str, str]:
        """Split docstring into sections."""
        sections = {"description": ""}

        # Find section headers
        section_headers = [
            ("args", ["Args:", "Arguments:", "Parameters:"]),
            ("returns", ["Returns:", "Return:"]),
            ("raises", ["Raises:", "Raise:", "Exceptions:"]),
            ("examples", ["Example:", "Examples:"]),
            ("attributes", ["Attributes:"]),
            ("notes", ["Note:", "Notes:"]),
            ("see_also", ["See Also:"]),
        ]

        # Build pattern
        all_headers = []
        for _, headers in section_headers:
            all_headers.extend(headers)

        pattern = "(" + "|".join(re.escape(h) for h in all_headers) + ")"
        parts = re.split(pattern, docstring, flags=re.MULTILINE)

        # First part is description
        if parts:
            sections["description"] = parts[0].strip()

        # Parse remaining parts
        i = 1
        while i < len(parts) - 1:
            header = parts[i].strip().rstrip(":")
            content = parts[i + 1] if i + 1 < len(parts) else ""

            for section_name, headers in section_headers:
                if header + ":" in headers:
                    sections[section_name] = content.strip()
                    break

            i += 2

        return sections

    def _parse_params(self, text: str) -> List[Parameter]:
        """Parse parameter documentation."""
        params = []
        lines = text.strip().split("\n")

        current_param = None
        current_desc = []

        for line in lines:
            # Check if this is a new parameter
            match = re.match(r"^\s*(\w+)(?:\s*\(([^)]+)\))?\s*:\s*(.*)$", line)

            if match:
                # Save previous parameter
                if current_param:
                    current_param.description = " ".join(current_desc).strip()
                    params.append(current_param)

                name = match.group(1)
                type_hint = match.group(2) or "Any"
                desc = match.group(3) or ""

                current_param = Parameter(
                    name=name,
                    type_hint=type_hint,
                    description=desc,
                )
                current_desc = [desc] if desc else []
            elif current_param and line.strip():
                # Continuation of description
                current_desc.append(line.strip())

        # Save last parameter
        if current_param:
            current_param.description = " ".join(current_desc).strip()
            params.append(current_param)

        return params

    def _parse_returns(self, text: str) -> Optional[ReturnDoc]:
        """Parse return documentation."""
        text = text.strip()
        if not text:
            return None

        # Try to parse type and description
        match = re.match(r"^(?:(\w+(?:\[.*?\])?)\s*:\s*)?(.+)$", text, re.DOTALL)

        if match:
            type_hint = match.group(1) or "Any"
            description = match.group(2).strip()
            return ReturnDoc(type_hint=type_hint, description=description)

        return ReturnDoc(type_hint="Any", description=text)

    def _parse_raises(self, text: str) -> List[ExceptionDoc]:
        """Parse raises documentation."""
        exceptions = []
        lines = text.strip().split("\n")

        for line in lines:
            match = re.match(r"^\s*(\w+)\s*:\s*(.*)$", line)
            if match:
                exceptions.append(ExceptionDoc(
                    exception_type=match.group(1),
                    description=match.group(2).strip(),
                ))

        return exceptions

    def _parse_examples(self, text: str) -> List[str]:
        """Parse example code blocks."""
        examples = []

        # Find code blocks (>>> style or indented)
        code_blocks = re.findall(
            r"(?:>>>.*(?:\n(?:>>>|\.\.\.).*)*)|(?:(?:^[ ]{4}.*\n?)+)",
            text,
            re.MULTILINE,
        )

        for block in code_blocks:
            # Clean up block
            clean = "\n".join(
                line.lstrip(">").lstrip(". ").strip()
                for line in block.split("\n")
            )
            if clean.strip():
                examples.append(clean.strip())

        return examples


class DocGenerator:
    """
    Generate documentation from Python modules.

    Features:
        - Automatic API documentation
        - Docstring parsing (Google/NumPy style)
        - Type hint extraction
        - Source file linking
        - Multiple output formats
    """

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
    ):
        """
        Initialize documentation generator.

        Args:
            config: Generator configuration
        """
        self.config = config or GeneratorConfig()
        self._parser = DocstringParser()

    def generate_module(
        self,
        module: Any,
        recursive: bool = True,
    ) -> ModuleDoc:
        """
        Generate documentation for a module.

        Args:
            module: Python module to document
            recursive: Include submodules

        Returns:
            Complete module documentation
        """
        module_name = getattr(module, "__name__", str(module))
        module_doc = getattr(module, "__doc__", "") or ""

        doc = ModuleDoc(
            name=module_name,
            description=module_doc,
            version=getattr(module, "__version__", ""),
        )

        # Get all members
        for name, obj in inspect.getmembers(module):
            # Skip private/dunder unless configured
            if name.startswith("_"):
                if name.startswith("__") and not self.config.include_dunder:
                    continue
                if not self.config.include_private:
                    continue

            # Skip imported objects (not defined in this module)
            if hasattr(obj, "__module__") and obj.__module__ != module_name:
                continue

            # Document based on type
            if inspect.isclass(obj):
                entry = self._document_class(obj, module_name)
                if entry:
                    doc.entries.append(entry)

            elif inspect.isfunction(obj):
                entry = self._document_function(obj, module_name)
                if entry:
                    doc.entries.append(entry)

        # Find submodules
        if recursive and hasattr(module, "__path__"):
            for path in module.__path__:
                for item in os.listdir(path):
                    if item.endswith(".py") and not item.startswith("_"):
                        submodule_name = item[:-3]
                        doc.submodules.append(f"{module_name}.{submodule_name}")

        return doc

    def generate_class(
        self,
        cls: Type,
        module_name: str = "",
    ) -> DocEntry:
        """
        Generate documentation for a class.

        Args:
            cls: Class to document
            module_name: Parent module name

        Returns:
            Class documentation entry
        """
        return self._document_class(cls, module_name)

    def generate_function(
        self,
        func: Callable,
        module_name: str = "",
    ) -> DocEntry:
        """
        Generate documentation for a function.

        Args:
            func: Function to document
            module_name: Parent module name

        Returns:
            Function documentation entry
        """
        return self._document_function(func, module_name)

    def generate_from_file(
        self,
        file_path: str,
    ) -> ModuleDoc:
        """
        Generate documentation from a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            Module documentation
        """
        path = Path(file_path)
        module_name = path.stem

        with open(path) as f:
            source = f.read()

        tree = ast.parse(source)

        doc = ModuleDoc(
            name=module_name,
            description=ast.get_docstring(tree) or "",
        )

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                entry = self._document_ast_class(node, module_name, str(path))
                doc.entries.append(entry)

            elif isinstance(node, ast.FunctionDef):
                entry = self._document_ast_function(node, module_name, str(path))
                doc.entries.append(entry)

        return doc

    def render(
        self,
        doc: ModuleDoc,
        format: Optional[DocFormat] = None,
    ) -> str:
        """
        Render documentation to string.

        Args:
            doc: Module documentation
            format: Output format (default from config)

        Returns:
            Rendered documentation
        """
        format = format or self.config.output_format

        if format == DocFormat.MARKDOWN:
            return doc.to_markdown()
        elif format == DocFormat.JSON:
            import json
            return json.dumps(doc.to_dict(), indent=2)
        elif format == DocFormat.HTML:
            return self._render_html(doc)
        elif format == DocFormat.RST:
            return self._render_rst(doc)
        else:
            return doc.to_markdown()

    def _document_class(
        self,
        cls: Type,
        module_name: str,
    ) -> DocEntry:
        """Document a class."""
        # Get source info
        try:
            source_file = inspect.getfile(cls)
            source_lines = inspect.getsourcelines(cls)
            line_number = source_lines[1] if source_lines else 0
        except (TypeError, OSError):
            source_file = ""
            line_number = 0

        # Parse docstring
        parsed = self._parser.parse(cls.__doc__)

        # Get signature
        try:
            init = getattr(cls, "__init__", None)
            if init and init is not object.__init__:
                sig = inspect.signature(init)
                signature = f"class {cls.__name__}{sig}"
            else:
                signature = f"class {cls.__name__}"
        except (ValueError, TypeError):
            signature = f"class {cls.__name__}"

        entry = DocEntry(
            name=cls.__name__,
            doc_type=DocType.CLASS,
            description=parsed["description"],
            module=module_name,
            signature=signature,
            parameters=parsed["parameters"],
            raises=parsed["raises"],
            examples=parsed["examples"],
            source_file=source_file,
            line_number=line_number,
        )

        return entry

    def _document_function(
        self,
        func: Callable,
        module_name: str,
    ) -> DocEntry:
        """Document a function."""
        # Get source info
        try:
            source_file = inspect.getfile(func)
            source_lines = inspect.getsourcelines(func)
            line_number = source_lines[1] if source_lines else 0
        except (TypeError, OSError):
            source_file = ""
            line_number = 0

        # Parse docstring
        parsed = self._parser.parse(func.__doc__)

        # Get signature
        try:
            sig = inspect.signature(func)
            signature = f"def {func.__name__}{sig}"
        except (ValueError, TypeError):
            signature = f"def {func.__name__}(...)"

        # Get type hints
        try:
            hints = get_type_hints(func)
            returns = hints.get("return")
            if returns and parsed["returns"]:
                parsed["returns"].type_hint = str(returns)
        except Exception:
            pass

        entry = DocEntry(
            name=func.__name__,
            doc_type=DocType.FUNCTION,
            description=parsed["description"],
            module=module_name,
            signature=signature,
            parameters=parsed["parameters"],
            returns=parsed["returns"],
            raises=parsed["raises"],
            examples=parsed["examples"],
            source_file=source_file,
            line_number=line_number,
        )

        return entry

    def _document_ast_class(
        self,
        node: ast.ClassDef,
        module_name: str,
        source_file: str,
    ) -> DocEntry:
        """Document a class from AST."""
        docstring = ast.get_docstring(node) or ""
        parsed = self._parser.parse(docstring)

        return DocEntry(
            name=node.name,
            doc_type=DocType.CLASS,
            description=parsed["description"],
            module=module_name,
            parameters=parsed["parameters"],
            raises=parsed["raises"],
            examples=parsed["examples"],
            source_file=source_file,
            line_number=node.lineno,
        )

    def _document_ast_function(
        self,
        node: ast.FunctionDef,
        module_name: str,
        source_file: str,
    ) -> DocEntry:
        """Document a function from AST."""
        docstring = ast.get_docstring(node) or ""
        parsed = self._parser.parse(docstring)

        # Build signature from AST
        args = []
        for arg in node.args.args:
            args.append(arg.arg)

        signature = f"def {node.name}({', '.join(args)})"

        return DocEntry(
            name=node.name,
            doc_type=DocType.FUNCTION,
            description=parsed["description"],
            module=module_name,
            signature=signature,
            parameters=parsed["parameters"],
            returns=parsed["returns"],
            raises=parsed["raises"],
            examples=parsed["examples"],
            source_file=source_file,
            line_number=node.lineno,
        )

    def _render_html(self, doc: ModuleDoc) -> str:
        """Render documentation as HTML."""
        # Simple HTML rendering
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{doc.name} Documentation</title>",
            "<style>",
            "body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }",
            "code { background: #f4f4f4; padding: 2px 6px; }",
            "pre { background: #f4f4f4; padding: 15px; overflow-x: auto; }",
            ".deprecated { background: #fff3cd; padding: 10px; border-left: 3px solid #ffc107; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{doc.name}</h1>",
            f"<p>{doc.description}</p>",
        ]

        for entry in doc.entries:
            tag = "h2" if entry.doc_type == DocType.CLASS else "h3"
            html.append(f"<{tag}>{entry.name}</{tag}>")

            if entry.deprecated:
                html.append(f'<div class="deprecated">Deprecated: {entry.deprecated}</div>')

            html.append(f"<p>{entry.description}</p>")

            if entry.signature:
                html.append(f"<pre><code>{entry.signature}</code></pre>")

        html.extend(["</body>", "</html>"])
        return "\n".join(html)

    def _render_rst(self, doc: ModuleDoc) -> str:
        """Render documentation as reStructuredText."""
        lines = [
            doc.name,
            "=" * len(doc.name),
            "",
            doc.description,
            "",
        ]

        for entry in doc.entries:
            underline = "-" if entry.doc_type == DocType.CLASS else "~"
            lines.append(entry.name)
            lines.append(underline * len(entry.name))
            lines.append("")
            lines.append(entry.description)
            lines.append("")

            if entry.signature:
                lines.append(".. code-block:: python")
                lines.append("")
                lines.append(f"    {entry.signature}")
                lines.append("")

        return "\n".join(lines)


def generate_api_docs(
    module: Any,
    output_path: Optional[str] = None,
    format: DocFormat = DocFormat.MARKDOWN,
) -> str:
    """
    Generate API documentation for a module.

    Args:
        module: Module to document
        output_path: Optional file to write to
        format: Output format

    Returns:
        Generated documentation
    """
    generator = DocGenerator(GeneratorConfig(output_format=format))
    doc = generator.generate_module(module)
    output = generator.render(doc, format)

    if output_path:
        with open(output_path, "w") as f:
            f.write(output)

    return output
