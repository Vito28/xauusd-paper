"""Generate deterministic Markdown code reference from Python source AST.

The scanner reports source facts and developer-authored docstrings only.  It
does not invent a purpose when documentation is missing, making coverage gaps
visible and actionable.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Component:
    """One AST-discovered module, class, function, or method."""

    kind: str
    name: str
    line: int
    signature: str
    docstring: str | None


@dataclass(frozen=True, slots=True)
class ModuleReference:
    """AST documentation facts for one Python source file."""

    path: Path
    docstring: str | None
    components: tuple[Component, ...]


def annotation_text(node: ast.expr | None) -> str:
    """Return stable source text for an annotation or a missing marker."""

    return ast.unparse(node) if node is not None else "Missing"


def function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Build a compact typed signature from a function AST node."""

    arguments = [
        f"{argument.arg}: {annotation_text(argument.annotation)}"
        for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    ]
    if node.args.vararg:
        arguments.append(f"*{node.args.vararg.arg}: {annotation_text(node.args.vararg.annotation)}")
    if node.args.kwarg:
        arguments.append(f"**{node.args.kwarg.arg}: {annotation_text(node.args.kwarg.annotation)}")
    return f"({', '.join(arguments)}) -> {annotation_text(node.returns)}"


def scan_module(path: Path) -> ModuleReference:
    """Extract module and public component documentation from one source file."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    components: list[Component] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            components.append(
                Component("Class", node.name, node.lineno, "", ast.get_docstring(node))
            )
            for child in node.body:
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and not child.name.startswith("_"):
                    components.append(
                        Component(
                            "Method",
                            f"{node.name}.{child.name}",
                            child.lineno,
                            function_signature(child),
                            ast.get_docstring(child),
                        )
                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith(
            "_"
        ):
            components.append(
                Component(
                    "Function",
                    node.name,
                    node.lineno,
                    function_signature(node),
                    ast.get_docstring(node),
                )
            )
    return ModuleReference(path, ast.get_docstring(tree), tuple(components))


def render_reference(modules: list[ModuleReference], root: Path) -> str:
    """Render module facts and documentation coverage as Markdown."""

    module_documented = sum(module.docstring is not None for module in modules)
    components = [component for module in modules for component in module.components]
    component_documented = sum(component.docstring is not None for component in components)
    lines = [
        "# Code Reference",
        "",
        "> Generated from source AST. Do not edit manually.",
        "",
        "## Documentation Coverage",
        "",
        f"- Modules: {module_documented}/{len(modules)}",
        f"- Public classes/functions/methods: {component_documented}/{len(components)}",
        "",
    ]
    for module in modules:
        relative = module.path.relative_to(root).as_posix()
        lines.extend(
            [
                f"## `{relative}`",
                "",
                (module.docstring or "Documentation Status: Missing").splitlines()[0],
                "",
            ]
        )
        for component in module.components:
            status = (
                component.docstring.splitlines()[0]
                if component.docstring
                else "Documentation Status: Missing"
            )
            signature = f" `{component.signature}`" if component.signature else ""
            lines.extend(
                [
                    f"- **{component.kind}** `{component.name}`{signature} (line {component.line})",
                    f"  - {status}",
                ]
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Scan project packages and write ``docs/reference/code-reference.md``."""

    root = Path(__file__).resolve().parents[1]
    paths = sorted(
        [
            *root.glob("market/**/*.py"),
            *root.glob("database/**/*.py"),
            *root.glob("analytics/**/*.py"),
            *root.glob("technical/**/*.py"),
            *root.glob("analisis_teknikall/*.py"),
        ]
    )
    modules = [scan_module(path) for path in paths]
    destination = root / "docs" / "reference" / "code-reference.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_reference(modules, root), encoding="utf-8")


if __name__ == "__main__":
    main()
