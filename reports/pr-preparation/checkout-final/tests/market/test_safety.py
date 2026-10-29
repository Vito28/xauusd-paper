"""Static guardrails proving the project remains an acquisition platform."""

from __future__ import annotations

import ast
from pathlib import Path

from market.collector import redact_sensitive_fields

ROOT = Path(__file__).parents[2]


def production_python_files() -> list[Path]:
    """Return collector source files while excluding tests and generated caches."""

    return sorted(
        [
            *ROOT.glob("market/**/*.py"),
            *ROOT.glob("database/**/*.py"),
            *ROOT.glob("analytics/**/*.py"),
        ]
    )


def test_no_trading_execution_calls_or_methods() -> None:
    """Production AST contains neither execution calls nor trading method names."""

    forbidden_methods = {
        "buy",
        "sell",
        "open_position",
        "close_position",
        "modify_position",
        "place_order",
    }
    forbidden_attribute = "order" + "_send"
    for path in production_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            assert not (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in forbidden_methods
            ), path
            assert not (isinstance(node, ast.Attribute) and node.attr == forbidden_attribute), path


def test_no_market_data_filesystem_persistence() -> None:
    """Production code contains no CSV, Parquet, pickle, or SQLite persistence calls."""

    forbidden_attributes = {"to_csv", "to_parquet", "to_pickle"}
    forbidden_literals = {"sqlite", ".parquet", ".csv", ".pickle"}
    for path in production_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_attributes, path
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not any(token in node.value.lower() for token in forbidden_literals), path


def test_sensitive_logging_fields_are_redacted() -> None:
    """Known password, DSN, token, login, and account fields never reach JSON logs."""

    event = {
        "password": "secret",
        "database_url": "postgresql://secret",
        "login": 123,
        "token": "secret",
        "account_number": 456,
        "source": "safe",
    }
    redacted = redact_sensitive_fields(None, "info", event)
    assert set(redacted.values()).isdisjoint({"secret", "postgresql://secret", 123, 456})
    assert redacted["source"] == "safe"


def test_every_python_module_has_docstring() -> None:
    """All project Python files start with AST-readable source documentation."""

    files = production_python_files() + sorted(ROOT.glob("tests/**/*.py"))
    for path in files:
        assert ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))), path
