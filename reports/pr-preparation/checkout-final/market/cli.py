"""Compatibility entry point for installed scripts and previous market CLI commands.

The CLI implementation and all collection stages live in market.collector.
This module only forwards the Typer app so existing launch commands still work.
"""

from market.collector import app as app

if __name__ == "__main__":
    app()
