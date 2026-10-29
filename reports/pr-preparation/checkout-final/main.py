"""Compatibility entry point forwarding to the documented market CLI."""

from market.collector import app


def main() -> None:
    """Run the market-data command-line application."""

    app()


if __name__ == "__main__":
    main()
