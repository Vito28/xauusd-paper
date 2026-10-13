"""XAUUSD market-data acquisition package.

Broker configuration lives in market.profile. The notebook-style workflow in
market.collector contains the MT5 boundary, validation, ingestion, coverage and
CLI for persisting auditable observations in PostgreSQL. Shared records and the
read-only dashboard remain separate. The package contains no trading or ML.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
