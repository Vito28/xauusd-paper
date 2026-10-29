"""Opt-in PostgreSQL integration smoke tests.

Set ``GOLD_TEST_DATABASE_URL`` to a disposable, migrated PostgreSQL database.
The test is skipped by default and never falls back to the runtime database.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text


@pytest.mark.integration
def test_required_schemas_and_views_exist() -> None:
    """Verify the migrated disposable database exposes required objects."""

    url = os.getenv("GOLD_TEST_DATABASE_URL")
    if not url:
        pytest.skip("GOLD_TEST_DATABASE_URL is not configured")
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            schemas = set(
                connection.scalars(
                    text(
                        "SELECT schema_name FROM information_schema.schemata "
                        "WHERE schema_name IN ('core','ingest','market','audit')"
                    )
                )
            )
            views = set(
                connection.scalars(
                    text(
                        "SELECT table_name FROM information_schema.views "
                        "WHERE table_schema='market'"
                    )
                )
            )
            processed_columns = set(
                connection.scalars(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema='market' AND table_name='processed_m5'"
                    )
                )
            )
        assert schemas == {"core", "ingest", "market", "audit"}
        assert {"bar_latest", "data_coverage", "quote_ohlc_m5", "processed_m5"} <= views
        assert {
            "open",
            "high",
            "low",
            "close",
            "bid_open",
            "bid_high",
            "bid_low",
            "bid_close",
            "ask_open",
            "ask_high",
            "ask_low",
            "ask_close",
            "spread_open",
            "spread_high",
            "spread_low",
            "spread_close",
            "spread_mean",
            "spread_median",
            "spread_std",
            "tick_count",
        } <= processed_columns
    finally:
        engine.dispose()
