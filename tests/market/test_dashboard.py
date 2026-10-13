"""Tests for the read-only processed M5 browser and query composition."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import pytest

from market.dashboard import dashboard_page, query_market_data, query_processed_m5


class FakeRows:
    """Minimal SQLAlchemy-like mapping result used by dashboard unit tests."""

    def mappings(self) -> FakeRows:
        """Return the mapping result interface expected by the query service."""

        return self

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield one representative processed M5 record."""

        yield {
            "actual_symbol": "XAUUSDm",
            "open_time_utc": datetime(2026, 1, 1, tzinfo=UTC),
            "bid_open": 2300,
            "ask_open": 2300.2,
        }


class FakeSession:
    """Capture parameterized statements without requiring PostgreSQL."""

    def __init__(self) -> None:
        """Initialize a statement log for assertions."""

        self.calls: list[tuple[str, dict[str, Any]]] = []

    def scalar(self, statement: object, parameters: dict[str, Any]) -> int:
        """Record the count statement and return one matching row."""

        self.calls.append((str(statement), parameters.copy()))
        return 1

    def execute(self, statement: object, parameters: dict[str, Any]) -> FakeRows:
        """Record the page statement and return a representative row."""

        self.calls.append((str(statement), parameters.copy()))
        return FakeRows()


class FakeDatabase:
    """Provide a reusable fake session through the production context API."""

    def __init__(self) -> None:
        """Create the single session inspected by the test."""

        self.fake_session = FakeSession()

    @contextmanager
    def session(self) -> Iterator[FakeSession]:
        """Yield the fake session without transaction side effects."""

        yield self.fake_session


def test_dashboard_page_exposes_filters_and_column_groups() -> None:
    """The browser page includes the requested table controls and field groups."""

    response = dashboard_page()
    html = response.body.decode()
    assert "XAUUSD Data Explorer" in html
    assert "Actual symbol" in html
    assert 'id="environment"' in html
    assert 'value="live"' in html
    assert '"Bid":' in html
    assert '"Ask":' in html
    assert '"Spread":' in html
    assert "Market bars" in html
    assert "Raw ticks" in html
    assert "/api/market-data" in html
    assert "sortDirection" in html


def test_query_processed_m5_uses_parameterized_filters() -> None:
    """User filters remain bound parameters while quote policy stays predefined."""

    database = FakeDatabase()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    result = query_processed_m5(
        database,  # type: ignore[arg-type]
        symbol=" XAUUSDm ",
        source_key="demo-source",
        start_utc=start,
        end_utc=None,
        quote_status="available",
        limit=50,
        offset=100,
    )

    assert result.total == 1
    assert result.rows[0]["bid_open"] == 2300
    statement, parameters = database.fake_session.calls[-1]
    assert "actual_symbol = :symbol" in statement
    assert "source_key = :source_key" in statement
    assert "bid_open IS NOT NULL AND ask_open IS NOT NULL" in statement
    assert parameters == {
        "limit": 50,
        "offset": 100,
        "symbol": "XAUUSDm",
        "source_key": "demo-source",
        "start_utc": start,
    }


def test_generic_query_selects_ticks_and_whitelisted_sorting() -> None:
    """The generic browser safely composes tick filters and ascending ordering."""

    database = FakeDatabase()
    result = query_market_data(
        database,  # type: ignore[arg-type]
        dataset="ticks",
        symbol="XAUUSD",
        source_key="demo-source",
        timeframe=None,
        start_utc=None,
        end_utc=None,
        quote_status="available",
        sort_by="bid",
        sort_direction="asc",
        limit=25,
        offset=0,
    )

    assert result.dataset == "ticks"
    assert result.sort_by == "bid"
    statement, parameters = database.fake_session.calls[-1]
    assert "market.tick_observation" in statement
    assert "is_valid_quote IS TRUE" in statement
    assert 'ORDER BY "bid" ASC NULLS LAST' in statement
    assert parameters["limit"] == 25


def test_generic_query_rejects_unknown_sort_column() -> None:
    """Untrusted sort identifiers cannot be composed into dashboard SQL."""

    with pytest.raises(ValueError, match="Kolom sort tidak tersedia"):
        query_market_data(
            FakeDatabase(),  # type: ignore[arg-type]
            dataset="bars",
            symbol=None,
            source_key=None,
            timeframe="M5",
            start_utc=None,
            end_utc=None,
            quote_status="all",
            sort_by="drop_table",
            sort_direction="desc",
            limit=50,
            offset=0,
        )
