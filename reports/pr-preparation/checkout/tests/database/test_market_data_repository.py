"""Unit tests for bounded market-data bulk inserts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

from database.storage import MarketDataRepository, _chunks
from market.contracts import BarObservation, TickObservation


class ScalarResult:
    """Minimal result object exposing SQLAlchemy's ``all`` contract."""

    def all(self) -> list[object]:
        """Return one representative inserted identifier."""

        return [object()]


class RecordingSession:
    """Capture statements issued by repository methods without PostgreSQL."""

    def __init__(self) -> None:
        """Initialize the statement capture list."""

        self.statements: list[Any] = []

    def scalars(self, statement: Any) -> ScalarResult:
        """Record one insert statement and return a fake scalar result."""

        self.statements.append(statement)
        return ScalarResult()


def _bar(index: int) -> BarObservation:
    """Build one normalized bar for batch-size tests."""

    start = datetime(2016, 1, 1, tzinfo=UTC) + timedelta(minutes=index * 5)
    return BarObservation(
        actual_symbol="XAUUSD",
        timeframe="M5",
        open_time_utc=start,
        close_time_utc=start + timedelta(minutes=5),
        open=Decimal("2000"),
        high=Decimal("2001"),
        low=Decimal("1999"),
        close=Decimal("2000.5"),
        tick_volume=10,
        real_volume=Decimal("0"),
        spread_points=20,
        spread_price=Decimal("0.2"),
        payload_hash=f"bar-{index}",
    )


def _tick(index: int) -> TickObservation:
    """Build one normalized tick for batch-size tests."""

    timestamp_ms = 1_451_606_400_000 + index * 1_000
    return TickObservation(
        actual_symbol="XAUUSD",
        tick_time_utc=datetime(2016, 1, 1, tzinfo=UTC) + timedelta(seconds=index),
        timestamp_ms=timestamp_ms,
        occurrence=0,
        bid=Decimal("2000"),
        ask=Decimal("2000.2"),
        last=None,
        volume=Decimal("1"),
        volume_real=Decimal("1"),
        flags=0,
        spread_price=Decimal("0.2"),
        is_valid_quote=True,
        payload_hash=f"tick-{index}",
    )


def test_chunks_split_large_sequences() -> None:
    """Chunking keeps every statement at or below the configured row limit."""

    chunks = list(_chunks(list(range(2_001))))
    assert [len(chunk) for chunk in chunks] == [1_000, 1_000, 1]


def test_insert_bars_and_ticks_use_multiple_statements() -> None:
    """Large bar and tick payloads are split before SQL execution."""

    session = RecordingSession()
    repository = MarketDataRepository()
    source_id = uuid4()

    assert (
        repository.insert_bars(
            session,
            source_id=source_id,
            symbol_spec_version_id=None,
            bars=[_bar(i) for i in range(1_001)],
        )
        == 2
    )
    assert (
        repository.insert_ticks(
            session, source_id=source_id, ticks=[_tick(i) for i in range(1_001)]
        )
        == 2
    )
    assert len(session.statements) == 4
