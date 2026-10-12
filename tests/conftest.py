"""Shared pytest fixtures for deterministic market-domain records."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from market.contracts import BarObservation


@pytest.fixture
def m5_bars() -> list[BarObservation]:
    """Build three complete M5 bars suitable for one M15 aggregation."""

    start = datetime(2026, 1, 5, 0, 0, tzinfo=UTC)
    values = [
        ("100", "103", "99", "102"),
        ("102", "105", "101", "104"),
        ("104", "106", "103", "105"),
    ]
    return [
        BarObservation(
            actual_symbol="XAUUSD",
            timeframe="M5",
            open_time_utc=start.replace(minute=index * 5),
            close_time_utc=start.replace(minute=(index + 1) * 5),
            open=Decimal(open_value),
            high=Decimal(high),
            low=Decimal(low),
            close=Decimal(close),
            tick_volume=10,
            real_volume=Decimal("0"),
            spread_points=20,
            spread_price=Decimal("0.20"),
            payload_hash=f"hash-{index}",
        )
        for index, (open_value, high, low, close) in enumerate(values)
    ]
