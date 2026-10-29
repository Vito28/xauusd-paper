"""Tests for deterministic historical partition and UTC behavior."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from market.collector import daily_partitions, monthly_partitions


def test_monthly_partitions_keep_partial_boundaries() -> None:
    """Arbitrary bar ranges split exactly at calendar-month boundaries."""

    parts = monthly_partitions(datetime(2025, 1, 15, tzinfo=UTC), datetime(2025, 3, 2, tzinfo=UTC))
    assert [(part.start_utc.day, part.end_utc.month, part.end_utc.day) for part in parts] == [
        (15, 2, 1),
        (1, 3, 1),
        (1, 3, 2),
    ]


def test_daily_partitions_split_at_utc_midnight() -> None:
    """Tick ranges use deterministic daily boundaries."""

    parts = daily_partitions(
        datetime(2025, 1, 1, 12, tzinfo=UTC), datetime(2025, 1, 3, 6, tzinfo=UTC)
    )
    assert len(parts) == 3
    assert parts[0].end_utc == datetime(2025, 1, 2, tzinfo=UTC)


def test_naive_datetime_is_rejected() -> None:
    """Local Windows timezone cannot silently influence collection ranges."""

    with pytest.raises(ValueError, match="timezone-aware"):
        monthly_partitions(datetime(2025, 1, 1), datetime(2025, 2, 1, tzinfo=UTC))
