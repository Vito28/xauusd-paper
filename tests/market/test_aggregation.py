"""Tests for derived M5 resampling and candidate gap detection."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from market.collector import aggregate_m5_bars, detect_candidate_gaps
from market.contracts import BarObservation


def test_m5_to_m15_uses_ohlc_and_volume_rules(m5_bars: list[BarObservation]) -> None:
    """M15 aggregation selects first/max/min/last and sums volumes only."""

    result = aggregate_m5_bars(m5_bars, "M15")
    assert len(result) == 1
    bar = result[0]
    assert (bar.open, bar.high, bar.low, bar.close) == tuple(
        map(Decimal, ("100", "106", "99", "105"))
    )
    assert bar.tick_volume == 30
    assert bar.real_volume == Decimal("0")
    assert bar.spread_points == 20


def test_incomplete_group_does_not_create_synthetic_bar(
    m5_bars: list[BarObservation],
) -> None:
    """Missing source candles cause the derived group to be omitted."""

    assert aggregate_m5_bars(m5_bars[:2], "M15") == []


def test_m5_to_m30_h1_require_complete_groups(m5_bars: list[BarObservation]) -> None:
    """Larger targets remain unavailable when their complete M5 inputs are absent."""

    assert aggregate_m5_bars(m5_bars, "M30") == []
    assert aggregate_m5_bars(m5_bars, "H1") == []


def test_m5_to_all_supported_higher_timeframes(
    m5_bars: list[BarObservation],
) -> None:
    """M15, M30, H1, and H4 produce one bar when every M5 slot is present."""

    template = m5_bars[0]
    for timeframe, count in {"M15": 3, "M30": 6, "H1": 12, "H4": 48}.items():
        bars = [
            BarObservation(
                actual_symbol=template.actual_symbol,
                timeframe="M5",
                open_time_utc=template.open_time_utc + timedelta(minutes=index * 5),
                close_time_utc=template.open_time_utc + timedelta(minutes=(index + 1) * 5),
                open=Decimal(100 + index),
                high=Decimal(102 + index),
                low=Decimal(99 + index),
                close=Decimal(101 + index),
                tick_volume=1,
                real_volume=None,
                spread_points=20,
                spread_price=Decimal("0.2"),
                payload_hash=f"{timeframe}-{index}",
            )
            for index in range(count)
        ]
        result = aggregate_m5_bars(bars, timeframe)
        assert len(result) == 1
        assert result[0].tick_volume == count
        assert result[0].real_volume is None


def test_gap_detection_reports_candidate_without_repair(
    m5_bars: list[BarObservation],
) -> None:
    """Gap output describes an interval and never inserts replacement candles."""

    third = m5_bars[2]
    shifted = BarObservation(
        **{
            **{field: getattr(third, field) for field in third.__dataclass_fields__},
            "open_time_utc": third.open_time_utc + timedelta(minutes=5),
            "close_time_utc": third.close_time_utc + timedelta(minutes=5),
        }
    )
    gaps = detect_candidate_gaps([m5_bars[0], m5_bars[1], shifted], "M5")
    assert gaps == [(m5_bars[1].close_time_utc, shifted.open_time_utc)]
