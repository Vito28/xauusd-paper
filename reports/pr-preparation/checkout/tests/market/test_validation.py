"""Tests for raw MT5 bar/tick normalization and integrity rules."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from market.collector import normalize_bar, normalize_ticks
from market.contracts import RejectedRecord


def valid_bar_payload() -> dict[str, float | int]:
    """Return a minimal well-formed native MT5 bar payload."""

    return {
        "time": int(datetime(2025, 1, 1, tzinfo=UTC).timestamp()),
        "open": 2600.0,
        "high": 2610.0,
        "low": 2595.0,
        "close": 2605.0,
        "tick_volume": 100,
        "spread": 20,
        "real_volume": 0,
    }


def test_valid_bar_preserves_native_values_and_spread() -> None:
    """Native OHLC and volume pass through while spread price uses symbol point."""

    result = normalize_bar(
        valid_bar_payload(),
        actual_symbol="XAUUSDm",
        timeframe="M5",
        point=Decimal("0.01"),
        now_utc=datetime(2025, 1, 2, tzinfo=UTC),
        finalization_delay_seconds=5,
    )
    assert not isinstance(result, RejectedRecord)
    assert result.actual_symbol == "XAUUSDm"
    assert result.spread_price == Decimal("0.20")
    assert result.real_volume == Decimal("0")


def test_invalid_ohlc_is_rejected_without_repair() -> None:
    """A high below close enters the audit path rather than being corrected."""

    payload = valid_bar_payload()
    payload["high"] = 2601.0
    result = normalize_bar(
        payload,
        actual_symbol="XAUUSD",
        timeframe="M5",
        point=Decimal("0.01"),
        now_utc=datetime(2025, 1, 2, tzinfo=UTC),
        finalization_delay_seconds=5,
    )
    assert isinstance(result, RejectedRecord)
    assert result.reason_code == "INVALID_OHLC"


def test_unfinished_bar_is_rejected() -> None:
    """The current candle cannot be persisted as finalized history."""

    payload = valid_bar_payload()
    payload["time"] = int(datetime(2025, 1, 2, 0, 0, tzinfo=UTC).timestamp())
    result = normalize_bar(
        payload,
        actual_symbol="XAUUSD",
        timeframe="M5",
        point=Decimal("0.01"),
        now_utc=datetime(2025, 1, 2, 0, 5, 4, tzinfo=UTC),
        finalization_delay_seconds=5,
    )
    assert isinstance(result, RejectedRecord)
    assert result.reason_code == "UNFINISHED_BAR"


def test_bar_payload_hash_is_idempotent_but_retains_revision() -> None:
    """Exact refetch hashes identically while a broker price revision changes hash."""

    payload = valid_bar_payload()
    arguments = {
        "actual_symbol": "XAUUSD",
        "timeframe": "M5",
        "point": Decimal("0.01"),
        "now_utc": datetime(2025, 1, 2, tzinfo=UTC),
        "finalization_delay_seconds": 0,
    }
    first = normalize_bar(payload, **arguments)
    repeated = normalize_bar(payload, **arguments)
    revised_payload = {**payload, "close": 2606.0, "high": 2611.0}
    revised = normalize_bar(revised_payload, **arguments)
    assert not isinstance(first, RejectedRecord)
    assert not isinstance(repeated, RejectedRecord)
    assert not isinstance(revised, RejectedRecord)
    assert first.payload_hash == repeated.payload_hash
    assert first.payload_hash != revised.payload_hash


def test_negative_volume_and_spread_are_rejected() -> None:
    """Negative native volume or spread cannot enter the observation table."""

    for field in ("tick_volume", "spread"):
        payload = valid_bar_payload()
        payload[field] = -1
        result = normalize_bar(
            payload,
            actual_symbol="XAUUSD",
            timeframe="M5",
            point=Decimal("0.01"),
            now_utc=datetime(2025, 1, 2, tzinfo=UTC),
            finalization_delay_seconds=0,
        )
        assert isinstance(result, RejectedRecord)


def test_ticks_calculate_spread_and_deterministic_occurrences() -> None:
    """Identical same-millisecond ticks retain separate stable occurrences."""

    payload = {
        "time": 1_735_689_600,
        "time_msc": 1_735_689_600_123,
        "bid": 2600.1,
        "ask": 2600.3,
        "last": 0,
        "volume": 0,
        "volume_real": 0,
        "flags": 6,
    }
    ticks, rejected = normalize_ticks([payload, payload], actual_symbol="GOLD")
    assert rejected == []
    assert [tick.occurrence for tick in ticks] == [0, 1]
    assert ticks[0].payload_hash == ticks[1].payload_hash
    assert ticks[0].spread_price == Decimal("0.2")


def test_ask_below_bid_is_rejected() -> None:
    """An inverted quote is audited rather than rewritten."""

    ticks, rejected = normalize_ticks(
        [{"time": 1, "time_msc": 1000, "bid": 2, "ask": 1, "flags": 6}],
        actual_symbol="XAUUSD",
    )
    assert ticks == []
    assert rejected[0].reason_code == "ASK_BELOW_BID"


def test_trade_only_tick_does_not_invent_quote() -> None:
    """A last-price tick remains quote-less instead of copying last into Bid/Ask."""

    ticks, rejected = normalize_ticks(
        [{"time": 1, "time_msc": 1000, "bid": 0, "ask": 0, "last": 2, "flags": 4}],
        actual_symbol="XAUUSD",
    )
    assert rejected == []
    assert ticks[0].bid == Decimal("0")
    assert ticks[0].ask == Decimal("0")
    assert ticks[0].spread_price is None
    assert not ticks[0].is_valid_quote
