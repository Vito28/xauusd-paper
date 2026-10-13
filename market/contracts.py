"""Typed contracts shared by the collector, validation, and persistence layers.

These immutable records form the internal boundary between provider-specific
MT5 payloads and the PostgreSQL repositories.  They contain only acquisition
data and provenance; feature engineering belongs to a later research project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any


class Environment(StrEnum):
    """Supported physical data environments."""

    DEMO = "demo"
    LIVE = "live"


class AccountKind(StrEnum):
    """Normalized MT5 account modes used by preflight validation."""

    DEMO = "demo"
    LIVE = "live"
    CONTEST = "contest"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TerminalSnapshot:
    """Non-sensitive terminal state captured during MT5 preflight."""

    connected: bool
    path: str | None
    company: str | None
    max_bars: int | None
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """Safe account metadata; the account login is intentionally excluded."""

    server: str
    company: str
    kind: AccountKind
    currency: str | None


@dataclass(frozen=True, slots=True)
class SymbolMetadata:
    """Versionable MT5 symbol specification associated with one source."""

    actual_symbol: str
    digits: int | None
    point: Decimal | None
    trade_tick_size: Decimal | None
    trade_tick_value: Decimal | None
    trade_contract_size: Decimal | None
    volume_min: Decimal | None
    volume_max: Decimal | None
    volume_step: Decimal | None
    currency_base: str | None
    currency_profit: str | None
    raw_payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BarObservation:
    """Normalized native MT5 OHLC bar ready for validation and persistence."""

    actual_symbol: str
    timeframe: str
    open_time_utc: datetime
    close_time_utc: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    tick_volume: int
    real_volume: Decimal | None
    spread_points: int | None
    spread_price: Decimal | None
    payload_hash: str
    quality_status: str = "valid"


@dataclass(frozen=True, slots=True)
class TickObservation:
    """Normalized native MT5 tick with deterministic identity fields."""

    actual_symbol: str
    tick_time_utc: datetime
    timestamp_ms: int
    occurrence: int
    bid: Decimal | None
    ask: Decimal | None
    last: Decimal | None
    volume: Decimal | None
    volume_real: Decimal | None
    flags: int | None
    spread_price: Decimal | None
    is_valid_quote: bool
    payload_hash: str
    quality_status: str = "valid"


@dataclass(frozen=True, slots=True)
class RejectedRecord:
    """Rejected provider payload retained for an auditable failure trail."""

    record_kind: str
    event_time_utc: datetime | None
    reason_code: str
    raw_payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TimePartition:
    """Half-open UTC interval used as a deterministic collection unit."""

    start_utc: datetime
    end_utc: datetime

    @property
    def label(self) -> str:
        """Return a compact date label suitable for logs."""

        if self.start_utc.day == 1 and self.end_utc.day == 1:
            return self.start_utc.strftime("%Y-%m")
        return self.start_utc.strftime("%Y-%m-%d")


@dataclass(slots=True)
class IngestionCounts:
    """Mutable row counters accumulated for a single ingestion run."""

    received: int = 0
    inserted: int = 0
    rejected: int = 0
    skipped: int = 0

    def add(self, other: IngestionCounts) -> None:
        """Accumulate another partition's counters into this run."""

        self.received += other.received
        self.inserted += other.inserted
        self.rejected += other.rejected
        self.skipped += other.skipped
