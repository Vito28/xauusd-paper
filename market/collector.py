"""Collect and audit MT5 market data in one notebook-style workflow.

Sections contain symbol resolution, lazy MT5 access, UTC partitions, bar/tick
validation, derived aggregation/gaps, retry/checkpoint ingestion, coverage/status
reports, redacted logging and CLI. Broker configuration lives in market.profile;
shared typed records remain in market.contracts for the persistence boundary.

Flow: profile -> MT5 preflight -> partitions -> normalization/validation ->
PostgreSQL observations/rejections/checkpoints -> coverage and audit reports.
No settings, terminal, or database are activated by importing this module.
Run: uv run python -m market.collector --help
"""

from __future__ import annotations

# %% 01. Import dan dependensi collector
import hashlib
import importlib
import json
import logging
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol
from uuid import UUID

import structlog
import typer
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.schema import IngestionRun
from database.storage import (
    Database,
    IngestionRepository,
    MarketDataRepository,
    RejectionRepository,
    SourceRepository,
)
from market.contracts import (
    AccountKind,
    AccountSnapshot,
    BarObservation,
    Environment,
    IngestionCounts,
    RejectedRecord,
    SymbolMetadata,
    TerminalSnapshot,
    TickObservation,
    TimePartition,
)
from market.profile import BrokerProfile, Settings, load_broker_profile, load_settings

# %% 02. Resolusi symbol broker
# SYMBOL-01 | Gunakan symbol persis dari profile; kandidat serupa dilaporkan tanpa menebak.

class SymbolResolutionError(ValueError):
    """Raised when a configured broker symbol is missing or ambiguous."""


def resolve_symbol(requested: str, available: Iterable[str]) -> tuple[str, list[str]]:
    """Resolve an exact symbol or report related candidates without guessing.

    Returns:
        The exact broker symbol and a list containing it.

    Raises:
        SymbolResolutionError: If no exact match exists.  The error includes
            candidate names but never silently chooses one.
    """

    symbols = sorted(set(available))
    if requested in symbols:
        return requested, [requested]
    requested_lower = requested.lower()
    candidates = [
        symbol
        for symbol in symbols
        if requested_lower in symbol.lower() or symbol.lower() in requested_lower
    ]
    if not candidates:
        raise SymbolResolutionError(f"symbol {requested!r} is unavailable; no candidates")
    raise SymbolResolutionError(f"symbol {requested!r} has no exact match; candidates={candidates}")


# %% 03. MT5 provider dan preflight terminal/account
# MT5-01 | Library MT5 diimpor hanya saat initialize() pada operasi eksplisit.
# MT5-02 | Verifikasi environment/account/broker/server, lalu baca symbol metadata/bars/ticks.
# Provider mengambil data; penyimpanan diatur oleh IngestionService pada section 07.

class MT5Error(RuntimeError):
    """Base error for terminal initialization and provider calls."""


class MT5PreflightError(MT5Error):
    """Permanent profile, environment, broker, or server mismatch."""


class MT5Provider:
    """Collect native bars and ticks from a locally installed MT5 terminal.

    Responsibilities:
        - initialize and shut down the Python terminal bridge;
        - validate terminal/account/source identity before collection;
        - retrieve native OHLC bars and historical ticks;
        - expose symbol specification snapshots.

    It does not persist data, compute indicators, infer signals, or execute
    trades.  Callers should use it as a context manager.
    """

    def __init__(self, terminal_path: Path | None = None) -> None:
        """Create a disconnected provider for an optional terminal executable."""

        self._terminal_path = terminal_path
        self._mt5: ModuleType | None = None
        self._initialized = False

    def __enter__(self) -> MT5Provider:
        """Initialize the terminal bridge for an explicit operation."""

        self.initialize()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Always release the terminal bridge after an operation."""

        self.shutdown()

    # MT5-03 | Buka bridge hanya saat provider dipakai; import module tetap tanpa koneksi.
    def initialize(self) -> None:
        """Lazily import and initialize the official MT5 Python API."""

        if self._initialized:
            return
        try:
            module = importlib.import_module("MetaTrader5")
        except ImportError as exc:
            raise MT5Error(
                "MetaTrader5 package is unavailable (requires Windows with an MT5 terminal)"
            ) from exc
        kwargs = {"path": str(self._terminal_path)} if self._terminal_path else {}
        if not module.initialize(**kwargs):
            raise MT5Error(f"MT5 initialization failed: {module.last_error()}")
        self._mt5 = module
        self._initialized = True

    def shutdown(self) -> None:
        """Shut down the initialized MT5 bridge safely."""

        if self._initialized and self._mt5 is not None:
            self._mt5.shutdown()
        self._initialized = False
        self._mt5 = None

    def _api(self) -> ModuleType:
        """Return the initialized API or fail with an explicit usage error."""

        if not self._initialized or self._mt5 is None:
            raise MT5Error("MT5 provider must be initialized by an explicit command")
        return self._mt5

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        """Convert MT5 namedtuple-like values to plain dictionaries."""

        if hasattr(value, "_asdict"):
            return dict(value._asdict())
        if isinstance(value, Mapping):
            return dict(value)
        raise MT5Error(f"unexpected MT5 response type: {type(value).__name__}")

    def terminal_snapshot(self) -> TerminalSnapshot:
        """Read non-sensitive terminal state and history-limit information."""

        info = self._api().terminal_info()
        if info is None:
            raise MT5Error(f"terminal_info failed: {self._api().last_error()}")
        raw = self._as_dict(info)
        return TerminalSnapshot(
            connected=bool(raw.get("connected")),
            path=str(raw["path"]) if raw.get("path") else None,
            company=str(raw["company"]) if raw.get("company") else None,
            max_bars=int(raw["maxbars"]) if raw.get("maxbars") is not None else None,
            raw={key: value for key, value in raw.items() if key.lower() != "login"},
        )

    def account_snapshot(self) -> AccountSnapshot:
        """Read safe account attributes without retaining the account number."""

        api = self._api()
        info = api.account_info()
        if info is None:
            raise MT5Error(f"account_info failed: {api.last_error()}")
        raw = self._as_dict(info)
        mode = raw.get("trade_mode")
        kind = AccountKind.UNKNOWN
        if mode == getattr(api, "ACCOUNT_TRADE_MODE_DEMO", object()):
            kind = AccountKind.DEMO
        elif mode == getattr(api, "ACCOUNT_TRADE_MODE_REAL", object()):
            kind = AccountKind.LIVE
        elif mode == getattr(api, "ACCOUNT_TRADE_MODE_CONTEST", object()):
            kind = AccountKind.CONTEST
        return AccountSnapshot(
            server=str(raw.get("server") or ""),
            company=str(raw.get("company") or ""),
            kind=kind,
            currency=str(raw["currency"]) if raw.get("currency") else None,
        )

    # MT5-04 | Pastikan terminal, account kind, broker, server, dan environment cocok.
    def preflight(
        self, profile: BrokerProfile, environment: Environment
    ) -> tuple[TerminalSnapshot, AccountSnapshot]:
        """Validate terminal, account mode, broker, and server before fetching."""

        profile.validate_environment(environment)
        terminal = self.terminal_snapshot()
        account = self.account_snapshot()
        if not terminal.connected:
            raise MT5PreflightError("MT5 terminal is not connected")
        if self._terminal_path and terminal.path:
            configured = self._terminal_path.resolve()
            actual = Path(terminal.path).resolve()
            configured_install = (
                configured.parent if configured.suffix.lower() == ".exe" else configured
            )
            actual_install = actual.parent if actual.suffix.lower() == ".exe" else actual
            if configured_install != actual_install:
                raise MT5PreflightError(
                    f"terminal path mismatch: configured={configured}, actual={actual}"
                )
        expected_kind = AccountKind.DEMO if environment is Environment.DEMO else AccountKind.LIVE
        if account.kind is not expected_kind:
            raise MT5PreflightError(
                f"account is {account.kind.value}, runtime requires {expected_kind.value}"
            )
        if account.server.casefold() != profile.source.server_name.casefold():
            raise MT5PreflightError(
                "server mismatch: "
                f"actual={account.server!r}, expected={profile.source.server_name!r}"
            )
        if profile.source.broker_name.casefold() not in account.company.casefold():
            raise MT5PreflightError(
                "broker mismatch: "
                f"actual={account.company!r}, expected={profile.source.broker_name!r}"
            )
        return terminal, account

    def available_symbols(self) -> list[str]:
        """List broker symbols for conservative exact symbol resolution."""

        result = self._api().symbols_get()
        if result is None:
            raise MT5Error(f"symbols_get failed: {self._api().last_error()}")
        return [str(item.name) for item in result]

    def ensure_symbol_selected(self, symbol: str) -> None:
        """Make a configured symbol visible for market-data retrieval."""

        if not self._api().symbol_select(symbol, True):
            raise MT5PreflightError(
                f"symbol {symbol!r} cannot be selected: {self._api().last_error()}"
            )

    def symbol_metadata(self, symbol: str) -> SymbolMetadata:
        """Capture a versionable subset and raw snapshot of symbol metadata."""

        info = self._api().symbol_info(symbol)
        if info is None:
            raise MT5PreflightError(
                f"symbol_info failed for {symbol!r}: {self._api().last_error()}"
            )
        raw = self._as_dict(info)

        def dec(name: str) -> Decimal | None:
            value = raw.get(name)
            return Decimal(str(value)) if value is not None else None

        return SymbolMetadata(
            actual_symbol=symbol,
            digits=int(raw["digits"]) if raw.get("digits") is not None else None,
            point=dec("point"),
            trade_tick_size=dec("trade_tick_size"),
            trade_tick_value=dec("trade_tick_value"),
            trade_contract_size=dec("trade_contract_size"),
            volume_min=dec("volume_min"),
            volume_max=dec("volume_max"),
            volume_step=dec("volume_step"),
            currency_base=str(raw["currency_base"]) if raw.get("currency_base") else None,
            currency_profit=(str(raw["currency_profit"]) if raw.get("currency_profit") else None),
            raw_payload=raw,
        )

    # MT5-05 | Fetch bars native dan filter waktu ke interval partition [start, end).
    def fetch_bars(
        self, symbol: str, timeframe: str, partition: TimePartition
    ) -> list[dict[str, Any]]:
        """Retrieve native broker OHLC bars for a half-open UTC partition."""

        api = self._api()
        timeframe_constant = getattr(api, f"TIMEFRAME_{timeframe}", None)
        if timeframe_constant is None:
            raise MT5PreflightError(f"unsupported MT5 timeframe: {timeframe}")
        result = api.copy_rates_range(
            symbol, timeframe_constant, partition.start_utc, partition.end_utc
        )
        if result is None:
            raise MT5Error(f"copy_rates_range failed: {api.last_error()}")
        names = result.dtype.names or ()
        start_epoch = int(partition.start_utc.timestamp())
        end_epoch = int(partition.end_utc.timestamp())
        rows = [{name: row[name].item() for name in names} for row in result]
        return [row for row in rows if start_epoch <= int(row["time"]) < end_epoch]

    # MT5-07 | Snapshot live untuk dashboard: tick terakhir dan N bar terakhir (read-only).
    def latest_tick(self, symbol: str) -> dict[str, Any]:
        """Return the newest Bid/Ask tick known by the terminal for ``symbol``."""

        api = self._api()
        tick = api.symbol_info_tick(symbol)
        if tick is None:
            raise MT5Error(f"symbol_info_tick failed: {api.last_error()}")
        return self._as_dict(tick)

    def recent_bars(self, symbol: str, timeframe: str, count: int) -> list[dict[str, Any]]:
        """Return the newest ``count`` native bars, oldest first.

        The last element is the bar that is still forming at request time.
        """

        api = self._api()
        timeframe_constant = getattr(api, f"TIMEFRAME_{timeframe}", None)
        if timeframe_constant is None:
            raise MT5PreflightError(f"unsupported MT5 timeframe: {timeframe}")
        result = api.copy_rates_from_pos(symbol, timeframe_constant, 0, count)
        if result is None:
            raise MT5Error(f"copy_rates_from_pos failed: {api.last_error()}")
        names = result.dtype.names or ()
        return [{name: row[name].item() for name in names} for row in result]

    # MT5-06 | Fetch historical ticks; identity milidetik dipertahankan untuk normalisasi.
    def fetch_ticks(self, symbol: str, partition: TimePartition) -> list[dict[str, Any]]:
        """Retrieve all native historical tick fields for a UTC partition."""

        api = self._api()
        result = api.copy_ticks_range(
            symbol, partition.start_utc, partition.end_utc, api.COPY_TICKS_ALL
        )
        if result is None:
            raise MT5Error(f"copy_ticks_range failed: {api.last_error()}")
        names = result.dtype.names or ()
        start_ms = int(partition.start_utc.timestamp() * 1000)
        end_ms = int(partition.end_utc.timestamp() * 1000)
        rows = [{name: row[name].item() for name in names} for row in result]
        return [
            row
            for row in rows
            if start_ms <= int(row.get("time_msc", int(row["time"]) * 1000)) < end_ms
        ]


# %% 04. Partition UTC untuk backfill dan resume
# PARTITION-01 | Bars dibagi per bulan, ticks per hari, memakai interval [start, end).
# Timestamp tanpa timezone ditolak; chunk tetap membuat checkpoint dapat dipakai ulang.

def require_aware_utc(value: datetime, name: str = "datetime") -> datetime:
    """Return a datetime in UTC and reject timezone-naive input."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def monthly_partitions(start: datetime, end: datetime) -> list[TimePartition]:
    """Split a requested range at UTC calendar-month boundaries.

    Partial first and last months are retained, which lets callers request an
    arbitrary historical range without fetching outside it.
    """

    cursor = require_aware_utc(start, "start")
    stop = require_aware_utc(end, "end")
    if cursor >= stop:
        raise ValueError("start must be earlier than end")
    partitions: list[TimePartition] = []
    while cursor < stop:
        if cursor.month == 12:
            boundary = datetime(cursor.year + 1, 1, 1, tzinfo=UTC)
        else:
            boundary = datetime(cursor.year, cursor.month + 1, 1, tzinfo=UTC)
        partition_end = min(boundary, stop)
        partitions.append(TimePartition(cursor, partition_end))
        cursor = partition_end
    return partitions


def daily_partitions(start: datetime, end: datetime) -> list[TimePartition]:
    """Split a requested range at UTC midnight boundaries."""

    cursor = require_aware_utc(start, "start")
    stop = require_aware_utc(end, "end")
    if cursor >= stop:
        raise ValueError("start must be earlier than end")
    partitions: list[TimePartition] = []
    while cursor < stop:
        midnight = datetime(cursor.year, cursor.month, cursor.day, tzinfo=UTC)
        boundary = midnight + timedelta(days=1)
        partition_end = min(boundary, stop)
        partitions.append(TimePartition(cursor, partition_end))
        cursor = partition_end
    return partitions


# %% 05. Normalisasi dan validasi bars/ticks
# VALIDATE-01 | Payload MT5 -> UTC/Decimal -> cek invariant -> accepted atau rejected.
# Harga tidak di-fill, diinterpolasi, atau disintesis. Rejection menyimpan alasan audit.

# VALIDATE-02 | Peta durasi candle untuk menurunkan close time dan validasi finalisasi.
TIMEFRAME_MINUTES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}


# VALIDATE-03 | Fingerprint deterministik dipakai untuk idempotency/provenance payload.
def canonical_payload_hash(payload: Mapping[str, Any]) -> str:
    """Create a stable SHA-256 fingerprint from provider-native fields."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def decimal_or_none(value: Any) -> Decimal | None:
    """Convert a provider numeric value to lossless decimal text semantics."""

    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid decimal value: {value!r}") from exc


def _utc_from_epoch(seconds: Any) -> datetime:
    """Convert epoch seconds to a timezone-aware UTC datetime."""

    return datetime.fromtimestamp(int(seconds), tz=UTC)


# VALIDATE-04 | Normalisasi satu candle: waktu UTC, OHLC/volume/spread, dan candle final.
def normalize_bar(
    payload: Mapping[str, Any],
    *,
    actual_symbol: str,
    timeframe: str,
    point: Decimal | None,
    now_utc: datetime,
    finalization_delay_seconds: int,
) -> BarObservation | RejectedRecord:
    """Normalize and validate one native MT5 OHLC bar.

    Native OHLC, tick volume, real volume, and broker spread are preserved.
    A still-open candle is rejected rather than treated as historical truth.
    """

    raw = dict(payload)
    try:
        minutes = TIMEFRAME_MINUTES[timeframe]
        open_time = _utc_from_epoch(payload["time"])
        close_time = open_time + timedelta(minutes=minutes)
        prices = {key: decimal_or_none(payload[key]) for key in ("open", "high", "low", "close")}
        if any(value is None or value <= 0 for value in prices.values()) or (
            prices["high"] < prices["open"]
            or prices["high"] < prices["close"]
            or prices["high"] < prices["low"]
            or prices["low"] > prices["open"]
            or prices["low"] > prices["close"]
        ):
            reason = "INVALID_OHLC"
        else:
            reason = ""
        tick_volume = int(payload.get("tick_volume", 0))
        real_volume = decimal_or_none(payload.get("real_volume"))
        spread_points = int(payload["spread"]) if payload.get("spread") is not None else None
        if tick_volume < 0 or (real_volume is not None and real_volume < 0):
            reason = "NEGATIVE_VOLUME"
        if spread_points is not None and spread_points < 0:
            reason = "NEGATIVE_SPREAD"
        if close_time + timedelta(seconds=finalization_delay_seconds) > now_utc:
            reason = "UNFINISHED_BAR"
        if reason:
            return RejectedRecord("bar", open_time, reason, raw)
        hash_fields = {
            key: payload.get(key)
            for key in (
                "time",
                "open",
                "high",
                "low",
                "close",
                "tick_volume",
                "spread",
                "real_volume",
            )
        }
        return BarObservation(
            actual_symbol=actual_symbol,
            timeframe=timeframe,
            open_time_utc=open_time,
            close_time_utc=close_time,
            open=prices["open"],  # type: ignore[arg-type]
            high=prices["high"],  # type: ignore[arg-type]
            low=prices["low"],  # type: ignore[arg-type]
            close=prices["close"],  # type: ignore[arg-type]
            tick_volume=tick_volume,
            real_volume=real_volume,
            spread_points=spread_points,
            spread_price=(
                Decimal(spread_points) * point
                if spread_points is not None and point is not None
                else None
            ),
            payload_hash=canonical_payload_hash(hash_fields),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raw["_normalization_error"] = str(exc)
        return RejectedRecord("bar", None, "MALFORMED_PAYLOAD", raw)


# VALIDATE-05 | Pisahkan batch bars menjadi accepted dan rejected beserta alasan audit.
def normalize_bars(
    payloads: Iterable[Mapping[str, Any]],
    *,
    actual_symbol: str,
    timeframe: str,
    point: Decimal | None,
    now_utc: datetime,
    finalization_delay_seconds: int,
) -> tuple[list[BarObservation], list[RejectedRecord]]:
    """Normalize a batch of bars without discarding invalid provider records."""

    accepted: list[BarObservation] = []
    rejected: list[RejectedRecord] = []
    for payload in payloads:
        result = normalize_bar(
            payload,
            actual_symbol=actual_symbol,
            timeframe=timeframe,
            point=point,
            now_utc=now_utc,
            finalization_delay_seconds=finalization_delay_seconds,
        )
        (rejected if isinstance(result, RejectedRecord) else accepted).append(result)
    return accepted, rejected


# VALIDATE-06 | Bentuk identity tick (timestamp/payload/occurrence), cek bid/ask/spread.
def normalize_ticks(
    payloads: Iterable[Mapping[str, Any]],
    *,
    actual_symbol: str,
) -> tuple[list[TickObservation], list[RejectedRecord]]:
    """Normalize ticks and assign deterministic duplicate occurrences.

    The occurrence counter is scoped to the fixed fetch partition and payload
    fingerprint.  Re-fetching the same ordered daily partition therefore yields
    the same identity, including multiple identical ticks in one millisecond.
    """

    accepted: list[TickObservation] = []
    rejected: list[RejectedRecord] = []
    occurrences: defaultdict[tuple[int, str], int] = defaultdict(int)
    for payload_value in payloads:
        payload = dict(payload_value)
        try:
            timestamp_ms = int(payload.get("time_msc", int(payload["time"]) * 1000))
            tick_time = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
            bid = decimal_or_none(payload.get("bid"))
            ask = decimal_or_none(payload.get("ask"))
            last = decimal_or_none(payload.get("last"))
            volume = decimal_or_none(payload.get("volume"))
            volume_real = decimal_or_none(payload.get("volume_real"))
            flags = int(payload["flags"]) if payload.get("flags") is not None else None
            if volume is not None and volume < 0 or volume_real is not None and volume_real < 0:
                rejected.append(RejectedRecord("tick", tick_time, "NEGATIVE_VOLUME", payload))
                continue
            has_bid_or_ask = bid not in (None, Decimal(0)) or ask not in (None, Decimal(0))
            if has_bid_or_ask and (bid is None or ask is None or bid <= 0 or ask <= 0):
                rejected.append(RejectedRecord("tick", tick_time, "INVALID_QUOTE", payload))
                continue
            if bid is not None and ask is not None and bid > 0 and ask > 0 and ask < bid:
                rejected.append(RejectedRecord("tick", tick_time, "ASK_BELOW_BID", payload))
                continue
            is_valid_quote = bool(bid and ask and bid > 0 and ask > 0)
            hash_fields = {
                key: payload.get(key)
                for key in ("time_msc", "bid", "ask", "last", "volume", "volume_real", "flags")
            }
            payload_hash = canonical_payload_hash(hash_fields)
            identity = (timestamp_ms, payload_hash)
            occurrence = occurrences[identity]
            occurrences[identity] += 1
            accepted.append(
                TickObservation(
                    actual_symbol=actual_symbol,
                    tick_time_utc=tick_time,
                    timestamp_ms=timestamp_ms,
                    occurrence=occurrence,
                    bid=bid,
                    ask=ask,
                    last=last,
                    volume=volume,
                    volume_real=volume_real,
                    flags=flags,
                    spread_price=ask - bid if is_valid_quote else None,  # type: ignore[operator]
                    is_valid_quote=is_valid_quote,
                    payload_hash=payload_hash,
                    quality_status="valid_quote" if is_valid_quote else "trade_only",
                )
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            payload["_normalization_error"] = str(exc)
            rejected.append(RejectedRecord("tick", None, "MALFORMED_PAYLOAD", payload))
    return accepted, rejected


# %% 06. Derived aggregation M5 dan pemeriksaan gap
# DERIVED-01 | Agregasi M5 ke M15/M30/H1/H4 hanya untuk grup yang lengkap.
# DERIVED-02 | Gap merupakan kandidat untuk diperiksa, termasuk market closure; tidak diisi.
# Data native M5 tidak diubah oleh utility ini.

def aggregate_m5_bars(bars: list[BarObservation], target_timeframe: str) -> list[BarObservation]:
    """Aggregate complete M5 groups into M15, M30, H1, or H4 bars.

    Spread is not summed.  Derived observations retain the first bar's spread
    as ``spread_points`` and ``spread_price``; richer spread OHLC statistics are
    exposed separately by the tick-derived SQL view.
    """

    allowed = {"M15", "M30", "H1", "H4"}
    if target_timeframe not in allowed:
        raise ValueError(f"target timeframe must be one of {sorted(allowed)}")
    target_minutes = TIMEFRAME_MINUTES[target_timeframe]
    expected = target_minutes // 5
    groups: defaultdict[tuple[str, datetime], list[BarObservation]] = defaultdict(list)
    for bar in sorted(bars, key=lambda item: item.open_time_utc):
        if bar.timeframe != "M5":
            raise ValueError("only M5 native bars may be aggregated")
        epoch_minutes = int(bar.open_time_utc.timestamp() // 60)
        bucket_minutes = epoch_minutes - (epoch_minutes % target_minutes)
        bucket = datetime.fromtimestamp(bucket_minutes * 60, tz=UTC)
        groups[(bar.actual_symbol, bucket)].append(bar)

    output: list[BarObservation] = []
    for (symbol, bucket), group in sorted(groups.items(), key=lambda item: item[0][1]):
        if len(group) != expected:
            continue
        expected_times = [bucket + timedelta(minutes=5 * index) for index in range(expected)]
        if [bar.open_time_utc for bar in group] != expected_times:
            continue
        first, last = group[0], group[-1]
        payload = {
            "source": [bar.payload_hash for bar in group],
            "target_timeframe": target_timeframe,
        }
        output.append(
            BarObservation(
                actual_symbol=symbol,
                timeframe=target_timeframe,
                open_time_utc=bucket,
                close_time_utc=bucket + timedelta(minutes=target_minutes),
                open=first.open,
                high=max(bar.high for bar in group),
                low=min(bar.low for bar in group),
                close=last.close,
                tick_volume=sum(bar.tick_volume for bar in group),
                real_volume=(
                    sum((bar.real_volume or Decimal(0) for bar in group), Decimal(0))
                    if any(bar.real_volume is not None for bar in group)
                    else None
                ),
                spread_points=first.spread_points,
                spread_price=first.spread_price,
                payload_hash=canonical_payload_hash(payload),
                quality_status="derived",
            )
        )
    return output


def detect_candidate_gaps(
    bars: list[BarObservation], timeframe: str
) -> list[tuple[datetime, datetime]]:
    """Return temporal gaps for investigation without attempting repair.

    Results are candidates only: weekends, holidays, and broker sessions may
    explain them.  No forward fill, interpolation, or synthetic candle occurs.
    """

    minutes = TIMEFRAME_MINUTES.get(timeframe)
    if minutes is None:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    ordered = sorted({bar.open_time_utc for bar in bars})
    interval = timedelta(minutes=minutes)
    return [
        (left + interval, right)
        for left, right in zip(ordered, ordered[1:], strict=False)
        if right - left > interval
    ]


# %% 07. Ingestion: retry, transaksi, rejection audit, dan checkpoint
# INGEST-01 | Pilih partition dan lewati yang sudah selesai.
# INGEST-02 | Fetch dengan retry terbatas, kemudian normalisasi/validasi.
# INGEST-03 | Simpan data, rejection, dan checkpoint dalam satu transaksi per partition.
# INGEST-04 | Akumulasi counter dan status run completed/partial/failed untuk audit.

class MarketProvider(Protocol):
    """Minimal read-only provider surface required by the ingestion service."""

    def fetch_bars(
        self, symbol: str, timeframe: str, partition: TimePartition
    ) -> list[dict[str, Any]]:
        """Fetch native bars for a half-open partition."""

    def fetch_ticks(self, symbol: str, partition: TimePartition) -> list[dict[str, Any]]:
        """Fetch native ticks for a half-open partition."""


class IngestionService:
    """Coordinate bounded, idempotent, partition-based market collection."""

    def __init__(
        self,
        *,
        database: Database,
        provider: MarketProvider,
        source_id: UUID,
        source_key: str,
        actual_symbol: str,
        symbol_spec_version_id: UUID | None,
        point: Decimal | None,
        collector_version: str,
        max_attempts: int,
        retry_delays_seconds: Sequence[float],
        finalization_delay_seconds: int,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        """Bind one validated source and its collection policy to the service."""

        self.database = database
        self.provider = provider
        self.source_id = source_id
        self.source_key = source_key
        self.actual_symbol = actual_symbol
        self.symbol_spec_version_id = symbol_spec_version_id
        self.point = point
        self.collector_version = collector_version
        self.max_attempts = max_attempts
        self.retry_delays_seconds = list(retry_delays_seconds)
        self.finalization_delay_seconds = finalization_delay_seconds
        self.sleeper = sleeper
        self.ingestion_repository = IngestionRepository()
        self.market_repository = MarketDataRepository()
        self.rejection_repository = RejectionRepository()
        self.log = structlog.get_logger(__name__)

    def ingest_bars(
        self, start_utc: datetime, end_utc: datetime, timeframe: str
    ) -> IngestionCounts:
        """Collect monthly native-bar partitions and resume completed work."""

        return self._collect(
            kind="bars",
            timeframe=timeframe,
            partitions=monthly_partitions(start_utc, end_utc),
        )

    def ingest_ticks(self, start_utc: datetime, end_utc: datetime) -> IngestionCounts:
        """Collect daily historical tick partitions and resume completed work."""

        return self._collect(
            kind="ticks",
            timeframe="",
            partitions=daily_partitions(start_utc, end_utc),
        )

    # INGEST-05 | Jalankan seluruh partition dan resume dari checkpoint yang sudah selesai.
    def _collect(
        self, *, kind: str, timeframe: str, partitions: list[TimePartition]
    ) -> IngestionCounts:
        """Execute a collection run while isolating each partition transaction."""

        start = require_aware_utc(partitions[0].start_utc)
        end = require_aware_utc(partitions[-1].end_utc)
        with self.database.transaction() as session:
            run = self.ingestion_repository.start_run(
                session,
                source_id=self.source_id,
                kind=kind,
                start_utc=start,
                end_utc=end,
                collector_version=self.collector_version,
            )
            run_id = run.id

        totals = IngestionCounts()
        failures: list[Exception] = []
        successful_partitions = 0
        for partition in partitions:
            with self.database.session() as session:
                if self.ingestion_repository.partition_completed(
                    session,
                    source_id=self.source_id,
                    kind=kind,
                    timeframe=timeframe,
                    partition=partition,
                ):
                    self.log.info(
                        "partition_skipped",
                        run_id=str(run_id),
                        source_id=str(self.source_id),
                        source=self.source_key,
                        actual_symbol=self.actual_symbol,
                        timeframe=timeframe or None,
                        operation=f"ingest_{kind}",
                        partition=partition.label,
                        status="already_completed",
                    )
                    continue
            try:
                counts = self._collect_partition(
                    run_id=run_id,
                    kind=kind,
                    timeframe=timeframe,
                    partition=partition,
                )
                totals.add(counts)
                successful_partitions += 1
            except Exception as exc:  # partition failure is recorded and collection resumes
                failures.append(exc)
                with self.database.transaction() as session:
                    self.ingestion_repository.save_checkpoint(
                        session,
                        source_id=self.source_id,
                        run_id=run_id,
                        kind=kind,
                        timeframe=timeframe,
                        partition=partition,
                        status="failed",
                        error=str(exc),
                    )
                self.log.error(
                    "partition_failed",
                    run_id=str(run_id),
                    source_id=str(self.source_id),
                    source=self.source_key,
                    actual_symbol=self.actual_symbol,
                    timeframe=timeframe or None,
                    operation=f"ingest_{kind}",
                    partition=partition.label,
                    status="failed",
                    reason_code=type(exc).__name__,
                )

        with self.database.transaction() as session:
            persisted_run = session.get(IngestionRun, run_id)
            if persisted_run is None:
                raise RuntimeError(f"ingestion run {run_id} disappeared")
            status = (
                "completed" if not failures else ("partial" if successful_partitions else "failed")
            )
            self.ingestion_repository.finish_run(
                session,
                persisted_run,
                status=status,
                counts=totals,
                error=failures[0] if failures else None,
            )
        if failures and not successful_partitions:
            raise failures[0]
        return totals

    # INGEST-06 | Retry fetch/storage sementara; mismatch preflight/ValueError langsung gagal.
    def _collect_partition(
        self,
        *,
        run_id: UUID,
        kind: str,
        timeframe: str,
        partition: TimePartition,
    ) -> IngestionCounts:
        """Fetch with bounded retry, then atomically insert and checkpoint."""

        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                payloads = (
                    self.provider.fetch_bars(self.actual_symbol, timeframe, partition)
                    if kind == "bars"
                    else self.provider.fetch_ticks(self.actual_symbol, partition)
                )
                return self._store_partition(
                    run_id=run_id,
                    kind=kind,
                    timeframe=timeframe,
                    partition=partition,
                    payloads=payloads,
                )
            except (MT5PreflightError, ValueError):
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    break
                delay = self.retry_delays_seconds[attempt - 1]
                self.log.warning(
                    "partition_retry",
                    run_id=str(run_id),
                    source_id=str(self.source_id),
                    source=self.source_key,
                    actual_symbol=self.actual_symbol,
                    timeframe=timeframe or None,
                    operation=f"ingest_{kind}",
                    partition=partition.label,
                    status="retrying",
                    attempt=attempt,
                    retry_delay_seconds=delay,
                    reason_code=type(exc).__name__,
                )
                self.sleeper(delay)
        if last_error is None:
            raise RuntimeError("partition failed without an exception")
        raise last_error

    # INGEST-07 | Normalisasi dulu, lalu insert+rejection+checkpoint secara atomik.
    # Jika transaksi gagal, data dan checkpoint tidak boleh hanya tersimpan sebagian.
    def _store_partition(
        self,
        *,
        run_id: UUID,
        kind: str,
        timeframe: str,
        partition: TimePartition,
        payloads: list[Mapping[str, Any]],
    ) -> IngestionCounts:
        """Validate, bulk insert, audit rejections, and checkpoint atomically."""

        if kind == "bars":
            accepted, rejected = normalize_bars(
                payloads,
                actual_symbol=self.actual_symbol,
                timeframe=timeframe,
                point=self.point,
                now_utc=datetime.now(UTC),
                finalization_delay_seconds=self.finalization_delay_seconds,
            )
        else:
            accepted, rejected = normalize_ticks(
                payloads,
                actual_symbol=self.actual_symbol,
            )
        with self.database.transaction() as session:
            self.market_repository.ensure_partitions(
                session,
                "bar_observation" if kind == "bars" else "tick_observation",
                partition,
            )
            inserted = (
                self.market_repository.insert_bars(
                    session,
                    source_id=self.source_id,
                    symbol_spec_version_id=self.symbol_spec_version_id,
                    bars=accepted,
                )
                if kind == "bars"
                else self.market_repository.insert_ticks(
                    session,
                    source_id=self.source_id,
                    ticks=accepted,
                )
            )
            rejected_count = self.rejection_repository.insert_many(
                session,
                run_id=run_id,
                source_id=self.source_id,
                records=rejected,
            )
            status = "empty" if not payloads else "fetched"
            self.ingestion_repository.save_checkpoint(
                session,
                source_id=self.source_id,
                run_id=run_id,
                kind=kind,
                timeframe=timeframe,
                partition=partition,
                status=status,
            )
        counts = IngestionCounts(
            received=len(payloads),
            inserted=inserted,
            rejected=rejected_count,
            skipped=len(accepted) - inserted,
        )
        self.log.info(
            "partition_completed",
            run_id=str(run_id),
            source_id=str(self.source_id),
            source=self.source_key,
            actual_symbol=self.actual_symbol,
            timeframe=timeframe or None,
            operation=f"ingest_{kind}",
            partition=partition.label,
            status=status,
            rows_received=counts.received,
            rows_inserted=counts.inserted,
            rows_skipped=counts.skipped,
            rows_rejected=counts.rejected,
        )
        return counts


# %% 08. Laporan coverage, status run, dan validasi tersimpan
# REPORT-01 | Bandingkan interval/partition yang diminta dengan actual data tersimpan.
# REPORT-02 | Tampilkan status ingestion dan pelanggaran invariant dari PostgreSQL.
# Query terparameterisasi; laporan ini tidak memanggil MT5 atau mengubah market data.

# REPORT-03 | Coverage menunjukkan requested vs actual serta partition fetched/empty/failed.
def coverage_report(session: Session, source_key: str) -> list[Mapping[str, Any]]:
    """Return requested versus actual coverage for one stable source key."""

    result = session.execute(
        text(
            """
            SELECT d.source_key, d.dataset_symbol, d.actual_symbol,
                   c.kind, c.timeframe, c.requested_start_utc, c.requested_end_utc,
                   c.actual_first_record, c.actual_last_record,
                   c.requested_partitions, c.fetched_partitions,
                   c.empty_partitions, c.failed_partitions, c.observation_count
            FROM market.data_coverage c
            JOIN core.data_source d ON d.id = c.source_id
            WHERE d.source_key = :source_key
            ORDER BY c.kind, c.timeframe
            """
        ),
        {"source_key": source_key},
    )
    return list(result.mappings())


# REPORT-04 | Status berisi run terakhir, counter baris, waktu selesai, dan error audit.
def run_status(session: Session, source_key: str, limit: int = 20) -> list[Mapping[str, Any]]:
    """Return recent ingestion audit rows without sensitive configuration."""

    result = session.execute(
        text(
            """
            SELECT r.id, r.kind, r.requested_start_utc, r.requested_end_utc,
                   r.started_at_utc, r.finished_at_utc, r.status,
                   r.rows_received, r.rows_inserted, r.rows_rejected, r.rows_skipped,
                   r.error_code, r.error_message, r.collector_version
            FROM ingest.run r
            JOIN core.data_source d ON d.id = r.source_id
            WHERE d.source_key = :source_key
            ORDER BY r.started_at_utc DESC
            LIMIT :limit
            """
        ),
        {"source_key": source_key, "limit": limit},
    )
    return list(result.mappings())


# REPORT-05 | Hitung invalid OHLC/quote serta jumlah rejection yang sudah tercatat.
def validation_report(session: Session, source_key: str) -> Mapping[str, Any]:
    """Count persisted invariant violations and recorded rejection reasons."""

    result = session.execute(
        text(
            """
            SELECT
              (SELECT count(*) FROM market.bar_observation b
               JOIN core.data_source d ON d.id = b.source_id
               WHERE d.source_key = :source_key AND (
                 b.open <= 0 OR b.high <= 0 OR b.low <= 0 OR b.close <= 0 OR
                 b.high < greatest(b.open, b.low, b.close) OR
                 b.low > least(b.open, b.close) OR b.tick_volume < 0 OR
                 coalesce(b.real_volume, 0) < 0 OR coalesce(b.spread_points, 0) < 0
               )) AS invalid_bars,
              (SELECT count(*) FROM market.tick_observation t
               JOIN core.data_source d ON d.id = t.source_id
               WHERE d.source_key = :source_key AND t.is_valid_quote AND (
                 t.bid <= 0 OR t.ask <= 0 OR t.ask < t.bid
               )) AS invalid_quotes,
              (SELECT count(*) FROM market.rejection r
               JOIN core.data_source d ON d.id = r.source_id
               WHERE d.source_key = :source_key) AS rejection_count
            """
        ),
        {"source_key": source_key},
    )
    return result.mappings().one()


# %% 09. Logging JSON dan redaksi credential
# LOG-01 | Hilangkan password/DSN/login/token dari event sebelum rendering log JSON.

SENSITIVE_KEYS = frozenset(
    {"password", "database_url", "dsn", "login", "account", "account_number", "token", "secret"}
)


def redact_sensitive_fields(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Remove sensitive values before the event reaches any renderer."""

    for key in list(event_dict):
        if key.lower() in SENSITIVE_KEYS:
            event_dict[key] = "[REDACTED]"
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    """Configure standard logging and structlog for newline-delimited JSON."""

    logging.basicConfig(stream=sys.stdout, level=level.upper(), format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            redact_sensitive_fields,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp_utc"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


# %% 10. CLI: register, inspect, backfill, sync, dan reports
# CLI-01 | load_runtime() membaca profile/settings hanya saat command dijalankan.
# CLI-02 | Perintah ingestion membuka MT5 eksplisit, memakai source yang terdaftar.
# CLI-03 | status/coverage/validate membaca PostgreSQL tanpa membuka terminal MT5.

app = typer.Typer(
    name="market",
    help="Read-only, auditable XAUUSD market-data ingestion for MetaTrader 5.",
    no_args_is_help=True,
)


def parse_utc(value: str) -> datetime:
    """Parse an ISO-8601 timestamp and require an explicit timezone."""

    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise typer.BadParameter("timestamp must include Z or an explicit UTC offset")
    return parsed.astimezone(UTC)


def _json_default(value: Any) -> str:
    """Serialize datetime, decimal, UUID, and path values for CLI output."""

    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, Decimal):
        return str(value)
    return str(value)


def print_json(value: Any) -> None:
    """Emit deterministic, human-readable JSON without exposing settings."""

    typer.echo(json.dumps(value, indent=2, sort_keys=True, default=_json_default))


def load_runtime(profile_name: str) -> tuple[Settings, BrokerProfile]:
    """Load settings/profile, validate environment, and configure safe logging."""

    settings = load_settings()
    profile = load_broker_profile(profile_name)
    profile.validate_environment(settings.environment)
    configure_logging(settings.log_level)
    return settings, profile


@contextmanager
def inspected_provider(
    settings: Settings, profile: BrokerProfile
) -> Iterator[tuple[MT5Provider, str, SymbolMetadata]]:
    """Yield a preflight-validated provider, exact symbol, and metadata.

    MT5 is guaranteed to shut down when the explicit command exits, including
    when preflight or collection fails.
    """

    with MT5Provider(settings.mt5_terminal_path) as provider:
        provider.preflight(profile, settings.environment)
        actual_symbol, _ = resolve_symbol(
            profile.source.requested_symbol, provider.available_symbols()
        )
        provider.ensure_symbol_selected(actual_symbol)
        metadata = provider.symbol_metadata(actual_symbol)
        yield provider, actual_symbol, metadata


# CLI-04 | Cocokkan provenance source terdaftar, simpan versi symbol, lalu bangun service.
def build_ingestion_service(
    *,
    database: Database,
    provider: MT5Provider,
    settings: Settings,
    profile: BrokerProfile,
    actual_symbol: str,
    metadata: SymbolMetadata,
) -> IngestionService:
    """Verify the registered source and assemble its ingestion service."""

    source_repository = SourceRepository()
    with database.transaction() as session:
        source = source_repository.get_by_key(session, profile.source.source_key)
        expected = (
            profile.source.broker_name,
            profile.source.server_name,
            profile.source.environment.value,
            profile.source.dataset_symbol,
            actual_symbol,
        )
        actual = (
            source.broker_name,
            source.server_name,
            source.environment,
            source.dataset_symbol,
            source.actual_symbol,
        )
        if actual != expected:
            raise ValueError("registered source provenance differs from active MT5 source")
        spec_id = source_repository.save_symbol_spec(session, source.id, metadata)
    return IngestionService(
        database=database,
        provider=provider,
        source_id=source.id,
        source_key=source.source_key,
        actual_symbol=actual_symbol,
        symbol_spec_version_id=spec_id,
        point=metadata.point,
        collector_version=settings.collector_version,
        max_attempts=profile.collection.max_attempts,
        retry_delays_seconds=profile.collection.retry_delays_seconds,
        finalization_delay_seconds=profile.collection.finalization_delay_seconds,
    )


@app.command("source-register", help="Validate MT5 identity and register a stable source.")
def source_register(
    profile: str = typer.Option(..., help="Broker profile stem in configs/brokers."),
) -> None:
    """Preflight the active terminal and persist source plus symbol provenance."""

    settings, broker_profile = load_runtime(profile)
    database = Database(settings.database_url)
    try:
        with inspected_provider(settings, broker_profile) as (_, symbol, metadata):
            with database.transaction() as session:
                repository = SourceRepository()
                source = repository.register(session, broker_profile, symbol)
                spec_id = repository.save_symbol_spec(session, source.id, metadata)
            print_json(
                {
                    "source_id": source.id,
                    "source_key": source.source_key,
                    "actual_symbol": symbol,
                    "symbol_spec_version_id": spec_id,
                    "status": "registered",
                }
            )
    finally:
        database.dispose()


@app.command("mt5-info", help="Validate and display safe MT5 terminal/account metadata.")
def mt5_info(
    profile: str = typer.Option(..., help="Broker profile stem in configs/brokers."),
) -> None:
    """Run mandatory preflight without displaying account number or secrets."""

    settings, broker_profile = load_runtime(profile)
    with MT5Provider(settings.mt5_terminal_path) as provider:
        terminal, account = provider.preflight(broker_profile, settings.environment)
        print_json(
            {
                "terminal": {
                    "connected": terminal.connected,
                    "path": terminal.path,
                    "company": terminal.company,
                    "max_bars": terminal.max_bars,
                },
                "account": asdict(account),
                "history_warning": (
                    "Terminal max-bars/history availability can limit backfill; "
                    "coverage must be verified after ingestion."
                ),
            }
        )


@app.command("symbol-info", help="Resolve and display configured symbol metadata.")
def symbol_info(
    profile: str = typer.Option(..., help="Broker profile stem in configs/brokers."),
) -> None:
    """Validate the configured exact symbol and display its current specification."""

    settings, broker_profile = load_runtime(profile)
    with inspected_provider(settings, broker_profile) as (_, _, metadata):
        safe = asdict(metadata)
        safe.pop("raw_payload", None)
        print_json(safe)


# CLI-05 | Backfill bars: load profile -> MT5 preflight -> ingestion -> tutup koneksi.
def _collect_bars(profile: str, start: datetime, end: datetime, timeframe: str) -> None:
    """Shared explicit bar-collection command implementation."""

    settings, broker_profile = load_runtime(profile)
    database = Database(settings.database_url)
    try:
        with inspected_provider(settings, broker_profile) as (provider, symbol, metadata):
            service = build_ingestion_service(
                database=database,
                provider=provider,
                settings=settings,
                profile=broker_profile,
                actual_symbol=symbol,
                metadata=metadata,
            )
            print_json(asdict(service.ingest_bars(start, end, timeframe)))
    finally:
        database.dispose()


# CLI-06 | Backfill ticks memakai alur yang sama dengan partition harian.
def _collect_ticks(profile: str, start: datetime, end: datetime) -> None:
    """Shared explicit tick-collection command implementation."""

    settings, broker_profile = load_runtime(profile)
    database = Database(settings.database_url)
    try:
        with inspected_provider(settings, broker_profile) as (provider, symbol, metadata):
            service = build_ingestion_service(
                database=database,
                provider=provider,
                settings=settings,
                profile=broker_profile,
                actual_symbol=symbol,
                metadata=metadata,
            )
            print_json(asdict(service.ingest_ticks(start, end)))
    finally:
        database.dispose()


@app.command("ingest-bars", help="Backfill native MT5 bars in monthly partitions.")
def ingest_bars(
    profile: str = typer.Option(..., help="Broker profile stem."),
    start: str = typer.Option(..., help="Inclusive ISO-8601 UTC start."),
    end: str = typer.Option(..., help="Exclusive ISO-8601 UTC end."),
    timeframe: str = typer.Option("M5", help="Native MT5 timeframe, normally M5."),
) -> None:
    """Collect a requested native-bar range with checkpoint/resume semantics."""

    _collect_bars(profile, parse_utc(start), parse_utc(end), timeframe.upper())


@app.command("ingest-ticks", help="Backfill native MT5 ticks in daily partitions.")
def ingest_ticks(
    profile: str = typer.Option(..., help="Broker profile stem."),
    start: str = typer.Option(..., help="Inclusive ISO-8601 UTC start."),
    end: str = typer.Option(..., help="Exclusive ISO-8601 UTC end."),
) -> None:
    """Collect all ticks available for the requested range and record coverage."""

    _collect_ticks(profile, parse_utc(start), parse_utc(end))


def _latest_timestamp(database: Database, source_key: str, kind: str, timeframe: str) -> datetime:
    """Read the latest stored event time used to seed incremental collection."""

    sql = (
        "SELECT max(b.open_time_utc) FROM market.bar_latest b "
        "JOIN core.data_source d ON d.id=b.source_id "
        "WHERE d.source_key=:source_key AND b.timeframe=:timeframe"
        if kind == "bars"
        else "SELECT max(t.tick_time_utc) FROM market.tick_observation t "
        "JOIN core.data_source d ON d.id=t.source_id WHERE d.source_key=:source_key"
    )
    with database.session() as session:
        latest = session.scalar(text(sql), {"source_key": source_key, "timeframe": timeframe})
    if latest is None:
        raise typer.BadParameter("no stored history; run the corresponding ingest command first")
    return latest.astimezone(UTC)


@app.command("sync-bars", help="Fetch finalized bars newer than existing history.")
def sync_bars(
    profile: str = typer.Option(..., help="Broker profile stem."),
    timeframe: str = typer.Option("M5", help="Native MT5 timeframe."),
) -> None:
    """Incrementally collect bars from the latest logical candle through now."""

    settings, broker_profile = load_runtime(profile)
    database = Database(settings.database_url)
    try:
        start = _latest_timestamp(
            database, broker_profile.source.source_key, "bars", timeframe.upper()
        )
    finally:
        database.dispose()
    _collect_bars(profile, start, datetime.now(UTC), timeframe.upper())


@app.command("sync-ticks", help="Fetch ticks from the current stored UTC day.")
def sync_ticks(
    profile: str = typer.Option(..., help="Broker profile stem."),
) -> None:
    """Incrementally re-fetch the latest UTC day for deterministic tick identity."""

    settings, broker_profile = load_runtime(profile)
    database = Database(settings.database_url)
    try:
        latest = _latest_timestamp(database, broker_profile.source.source_key, "ticks", "")
        start = datetime(latest.year, latest.month, latest.day, tzinfo=UTC)
    finally:
        database.dispose()
    _collect_ticks(profile, start, datetime.now(UTC))


# CLI-07 | Laporan database menggunakan source_key profile dan tidak membuka MT5.
def _database_report(profile: str, report: str) -> None:
    """Execute a read-only report command without contacting MT5."""

    settings, broker_profile = load_runtime(profile)
    database = Database(settings.database_url)
    try:
        with database.session() as session:
            if report == "status":
                rows: Any = run_status(session, broker_profile.source.source_key)
            elif report == "coverage":
                rows = coverage_report(session, broker_profile.source.source_key)
            else:
                rows = validation_report(session, broker_profile.source.source_key)
            print_json([dict(row) for row in rows] if isinstance(rows, list) else dict(rows))
    finally:
        database.dispose()


@app.command("status", help="Show recent ingestion runs from PostgreSQL.")
def status(profile: str = typer.Option(..., help="Broker profile stem.")) -> None:
    """Display recent run states and counters for a source."""

    _database_report(profile, "status")


@app.command("coverage", help="Show requested and actual data coverage.")
def coverage(profile: str = typer.Option(..., help="Broker profile stem.")) -> None:
    """Display partition and observation coverage for a source."""

    _database_report(profile, "coverage")


@app.command("validate", help="Check persisted invariants and rejection counts.")
def validate(profile: str = typer.Option(..., help="Broker profile stem.")) -> None:
    """Report stored invalid-row counts without mutating the dataset."""

    _database_report(profile, "validate")


if __name__ == "__main__":
    # Run the canonical module so repository imports share the same collector classes.
    from market.collector import app as cli_app

    cli_app()
