"""Manage database sessions and persistence in notebook-style sections.

Sections cover lazy engine creation, explicit session/transaction contexts,
source/spec registration, bounded bars/ticks inserts, run/checkpoint updates
and rejection audit. Table mappings live in database.schema; versioned schema
changes remain under database/migrations for Alembic.

Flow: normalized market records -> repositories -> caller-owned transaction ->
PostgreSQL observations/audit/checkpoint. Imports never create an engine, read
runtime credentials, initialize MT5, connect to PostgreSQL or apply migrations.
"""

from __future__ import annotations

# %% 01. Import dan dependensi storage
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy import insert as generic_insert
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from database.schema import (
    BarObservationModel,
    Checkpoint,
    DataSource,
    IngestionRun,
    Rejection,
    SymbolSpecVersion,
    TickObservationModel,
)
from market.contracts import (
    BarObservation,
    IngestionCounts,
    RejectedRecord,
    SymbolMetadata,
    TickObservation,
    TimePartition,
)
from market.profile import BrokerProfile, load_settings

# %% 02. Engine, session, transaksi, dan cleanup koneksi
# CONNECT-01 | get_engine() membentuk engine lazy; Database memiliki lifecycle koneksi.
# TRANSACTION-01 | session() hanya menutup session; transaction() melakukan commit/rollback.

def get_engine(
    database_url: str | None = None,
    *,
    connect_timeout_seconds: int | None = None,
) -> Engine:
    """Create a lazy PostgreSQL engine from a URL or validated settings.

    Analytics uses this shared entry point and opens read-only transactions.
    Creating the engine does not establish a database connection immediately.

    Args:
        database_url: Explicit SQLAlchemy URL, or ``None`` to read
            ``GOLD_DATABASE_URL`` through application settings.
        connect_timeout_seconds: Optional driver-level connection timeout.

    Returns:
        SQLAlchemy engine with connection health checks and hidden parameters.
    """

    connect_args: dict[str, Any] = {}
    if connect_timeout_seconds is not None:
        connect_args["connect_timeout"] = connect_timeout_seconds
    # CONNECT-02 | Gunakan URL eksplisit; baca settings hanya jika URL tidak diberikan.
    resolved_url = database_url or load_settings().database_url
    return create_engine(
        resolved_url,
        pool_pre_ping=True,
        hide_parameters=True,
        connect_args=connect_args,
    )


class Database:
    """Own the SQLAlchemy engine and produce explicit transaction scopes."""

    def __init__(
        self,
        database_url: str,
        *,
        connect_timeout_seconds: int | None = None,
    ) -> None:
        """Configure a PostgreSQL engine without eagerly connecting.

        Args:
            database_url:
                SQLAlchemy PostgreSQL connection URL.

            connect_timeout_seconds:
                Optional driver-level connection timeout. Interactive callers
                can fail quickly while ingestion keeps the driver default.
        """

        self.engine: Engine = get_engine(
            database_url,
            connect_timeout_seconds=connect_timeout_seconds,
        )
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    @contextmanager
    # TRANSACTION-02 | Context read/query menutup session tanpa commit implisit.
    def session(self) -> Iterator[Session]:
        """Yield a session and always close it without implicit commit."""

        with self._sessions() as session:
            yield session

    @contextmanager
    # TRANSACTION-03 | Unit kerja atomik: commit saat berhasil, rollback saat exception.
    def transaction(self) -> Iterator[Session]:
        """Commit a complete unit of work or roll it back on any exception."""

        with self._sessions.begin() as session:
            yield session

    def dispose(self) -> None:
        """Release pooled PostgreSQL connections during CLI shutdown."""

        self.engine.dispose()


# %% 03. Source identity dan versi symbol specification
# SOURCE-01 | Verifikasi provenance source_key dan tolak pemakaian identitas yang berbeda.
# SOURCE-02 | Simpan versi metadata baru atau gunakan versi payload yang identik.

class SourceIdentityConflict(ValueError):
    """Raised when a stable source key is reused for different provenance."""


class SourceRepository:
    """Register source identities and append versioned symbol metadata."""

    # SOURCE-03 | Source lama wajib memiliki broker/server/environment/symbol yang identik.
    def register(self, session: Session, profile: BrokerProfile, actual_symbol: str) -> DataSource:
        """Create or verify a source without changing its provenance identity."""

        source = session.scalar(
            select(DataSource).where(DataSource.source_key == profile.source.source_key)
        )
        identity = (
            profile.source.broker_name,
            profile.source.server_name,
            profile.source.environment.value,
            profile.source.dataset_symbol,
            actual_symbol,
        )
        if source is not None:
            existing = (
                source.broker_name,
                source.server_name,
                source.environment,
                source.dataset_symbol,
                source.actual_symbol,
            )
            if existing != identity:
                raise SourceIdentityConflict(
                    f"source_key {source.source_key!r} already has different provenance"
                )
            return source
        source = DataSource(
            source_key=profile.source.source_key,
            broker_name=identity[0],
            server_name=identity[1],
            environment=identity[2],
            dataset_symbol=identity[3],
            actual_symbol=identity[4],
            created_at_utc=datetime.now(UTC),
            active=True,
        )
        session.add(source)
        session.flush()
        return source

    def get_by_key(self, session: Session, source_key: str) -> DataSource:
        """Return an active source by stable key or raise a clear error."""

        source = session.scalar(
            select(DataSource).where(
                DataSource.source_key == source_key, DataSource.active.is_(True)
            )
        )
        if source is None:
            raise LookupError(f"source {source_key!r} is not registered")
        return source

    # SOURCE-04 | Payload hash identik memakai versi lama; perubahan menghasilkan versi baru.
    def save_symbol_spec(self, session: Session, source_id: UUID, metadata: SymbolMetadata) -> UUID:
        """Insert a changed symbol specification or reuse an identical version."""

        # Delay this import: the collector also imports SourceRepository for its CLI.
        from market.collector import canonical_payload_hash

        payload_hash = canonical_payload_hash(metadata.raw_payload)
        existing = session.scalar(
            select(SymbolSpecVersion.id).where(
                SymbolSpecVersion.source_id == source_id,
                SymbolSpecVersion.payload_hash == payload_hash,
            )
        )
        if existing is not None:
            return existing
        values = {
            "source_id": source_id,
            "observed_at_utc": datetime.now(UTC),
            "actual_symbol": metadata.actual_symbol,
            "digits": metadata.digits,
            "point": metadata.point,
            "trade_tick_size": metadata.trade_tick_size,
            "trade_tick_value": metadata.trade_tick_value,
            "trade_contract_size": metadata.trade_contract_size,
            "volume_min": metadata.volume_min,
            "volume_max": metadata.volume_max,
            "volume_step": metadata.volume_step,
            "currency_base": metadata.currency_base,
            "currency_profit": metadata.currency_profit,
            "payload_hash": payload_hash,
            "raw_payload": metadata.raw_payload,
        }
        statement = (
            insert(SymbolSpecVersion)
            .values(**values)
            .on_conflict_do_nothing(constraint="uq_symbol_spec_payload")
            .returning(SymbolSpecVersion.id)
        )
        inserted = session.scalar(statement)
        if inserted is not None:
            return inserted
        existing = session.scalar(
            select(SymbolSpecVersion.id).where(
                SymbolSpecVersion.source_id == source_id,
                SymbolSpecVersion.payload_hash == payload_hash,
            )
        )
        if existing is None:
            raise RuntimeError("symbol specification insert did not return an identity")
        return existing


# %% 04. Partition tabel dan bulk insert bars/ticks
# STORE-01 | Pastikan partition bulanan tersedia sebelum insert data.
# STORE-02 | Bagi batch maksimal 1.000 baris agar tidak melewati batas parameter PostgreSQL.
# STORE-03 | Lewati payload identik; revisi bars dan occurrence ticks tetap dipertahankan.

_MAX_INSERT_ROWS = 1_000


def _chunks[T](items: Sequence[T], size: int = _MAX_INSERT_ROWS) -> Iterator[Sequence[T]]:
    """Yield bounded slices used to keep bulk statements below PostgreSQL limits.

    Args:
        items: Materialized records ready for one insert operation.
        size: Maximum records per SQL statement.

    Yields:
        Non-empty slices preserving the source order.

    Raises:
        ValueError: If ``size`` is not positive.
    """

    if size <= 0:
        raise ValueError("chunk size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


class MarketDataRepository:
    """Ensure monthly partitions and insert idempotent observation batches.

    Data flow:
        MetaTrader 5 -> normalization -> this repository -> PostgreSQL

    Every insert method uses bounded statements.  PostgreSQL commits the
    complete partition transaction in the ingestion service, so a failed
    chunk rolls back the partition rather than leaving a partial checkpoint.
    """

    # STORE-04 | Helper PostgreSQL membuat partition bulanan untuk interval data yang diminta.
    def ensure_partitions(
        self, session: Session, table_name: str, partition: TimePartition
    ) -> None:
        """Ask the database helper to create all required monthly partitions."""

        if table_name not in {"bar_observation", "tick_observation"}:
            raise ValueError("unsupported partitioned table")
        session.execute(
            text("SELECT market.ensure_monthly_partitions(:table_name, :start, :end)"),
            {
                "table_name": table_name,
                "start": partition.start_utc,
                "end": partition.end_utc,
            },
        )

    # STORE-05 | Append bars per batch; constraint payload identik mencegah duplikasi refetch.
    def insert_bars(
        self,
        session: Session,
        *,
        source_id: UUID,
        symbol_spec_version_id: UUID | None,
        bars: list[BarObservation],
    ) -> int:
        """Bulk insert bars, skipping exact duplicates while retaining revisions."""

        if not bars:
            return 0
        now = datetime.now(UTC)
        rows = [
            {
                "id": uuid4(),
                "source_id": source_id,
                "symbol_spec_version_id": symbol_spec_version_id,
                "actual_symbol": bar.actual_symbol,
                "timeframe": bar.timeframe,
                "open_time_utc": bar.open_time_utc,
                "close_time_utc": bar.close_time_utc,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "tick_volume": bar.tick_volume,
                "real_volume": bar.real_volume,
                "spread_points": bar.spread_points,
                "spread_price": bar.spread_price,
                "ingested_at_utc": now,
                "payload_hash": bar.payload_hash,
                "quality_status": bar.quality_status,
            }
            for bar in bars
        ]
        inserted = 0
        for batch in _chunks(rows):
            statement = (
                insert(BarObservationModel)
                .values(batch)
                .on_conflict_do_nothing(constraint="uq_bar_exact_payload")
                .returning(BarObservationModel.id)
            )
            inserted += len(session.scalars(statement).all())
        return inserted

    # STORE-06 | Append ticks dengan hash+occurrence; tick identik pada milidetik sama tetap unik.
    def insert_ticks(
        self,
        session: Session,
        *,
        source_id: UUID,
        ticks: list[TickObservation],
    ) -> int:
        """Bulk insert ticks using deterministic fingerprint and occurrence identity."""

        if not ticks:
            return 0
        now = datetime.now(UTC)
        rows = [
            {
                "id": uuid4(),
                "source_id": source_id,
                "actual_symbol": tick.actual_symbol,
                "tick_time_utc": tick.tick_time_utc,
                "timestamp_ms": tick.timestamp_ms,
                "occurrence": tick.occurrence,
                "bid": tick.bid,
                "ask": tick.ask,
                "last": tick.last,
                "volume": tick.volume,
                "volume_real": tick.volume_real,
                "flags": tick.flags,
                "spread_price": tick.spread_price,
                "is_valid_quote": tick.is_valid_quote,
                "payload_hash": tick.payload_hash,
                "quality_status": tick.quality_status,
                "ingested_at_utc": now,
            }
            for tick in ticks
        ]
        inserted = 0
        for batch in _chunks(rows):
            statement = (
                insert(TickObservationModel)
                .values(batch)
                .on_conflict_do_nothing(constraint="uq_tick_deterministic_identity")
                .returning(TickObservationModel.id)
            )
            inserted += len(session.scalars(statement).all())
        return inserted


# %% 05. Ingestion run, counter, dan upsert checkpoint
# CHECKPOINT-01 | Start/finish run menyimpan status, counter, dan error audit.
# CHECKPOINT-02 | Hanya checkpoint fetched/empty yang dianggap selesai untuk resume.
# CHECKPOINT-03 | Upsert checkpoint memakai session transaksi yang diberikan collector.

class IngestionRepository:
    """Maintain collection audit records and partition resume state."""

    def start_run(
        self,
        session: Session,
        *,
        source_id: UUID,
        kind: str,
        start_utc: datetime | None,
        end_utc: datetime | None,
        collector_version: str,
    ) -> IngestionRun:
        """Create a running ingestion audit row before provider access."""

        run = IngestionRun(
            source_id=source_id,
            kind=kind,
            requested_start_utc=start_utc,
            requested_end_utc=end_utc,
            started_at_utc=datetime.now(UTC),
            status="running",
            rows_received=0,
            rows_inserted=0,
            rows_rejected=0,
            rows_skipped=0,
            collector_version=collector_version,
        )
        session.add(run)
        session.flush()
        return run

    def finish_run(
        self,
        session: Session,
        run: IngestionRun,
        *,
        status: str,
        counts: IngestionCounts,
        error: Exception | None = None,
    ) -> None:
        """Finalize run counters and sanitized error details."""

        run.finished_at_utc = datetime.now(UTC)
        run.status = status
        run.rows_received = counts.received
        run.rows_inserted = counts.inserted
        run.rows_rejected = counts.rejected
        run.rows_skipped = counts.skipped
        if error is not None:
            run.error_code = type(error).__name__
            run.error_message = str(error)[:2000]

    # CHECKPOINT-04 | Partition failed/partial masih dapat diulang; fetched/empty dilewati.
    def partition_completed(
        self,
        session: Session,
        *,
        source_id: UUID,
        kind: str,
        timeframe: str,
        partition: TimePartition,
    ) -> bool:
        """Return whether a partition is already fetched or confirmed empty."""

        status = session.scalar(
            select(Checkpoint.status).where(
                Checkpoint.source_id == source_id,
                Checkpoint.kind == kind,
                Checkpoint.timeframe == timeframe,
                Checkpoint.partition_start_utc == partition.start_utc,
                Checkpoint.partition_end_utc == partition.end_utc,
            )
        )
        return status in {"fetched", "empty"}

    # CHECKPOINT-05 | Upsert status partition dalam transaksi caller, bersama data dan audit.
    def save_checkpoint(
        self,
        session: Session,
        *,
        source_id: UUID,
        run_id: UUID,
        kind: str,
        timeframe: str,
        partition: TimePartition,
        status: str,
        error: str | None = None,
    ) -> None:
        """Upsert partition state within the same transaction as inserted data."""

        now = datetime.now(UTC)
        values = {
            "source_id": source_id,
            "run_id": run_id,
            "kind": kind,
            "timeframe": timeframe,
            "partition_start_utc": partition.start_utc,
            "partition_end_utc": partition.end_utc,
            "status": status,
            "last_success_at_utc": now if status in {"fetched", "empty"} else None,
            "last_error": error[:2000] if error else None,
        }
        statement = insert(Checkpoint).values(**values)
        statement = statement.on_conflict_do_update(
            constraint="uq_checkpoint_partition",
            set_={
                "run_id": statement.excluded.run_id,
                "status": statement.excluded.status,
                "last_success_at_utc": statement.excluded.last_success_at_utc,
                "last_error": statement.excluded.last_error,
            },
        )
        session.execute(statement)


# %% 06. Penyimpanan payload yang ditolak
# AUDIT-01 | Simpan rejection beserta reason code/run/source dalam transaksi caller.
# Record invalid tidak masuk ke tabel bars/ticks valid.

class RejectionRepository:
    """Persist rejected payloads without contaminating valid market tables."""

    # AUDIT-02 | Payload invalid disimpan bersama alasan penolakan serta source/run terkait.
    def insert_many(
        self,
        session: Session,
        *,
        run_id: UUID,
        source_id: UUID,
        records: list[RejectedRecord],
    ) -> int:
        """Bulk insert rejected records and their machine-readable reason codes."""

        if not records:
            return 0
        rejected_at = datetime.now(UTC)
        rows = [
            {
                "id": uuid4(),
                "run_id": run_id,
                "source_id": source_id,
                "record_kind": record.record_kind,
                "event_time_utc": record.event_time_utc,
                "reason_code": record.reason_code,
                "raw_payload": record.raw_payload,
                "rejected_at_utc": rejected_at,
            }
            for record in records
        ]
        session.execute(generic_insert(Rejection), rows)
        return len(rows)
