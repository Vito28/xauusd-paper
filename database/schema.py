"""Define PostgreSQL tables and relationships in notebook-style sections.

The mappings mirror the Alembic DDL.  Raw bars and ticks are append-only
partitioned observations; current-state and aggregate access is provided by SQL
views defined in the migration rather than hidden ORM behavior.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# %% 01. Metadata root untuk ORM dan Alembic
# SCHEMA-01 | Deklarasi tabel tidak membuka koneksi atau membuat tabel saat import.

class Base(DeclarativeBase):
    """Declarative metadata root used by Alembic and repository mappings."""


# %% 02. Identitas source broker/server/environment
# SCHEMA-02 | Satu source_key mengikat provenance dataset ke broker dan symbol aktual.

class DataSource(Base):
    """Stable broker/server/environment/symbol provenance identity."""

    __tablename__ = "data_source"
    __table_args__ = (
        CheckConstraint("environment IN ('demo', 'live')", name="ck_source_environment"),
        {"schema": "core"},
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    source_key: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    broker_name: Mapped[str] = mapped_column(Text, nullable=False)
    server_name: Mapped[str] = mapped_column(Text, nullable=False)
    environment: Mapped[str] = mapped_column(Text, nullable=False)
    dataset_symbol: Mapped[str] = mapped_column(Text, nullable=False)
    actual_symbol: Mapped[str] = mapped_column(Text, nullable=False)
    created_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


# %% 03. Versi spesifikasi symbol
# SCHEMA-03 | Metadata symbol disimpan per versi payload hash, bukan ditimpa.

class SymbolSpecVersion(Base):
    """Versioned broker symbol specification observed during preflight."""

    __tablename__ = "symbol_spec_version"
    __table_args__ = (
        UniqueConstraint("source_id", "payload_hash", name="uq_symbol_spec_payload"),
        {"schema": "core"},
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    source_id: Mapped[UUID] = mapped_column(ForeignKey("core.data_source.id"), nullable=False)
    observed_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    actual_symbol: Mapped[str] = mapped_column(Text, nullable=False)
    digits: Mapped[int | None] = mapped_column(Integer)
    point: Mapped[Decimal | None] = mapped_column(Numeric(20, 10))
    trade_tick_size: Mapped[Decimal | None] = mapped_column(Numeric(20, 10))
    trade_tick_value: Mapped[Decimal | None] = mapped_column(Numeric(30, 10))
    trade_contract_size: Mapped[Decimal | None] = mapped_column(Numeric(30, 10))
    volume_min: Mapped[Decimal | None] = mapped_column(Numeric(30, 10))
    volume_max: Mapped[Decimal | None] = mapped_column(Numeric(30, 10))
    volume_step: Mapped[Decimal | None] = mapped_column(Numeric(30, 10))
    currency_base: Mapped[str | None] = mapped_column(Text)
    currency_profit: Mapped[str | None] = mapped_column(Text)
    payload_hash: Mapped[str] = mapped_column(Text, nullable=False)
    raw_payload: Mapped[dict[str, object] | None] = mapped_column(JSONB)


# %% 04. Audit ingestion run
# SCHEMA-04 | Simpan rentang request, waktu run, status, counter baris, dan error.

class IngestionRun(Base):
    """Audit record and aggregate counters for one collection invocation."""

    __tablename__ = "run"
    __table_args__ = (
        CheckConstraint("kind IN ('bars', 'ticks', 'symbol_spec')", name="ck_run_kind"),
        CheckConstraint(
            "status IN ('running', 'completed', 'partial', 'failed')",
            name="ck_run_status",
        ),
        {"schema": "ingest"},
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    source_id: Mapped[UUID] = mapped_column(ForeignKey("core.data_source.id"), nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    requested_start_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    requested_end_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(Text, nullable=False)
    rows_received: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    rows_inserted: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    rows_rejected: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    rows_skipped: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    error_code: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    collector_version: Mapped[str | None] = mapped_column(Text)


# %% 05. Checkpoint partition dan resume state
# SCHEMA-05 | Identitas partition unik memastikan status resume dapat diperiksa ulang.

class Checkpoint(Base):
    """Authoritative resume state for one deterministic fetch partition."""

    __tablename__ = "checkpoint"
    __table_args__ = (
        CheckConstraint(
            "status IN ('fetched', 'empty', 'partial', 'failed')",
            name="ck_checkpoint_status",
        ),
        UniqueConstraint(
            "source_id",
            "kind",
            "timeframe",
            "partition_start_utc",
            "partition_end_utc",
            name="uq_checkpoint_partition",
        ),
        {"schema": "ingest"},
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    source_id: Mapped[UUID] = mapped_column(ForeignKey("core.data_source.id"), nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    timeframe: Mapped[str] = mapped_column(Text, nullable=False, default="")
    partition_start_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    partition_end_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    last_success_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("ingest.run.id"), nullable=False)


# %% 06. Observasi bars append-only
# SCHEMA-06 | OHLC native/derived dipartisi menurut open_time_utc.
# Hash payload membedakan revisi harga dari refetch payload identik.

class BarObservationModel(Base):
    """Append-only native or explicitly derived OHLC observation."""

    __tablename__ = "bar_observation"
    __table_args__ = (
        PrimaryKeyConstraint("open_time_utc", "id", name="pk_bar_observation"),
        UniqueConstraint(
            "open_time_utc",
            "source_id",
            "actual_symbol",
            "timeframe",
            "payload_hash",
            name="uq_bar_exact_payload",
        ),
        {"schema": "market", "postgresql_partition_by": "RANGE (open_time_utc)"},
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), default=uuid4)
    source_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    symbol_spec_version_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True))
    actual_symbol: Mapped[str] = mapped_column(Text, nullable=False)
    timeframe: Mapped[str] = mapped_column(Text, nullable=False)
    open_time_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    close_time_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[Decimal] = mapped_column(Numeric(20, 10), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(20, 10), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(20, 10), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(20, 10), nullable=False)
    tick_volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    real_volume: Mapped[Decimal | None] = mapped_column(Numeric(30, 10))
    spread_points: Mapped[int | None] = mapped_column(BigInteger)
    spread_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 10))
    ingested_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload_hash: Mapped[str] = mapped_column(Text, nullable=False)
    quality_status: Mapped[str] = mapped_column(Text, nullable=False)


# %% 07. Observasi ticks append-only
# SCHEMA-07 | Timestamp, hash, dan occurrence menjaga identity tick secara deterministik.

class TickObservationModel(Base):
    """Append-only native tick with fingerprint plus occurrence identity."""

    __tablename__ = "tick_observation"
    __table_args__ = (
        PrimaryKeyConstraint("tick_time_utc", "id", name="pk_tick_observation"),
        UniqueConstraint(
            "tick_time_utc",
            "source_id",
            "actual_symbol",
            "payload_hash",
            "occurrence",
            name="uq_tick_deterministic_identity",
        ),
        {"schema": "market", "postgresql_partition_by": "RANGE (tick_time_utc)"},
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), default=uuid4)
    source_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    actual_symbol: Mapped[str] = mapped_column(Text, nullable=False)
    tick_time_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    occurrence: Mapped[int] = mapped_column(Integer, nullable=False)
    bid: Mapped[Decimal | None] = mapped_column(Numeric(20, 10))
    ask: Mapped[Decimal | None] = mapped_column(Numeric(20, 10))
    last: Mapped[Decimal | None] = mapped_column(Numeric(20, 10))
    volume: Mapped[Decimal | None] = mapped_column(Numeric(30, 10))
    volume_real: Mapped[Decimal | None] = mapped_column(Numeric(30, 10))
    flags: Mapped[int | None] = mapped_column(BigInteger)
    spread_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 10))
    is_valid_quote: Mapped[bool] = mapped_column(Boolean, nullable=False)
    payload_hash: Mapped[str] = mapped_column(Text, nullable=False)
    quality_status: Mapped[str] = mapped_column(Text, nullable=False)
    ingested_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


# %% 08. Rejection audit
# SCHEMA-08 | Simpan payload yang ditolak dan reason code terpisah dari data valid.

class Rejection(Base):
    """Invalid provider record retained with reason and raw diagnostic payload."""

    __tablename__ = "rejection"
    __table_args__ = ({"schema": "market"},)

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("ingest.run.id"), nullable=False)
    source_id: Mapped[UUID] = mapped_column(ForeignKey("core.data_source.id"), nullable=False)
    record_kind: Mapped[str] = mapped_column(Text, nullable=False)
    event_time_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reason_code: Mapped[str] = mapped_column(Text, nullable=False)
    raw_payload: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    rejected_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
