"""Create the auditable multi-broker XAUUSD ingestion schema.

Revision ID: 20260929_0001
Revises: None
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260929_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create schemas, partitioned observations, helpers, indexes, and views."""

    op.execute("CREATE SCHEMA IF NOT EXISTS core")
    op.execute("CREATE SCHEMA IF NOT EXISTS ingest")
    op.execute("CREATE SCHEMA IF NOT EXISTS market")
    op.execute("CREATE SCHEMA IF NOT EXISTS audit")
    op.execute(
        """
        CREATE TABLE core.data_source (
            id UUID PRIMARY KEY,
            source_key TEXT UNIQUE NOT NULL,
            broker_name TEXT NOT NULL,
            server_name TEXT NOT NULL,
            environment TEXT NOT NULL CHECK (environment IN ('demo', 'live')),
            dataset_symbol TEXT NOT NULL,
            actual_symbol TEXT NOT NULL,
            created_at_utc TIMESTAMPTZ NOT NULL,
            active BOOLEAN NOT NULL DEFAULT TRUE
        )
        """
    )
    op.execute(
        """
        CREATE TABLE core.symbol_spec_version (
            id UUID PRIMARY KEY,
            source_id UUID NOT NULL REFERENCES core.data_source(id),
            observed_at_utc TIMESTAMPTZ NOT NULL,
            actual_symbol TEXT NOT NULL,
            digits INTEGER,
            point NUMERIC(20,10),
            trade_tick_size NUMERIC(20,10),
            trade_tick_value NUMERIC(30,10),
            trade_contract_size NUMERIC(30,10),
            volume_min NUMERIC(30,10),
            volume_max NUMERIC(30,10),
            volume_step NUMERIC(30,10),
            currency_base TEXT,
            currency_profit TEXT,
            payload_hash TEXT NOT NULL,
            raw_payload JSONB,
            CONSTRAINT uq_symbol_spec_payload UNIQUE (source_id, payload_hash)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE ingest.run (
            id UUID PRIMARY KEY,
            source_id UUID NOT NULL REFERENCES core.data_source(id),
            kind TEXT NOT NULL CHECK (kind IN ('bars', 'ticks', 'symbol_spec')),
            requested_start_utc TIMESTAMPTZ,
            requested_end_utc TIMESTAMPTZ,
            started_at_utc TIMESTAMPTZ NOT NULL,
            finished_at_utc TIMESTAMPTZ,
            status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'partial', 'failed')),
            rows_received BIGINT NOT NULL DEFAULT 0,
            rows_inserted BIGINT NOT NULL DEFAULT 0,
            rows_rejected BIGINT NOT NULL DEFAULT 0,
            rows_skipped BIGINT NOT NULL DEFAULT 0,
            error_code TEXT,
            error_message TEXT,
            collector_version TEXT
        )
        """
    )
    op.execute(
        """
        CREATE TABLE ingest.checkpoint (
            id UUID PRIMARY KEY,
            source_id UUID NOT NULL REFERENCES core.data_source(id),
            kind TEXT NOT NULL,
            timeframe TEXT NOT NULL DEFAULT '',
            partition_start_utc TIMESTAMPTZ NOT NULL,
            partition_end_utc TIMESTAMPTZ NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('fetched', 'empty', 'partial', 'failed')),
            last_success_at_utc TIMESTAMPTZ,
            last_error TEXT,
            run_id UUID NOT NULL REFERENCES ingest.run(id),
            CONSTRAINT uq_checkpoint_partition UNIQUE (
                source_id, kind, timeframe, partition_start_utc, partition_end_utc
            )
        )
        """
    )
    op.execute(
        """
        CREATE TABLE market.bar_observation (
            id UUID NOT NULL,
            source_id UUID NOT NULL REFERENCES core.data_source(id),
            symbol_spec_version_id UUID REFERENCES core.symbol_spec_version(id),
            actual_symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            open_time_utc TIMESTAMPTZ NOT NULL,
            close_time_utc TIMESTAMPTZ NOT NULL,
            open NUMERIC(20,10) NOT NULL,
            high NUMERIC(20,10) NOT NULL,
            low NUMERIC(20,10) NOT NULL,
            close NUMERIC(20,10) NOT NULL,
            tick_volume BIGINT NOT NULL,
            real_volume NUMERIC(30,10),
            spread_points BIGINT,
            spread_price NUMERIC(20,10),
            ingested_at_utc TIMESTAMPTZ NOT NULL,
            payload_hash TEXT NOT NULL,
            quality_status TEXT NOT NULL,
            CONSTRAINT pk_bar_observation PRIMARY KEY (open_time_utc, id),
            CONSTRAINT uq_bar_exact_payload UNIQUE (
                open_time_utc, source_id, actual_symbol, timeframe, payload_hash
            )
        ) PARTITION BY RANGE (open_time_utc)
        """
    )
    op.execute(
        """
        CREATE TABLE market.tick_observation (
            id UUID NOT NULL,
            source_id UUID NOT NULL REFERENCES core.data_source(id),
            actual_symbol TEXT NOT NULL,
            tick_time_utc TIMESTAMPTZ NOT NULL,
            timestamp_ms BIGINT NOT NULL,
            occurrence INTEGER NOT NULL,
            bid NUMERIC(20,10),
            ask NUMERIC(20,10),
            last NUMERIC(20,10),
            volume NUMERIC(30,10),
            volume_real NUMERIC(30,10),
            flags BIGINT,
            spread_price NUMERIC(20,10),
            is_valid_quote BOOLEAN NOT NULL,
            payload_hash TEXT NOT NULL,
            quality_status TEXT NOT NULL,
            ingested_at_utc TIMESTAMPTZ NOT NULL,
            CONSTRAINT pk_tick_observation PRIMARY KEY (tick_time_utc, id),
            CONSTRAINT uq_tick_deterministic_identity UNIQUE (
                tick_time_utc, source_id, actual_symbol, payload_hash, occurrence
            )
        ) PARTITION BY RANGE (tick_time_utc)
        """
    )
    op.execute(
        """
        CREATE TABLE market.rejection (
            id UUID PRIMARY KEY,
            run_id UUID NOT NULL REFERENCES ingest.run(id),
            source_id UUID NOT NULL REFERENCES core.data_source(id),
            record_kind TEXT NOT NULL,
            event_time_utc TIMESTAMPTZ,
            reason_code TEXT NOT NULL,
            raw_payload JSONB NOT NULL,
            rejected_at_utc TIMESTAMPTZ NOT NULL
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_bar_source_timeframe_time "
        "ON market.bar_observation (source_id, timeframe, open_time_utc)"
    )
    op.execute(
        "CREATE INDEX ix_tick_source_time ON market.tick_observation (source_id, tick_time_utc)"
    )
    op.execute(
        "CREATE INDEX ix_checkpoint_source_status "
        "ON ingest.checkpoint (source_id, kind, timeframe, status)"
    )
    op.execute(
        """
        CREATE FUNCTION market.ensure_monthly_partitions(
            p_table_name TEXT,
            p_start TIMESTAMPTZ,
            p_end TIMESTAMPTZ
        ) RETURNS VOID LANGUAGE plpgsql AS $$
        DECLARE
            month_start TIMESTAMPTZ;
            month_end TIMESTAMPTZ;
            child_name TEXT;
        BEGIN
            IF p_table_name NOT IN ('bar_observation', 'tick_observation') THEN
                RAISE EXCEPTION 'unsupported partitioned table: %', p_table_name;
            END IF;
            month_start := date_trunc('month', p_start AT TIME ZONE 'UTC') AT TIME ZONE 'UTC';
            WHILE month_start < p_end LOOP
                month_end := month_start + INTERVAL '1 month';
                child_name := p_table_name || '_' || to_char(month_start, 'YYYY_MM');
                EXECUTE format(
                    'CREATE TABLE IF NOT EXISTS market.%I PARTITION OF market.%I '
                    'FOR VALUES FROM (%L) TO (%L)',
                    child_name, p_table_name, month_start, month_end
                );
                month_start := month_end;
            END LOOP;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE VIEW market.bar_latest AS
        SELECT * FROM (
            SELECT b.*,
                   row_number() OVER (
                       PARTITION BY source_id, actual_symbol, timeframe, open_time_utc
                       ORDER BY ingested_at_utc DESC, id DESC
                   ) AS revision_rank
            FROM market.bar_observation b
        ) ranked
        WHERE revision_rank = 1
        """
    )
    op.execute(
        """
        CREATE VIEW market.data_coverage AS
        WITH checkpoints AS (
            SELECT source_id, kind, timeframe,
                   min(partition_start_utc) AS requested_start_utc,
                   max(partition_end_utc) AS requested_end_utc,
                   count(*) AS requested_partitions,
                   count(*) FILTER (WHERE status = 'fetched') AS fetched_partitions,
                   count(*) FILTER (WHERE status = 'empty') AS empty_partitions,
                   count(*) FILTER (WHERE status = 'failed') AS failed_partitions
            FROM ingest.checkpoint
            GROUP BY source_id, kind, timeframe
        ), actual AS (
            SELECT source_id, 'bars'::TEXT AS kind, timeframe,
                   min(open_time_utc) AS actual_first_record,
                   max(open_time_utc) AS actual_last_record,
                   count(*) AS observation_count
            FROM market.bar_latest
            GROUP BY source_id, timeframe
            UNION ALL
            SELECT source_id, 'ticks'::TEXT AS kind, ''::TEXT AS timeframe,
                   min(tick_time_utc), max(tick_time_utc), count(*)
            FROM market.tick_observation
            GROUP BY source_id
        )
        SELECT c.*, a.actual_first_record, a.actual_last_record,
               coalesce(a.observation_count, 0) AS observation_count
        FROM checkpoints c
        LEFT JOIN actual a USING (source_id, kind, timeframe)
        """
    )
    op.execute(
        """
        CREATE VIEW market.quote_ohlc_m5 AS
        SELECT source_id, actual_symbol,
               date_bin(INTERVAL '5 minutes', tick_time_utc,
                        TIMESTAMPTZ '1970-01-01 00:00:00+00') AS open_time_utc,
               (array_agg(bid ORDER BY tick_time_utc, occurrence) FILTER (WHERE bid IS NOT NULL))[1] AS bid_open,
               max(bid) AS bid_high, min(bid) AS bid_low,
               (array_agg(bid ORDER BY tick_time_utc DESC, occurrence DESC) FILTER (WHERE bid IS NOT NULL))[1] AS bid_close,
               (array_agg(ask ORDER BY tick_time_utc, occurrence) FILTER (WHERE ask IS NOT NULL))[1] AS ask_open,
               max(ask) AS ask_high, min(ask) AS ask_low,
               (array_agg(ask ORDER BY tick_time_utc DESC, occurrence DESC) FILTER (WHERE ask IS NOT NULL))[1] AS ask_close,
               (array_agg(spread_price ORDER BY tick_time_utc, occurrence) FILTER (WHERE spread_price IS NOT NULL))[1] AS spread_open,
               max(spread_price) AS spread_high, min(spread_price) AS spread_low,
               (array_agg(spread_price ORDER BY tick_time_utc DESC, occurrence DESC) FILTER (WHERE spread_price IS NOT NULL))[1] AS spread_close,
               avg(spread_price) AS spread_mean,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY spread_price) AS spread_median,
               stddev_pop(spread_price) AS spread_std,
               count(*) AS tick_count,
               min(tick_time_utc) AS first_tick_time_utc,
               max(tick_time_utc) AS last_tick_time_utc
        FROM market.tick_observation
        WHERE is_valid_quote
        GROUP BY source_id, actual_symbol,
                 date_bin(INTERVAL '5 minutes', tick_time_utc,
                          TIMESTAMPTZ '1970-01-01 00:00:00+00')
        """
    )


def downgrade() -> None:
    """Remove all collector-owned database objects in dependency order."""

    op.execute("DROP VIEW IF EXISTS market.quote_ohlc_m5")
    op.execute("DROP VIEW IF EXISTS market.data_coverage")
    op.execute("DROP VIEW IF EXISTS market.bar_latest")
    op.execute(
        "DROP FUNCTION IF EXISTS market.ensure_monthly_partitions(TEXT, TIMESTAMPTZ, TIMESTAMPTZ)"
    )
    op.execute("DROP TABLE IF EXISTS market.rejection")
    op.execute("DROP TABLE IF EXISTS market.tick_observation CASCADE")
    op.execute("DROP TABLE IF EXISTS market.bar_observation CASCADE")
    op.execute("DROP TABLE IF EXISTS ingest.checkpoint")
    op.execute("DROP TABLE IF EXISTS ingest.run")
    op.execute("DROP TABLE IF EXISTS core.symbol_spec_version")
    op.execute("DROP TABLE IF EXISTS core.data_source")
    op.execute("DROP SCHEMA IF EXISTS audit")
    op.execute("DROP SCHEMA IF EXISTS market")
    op.execute("DROP SCHEMA IF EXISTS ingest")
    op.execute("DROP SCHEMA IF EXISTS core")
