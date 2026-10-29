"""Create a research-ready M5 view combining native bars and quote aggregates.

Revision ID: 20260929_0002
Revises: 20260929_0001

The view keeps native MT5 candle fields and tick-derived Bid/Ask/spread fields
separate.  A left join preserves every latest native M5 candle even when the
broker does not provide historical ticks for the same interval.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260929_0002"
down_revision: str | None = "20260929_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the unified, read-only M5 research view."""

    op.execute(
        """
        CREATE VIEW market.processed_m5 AS
        SELECT
            b.id AS bar_observation_id,
            b.source_id,
            b.symbol_spec_version_id,
            d.source_key,
            d.broker_name,
            d.server_name,
            d.environment,
            d.dataset_symbol,
            b.actual_symbol,
            b.timeframe,
            b.open_time_utc,
            b.close_time_utc,
            b.open,
            b.high,
            b.low,
            b.close,
            b.tick_volume,
            b.real_volume,
            b.spread_points,
            b.spread_price AS native_spread_price,
            q.bid_open,
            q.bid_high,
            q.bid_low,
            q.bid_close,
            q.ask_open,
            q.ask_high,
            q.ask_low,
            q.ask_close,
            q.spread_open,
            q.spread_high,
            q.spread_low,
            q.spread_close,
            q.spread_mean,
            q.spread_median,
            q.spread_std,
            q.tick_count,
            q.first_tick_time_utc,
            q.last_tick_time_utc,
            b.payload_hash AS bar_payload_hash,
            b.quality_status AS bar_quality_status,
            b.ingested_at_utc AS bar_ingested_at_utc
        FROM market.bar_latest AS b
        JOIN core.data_source AS d
          ON d.id = b.source_id
        LEFT JOIN market.quote_ohlc_m5 AS q
          ON q.source_id = b.source_id
         AND q.actual_symbol = b.actual_symbol
         AND q.open_time_utc = b.open_time_utc
        WHERE b.timeframe = 'M5'
        """
    )


def downgrade() -> None:
    """Remove the unified M5 research view without touching source data."""

    op.execute("DROP VIEW IF EXISTS market.processed_m5")
