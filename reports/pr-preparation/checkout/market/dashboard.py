"""Read-only FastAPI browser for bars, ticks, and unified M5 data.

The dashboard exposes one responsive HTML interface and paginated JSON APIs
backed by PostgreSQL. Dataset names, selected columns, and sort columns come
from server-side whitelists; user values remain bound SQL parameters. This
module never initializes MetaTrader 5 and never mutates collected market data.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from database.storage import Database
from market.contracts import Environment
from market.profile import load_settings

DatasetKey = Literal["processed_m5", "bars", "ticks"]
SortDirection = Literal["asc", "desc"]
QuoteStatus = Literal["all", "available", "missing"]

PROCESSED_M5_COLUMNS = (
    "bar_observation_id",
    "source_id",
    "symbol_spec_version_id",
    "source_key",
    "broker_name",
    "server_name",
    "environment",
    "dataset_symbol",
    "actual_symbol",
    "timeframe",
    "open_time_utc",
    "close_time_utc",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "real_volume",
    "spread_points",
    "native_spread_price",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
    "spread_open",
    "spread_high",
    "spread_low",
    "spread_close",
    "spread_mean",
    "spread_median",
    "spread_std",
    "tick_count",
    "first_tick_time_utc",
    "last_tick_time_utc",
    "bar_payload_hash",
    "bar_quality_status",
    "bar_ingested_at_utc",
)

BAR_COLUMNS = (
    "id",
    "source_id",
    "symbol_spec_version_id",
    "source_key",
    "broker_name",
    "server_name",
    "environment",
    "dataset_symbol",
    "actual_symbol",
    "timeframe",
    "open_time_utc",
    "close_time_utc",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "real_volume",
    "spread_points",
    "spread_price",
    "ingested_at_utc",
    "payload_hash",
    "quality_status",
    "revision_rank",
)

TICK_COLUMNS = (
    "id",
    "source_id",
    "source_key",
    "broker_name",
    "server_name",
    "environment",
    "dataset_symbol",
    "actual_symbol",
    "tick_time_utc",
    "timestamp_ms",
    "occurrence",
    "bid",
    "ask",
    "last",
    "volume",
    "volume_real",
    "flags",
    "spread_price",
    "is_valid_quote",
    "payload_hash",
    "quality_status",
    "ingested_at_utc",
)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Whitelisted SQL and presentation metadata for one dashboard dataset."""

    key: DatasetKey
    title: str
    description: str
    timestamp_column: str
    columns: tuple[str, ...]
    select_sql: str
    supports_timeframe: bool = False
    quote_available_clause: str | None = None
    quote_missing_clause: str | None = None


DATASETS: dict[DatasetKey, DatasetSpec] = {
    "processed_m5": DatasetSpec(
        key="processed_m5",
        title="Gabungan M5",
        description="Native M5 dengan agregasi Bid, Ask, dan spread dari tick.",
        timestamp_column="open_time_utc",
        columns=PROCESSED_M5_COLUMNS,
        select_sql="SELECT * FROM market.processed_m5",
        supports_timeframe=True,
        quote_available_clause="bid_open IS NOT NULL AND ask_open IS NOT NULL",
        quote_missing_clause="(bid_open IS NULL OR ask_open IS NULL)",
    ),
    "bars": DatasetSpec(
        key="bars",
        title="Market bars",
        description="Revision terbaru candle OHLC dan volume native broker.",
        timestamp_column="open_time_utc",
        columns=BAR_COLUMNS,
        select_sql="""
            SELECT b.id, b.source_id, b.symbol_spec_version_id,
                   d.source_key, d.broker_name, d.server_name, d.environment,
                   d.dataset_symbol, b.actual_symbol, b.timeframe,
                   b.open_time_utc, b.close_time_utc,
                   b.open, b.high, b.low, b.close,
                   b.tick_volume, b.real_volume, b.spread_points, b.spread_price,
                   b.ingested_at_utc, b.payload_hash, b.quality_status,
                   b.revision_rank
            FROM market.bar_latest AS b
            JOIN core.data_source AS d ON d.id = b.source_id
        """,
        supports_timeframe=True,
    ),
    "ticks": DatasetSpec(
        key="ticks",
        title="Raw ticks",
        description="Tick native broker beserta Bid, Ask, last, volume, dan audit.",
        timestamp_column="tick_time_utc",
        columns=TICK_COLUMNS,
        select_sql="""
            SELECT t.id, t.source_id,
                   d.source_key, d.broker_name, d.server_name, d.environment,
                   d.dataset_symbol, t.actual_symbol,
                   t.tick_time_utc, t.timestamp_ms, t.occurrence,
                   t.bid, t.ask, t.last, t.volume, t.volume_real, t.flags,
                   t.spread_price, t.is_valid_quote,
                   t.payload_hash, t.quality_status, t.ingested_at_utc
            FROM market.tick_observation AS t
            JOIN core.data_source AS d ON d.id = t.source_id
        """,
        quote_available_clause="is_valid_quote IS TRUE",
        quote_missing_clause="is_valid_quote IS FALSE",
    ),
}


class MarketDataPage(BaseModel):
    """One filtered and sorted page from a whitelisted market dataset."""

    dataset: DatasetKey = Field(description="Selected server-side dataset key.")
    title: str = Field(description="Human-readable dataset name.")
    description: str = Field(description="Short explanation of dataset semantics.")
    timestamp_column: str = Field(description="Primary UTC column used by time filters.")
    columns: list[str] = Field(description="Stable display order for available columns.")
    rows: list[dict[str, Any]] = Field(description="Requested rows in the selected order.")
    total: int = Field(description="Total rows matching the active filters.", ge=0)
    limit: int = Field(description="Maximum rows returned on this page.", ge=1)
    offset: int = Field(description="Number of matching rows skipped.", ge=0)
    sort_by: str = Field(description="Whitelisted column used to sort the page.")
    sort_direction: SortDirection = Field(description="Ascending or descending sort order.")


class ProcessedM5Page(BaseModel):
    """Backward-compatible page returned by the original processed-M5 API."""

    columns: list[str] = Field(description="Stable display order for all view columns.")
    rows: list[dict[str, Any]] = Field(description="Requested M5 rows in selected order.")
    total: int = Field(description="Total rows matching the active filters.", ge=0)
    limit: int = Field(description="Maximum rows returned on this page.", ge=1)
    offset: int = Field(description="Number of matching rows skipped.", ge=0)


class DashboardConfigurationError(RuntimeError):
    """Raised when the selected read-only demo or live database is not configured."""


_databases: dict[Environment, Database] = {}
_page_path = Path(__file__).with_name("static") / "processed_m5.html"


def get_database(environment: Environment) -> Database:
    """Return the lazy read-only database manager selected by a safe environment value.

    The browser can select only the ``Environment`` enum. Connection URLs remain
    on the server in ``GOLD_DASHBOARD_DEMO_DATABASE_URL`` and
    ``GOLD_DASHBOARD_LIVE_DATABASE_URL``.
    """

    database = _databases.get(environment)
    if database is None:
        try:
            database_url = load_settings().dashboard_database_url(environment)
        except ValueError as exc:
            raise DashboardConfigurationError(str(exc)) from exc
        database = Database(database_url, connect_timeout_seconds=5)
        _databases[environment] = database
    return database


@asynccontextmanager
async def lifespan(_: FastAPI) -> Iterator[None]:
    """Dispose every lazily created demo/live engine when the web server stops."""

    yield
    for database in _databases.values():
        database.dispose()
    _databases.clear()


app = FastAPI(
    title="XAUUSD Market Data Browser",
    summary="Inspect native bars, raw ticks, and unified M5 observations.",
    description=(
        "A local, read-only browser for PostgreSQL market data. It exposes "
        "whitelisted filters and sorting and never communicates with MetaTrader 5."
    ),
    version="0.3.0",
    lifespan=lifespan,
)


def query_market_data(
    database: Database,
    *,
    dataset: DatasetKey,
    symbol: str | None,
    source_key: str | None,
    timeframe: str | None,
    start_utc: datetime | None,
    end_utc: datetime | None,
    quote_status: QuoteStatus,
    sort_by: str | None,
    sort_direction: SortDirection,
    limit: int,
    offset: int,
) -> MarketDataPage:
    """Read one safe page from bars, ticks, or the unified M5 view.

    Dataset SQL and sortable columns are selected exclusively from ``DATASETS``.
    Filter values remain SQL bind parameters. The function is read-only and
    performs no collection, migration, insert, update, or delete operation.

    Raises:
        ValueError: If ``sort_by`` is not present in the selected dataset.
    """

    spec = DATASETS[dataset]
    selected_sort = sort_by or spec.timestamp_column
    if selected_sort not in spec.columns:
        raise ValueError(f"Kolom sort tidak tersedia untuk {dataset}: {selected_sort}")

    clauses: list[str] = []
    parameters: dict[str, Any] = {}
    normalized_symbol = symbol.strip() if symbol else None
    normalized_source = source_key.strip() if source_key else None
    normalized_timeframe = timeframe.strip().upper() if timeframe else None
    if normalized_symbol:
        clauses.append("actual_symbol = :symbol")
        parameters["symbol"] = normalized_symbol
    if normalized_source:
        clauses.append("source_key = :source_key")
        parameters["source_key"] = normalized_source
    if normalized_timeframe and spec.supports_timeframe:
        clauses.append("timeframe = :timeframe")
        parameters["timeframe"] = normalized_timeframe
    if start_utc is not None:
        clauses.append(f"{spec.timestamp_column} >= :start_utc")
        parameters["start_utc"] = start_utc
    if end_utc is not None:
        clauses.append(f"{spec.timestamp_column} < :end_utc")
        parameters["end_utc"] = end_utc
    if quote_status == "available" and spec.quote_available_clause:
        clauses.append(spec.quote_available_clause)
    elif quote_status == "missing" and spec.quote_missing_clause:
        clauses.append(spec.quote_missing_clause)

    where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    dataset_sql = f"({spec.select_sql}) AS dataset"
    count_sql = text(f"SELECT count(*) FROM {dataset_sql}{where_sql}")
    direction_sql = sort_direction.upper()
    page_sql = text(
        f'SELECT * FROM {dataset_sql}{where_sql} ORDER BY "{selected_sort}" '
        f"{direction_sql} NULLS LAST LIMIT :limit OFFSET :offset"
    )
    with database.session() as session:
        total = int(session.scalar(count_sql, parameters) or 0)
        page_parameters = {**parameters, "limit": limit, "offset": offset}
        rows = [dict(row) for row in session.execute(page_sql, page_parameters).mappings()]
    return MarketDataPage(
        dataset=dataset,
        title=spec.title,
        description=spec.description,
        timestamp_column=spec.timestamp_column,
        columns=list(spec.columns),
        rows=rows,
        total=total,
        limit=limit,
        offset=offset,
        sort_by=selected_sort,
        sort_direction=sort_direction,
    )


def query_processed_m5(
    database: Database,
    *,
    symbol: str | None,
    source_key: str | None,
    start_utc: datetime | None,
    end_utc: datetime | None,
    quote_status: QuoteStatus,
    limit: int,
    offset: int,
) -> ProcessedM5Page:
    """Read one backward-compatible page from ``market.processed_m5``."""

    page = query_market_data(
        database,
        dataset="processed_m5",
        symbol=symbol,
        source_key=source_key,
        timeframe=None,
        start_utc=start_utc,
        end_utc=end_utc,
        quote_status=quote_status,
        sort_by="open_time_utc",
        sort_direction="desc",
        limit=limit,
        offset=offset,
    )
    return ProcessedM5Page(
        columns=page.columns,
        rows=page.rows,
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard_page() -> HTMLResponse:
    """Serve the interactive market-data interface without querying PostgreSQL."""

    return HTMLResponse(_page_path.read_text(encoding="utf-8"))


@app.get(
    "/api/market-data",
    response_model=MarketDataPage,
    summary="Browse a market dataset",
    description=(
        "Return a filtered, paginated, and safely sorted page from latest native bars, "
        "raw ticks, or the unified processed M5 view. This endpoint is read-only."
    ),
    tags=["Market Data"],
    responses={503: {"description": "PostgreSQL or required migrated views are unavailable."}},
)
def market_data(
    environment: Annotated[
        Environment,
        Query(description="Configured database to browse: demo or live."),
    ] = Environment.DEMO,
    dataset: Annotated[
        DatasetKey,
        Query(description="Dataset: processed_m5, bars, or ticks."),
    ] = "processed_m5",
    symbol: Annotated[
        str | None,
        Query(description="Exact broker symbol, for example XAUUSD or XAUUSDm."),
    ] = None,
    source_key: Annotated[
        str | None,
        Query(description="Stable configured data-source key."),
    ] = None,
    timeframe: Annotated[
        str | None,
        Query(description="Exact timeframe for bars, for example M5, M15, or H1."),
    ] = None,
    start_utc: Annotated[
        datetime | None,
        Query(description="Inclusive lower UTC timestamp."),
    ] = None,
    end_utc: Annotated[
        datetime | None,
        Query(description="Exclusive upper UTC timestamp."),
    ] = None,
    quote_status: Annotated[
        QuoteStatus,
        Query(description="All rows, valid/available quotes, or missing/invalid quotes."),
    ] = "all",
    sort_by: Annotated[
        str | None,
        Query(description="Column to sort; must exist in the selected dataset."),
    ] = None,
    sort_direction: Annotated[
        SortDirection,
        Query(description="Ascending or descending ordering."),
    ] = "desc",
    limit: Annotated[int, Query(ge=10, le=200, description="Rows per page.")] = 50,
    offset: Annotated[int, Query(ge=0, description="Rows skipped for pagination.")] = 0,
) -> MarketDataPage:
    """Return a validated page from the selected demo or live market database."""

    if start_utc is not None and end_utc is not None and start_utc >= end_utc:
        raise HTTPException(status_code=422, detail="start_utc must be earlier than end_utc")
    try:
        database = get_database(environment)
        return query_market_data(
            database,
            dataset=dataset,
            symbol=symbol,
            source_key=source_key,
            timeframe=timeframe,
            start_utc=start_utc,
            end_utc=end_utc,
            quote_status=quote_status,
            sort_by=sort_by,
            sort_direction=sort_direction,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (DashboardConfigurationError, SQLAlchemyError) as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Database unavailable or required market views are not installed. "
                "Check the selected GOLD_DASHBOARD_*_DATABASE_URL and run "
                "`uv run alembic upgrade head` for that database."
            ),
        ) from exc


@app.get(
    "/api/processed-m5",
    response_model=ProcessedM5Page,
    summary="Browse processed M5 market data",
    description=(
        "Backward-compatible endpoint returning native M5 plus tick-derived "
        "Bid/Ask and spread fields in newest-first order."
    ),
    tags=["Market Data"],
)
def processed_m5_data(
    environment: Annotated[
        Environment,
        Query(description="Configured database to browse: demo or live."),
    ] = Environment.DEMO,
    symbol: Annotated[str | None, Query(description="Exact broker symbol.")] = None,
    source_key: Annotated[str | None, Query(description="Stable source key.")] = None,
    start_utc: Annotated[
        datetime | None,
        Query(description="Inclusive lower UTC candle timestamp."),
    ] = None,
    end_utc: Annotated[
        datetime | None,
        Query(description="Exclusive upper UTC candle timestamp."),
    ] = None,
    quote_status: Annotated[
        QuoteStatus,
        Query(description="Filter by paired Bid/Ask availability."),
    ] = "all",
    limit: Annotated[int, Query(ge=10, le=200, description="Rows per page.")] = 50,
    offset: Annotated[int, Query(ge=0, description="Rows skipped for pagination.")] = 0,
) -> ProcessedM5Page:
    """Return the original processed-M5 contract from the selected environment."""

    if start_utc is not None and end_utc is not None and start_utc >= end_utc:
        raise HTTPException(status_code=422, detail="start_utc must be earlier than end_utc")
    try:
        database = get_database(environment)
        return query_processed_m5(
            database,
            symbol=symbol,
            source_key=source_key,
            start_utc=start_utc,
            end_utc=end_utc,
            quote_status=quote_status,
            limit=limit,
            offset=offset,
        )
    except (DashboardConfigurationError, SQLAlchemyError) as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Database unavailable or market.processed_m5 is not installed. "
                "Check the selected GOLD_DASHBOARD_*_DATABASE_URL and run "
                "`uv run alembic upgrade head` for that database."
            ),
        ) from exc


def run() -> None:
    """Start the read-only dashboard on the local loopback interface."""

    uvicorn.run("market.dashboard:app", host="127.0.0.1", port=8080)


if __name__ == "__main__":
    run()
