"""Read analytics inputs from PostgreSQL through read-only transactions.

This module is the only database boundary in the analytics folder. It loads
market bars, engineered features, predictions, and feature importance with
bound SQL parameters and mandatory UTC ranges. It never writes to PostgreSQL.
Analysis copies retain missing observations, mask returns across native gaps,
and aggregate observed UTC sessions without inventing candles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

from database.storage import get_engine

# %% 01 | UTC, rangkaian query read-only dan parameter terikat


def parse_utc(value: str) -> pd.Timestamp:
    """Parse a timezone-aware timestamp and normalize it to UTC."""

    try:
        parsed = pd.Timestamp(value)
    except ValueError as exc:
        raise ValueError(f"Timestamp tidak valid: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp wajib memiliki timezone: {value!r}")
    return parsed.tz_convert("UTC")


def validate_time_range(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate and return a non-empty half-open UTC query interval."""

    start_utc = parse_utc(start)
    end_utc = parse_utc(end)
    if start_utc >= end_utc:
        raise ValueError("Start harus lebih awal daripada end")
    return start_utc, end_utc


def require_data(frame: pd.DataFrame, context: str) -> None:
    """Raise a clear error when a PostgreSQL query returns no observations."""

    if frame.empty:
        raise ValueError(f"Data kosong untuk {context}")


def read_postgres(
    query: str,
    parameters: Mapping[str, Any],
    datetime_columns: Sequence[str],
    *,
    engine: Engine | None = None,
) -> pd.DataFrame:
    """Execute bound SQL inside an explicit read-only transaction.

    User values are passed through ``parameters`` rather than interpolated into
    SQL. The optional engine exists for isolated tests; production uses the
    environment-backed engine from ``database.storage``.
    """

    active_engine = engine or get_engine()
    owns_engine = engine is None
    try:
        with active_engine.connect() as connection:
            transaction = connection.begin()
            try:
                connection.execute(text("SET TRANSACTION READ ONLY"))
                frame = pd.read_sql_query(text(query), connection, params=dict(parameters))
                transaction.commit()
            except Exception:
                transaction.rollback()
                raise
    except SQLAlchemyError as exc:
        raise RuntimeError(
            "Query analytics gagal. Pastikan migration dan tabel sumber tersedia."
        ) from exc
    finally:
        if owns_engine:
            active_engine.dispose()
    for column in datetime_columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    return frame


# %% 02 | Data pasar: satu source, symbol, timeframe dan rentang UTC


def load_market_data(
    source_id: str,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Load revision-resolved native bars for one source and UTC range.

    The symbol may be canonical or broker-specific. The query reads
    ``market.bar_latest`` and never updates PostgreSQL.
    """

    start_utc, end_utc = validate_time_range(start, end)
    frame = read_postgres(
        """
        SELECT b.open_time_utc, b.close_time_utc, b.source_id,
               d.dataset_symbol AS symbol, b.actual_symbol, b.timeframe,
               b.open, b.high, b.low, b.close, b.tick_volume,
               b.real_volume, b.spread_points, b.spread_price,
               b.quality_status
        FROM market.bar_latest AS b
        JOIN core.data_source AS d ON d.id = b.source_id
        WHERE b.source_id = :source_id
          AND (d.dataset_symbol = :symbol OR b.actual_symbol = :symbol)
          AND b.timeframe = :timeframe
          AND b.open_time_utc >= :start_utc
          AND b.open_time_utc < :end_utc
        ORDER BY b.open_time_utc
        """,
        {
            "source_id": UUID(source_id),
            "symbol": symbol,
            "timeframe": timeframe,
            "start_utc": start_utc.to_pydatetime(),
            "end_utc": end_utc.to_pydatetime(),
        },
        ("open_time_utc", "close_time_utc"),
    )
    require_data(frame, "market data")
    return frame


# %% 03 | Kontrak fitur dan prediksi dari pipeline berikutnya


def load_features(
    source_id: str,
    feature_version: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Load observations from the future feature pipeline.

    The fixed table contract is ``feature.feature_observation``. This market
    acquisition project does not create that future pipeline table.
    """

    start_utc, end_utc = validate_time_range(start, end)
    frame = read_postgres(
        """
        SELECT *
        FROM feature.feature_observation
        WHERE source_id = :source_id
          AND feature_version = :feature_version
          AND decision_time >= :start_utc
          AND decision_time < :end_utc
        ORDER BY decision_time
        """,
        {
            "source_id": UUID(source_id),
            "feature_version": feature_version,
            "start_utc": start_utc.to_pydatetime(),
            "end_utc": end_utc.to_pydatetime(),
        },
        ("timestamp_utc", "decision_time"),
    )
    require_data(frame, "feature data")
    return frame


def load_predictions(
    source_id: str,
    model_version: str,
    evaluation_split: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Load one explicit split from frozen model predictions.

    The fixed table contract is ``model.prediction_observation``. Train and test
    rows are never combined implicitly.
    """

    if not evaluation_split.strip():
        raise ValueError("evaluation_split wajib diisi secara eksplisit")
    start_utc, end_utc = validate_time_range(start, end)
    frame = read_postgres(
        """
        SELECT decision_time, source_id, model_version, fold_id,
               evaluation_split, target_class, predicted_class,
               p_sell, p_no_trade, p_buy
        FROM model.prediction_observation
        WHERE source_id = :source_id
          AND model_version = :model_version
          AND evaluation_split = :evaluation_split
          AND decision_time >= :start_utc
          AND decision_time < :end_utc
        ORDER BY decision_time
        """,
        {
            "source_id": UUID(source_id),
            "model_version": model_version,
            "evaluation_split": evaluation_split,
            "start_utc": start_utc.to_pydatetime(),
            "end_utc": end_utc.to_pydatetime(),
        },
        ("decision_time",),
    )
    require_data(frame, "prediction data")
    return frame


def load_feature_importance(source_id: str, model_version: str) -> pd.DataFrame:
    """Load importance values for one frozen model from a read-only table."""

    frame = read_postgres(
        """
        SELECT feature, importance
        FROM model.feature_importance
        WHERE source_id = :source_id
          AND model_version = :model_version
        ORDER BY importance DESC
        """,
        {"source_id": UUID(source_id), "model_version": model_version},
        (),
    )
    require_data(frame, "feature importance")
    return frame


# %% 04 | Agregasi close harian untuk eksperimen forecast


def prepare_daily_close(bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate valid intraday bars to final UTC daily closing prices.

    Args:
        bars: Market frame from analytics.data with open_time_utc and close.

    Returns:
        A new ordered frame containing timestamp_utc and close. Missing days
        remain missing; this function never interpolates market prices.
    """

    require_columns(bars, ("open_time_utc", "close"))
    daily = bars[["open_time_utc", "close"]].copy()
    daily["open_time_utc"] = pd.to_datetime(daily["open_time_utc"], utc=True, errors="coerce")
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    daily = daily.replace([np.inf, -np.inf], np.nan).dropna()
    daily = daily.loc[daily["close"] > 0].sort_values("open_time_utc")
    if daily.empty:
        raise ValueError("Tidak ada close harga yang valid untuk agregasi harian")
    return (
        daily.set_index("open_time_utc")["close"]
        .resample("1D")
        .last()
        .dropna()
        .rename_axis("timestamp_utc")
        .reset_index()
    )


# %% 05 | Persiapan time series: provenance, candle tertutup dan kualitas
def require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Reject missing input columns before transforming an analysis copy."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")


def timeframe_interval(timeframe: str) -> pd.Timedelta:
    """Return the native candle duration for supported intraday timeframes."""

    durations = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60}
    if timeframe not in durations:
        raise ValueError(f"Timeframe analytics tidak didukung: {timeframe}")
    return pd.Timedelta(minutes=durations[timeframe])


def utc_observations(values: pd.Series) -> pd.Series:
    """Validate explicit timezone information and return UTC timestamps.

    Naive observations are rejected rather than silently assigned a timezone.
    This preserves broker-time provenance at the analysis boundary.
    """

    if isinstance(values.dtype, pd.DatetimeTZDtype):
        return values.dt.tz_convert("UTC")
    if values.isna().any() or any(pd.Timestamp(value).tzinfo is None for value in values):
        raise ValueError("Timestamp observasi wajib valid dan memiliki timezone")
    return pd.to_datetime(values, utc=True, errors="raise")


def prepare_time_series(
    frame: pd.DataFrame,
    timeframe: str = "M5",
    *,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build an ordered, gap-aware analysis copy for one native market stream.

    Args:
        frame: Native bars; optional OHLC, identity, volume and spread columns.
        timeframe: Duration used to distinguish one-bar returns from gap moves.
        as_of: Inclusive close availability boundary, defaulting to current UTC.

    Returns:
        A new frame with log_return, absolute_return, relative_range and
        relative_spread. Quality counts are stored in attrs['quality'].
        Invalid prices remain missing; no imputation, clipping or smoothing
        is applied. Returns across noncontiguous candles are unavailable.

    Raises:
        ValueError: For mixed identities, duplicate times, or naive timestamps.
    """

    require_columns(frame, ("open_time_utc", "close"))
    require_data(frame, "time series")
    interval = timeframe_interval(timeframe)
    local = frame.copy(deep=True)
    for identity in ("source_id", "actual_symbol", "timeframe"):
        if identity in local and (local[identity].isna().any() or local[identity].nunique() != 1):
            raise ValueError(f"Time series wajib satu {identity}")
    if "timeframe" in local and str(local["timeframe"].iloc[0]) != timeframe:
        raise ValueError("Timeframe data berbeda dengan interval analisis")
    local["open_time_utc"] = utc_observations(local["open_time_utc"])
    local = local.sort_values("open_time_utc", kind="stable").reset_index(drop=True)
    if local["open_time_utc"].duplicated().any():
        raise ValueError("Timestamp duplikat: gunakan bar_latest untuk resolusi revisi")
    cutoff = pd.Timestamp.now(tz="UTC") if as_of is None else parse_utc(str(as_of))
    closes = (
        utc_observations(local["close_time_utc"])
        if "close_time_utc" in local
        else local["open_time_utc"] + interval
    )
    if not (closes == local["open_time_utc"] + interval).all():
        raise ValueError("Batas close_time_utc tidak sesuai timeframe")
    quality: dict[str, Any] = {
        "input_bars": len(local),
        "unclosed_bars": int((closes > cutoff).sum()),
    }
    local = local.loc[closes <= cutoff].copy().reset_index(drop=True)
    require_data(local, "closed time series")
    price_columns = [column for column in ("open", "high", "low", "close") if column in local]
    for column in price_columns + [
        name for name in ("tick_volume", "real_volume", "spread_price") if name in local
    ]:
        values = pd.to_numeric(local[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        local[column] = values.where(values > 0 if column in price_columns else values >= 0)
    # DATA-01 | Pertahankan bar invalid sebagai NaN supaya gap tidak tersambung palsu.
    invalid = local[price_columns].isna().any(axis=1)
    if len(price_columns) == 4:
        invalid |= (local["high"] < local[["open", "close", "low"]].max(axis=1)) | (
            local["low"] > local[["open", "close", "high"]].min(axis=1)
        )
    local.loc[invalid, price_columns] = np.nan
    if "quality_status" in local:
        quality["quality_status_counts"] = (
            local["quality_status"].astype(str).value_counts().to_dict()
        )
    quality["invalid_price_bars"] = int(invalid.sum())
    # DATA-02 | Return M5 hanya berlaku untuk pasangan candle yang tepat berjarak 5 menit.
    local = calculate_return(local, timeframe)
    local["log_return"] = np.log(local["close"]).diff().where(local["contiguous_previous"])
    local["absolute_return"] = local["log_return"].abs()
    if {"high", "low"}.issubset(local):
        local["relative_range"] = (local["high"] - local["low"]) / local["close"]
    if "spread_price" in local:
        local["relative_spread"] = local["spread_price"] / local["close"]
    if "tick_volume" in local:
        local["log_tick_volume"] = np.log1p(local["tick_volume"])
    quality["retained_bars"] = len(local)
    quality["gap_transitions"] = int((local["open_time_utc"].diff() > interval).sum())
    quality["available_native_returns"] = int(local["log_return"].notna().sum())
    quality["note"] = "Gap mencakup penutupan pasar; bukan persentase coverage kalender trading."
    local.attrs["quality"] = quality
    return local


def prepare_univariate_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare an isolated native stream for the ten existing UNI plots."""

    require_columns(frame, ("open_time_utc", "close", "tick_volume", "spread_price"))
    timeframe = str(frame["timeframe"].iloc[0]) if "timeframe" in frame else "M5"
    return prepare_time_series(frame, timeframe)


# %% 06 | Feature engineering deskriptif: return dan agregat harian kausal
def calculate_return(frame: pd.DataFrame, timeframe: str | None = None) -> pd.DataFrame:
    """Add percentage returns only for adjacent native bars, preserving missing values."""

    require_columns(frame, ("open_time_utc", "close"))
    local = frame.copy(deep=True)
    times = utc_observations(local["open_time_utc"])
    if not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError("Return membutuhkan timestamp unik dan berurutan")
    selected = timeframe or (str(local["timeframe"].iloc[0]) if "timeframe" in local else "M5")
    local["contiguous_previous"] = times.diff().eq(timeframe_interval(selected))
    local["return"] = (
        local["close"].pct_change(fill_method=None).where(local["contiguous_previous"])
    )
    return local


def prepare_daily_summary(native: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Aggregate observed UTC dates for descriptive insight, keeping absent dates missing.

    Daily returns compare adjacent observed UTC sessions, including weekends
    up to four calendar days apart. Longer gaps invalidate the return. They
    are not M5 targets. Volatility uses a trailing window of observed session
    returns without annualisation; partial sessions remain labelled by counts.
    """

    require_columns(native, ("open_time_utc", "close", "log_return"))
    if window < 2:
        raise ValueError("Window volatilitas minimal dua observasi")
    indexed = native.set_index("open_time_utc")
    groups = indexed.resample("1D")
    daily = pd.DataFrame(
        {
            "bars": groups.size(),
            "close": groups["close"].agg(lambda x: x.iloc[-1] if len(x) else np.nan),
        }
    )
    for column in ("tick_volume", "spread_price", "relative_range", "relative_spread"):
        if column in indexed:
            daily[column] = (
                groups[column].sum(min_count=1)
                if column == "tick_volume"
                else groups[column].mean()
            )
    daily["valid_prices"] = groups["close"].count()
    daily["native_return_count"] = groups["log_return"].count()
    # DATA-03 | Jangan menganggap nol return pada sesi yang tidak memiliki return valid.
    daily["realized_volatility"] = np.sqrt(
        groups["log_return"].agg(lambda x: x.pow(2).sum(min_count=1))
    )
    daily = daily.loc[daily["bars"] > 0].copy()
    daily["log_return"] = (
        np.log(daily["close"]).diff().where(daily.index.to_series().diff() <= pd.Timedelta(days=4))
    )
    daily["absolute_return"] = daily["log_return"].abs()
    daily["rolling_volatility"] = daily["log_return"].rolling(window, min_periods=window).std()
    daily["drawdown"] = daily["close"] / daily["close"].cummax() - 1
    daily.attrs["window"] = window
    daily.attrs["quality"] = native.attrs.get("quality", {}).copy()
    return daily.rename_axis("timestamp_utc").reset_index()


def monthly_returns(daily: pd.DataFrame) -> pd.Series:
    """Calculate adjacent calendar-month close returns without filling absent months.

    The first month has no prior-month anchor and stays unavailable. The last
    month can be partial; report its observed end date alongside the heatmap.
    """

    require_columns(daily, ("timestamp_utc", "close"))
    closes = (
        daily.set_index("timestamp_utc")["close"]
        .resample("MS")
        .agg(lambda values: values.iloc[-1] if len(values) else np.nan)
    )
    return closes.pct_change(fill_method=None)
