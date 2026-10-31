"""API dashboard backtesting: membaca hasil run di folder runs/ (read-only).

Run, model, dan periode hanya boleh dipilih dari daftar yang ditemukan di disk,
sehingga parameter URL tidak dapat dipakai untuk membaca file lain.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from dashboard.settings import load_dashboard_settings
from technical.backtest import list_backtests, read_candles, read_predictions, summarize_period
from technical.run import clean_for_json
from technical.simulation import InstrumentSpec

router = APIRouter(prefix="/api/backtest", tags=["Backtest"])

MAX_CANDLES = 30000
DEFAULT_WINDOW_DAYS = 3


def runs_root() -> Path:
    return load_dashboard_settings().runs_dir


def resolve(run_id: str, model: str, period: str) -> dict:
    for entry in list_backtests(runs_root()):
        if entry["run_id"] != run_id:
            continue
        periods = entry["models"].get(model)
        if periods is None or period not in periods:
            break
        run_directory = runs_root() / run_id
        return {
            "run": run_directory,
            "predictions": run_directory / "backtest" / model / period / "predictions.csv",
            "candles": run_directory / "backtest" / "candles" / f"{period}.csv",
        }
    raise HTTPException(status_code=404, detail="Run, model, atau periode tidak ditemukan")


def horizon_bars(run_directory: Path) -> int:
    metadata_path = run_directory / "run.json"
    if not metadata_path.exists():
        return 6
    with open(metadata_path, encoding="utf-8") as file:
        metadata = json.load(file)
    return int(metadata["config"]["labels"]["horizon_bars"])


# Cache per (path, waktu modifikasi): file yang ditulis ulang otomatis dibaca ulang.
@lru_cache(maxsize=32)
def cached_summary(
    path: str, modified: float, horizon: int, lot: float, leverage: float, contract_size: float
) -> dict:
    predictions = cached_predictions(path, modified)
    spec = InstrumentSpec(lot=lot, leverage=leverage, contract_size=contract_size)
    return clean_for_json(summarize_period(predictions, horizon, spec))


@lru_cache(maxsize=16)
def cached_predictions(path: str, modified: float) -> pd.DataFrame:
    return read_predictions(Path(path))


@lru_cache(maxsize=16)
def cached_candles(path: str, modified: float) -> pd.DataFrame:
    return read_candles(Path(path))


def load_predictions(path: Path) -> pd.DataFrame:
    return cached_predictions(str(path), path.stat().st_mtime)


def load_candles(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise HTTPException(status_code=404, detail="File candle periode ini tidak ditemukan")
    return cached_candles(str(path), path.stat().st_mtime)


def parse_time(value: str | None) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


@router.get("/runs", summary="Daftar run yang memiliki prediksi backtest")
def backtest_runs() -> dict:
    return {"runs_root": str(runs_root()), "runs": list_backtests(runs_root())}


@router.get("/{run_id}/{model}/{period}/summary", summary="Metrik klasifikasi, arah, dan simulasi")
def backtest_summary(
    run_id: str,
    model: str,
    period: str,
    lot: Annotated[float, Query(gt=0, le=100)] = 0.01,
    leverage: Annotated[float, Query(ge=1, le=1000)] = 100.0,
    contract_size: Annotated[float, Query(gt=0, le=100000)] = 100.0,
) -> dict:
    paths = resolve(run_id, model, period)
    path = paths["predictions"]
    predictions = load_predictions(path)
    horizon = horizon_bars(paths["run"])
    summary = dict(
        cached_summary(str(path), path.stat().st_mtime, horizon, lot, leverage, contract_size)
    )
    summary["horizon_bars"] = horizon
    summary["horizon_minutes"] = horizon * 5
    summary["model_version"] = (
        str(predictions["model_version"].iloc[0]) if len(predictions) else None
    )
    return summary


@router.get("/{run_id}/{model}/{period}/candles", summary="Candle OHLC untuk grafik")
def backtest_candles(
    run_id: str,
    model: str,
    period: str,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    paths = resolve(run_id, model, period)
    candles = load_candles(paths["candles"])
    if candles.empty:
        return {"bars": [], "first": None, "last": None}

    first = candles["open_time_utc"].iloc[0]
    last = candles["open_time_utc"].iloc[-1]
    start_time = parse_time(start)
    end_time = parse_time(end)
    if start_time is None and end_time is None:
        end_time = last + pd.Timedelta(minutes=5)
        start_time = end_time - pd.Timedelta(days=DEFAULT_WINDOW_DAYS)
    if start_time is None:
        start_time = first
    if end_time is None:
        end_time = last + pd.Timedelta(minutes=5)

    window = candles[
        (candles["open_time_utc"] >= start_time) & (candles["open_time_utc"] < end_time)
    ]
    truncated = len(window) > MAX_CANDLES
    window = window.tail(MAX_CANDLES)

    bars = []
    for row in window.itertuples(index=False):
        bars.append(
            {
                "time": int(row.open_time_utc.timestamp()),
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.tick_volume,
                "spread": row.spread,
            }
        )
    return clean_for_json(
        {
            "bars": bars,
            "first": first.isoformat(),
            "last": last.isoformat(),
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "truncated": truncated,
        }
    )


PREDICTION_COLUMNS = [
    "prediction_time",
    "candle_open_time",
    "label_end_time",
    "signal",
    "confidence",
    "probability_sell",
    "probability_no_trade",
    "probability_buy",
    "close",
    "spread",
    "bid_reference",
    "ask_reference",
    "barrier",
    "future_close",
    "actual_return",
    "actual_label",
    "label_status",
    "prediction_result",
    "direction_result",
    "up_touch_step",
    "down_touch_step",
    "reason_code",
    "model_version",
]


@router.get("/{run_id}/{model}/{period}/predictions", summary="Riwayat sinyal BUY/SELL")
def backtest_predictions(
    run_id: str,
    model: str,
    period: str,
    start: str | None = None,
    end: str | None = None,
    signal: Literal["ALL", "BUY", "SELL"] = "ALL",
    result: Literal["ALL", "CORRECT", "INCORRECT", "NEUTRAL", "PENDING", "UNAVAILABLE"] = "ALL",
) -> dict:
    """Hanya baris BUY/SELL. NO_TRADE tidak dikirim (lihat ringkasan untuk jumlahnya)."""
    paths = resolve(run_id, model, period)
    predictions = load_predictions(paths["predictions"])
    rows = predictions[predictions["signal"].isin(["BUY", "SELL"])]

    start_time = parse_time(start)
    end_time = parse_time(end)
    if start_time is not None:
        rows = rows[rows["prediction_time"] >= start_time]
    if end_time is not None:
        rows = rows[rows["prediction_time"] < end_time]
    if signal != "ALL":
        rows = rows[rows["signal"] == signal]
    if result != "ALL":
        rows = rows[rows["prediction_result"] == result]

    # Format kolom: nama kolom dikirim sekali, setiap baris berupa array (payload jauh lebih kecil).
    table = rows[PREDICTION_COLUMNS].copy()
    for key in ["prediction_time", "candle_open_time", "label_end_time"]:
        table[key] = table[key].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    float_columns = table.select_dtypes(include="float").columns
    table[float_columns] = table[float_columns].round(6)
    model_version = str(rows["model_version"].iloc[0]) if len(rows) else None
    table = table.drop(columns=["model_version"])
    table = table.astype(object).where(table.notna(), None)
    return {
        "count": len(table),
        "model_version": model_version,
        "columns": list(table.columns),
        "rows": table.to_numpy().tolist(),
    }
