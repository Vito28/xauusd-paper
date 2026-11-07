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
from technical.simulation import InstrumentSpec, trade_details

router = APIRouter(prefix="/api/backtest", tags=["Backtest"])

MAX_CANDLES = 30000
DEFAULT_WINDOW_DAYS = 3
DEFAULT_BARRIER_ATR_MULTIPLE = 1.0

Lot = Annotated[float, Query(gt=0, le=100)]
Leverage = Annotated[float, Query(ge=1, le=1000)]
ContractSize = Annotated[float, Query(gt=0, le=100000)]


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


def label_settings(run_directory: Path) -> dict:
    """Horizon dan pengali barrier yang dipakai run (dari run.json)."""
    settings = {"horizon_bars": 6, "barrier_atr_multiple": DEFAULT_BARRIER_ATR_MULTIPLE}
    metadata_path = run_directory / "run.json"
    if not metadata_path.exists():
        return settings
    with open(metadata_path, encoding="utf-8") as file:
        labels = json.load(file)["config"]["labels"]
    settings["horizon_bars"] = int(labels["horizon_bars"])
    settings["barrier_atr_multiple"] = float(
        labels.get("barrier_atr_multiple", DEFAULT_BARRIER_ATR_MULTIPLE)
    )
    return settings


def horizon_bars(run_directory: Path) -> int:
    return label_settings(run_directory)["horizon_bars"]


def file_version(path: Path) -> tuple[str, float]:
    """(path, waktu modifikasi) untuk kunci cache; ("", 0) jika file tidak ada."""
    if not path.exists():
        return "", 0.0
    return str(path), path.stat().st_mtime


# Cache per (path, waktu modifikasi): file yang ditulis ulang otomatis dibaca ulang.
@lru_cache(maxsize=32)
def cached_summary(
    path: str,
    modified: float,
    candle_path: str,
    candle_modified: float,
    horizon: int,
    lot: float,
    leverage: float,
    contract_size: float,
) -> dict:
    predictions = cached_predictions(path, modified)
    candles = None
    if candle_path:
        candles = cached_candles(candle_path, candle_modified)
    spec = InstrumentSpec(lot=lot, leverage=leverage, contract_size=contract_size)
    return clean_for_json(summarize_period(predictions, horizon, spec, candles))


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
    lot: Lot = 0.01,
    leverage: Leverage = 100.0,
    contract_size: ContractSize = 100.0,
) -> dict:
    paths = resolve(run_id, model, period)
    path = paths["predictions"]
    predictions = load_predictions(path)
    settings = label_settings(paths["run"])
    horizon = settings["horizon_bars"]
    candle_path, candle_modified = file_version(paths["candles"])
    summary = dict(
        cached_summary(
            str(path),
            path.stat().st_mtime,
            candle_path,
            candle_modified,
            horizon,
            lot,
            leverage,
            contract_size,
        )
    )
    summary["horizon_bars"] = horizon
    summary["horizon_minutes"] = horizon * 5
    summary["barrier_atr_multiple"] = settings["barrier_atr_multiple"]
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

# Rincian transaksi dari technical.simulation.trade_details (kosong jika belum bisa dinilai).
TRADE_COLUMNS = [
    "exit_reason",
    "exit_step",
    "holding_minutes",
    "exit_candle_open_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "gross_move",
    "net_move",
    "pnl",
    "net_return",
    "spread_to_price",
    "spread_to_barrier",
    "exit_candle_spread",
]
TIME_COLUMNS = [
    "prediction_time",
    "candle_open_time",
    "label_end_time",
    "exit_candle_open_time",
    "exit_time",
]
NUMERIC_TRADE_COLUMNS = [
    "exit_step",
    "holding_minutes",
    "entry_price",
    "exit_price",
    "gross_move",
    "net_move",
    "pnl",
    "net_return",
    "spread_to_price",
    "spread_to_barrier",
    "exit_candle_spread",
]
# Rasio kecil (mis. spread/harga ~ 0,0001) dibulatkan lebih halus agar persentase tetap terbaca.
RATIO_COLUMNS = ["net_return", "spread_to_price", "spread_to_barrier"]


@router.get("/{run_id}/{model}/{period}/predictions", summary="Riwayat sinyal BUY/SELL")
def backtest_predictions(
    run_id: str,
    model: str,
    period: str,
    start: str | None = None,
    end: str | None = None,
    signal: Literal["ALL", "BUY", "SELL"] = "ALL",
    result: Literal["ALL", "CORRECT", "INCORRECT", "NEUTRAL", "PENDING", "UNAVAILABLE"] = "ALL",
    lot: Lot = 0.01,
    leverage: Leverage = 100.0,
    contract_size: ContractSize = 100.0,
) -> dict:
    """Hanya baris BUY/SELL, lengkap dengan waktu buka/tutup, P&L, dan persentase spread.

    NO_TRADE tidak dikirim (lihat ringkasan untuk jumlahnya).
    """
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

    candles = None
    if paths["candles"].exists():
        candles = load_candles(paths["candles"])
    spec = InstrumentSpec(lot=lot, leverage=leverage, contract_size=contract_size)
    details = trade_details(rows, horizon_bars(paths["run"]), spec, candles)

    # Format kolom: nama kolom dikirim sekali, setiap baris berupa array (payload jauh lebih kecil).
    table = pd.concat([rows[PREDICTION_COLUMNS], details[TRADE_COLUMNS]], axis=1)
    for key in TIME_COLUMNS:
        table[key] = pd.to_datetime(table[key], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    for key in NUMERIC_TRADE_COLUMNS:
        table[key] = pd.to_numeric(table[key], errors="coerce")
    for name in table.select_dtypes(include="float").columns:
        if name in RATIO_COLUMNS:
            table[name] = table[name].round(8)
        else:
            table[name] = table[name].round(6)
    model_version = str(rows["model_version"].iloc[0]) if len(rows) else None
    table = table.drop(columns=["model_version"])
    table = table.astype(object).where(table.notna(), None)
    return {
        "count": len(table),
        "model_version": model_version,
        "columns": list(table.columns),
        "rows": table.to_numpy().tolist(),
    }
