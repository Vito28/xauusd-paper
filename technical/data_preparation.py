"""Prepare canonical M5 data and temporal folds in notebook-style sections.

Flow: market.processed_m5 -> closed UTC candles -> quality report -> spread-aware
labels -> calendar folds and purged CV. Prices are never filled or written back
into PostgreSQL. The modeling workflow consumes these functions from
technical.xgboost_model; no model is fitted when this module is imported.
"""

# %% 01. Import dan dependensi data

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import Engine

from analytics.data import parse_utc, read_postgres, require_data, validate_time_range

# %% 02. Load canonical M5 melalui transaksi read-only
# Input: identitas source/symbol dan batas waktu UTC; output: candle dari processed_m5.
# Query hanya mengambil M5 yang sudah tutup melalui transaksi PostgreSQL read-only.


def load_data(
    source_key: str, symbol: str, start: str, end: str, *, engine: Engine | None = None
) -> pd.DataFrame:
    """Read one source/symbol M5 interval with bound SQL in a read-only transaction.

    Args: source_key/symbol identify one instrument; start/end are aware UTC bounds.
    Returns: closed canonical bars, preserving spread provenance and source identity.
    Raises: ValueError for invalid bounds/empty data; RuntimeError on query failure.
    """
    lower, upper = validate_time_range(start, end)
    if not source_key.strip() or not symbol.strip():
        raise ValueError("source_key dan symbol wajib diisi")
    frame = read_postgres(
        """SELECT source_id, source_key, dataset_symbol, actual_symbol, timeframe,
                  open_time_utc, close_time_utc, open, high, low, close, tick_volume,
                  native_spread_price, spread_mean, bar_observation_id
           FROM market.processed_m5
           WHERE source_key = :source_key
             AND (dataset_symbol = :symbol OR actual_symbol = :symbol)
             AND timeframe = 'M5'
             AND open_time_utc >= :start
             AND close_time_utc < :end
             AND close_time_utc <= :now
           ORDER BY open_time_utc, bar_observation_id""",
        {
            "source_key": source_key.strip(),
            "symbol": symbol.strip(),
            "start": lower.to_pydatetime(),
            "end": upper.to_pydatetime(),
            "now": pd.Timestamp.now(tz="UTC").to_pydatetime(),
        },
        ("open_time_utc", "close_time_utc"),
        engine=engine,
    )
    require_data(frame, "XAUUSD canonical M5")
    return frame


# %% 03. Preparation dan laporan kualitas


def prepare_data(
    frame: pd.DataFrame, *, as_of: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize UTC/numbers, keep last stable duplicate, reject invalid OHLC and report gaps.

    Missing/negative volume and spread become NaN, never synthetic values. Native
    spread is preferred; same-bar spread_mean is a fallback with explicit provenance.
    Missing M5 slots include market closures; they are recorded, never interpolated.
    Rows cannot mix source or actual symbol. Naive input timestamps are rejected.
    """
    # PREP-01 | Validasi kolom wajib, satu source/symbol, dan base timeframe M5.
    required = {"open_time_utc", "open", "high", "low", "close"}
    if not required <= set(frame):
        raise ValueError(f"Kolom wajib: {sorted(required)}")
    data = frame.copy()
    for column in ("source_id", "source_key", "actual_symbol", "dataset_symbol"):
        if column in data and (data[column].isna().any() or data[column].nunique() != 1):
            raise ValueError(f"Dataset harus memiliki tepat satu {column}")
    if "timeframe" in data and not data["timeframe"].eq("M5").all():
        raise ValueError("Base timeframe wajib M5")
    # PREP-02 | Normalisasi timestamp ke UTC; waktu keputusan adalah close time candle.
    for column in ("open_time_utc", "close_time_utc"):
        if column in data:
            # Reject ambiguous local timestamps before pandas can implicitly assign UTC.
            if any(pd.Timestamp(v).tzinfo is None for v in data[column].dropna()):
                raise ValueError(f"{column} wajib memiliki timezone")
            data[column] = pd.to_datetime(data[column], utc=True, errors="coerce")
    if "close_time_utc" not in data:
        data["close_time_utc"] = data["open_time_utc"] + pd.Timedelta(minutes=5)
    data["prediction_timestamp"] = data["close_time_utc"]
    prices = ["open", "high", "low", "close"]
    for column in (*prices, "tick_volume", "native_spread_price", "spread_mean", "spread_price"):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
    # PREP-03 | Pilih spread native atau fallback tick candle yang sama, simpan asalnya.
    # Spread/volume negatif atau tidak tersedia menjadi NaN.
    empty = pd.Series(np.nan, index=data.index, dtype=float)
    native = data.get("native_spread_price", data.get("spread_price", empty)).where(
        lambda x: x >= 0
    )
    tick = data.get("spread_mean", empty).where(lambda x: x >= 0)
    data["spread_price"] = native.combine_first(tick)
    data["spread_source"] = np.select(
        [native.notna(), tick.notna()], ["native", "same_candle_ticks"], default="missing"
    )
    data["tick_volume"] = data.get("tick_volume", empty).where(lambda x: x >= 0)
    # PREP-04 | Tolak OHLC tidak valid, timestamp tidak sesuai grid M5, dan candle belum tutup.
    valid_ohlc = (
        data[prices].notna().all(axis=1)
        & (data[prices] > 0).all(axis=1)
        & (data.high >= data[["open", "close", "low"]].max(axis=1))
        & (data.low <= data[["open", "close", "high"]].min(axis=1))
    )
    valid_time = (
        data.open_time_utc.notna()
        & data.close_time_utc.notna()
        & ((data.close_time_utc - data.open_time_utc) == pd.Timedelta(minutes=5))
        & (data.open_time_utc.dt.minute % 5 == 0)
        & (data.open_time_utc.dt.second == 0)
        & (data.open_time_utc.dt.microsecond == 0)
        & (data.open_time_utc.dt.nanosecond == 0)
    )
    cutoff = as_of if as_of is not None else pd.Timestamp.now(tz="UTC")
    if cutoff.tzinfo is None:
        raise ValueError("as_of wajib memiliki timezone")
    closed = data.prediction_timestamp <= cutoff
    duplicate_rows = int(data.open_time_utc.duplicated().sum())
    report: dict[str, Any] = {
        "total_rows": len(data),
        "duplicate_rows": duplicate_rows,
        "invalid_ohlc_rows": int((~valid_ohlc).sum()),
        "invalid_timestamp_rows": int((~valid_time).sum()),
        "unclosed_rows": int((~closed).sum()),
    }
    # PREP-05 | Urutkan candle dan ambil duplicate valid terakhir secara deterministik.
    data = (
        data.loc[valid_ohlc & valid_time & closed]
        .sort_values("open_time_utc", kind="stable")
        .drop_duplicates("open_time_utc", keep="last")
    )
    data = data.reset_index(drop=True)
    require_data(data, "valid closed M5 candles")
    # PREP-06 | Catat gap/missing slots untuk audit; harga tidak di-fill atau diinterpolasi.
    differences = data.open_time_utc.diff()
    gaps = []
    for i in data.index[differences > pd.Timedelta(minutes=5)]:
        gaps.append(
            {
                "after": data.open_time_utc.iloc[i - 1].isoformat(),
                "before": data.open_time_utc.iloc[i].isoformat(),
                "missing_slots": int(differences.iloc[i].total_seconds() // 300 - 1),
            }
        )
    report.update(
        {
            "prepared_rows": len(data),
            "detected_gaps": len(gaps),
            "gaps": gaps,
            "missing_rows": sum(g["missing_slots"] for g in gaps),
            "spread_missing": int(data.spread_price.isna().sum()),
            "tick_volume_missing": int(data.tick_volume.isna().sum()),
            "first_timestamp": data.open_time_utc.iloc[0].isoformat(),
            "last_timestamp": data.prediction_timestamp.iloc[-1].isoformat(),
            "gap_note": "Missing slots include market closures; no price filling.",
        }
    )
    return data, report


# %% 04. Target klasifikasi dengan biaya spread
# Tahap ini membuat target y, terpisah dari input X dalam feature engineering.
# BUY jika future return > biaya+edge; SELL jika < -(biaya+edge); sisanya NO_TRADE.
# Label belum diketahui bila future candle/spread tidak tersedia atau target melintasi gap.


def make_labels(
    data: pd.DataFrame,
    horizon_bars: int = 3,
    minimum_edge_bps: float = 1.0,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    """Attach spread/slippage-aware training labels separately from direction outcomes.

    H bars must also equal H*5 elapsed minutes. A target crossing a gap is unknown.
    Last H rows retain a target timestamp and null actuals for pending predictions.
    Labels without known decision-time spread are unavailable, never NO_TRADE.
    Slippage is a configured round-trip bps assumption, included exactly once.
    actual_status separates missing future targets, gaps and unavailable costs.
    """
    if (
        horizon_bars < 1
        or not np.isfinite(minimum_edge_bps)
        or minimum_edge_bps < 0
        or not np.isfinite(slippage_bps)
        or slippage_bps < 0
    ):
        raise ValueError("Horizon/edge tidak valid")
    result = data.copy()
    result["target_timestamp"] = result.prediction_timestamp + pd.Timedelta(
        minutes=5 * horizon_bars
    )
    future_time = result.prediction_timestamp.shift(-horizon_bars)
    contiguous = future_time.eq(result.target_timestamp)
    result["future_close"] = result.close.shift(-horizon_bars).where(contiguous)
    result["future_return"] = result.future_close / result.close - 1
    result["spread_return"] = result.spread_price / result.close
    result["slippage_return"] = slippage_bps / 10000
    result["threshold_return"] = (
        result.spread_return + minimum_edge_bps / 10000 + result.slippage_return
    )
    result["actual_status"] = np.select(
        [
            result.target_timestamp > data.prediction_timestamp.max(),
            ~contiguous,
            result.spread_price.isna(),
        ],
        ["PENDING", "GAP_INVALID", "COST_UNAVAILABLE"],
        default="AVAILABLE",
    )
    result["target_class"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    known = result.future_return.notna() & result.threshold_return.notna()
    result.loc[known, "target_class"] = np.select(
        [
            result.loc[known, "future_return"] < -result.loc[known, "threshold_return"],
            result.loc[known, "future_return"] > result.loc[known, "threshold_return"],
        ],
        [0, 2],
        default=1,
    )
    return result


# %% 05. Walk-forward folds dan purged temporal validation
# Pemisahan kronologis: TRAIN -> TUNE -> CALIBRATION -> POLICY -> TEST.
# Label development harus selesai sebelum akhir blok agar tidak bocor ke blok berikutnya.


@dataclass(frozen=True)
class Fold:
    """Five disjoint half-open UTC blocks: TRAIN/TUNE/CALIBRATION/POLICY/TEST."""

    name: str
    boundaries: tuple[pd.Timestamp, ...]

    def split(self, labels: pd.DataFrame) -> dict[str, pd.Index]:
        """Select known development labels whose target closes strictly inside their block.

        TEST keeps pending predictions, while its out-of-range actuals are masked
        later by the evaluator. Purge uses true target times, so gaps cannot leak.
        """
        blocks = {}
        for i, name in enumerate(("train", "tune", "calibration", "policy", "test")):
            start, end = self.boundaries[i : i + 2]
            mask = labels.prediction_timestamp.ge(start) & labels.prediction_timestamp.lt(end)
            if name != "test":
                mask &= labels.target_timestamp.lt(end) & labels.target_class.notna()
            blocks[name] = labels.index[mask]
        return blocks


def make_folds(
    start: str = "2016-01-01T00:00:00Z",
    evaluation_end: str = "2026-10-01T00:00:00Z",
    *,
    tune_start: str = "2025-01-01T00:00:00Z",
    calibration_start: str = "2025-07-01T00:00:00Z",
    policy_start: str = "2025-10-01T00:00:00Z",
    test_start: str = "2026-01-01T00:00:00Z",
) -> list[Fold]:
    """Build annual tests 2020–2025 and a configurable frozen final evaluation.

    Jan–Jun/Jul–Sep/Oct–Dec of the preceding year are development blocks.
    Default final development uses 2025, leaving January–September 2026 for
    sequential inference on actual closed candles with no model updates.
    """
    lower, end = parse_utc(start), parse_utc(evaluation_end)
    if lower >= pd.Timestamp("2019-01-01", tz="UTC"):
        raise ValueError("History start harus sebelum 2019 untuk fold 2020")
    final_dates = (
        lower,
        *(
            parse_utc(value)
            for value in (
                tune_start,
                calibration_start,
                policy_start,
                test_start,
            )
        ),
        end,
    )
    if any(a >= b for a, b in zip(final_dates, final_dates[1:], strict=False)):
        raise ValueError("Batas final TRAIN/TUNE/CALIBRATION/POLICY/TEST harus berurutan")
    folds = []
    for year in range(2020, 2026):
        previous = year - 1
        dates = (
            lower,
            pd.Timestamp(f"{previous}-01-01", tz="UTC"),
            pd.Timestamp(f"{previous}-07-01", tz="UTC"),
            pd.Timestamp(f"{previous}-10-01", tz="UTC"),
            pd.Timestamp(f"{year}-01-01", tz="UTC"),
            pd.Timestamp(f"{year + 1}-01-01", tz="UTC"),
        )
        folds.append(Fold(f"fold_{year}", dates))
    folds.append(
        Fold(
            "final_2026",
            final_dates,
        )
    )
    return folds


# %% 06. Purged CV untuk seleksi fitur TRAIN
# Validation internal memakai expanding window hanya di TRAIN.
# Sampel train dibuang jika target timestamp mencapai awal validation.


def purged_cv(
    times: pd.Series, targets: pd.Series, n_splits: int = 3
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return expanding CV indices, removing any train target at/after validation start.

    Operates only on the provided TRAIN data. No imputer is fit before CV; XGBoost
    receives missing values natively, avoiding preprocessing leakage within folds.
    """
    result = []
    for train, test in TimeSeriesSplit(n_splits=n_splits).split(times):
        train = train[(targets.iloc[train] < times.iloc[test[0]]).to_numpy()]
        if len(train) < 10:
            raise ValueError("TRAIN terlalu pendek untuk purged CV")
        result.append((train, test))
    return result
