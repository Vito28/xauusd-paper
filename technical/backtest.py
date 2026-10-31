"""Tabel prediksi out-of-sample vs harga aktual untuk dashboard backtesting.

Setiap periode backtest berasal dari SATU fold walk-forward:
  validation_2021 ... validation_2025 : model dilatih hanya dengan data sebelum tahun itu.
  final_test                          : model dilatih dengan data <= 2025, diuji di 2026.
Jadi setiap prediksi dibuat oleh model yang memang tersedia pada waktu itu.

Tiga ukuran dipisahkan dengan sengaja:
  1. Akurasi klasifikasi : sinyal dibandingkan label triple-barrier (technical/evaluation.py).
  2. Ketepatan arah      : tanda close[t+H] - close[t] dibandingkan arah sinyal.
  3. Simulasi trading    : P&L setelah spread (technical/simulation.py).

Hasil prediksi per sinyal (prediction_result), konsisten dengan definisi label:
  CORRECT   : label aktual sama dengan sinyal (barrier searah tersentuh lebih dulu).
  INCORRECT : label aktual berlawanan (barrier lawan tersentuh lebih dulu).
  NEUTRAL   : label aktual NO_TRADE (tidak ada barrier yang tersentuh, atau dua-duanya
              di candle yang sama). Kenaikan kecil di bawah barrier BUKAN "BUY benar".
  PENDING     : horizon belum selesai (candle t+H belum ada).
  UNAVAILABLE : horizon melewati gap pasar/data, atau ATR/spread tidak tersedia;
                sinyal ini tidak pernah bisa dinilai.
  NO_SIGNAL   : model memilih NO_TRADE.

Struktur file di dalam folder run:
  backtest/candles/<periode>.csv
  backtest/<model>/<periode>/predictions.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from technical.config import CLASS_NAMES
from technical.evaluation import evaluate_predictions
from technical.simulation import InstrumentSpec, simulate_predictions, summarize_trades

CANDLE_COLUMNS = ["open_time_utc", "open", "high", "low", "close", "tick_volume", "spread"]
CLASS_INDEX = {"SELL": 0, "NO_TRADE": 1, "BUY": 2}


def prediction_result(signal: str, actual_label: str, label_status: str) -> str:
    if signal == "NO_TRADE":
        return "NO_SIGNAL"
    if label_status == "PENDING":
        return "PENDING"
    if label_status != "AVAILABLE" or actual_label == "":
        return "UNAVAILABLE"
    if actual_label == signal:
        return "CORRECT"
    if actual_label == "NO_TRADE":
        return "NEUTRAL"
    return "INCORRECT"


def direction_result(signal: str, close: float, future_close: float, label_status: str) -> str:
    """Apakah harga di akhir horizon bergerak searah sinyal (tanpa ambang barrier)."""
    if signal == "NO_TRADE":
        return "NO_SIGNAL"
    if label_status == "PENDING":
        return "PENDING"
    if label_status != "AVAILABLE" or not np.isfinite(future_close):
        return "UNAVAILABLE"
    move = future_close - close
    if move == 0:
        return "FLAT"
    if signal == "BUY" and move > 0:
        return "HIT"
    if signal == "SELL" and move < 0:
        return "HIT"
    return "MISS"


def build_prediction_frame(
    candles: pd.DataFrame,
    labels: pd.DataFrame,
    rows: pd.Index,
    calibrated: np.ndarray,
    decisions: np.ndarray,
    reasons: np.ndarray,
    period: str,
    model_version: str,
) -> pd.DataFrame:
    """Gabungkan prediksi, harga referensi, dan hasil aktual untuk satu periode."""
    frame = pd.DataFrame(index=rows)
    frame["prediction_time"] = labels.loc[rows, "prediction_time"]
    frame["candle_open_time"] = candles.loc[rows, "open_time_utc"]
    frame["label_end_time"] = labels.loc[rows, "label_end_time"]
    frame["close"] = candles.loc[rows, "close"]
    frame["spread"] = candles.loc[rows, "spread"]
    frame["bid_reference"] = frame["close"]
    frame["ask_reference"] = frame["close"] + frame["spread"]
    frame["barrier"] = labels.loc[rows, "barrier"]
    frame["future_close"] = labels.loc[rows, "future_close"]
    frame["up_touch_step"] = labels.loc[rows, "up_touch_step"]
    frame["down_touch_step"] = labels.loc[rows, "down_touch_step"]
    frame["label_status"] = labels.loc[rows, "label_status"]

    actual = []
    for value in labels.loc[rows, "label"]:
        if pd.notna(value):
            actual.append(CLASS_NAMES[int(value)])
        else:
            actual.append("")
    frame["actual_label"] = actual

    frame["probability_sell"] = calibrated[:, 0]
    frame["probability_no_trade"] = calibrated[:, 1]
    frame["probability_buy"] = calibrated[:, 2]
    signals = []
    confidences = []
    for position, decision in enumerate(decisions):
        signals.append(CLASS_NAMES[int(decision)])
        confidences.append(float(calibrated[position, int(decision)]))
    frame["signal"] = signals
    frame["confidence"] = confidences
    frame["reason_code"] = reasons

    frame["actual_return"] = (frame["future_close"] - frame["close"]) / frame["close"]
    results = []
    directions = []
    for _, row in frame.iterrows():
        results.append(prediction_result(row["signal"], row["actual_label"], row["label_status"]))
        directions.append(
            direction_result(row["signal"], row["close"], row["future_close"], row["label_status"])
        )
    frame["prediction_result"] = results
    frame["direction_result"] = directions
    frame["period"] = period
    frame["model_version"] = model_version
    return frame.reset_index(drop=True)


def save_period(
    run_directory: Path,
    model_name: str,
    period: str,
    predictions: pd.DataFrame,
    candles: pd.DataFrame,
) -> None:
    candle_directory = run_directory / "backtest" / "candles"
    candle_directory.mkdir(parents=True, exist_ok=True)
    candle_path = candle_directory / f"{period}.csv"
    if not candle_path.exists():
        candles[CANDLE_COLUMNS].to_csv(candle_path, index=False)

    prediction_directory = run_directory / "backtest" / model_name / period
    prediction_directory.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_directory / "predictions.csv", index=False)


def period_candles(candles: pd.DataFrame, start, end, horizon_bars: int) -> pd.DataFrame:
    """Candle periode evaluasi ditambah H candle sesudahnya (untuk melihat hasil horizon)."""
    inside = (candles["prediction_time"] >= start) & (candles["prediction_time"] < end)
    positions = np.flatnonzero(inside.to_numpy())
    if len(positions) == 0:
        return candles.iloc[0:0]
    last = min(positions[-1] + horizon_bars + 1, len(candles))
    return candles.iloc[positions[0] : last]


# ---------------------------------------------------------------------------
# Membaca hasil untuk dashboard
# ---------------------------------------------------------------------------


def list_backtests(runs_root: Path) -> list[dict]:
    """Daftar run yang memiliki file prediksi backtest, terbaru lebih dulu."""
    found = []
    if not runs_root.exists():
        return found
    for run_directory in sorted(runs_root.iterdir(), reverse=True):
        backtest_directory = run_directory / "backtest"
        if not backtest_directory.is_dir():
            continue
        models = {}
        for model_directory in sorted(backtest_directory.iterdir()):
            if model_directory.name == "candles" or not model_directory.is_dir():
                continue
            periods = []
            for period_directory in sorted(model_directory.iterdir()):
                if (period_directory / "predictions.csv").exists():
                    periods.append(period_directory.name)
            if periods:
                models[model_directory.name] = periods
        if models:
            found.append({"run_id": run_directory.name, "models": models})
    return found


def read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False, na_values=[""])
    for column in ["prediction_time", "candle_open_time", "label_end_time"]:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    for column in ["signal", "actual_label", "label_status", "prediction_result"]:
        frame[column] = frame[column].fillna("").astype(str)
    return frame


def read_candles(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["open_time_utc"] = pd.to_datetime(frame["open_time_utc"], utc=True)
    return frame


def confidence_histogram(predictions: pd.DataFrame, bins: int = 10) -> dict:
    """Distribusi confidence sinyal BUY/SELL, dipisah benar / tidak benar."""
    signals = predictions[predictions["signal"].isin(["BUY", "SELL"])]
    edges = np.linspace(0.3, 1.0, bins + 1)
    result = {"edges": edges.tolist(), "CORRECT": [], "INCORRECT": [], "NEUTRAL": []}
    for outcome in ["CORRECT", "INCORRECT", "NEUTRAL"]:
        values = signals.loc[signals["prediction_result"] == outcome, "confidence"]
        counts, _ = np.histogram(values.clip(0.3, 1.0), bins=edges)
        result[outcome] = counts.tolist()
    return result


def summarize_period(predictions: pd.DataFrame, horizon_bars: int, spec: InstrumentSpec) -> dict:
    """Ringkasan untuk panel Performance Analytics."""
    counts = {}
    for name in CLASS_NAMES:
        counts[name] = int((predictions["signal"] == name).sum())

    evaluated = predictions[predictions["label_status"] == "AVAILABLE"]
    classification = None
    if len(evaluated) > 0:
        actual = evaluated["actual_label"].map(CLASS_INDEX).to_numpy(dtype=int)
        predicted = evaluated["signal"].map(CLASS_INDEX).to_numpy(dtype=int)
        probabilities = evaluated[
            ["probability_sell", "probability_no_trade", "probability_buy"]
        ].to_numpy(dtype=float)
        classification = evaluate_predictions(actual, predicted, probabilities)

    outcome_counts = {}
    for name in ["CORRECT", "INCORRECT", "NEUTRAL", "PENDING", "UNAVAILABLE"]:
        outcome_counts[name] = int((predictions["prediction_result"] == name).sum())

    direction = {}
    for signal in ["BUY", "SELL", "ALL"]:
        if signal == "ALL":
            subset = predictions[predictions["direction_result"].isin(["HIT", "MISS", "FLAT"])]
        else:
            subset = predictions[
                (predictions["signal"] == signal)
                & predictions["direction_result"].isin(["HIT", "MISS", "FLAT"])
            ]
        hits = int((subset["direction_result"] == "HIT").sum())
        total = int(len(subset))
        direction[signal] = {
            "signals": total,
            "hits": hits,
            "hit_rate": hits / total if total > 0 else None,
        }

    trades = simulate_predictions(predictions, horizon_bars, spec)
    simulation = summarize_trades(trades)
    equity = []
    if not trades.empty:
        cumulative = trades["pnl"].cumsum().to_numpy()
        times = pd.to_datetime(trades["prediction_time"], utc=True)
        # Tidak bergantung pada resolusi internal datetime (ns/us/ms) pandas.
        seconds = (times - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
        for position, second in enumerate(seconds.to_numpy()):
            equity.append({"time": int(second), "value": float(cumulative[position])})
    simulation["equity_curve"] = equity
    simulation["instrument"] = spec.to_dict()

    first_time = predictions["prediction_time"].min()
    last_time = predictions["prediction_time"].max()
    return {
        "rows": int(len(predictions)),
        "first_prediction": first_time.isoformat() if len(predictions) else None,
        "last_prediction": last_time.isoformat() if len(predictions) else None,
        "signal_counts": counts,
        "evaluated_rows": int(len(evaluated)),
        "classification": classification,
        "prediction_results": outcome_counts,
        "direction": direction,
        "simulation": simulation,
        "confidence_histogram": confidence_histogram(predictions),
    }
