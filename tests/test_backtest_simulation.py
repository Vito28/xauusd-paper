"""Uji tabel prediksi backtest dan simulasi trading offline."""

import numpy as np
import pandas as pd
import pytest

from technical.backtest import (
    build_prediction_frame,
    direction_result,
    prediction_result,
    summarize_period,
)
from technical.config import TechnicalConfig
from technical.data import prepare_candles
from technical.run import build_dataset, predict_period, run_fold
from technical.simulation import InstrumentSpec, simulate_predictions, simulate_trade
from technical.splits import Fold
from tests.test_technical_pipeline import make_candles

SPEC = InstrumentSpec(lot=0.01, contract_size=100.0, leverage=100.0)


# ---------------------------------------------------------------------------
# Simulasi satu transaksi
# ---------------------------------------------------------------------------


def trade(direction, up_step, down_step, future_close=100.5):
    return simulate_trade(
        direction=direction,
        close=100.0,
        spread=0.2,
        barrier=1.2,
        future_close=future_close,
        up_step=up_step,
        down_step=down_step,
        horizon_bars=6,
        spec=SPEC,
    )


def test_buy_take_profit_earns_barrier_minus_spread() -> None:
    result = trade("BUY", up_step=2, down_step=np.nan)
    assert result.exit_reason == "TP"
    assert result.entry_price == pytest.approx(100.2)  # ask
    assert result.net_move == pytest.approx(1.0)  # barrier 1.2 - spread 0.2 = k*ATR
    assert result.pnl == pytest.approx(1.0 * 0.01 * 100)


def test_buy_stop_loss_costs_barrier_plus_spread() -> None:
    result = trade("BUY", up_step=5, down_step=3)
    assert result.exit_reason == "SL"
    assert result.net_move == pytest.approx(-1.4)


def test_sell_take_profit_buys_back_at_ask() -> None:
    result = trade("SELL", up_step=np.nan, down_step=1)
    assert result.exit_reason == "TP"
    assert result.entry_price == pytest.approx(100.0)  # bid
    assert result.exit_price == pytest.approx(100.0 - 1.2 + 0.2)
    assert result.net_move == pytest.approx(1.0)


def test_both_sides_same_candle_is_conservative_loss() -> None:
    result = trade("SELL", up_step=2, down_step=2)
    assert result.exit_reason == "BOTH_SAME_CANDLE"
    assert result.net_move == pytest.approx(-1.4)


def test_time_exit_uses_close_at_horizon() -> None:
    result = trade("BUY", up_step=np.nan, down_step=np.nan, future_close=100.5)
    assert result.exit_reason == "TIME"
    assert result.exit_step == 6
    assert result.net_move == pytest.approx(100.5 - 100.2)
    assert result.margin == pytest.approx(100.2 * 0.01 * 100 / 100)


def test_vectorized_simulation_matches_single_trade_function() -> None:
    rng = np.random.default_rng(3)
    count = 300
    up = rng.choice([np.nan, 1, 2, 3, 4, 5, 6], size=count)
    down = rng.choice([np.nan, 1, 2, 3, 4, 5, 6], size=count)
    predictions = pd.DataFrame(
        {
            "prediction_time": pd.date_range("2026-01-01", periods=count, freq="5min", tz="UTC"),
            "signal": rng.choice(["BUY", "SELL", "NO_TRADE"], size=count),
            "label_status": rng.choice(["AVAILABLE", "AVAILABLE", "GAP"], size=count),
            "close": 2000 + rng.normal(0, 5, count),
            "spread": rng.uniform(0.1, 0.4, count),
            "barrier": rng.uniform(0.5, 2.0, count),
            "future_close": 2000 + rng.normal(0, 5, count),
            "up_touch_step": up,
            "down_touch_step": down,
        }
    )
    trades = simulate_predictions(predictions, 6, SPEC)

    expected = []
    for _, row in predictions.iterrows():
        if row["signal"] == "NO_TRADE" or row["label_status"] != "AVAILABLE":
            continue
        expected.append(trade_from_row(row))
    assert len(trades) == len(expected)
    for position, single in enumerate(expected):
        assert trades["exit_reason"].iloc[position] == single.exit_reason
        assert trades["net_move"].iloc[position] == pytest.approx(single.net_move)
        assert trades["exit_step"].iloc[position] == single.exit_step


def trade_from_row(row):
    return simulate_trade(
        direction=row["signal"],
        close=row["close"],
        spread=row["spread"],
        barrier=row["barrier"],
        future_close=row["future_close"],
        up_step=row["up_touch_step"],
        down_step=row["down_touch_step"],
        horizon_bars=6,
        spec=SPEC,
    )


# ---------------------------------------------------------------------------
# Hasil prediksi
# ---------------------------------------------------------------------------


def test_prediction_result_follows_label_definition() -> None:
    assert prediction_result("BUY", "BUY", "AVAILABLE") == "CORRECT"
    assert prediction_result("BUY", "SELL", "AVAILABLE") == "INCORRECT"
    # Naik sedikit tetapi barrier tidak tersentuh -> label NO_TRADE -> bukan "BUY benar".
    assert prediction_result("BUY", "NO_TRADE", "AVAILABLE") == "NEUTRAL"
    assert prediction_result("SELL", "", "PENDING") == "PENDING"
    assert prediction_result("SELL", "", "GAP") == "UNAVAILABLE"
    assert prediction_result("NO_TRADE", "BUY", "AVAILABLE") == "NO_SIGNAL"


def test_direction_result_ignores_barrier() -> None:
    assert direction_result("BUY", 100.0, 100.1, "AVAILABLE") == "HIT"
    assert direction_result("SELL", 100.0, 100.1, "AVAILABLE") == "MISS"
    assert direction_result("SELL", 100.0, 100.0, "AVAILABLE") == "FLAT"
    assert direction_result("BUY", 100.0, np.nan, "PENDING") == "PENDING"


def test_build_prediction_frame_and_summary() -> None:
    candles, _ = prepare_candles(make_candles(400))
    config = TechnicalConfig()
    _, _, labels, _ = build_dataset(candles, config, config.features.groups)
    rows = labels.index[100:200]
    probabilities = np.tile([0.2, 0.2, 0.6], (len(rows), 1))
    decisions = np.full(len(rows), 2)
    reasons = np.full(len(rows), "BUY_PROBABILITY_ABOVE_THRESHOLD", dtype=object)
    frame = build_prediction_frame(
        candles, labels, rows, probabilities, decisions, reasons, "test", "model-x"
    )
    assert (frame["ask_reference"] - frame["bid_reference"]).round(6).eq(0.2).all()
    assert set(frame["prediction_result"]) <= {"CORRECT", "INCORRECT", "NEUTRAL"}

    summary = summarize_period(frame, config.labels.horizon_bars, SPEC)
    outcomes = summary["prediction_results"]
    assert outcomes["CORRECT"] + outcomes["INCORRECT"] + outcomes["NEUTRAL"] == len(rows)
    # TP <-> CORRECT dan SL <-> INCORRECT (tanpa sentuhan dua sisi pada data ini).
    exits = summary["simulation"]["exit_reasons"]
    assert exits.get("TP", 0) == outcomes["CORRECT"]
    assert exits.get("SL", 0) + exits.get("BOTH_SAME_CANDLE", 0) >= outcomes["INCORRECT"]
    times = [point["time"] for point in summary["simulation"]["equity_curve"]]
    assert times == sorted(set(times))
    assert times[0] == int(frame["prediction_time"].iloc[0].timestamp())


def test_predict_period_only_covers_evaluation_period() -> None:
    raw = make_candles(3000, start="2024-01-01T00:00:00Z", seed=5)
    candles, _ = prepare_candles(raw)
    config = TechnicalConfig()
    config.xgboost.n_estimators = 20
    config.decision.min_signals = 1
    features, _, labels, baseline_inputs = build_dataset(candles, config, config.features.groups)
    fold = Fold(
        name="mini",
        train_start=pd.Timestamp("2024-01-01", tz="UTC"),
        calibration_start=pd.Timestamp("2024-01-05", tz="UTC"),
        threshold_start=pd.Timestamp("2024-01-07", tz="UTC"),
        evaluation_start=pd.Timestamp("2024-01-09", tz="UTC"),
        evaluation_end=pd.Timestamp("2024-01-11", tz="UTC"),
    )
    result = run_fold("xgboost", fold, features, labels, baseline_inputs, config)
    frame, chart_candles, *_ = predict_period(result, fold, candles, features, labels, config, "v")
    assert frame["prediction_time"].min() >= fold.evaluation_start
    assert frame["prediction_time"].max() < fold.evaluation_end
    assert frame["period"].eq("mini").all()
    # Grafik ikut menampilkan H candle setelah periode agar hasil horizon terlihat.
    assert chart_candles["prediction_time"].max() >= fold.evaluation_end
