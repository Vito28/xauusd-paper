"""Uji data, fitur, label, split, decision layer, dan evaluasi Technical Agent."""

import numpy as np
import pandas as pd
import pytest

from technical.config import BUY, NO_TRADE, SELL, DecisionConfig, TechnicalConfig
from technical.data import prepare_candles
from technical.decision import apply_temperature, decide, fit_temperature, select_thresholds
from technical.evaluation import evaluate_predictions
from technical.features import build_features
from technical.labels import make_labels
from technical.run import build_dataset, run_fold
from technical.splits import Fold, assert_no_overlap, make_final_fold, split_fold


def make_candles(count: int, start: str = "2024-01-02T00:00:00Z", seed: int = 0) -> pd.DataFrame:
    """Random walk M5 sintetis dengan OHLC valid dan spread konstan."""
    rng = np.random.default_rng(seed)
    open_time = pd.date_range(start, periods=count, freq="5min", tz="UTC")
    close = 2000 + np.cumsum(rng.normal(0, 1.0, count))
    open_ = np.concatenate([[2000.0], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0, 0.8, count)
    low = np.minimum(open_, close) - rng.uniform(0, 0.8, count)
    return pd.DataFrame(
        {
            "open_time_utc": open_time,
            "close_time_utc": open_time + pd.Timedelta(minutes=5),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": rng.integers(50, 500, count),
            "native_spread_price": np.full(count, 0.2),
        }
    )


def prepared(count: int = 600, seed: int = 0) -> pd.DataFrame:
    candles, _ = prepare_candles(make_candles(count, seed=seed))
    return candles


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def test_prepare_rejects_invalid_ohlc_and_marks_gaps() -> None:
    raw = make_candles(10)
    raw.loc[3, "high"] = raw.loc[3, "low"] - 1  # OHLC tidak valid
    raw = raw.drop(index=6)  # membuat gap
    candles, report = prepare_candles(raw)
    assert report["invalid_ohlc_rows"] == 1
    assert len(candles) == 8
    assert candles["gap_before"].sum() == 3  # baris pertama + dua gap


# ---------------------------------------------------------------------------
# Fitur: tidak boleh memakai masa depan
# ---------------------------------------------------------------------------


def test_features_do_not_change_when_future_candles_change() -> None:
    candles = prepared(800)
    groups = ["price_action", "trend", "momentum", "volatility", "volume", "spread", "time"]
    original, _ = build_features(candles, groups)

    modified = candles.copy()
    cut = 500
    modified.loc[cut:, ["open", "high", "low", "close"]] *= 1.5
    modified.loc[cut:, "tick_volume"] *= 3
    changed, _ = build_features(modified, groups)

    pd.testing.assert_frame_equal(original.iloc[:cut], changed.iloc[:cut])


def test_feature_groups_are_selectable() -> None:
    candles = prepared(400)
    features, groups = build_features(candles, ["time"])
    assert set(groups.values()) == {"time"}
    assert "rsi_14" not in features.columns


# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------


def flat_candles(count: int) -> pd.DataFrame:
    """Candle datar 2000 dengan range kecil: ATR stabil sekitar 1."""
    open_time = pd.date_range("2024-01-02", periods=count, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open_time_utc": open_time,
            "open": 2000.0,
            "high": 2000.5,
            "low": 1999.5,
            "close": 2000.0,
            "tick_volume": 100,
            "native_spread_price": 0.0,
        }
    )
    candles, _ = prepare_candles(frame)
    return candles


def test_triple_barrier_first_touch_decides_label() -> None:
    candles = flat_candles(60)
    t = 40
    candles.loc[t + 2, "high"] = 2005.0  # barrier atas tersentuh di langkah 2
    candles.loc[t + 4, "low"] = 1995.0  # barrier bawah baru di langkah 4
    labels = make_labels(
        candles, "triple_barrier", horizon_bars=6, atr_period=14, barrier_atr_multiple=1.0
    )
    assert labels.loc[t, "label"] == BUY


def test_triple_barrier_both_sides_same_candle_is_no_trade() -> None:
    candles = flat_candles(60)
    t = 40
    candles.loc[t + 1, "high"] = 2005.0
    candles.loc[t + 1, "low"] = 1995.0
    labels = make_labels(
        candles, "triple_barrier", horizon_bars=6, atr_period=14, barrier_atr_multiple=1.0
    )
    assert labels.loc[t, "label"] == NO_TRADE


def test_spread_widens_barrier() -> None:
    candles = flat_candles(60)
    t = 40
    candles.loc[t + 1, "low"] = 1998.7  # turun 1.3 dari close
    no_spread = make_labels(candles, "triple_barrier", 6, 14, 1.0)
    candles["spread"] = 0.5  # barrier = ~1.0 + 0.5 = 1.5 -> tidak tersentuh
    with_spread = make_labels(candles, "triple_barrier", 6, 14, 1.0)
    assert no_spread.loc[t, "label"] == SELL
    assert with_spread.loc[t, "label"] == NO_TRADE


def test_labels_unknown_across_gap_and_at_end() -> None:
    raw = make_candles(80)
    raw = raw.drop(index=50)
    candles, _ = prepare_candles(raw)
    labels = make_labels(candles, "triple_barrier", 6, 14, 1.0)
    # Baris 45..49 (sebelum gap) memiliki horizon yang melewati gap.
    assert labels.loc[45:49, "label"].isna().all()
    assert (labels.loc[45:49, "label_status"] == "GAP").all()
    assert labels["label"].iloc[-6:].isna().all()
    assert (labels["label_status"].iloc[-6:] == "PENDING").all()


def test_fixed_horizon_uses_close_at_horizon() -> None:
    candles = flat_candles(60)
    t = 40
    candles.loc[t + 6, "close"] = 2003.0
    candles.loc[t + 6, "high"] = 2003.0
    labels = make_labels(candles, "fixed_horizon", 6, 14, 1.0)
    assert labels.loc[t, "label"] == BUY


# ---------------------------------------------------------------------------
# Split: kronologis, purging, embargo
# ---------------------------------------------------------------------------


def test_purging_removes_labels_that_cross_block_boundary() -> None:
    candles = prepared(3000)
    labels = make_labels(candles, "triple_barrier", 6, 14, 1.0)
    start = candles["prediction_time"].iloc[0]
    fold = Fold(
        name="test",
        train_start=start,
        calibration_start=start + pd.Timedelta(days=4),
        threshold_start=start + pd.Timedelta(days=6),
        evaluation_start=start + pd.Timedelta(days=8),
        evaluation_end=start + pd.Timedelta(days=10),
    )
    blocks = split_fold(labels, fold, embargo_bars=6)
    assert_no_overlap(labels, blocks)
    train_end = labels.loc[blocks["train"], "label_end_time"].max()
    assert train_end + pd.Timedelta(minutes=30) <= fold.calibration_start


def test_final_fold_fits_only_on_data_before_development_end() -> None:
    fold = make_final_fold("2016-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2027-01-01T00:00:00Z")
    bounds = fold.block_bounds()
    development_end = pd.Timestamp("2026-01-01", tz="UTC")
    for block_name in ["train", "calibration", "threshold"]:
        assert bounds[block_name][1] <= development_end
    assert bounds["evaluation"][0] == development_end


# ---------------------------------------------------------------------------
# Decision layer
# ---------------------------------------------------------------------------


def test_temperature_scaling_reduces_log_loss_of_overconfident_model() -> None:
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 3, 2000)
    probabilities = np.full((2000, 3), 0.05)
    noisy = np.where(rng.random(2000) < 0.5, labels, rng.integers(0, 3, 2000))
    probabilities[np.arange(2000), noisy] = 0.9
    result = fit_temperature(probabilities, labels)
    assert result["temperature"] > 1.0
    assert result["log_loss_after"] < result["log_loss_before"]
    calibrated = apply_temperature(probabilities, result["temperature"])
    assert np.allclose(calibrated.sum(axis=1), 1.0)


def test_threshold_disabled_when_precision_is_not_better_than_prevalence() -> None:
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 3, 3000)
    probabilities = rng.dirichlet([1, 1, 1], 3000)  # tidak informatif
    rules = select_thresholds(
        probabilities, labels, DecisionConfig(min_precision_lift=1.5, min_signals=30)
    )
    decisions, reasons = decide(probabilities, rules)
    assert not rules["BUY"]["enabled"]
    assert not rules["SELL"]["enabled"]
    assert (decisions == NO_TRADE).all()
    assert "BUY_DISABLED_NO_VALID_THRESHOLD" in set(reasons)


def test_threshold_enabled_for_informative_probabilities() -> None:
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 3, 3000)
    probabilities = rng.dirichlet([1, 1, 1], 3000) * 0.3
    probabilities[np.arange(3000), labels] += 0.7
    rules = select_thresholds(probabilities, labels, DecisionConfig())
    assert rules["BUY"]["enabled"]
    assert rules["BUY"]["chosen"]["precision"] >= rules["BUY"]["required_precision"]


# ---------------------------------------------------------------------------
# Evaluasi
# ---------------------------------------------------------------------------


def test_buy_on_no_trade_label_is_false_positive() -> None:
    actual = np.array([BUY, NO_TRADE, SELL, NO_TRADE])
    predicted = np.array([BUY, BUY, BUY, NO_TRADE])
    metrics = evaluate_predictions(actual, predicted)
    buy = metrics["per_class"]["BUY"]
    assert buy["precision"] == pytest.approx(1 / 3)
    assert buy["false_positive"] == 2
    assert metrics["opposite_direction_errors"]["BUY_when_SELL"] == 1
    assert metrics["signal_coverage"] == pytest.approx(0.75)


def test_no_signal_gives_undefined_precision_not_perfect() -> None:
    actual = np.array([BUY, SELL, NO_TRADE])
    predicted = np.array([NO_TRADE, NO_TRADE, NO_TRADE])
    metrics = evaluate_predictions(actual, predicted)
    assert metrics["per_class"]["BUY"]["precision"] is None
    assert metrics["per_class"]["BUY"]["f1"] == 0.0
    assert metrics["signal_coverage"] == 0.0


# ---------------------------------------------------------------------------
# Smoke test seluruh fold (data sintetis, bukan hasil penelitian)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name",
    [
        "always_no_trade",
        "persistence",
        "logistic",
        "xgboost",
        "xgboost_balanced",
        "xgboost_scratch",
        "random_forest",
        "ensemble",
    ],
)
def test_run_fold_end_to_end_on_synthetic_data(model_name: str) -> None:
    config = TechnicalConfig()
    config.xgboost.n_estimators = 20
    config.scratch.n_estimators = 5
    config.random_forest.n_estimators = 20
    config.decision.min_signals = 5
    config.run.n_jobs = 1

    candles = prepared(6000, seed=7)
    features, _, labels, baseline_inputs = build_dataset(candles, config, config.features.groups)
    start = candles["prediction_time"].iloc[0]
    fold = Fold(
        name="synthetic",
        train_start=start,
        calibration_start=start + pd.Timedelta(days=12),
        threshold_start=start + pd.Timedelta(days=15),
        evaluation_start=start + pd.Timedelta(days=18),
        evaluation_end=start + pd.Timedelta(days=21),
    )
    result = run_fold(model_name, fold, features, labels, baseline_inputs, config)
    evaluation = result["evaluation"]
    assert evaluation["samples"] == result["block_sizes"]["evaluation"]
    assert sum(evaluation["predicted_distribution"].values()) == evaluation["samples"]
