"""Verify selective precision, model-direction consistency and configurable temporal boundaries."""

import numpy as np
import pandas as pd
import pytest

from technical.config import Config
from technical.data_preparation import make_folds, make_labels
from technical.xgboost_model import (
    apply_selective_policy,
    attach_direction_outcomes,
    decision_costs,
    non_overlapping_positions,
    precision_statistics,
    replay_single_position,
    select_precision_policy,
)


def confirmed_policy() -> dict:
    """Return fixed high-confidence rules with regression interval agreement enabled."""
    return {
        "require_interval_agreement": True,
        "classes": {
            name: {
                "enabled": True,
                "threshold": 0.8,
                "probability_margin": 0.2,
                "edge_multiple": 1.0,
            }
            for name in ("BUY", "SELL")
        },
    }


def test_buy_sell_must_agree_with_regression_cost_and_interval() -> None:
    """Confident contradictory, uncertain or too-small return predictions always abstain."""
    probabilities = np.asarray([[0.02, 0.03, 0.95], [0.95, 0.03, 0.02]] * 4)
    point = np.asarray([-0.002, 0.002, 0.002, -0.002, 0.00005, -0.00005, 0.002, -0.002])
    lower = point - 0.0002
    upper = point + 0.0002
    lower[2], upper[3] = -0.001, 0.001
    result, reasons = apply_selective_policy(
        probabilities, point, np.full(8, 0.0001), confirmed_policy(), lower, upper
    )
    assert result.tolist() == [1, 1, 1, 1, 1, 1, 2, 0]
    assert reasons[0] == reasons[1] == "regression_edge_disagreement"
    assert reasons[2] == reasons[3] == "regression_uncertain"
    assert not ((result == 2) & (point <= 0)).any()
    assert not ((result == 0) & (point >= 0)).any()


def test_zero_signals_do_not_claim_perfect_precision() -> None:
    """Abstention has undefined precision and zero directional recall/F1."""
    actual = np.asarray([0, 2, 0, 2])
    stats = precision_statistics(actual, np.ones(4), 2)
    assert stats["signals"] == 0
    assert stats["precision"] is None and stats["wilson95_lower"] is None
    assert stats["recall"] == stats["f1"] == 0


def test_policy_disables_selection_winner_when_later_confirmation_fails() -> None:
    """Later POLICY outcomes can reject a rule but cannot trigger a new threshold search."""
    dates = pd.date_range("2025-10-01", periods=240, freq="5min", tz="UTC")
    raw = np.tile([0, 2], 120)
    actual = raw.copy()
    actual[120:] = 2 - actual[120:]
    labels = pd.DataFrame(
        {
            "prediction_timestamp": dates,
            "target_timestamp": dates + pd.Timedelta(minutes=15),
            "target_class": actual,
            "threshold_return": 0.0001,
        }
    )
    probabilities = np.full((240, 3), 0.02)
    probabilities[np.arange(240), raw] = 0.96
    returns = np.where(raw == 2, 0.002, -0.002)
    policy = select_precision_policy(
        probabilities,
        returns,
        labels,
        returns - 0.0002,
        returns + 0.0002,
        Config(min_policy_signals=8),
    )
    for rule in policy["classes"].values():
        assert rule["selection"]["precision"] == 1
        assert rule["confirmation"]["precision"] == 0
        assert rule["status"] == "confirmation_failed"
        assert not rule["enabled"]
    predictions, _ = apply_selective_policy(
        probabilities,
        returns,
        labels.threshold_return.to_numpy(),
        policy,
        returns - 0.0002,
        returns + 0.0002,
    )
    assert (predictions == 1).all()
    positions = non_overlapping_positions(labels)
    assert (
        labels.target_timestamp.iloc[positions[:-1]].to_numpy()
        <= labels.prediction_timestamp.iloc[positions[1:]].to_numpy()
    ).all()


def test_final_development_dates_are_configurable_and_ordered() -> None:
    """The final period can start in January with all fitting blocks before that date."""
    fold = make_folds(
        tune_start="2024-07-01T00:00:00Z",
        calibration_start="2025-01-01T00:00:00Z",
        policy_start="2025-07-01T00:00:00Z",
        test_start="2026-01-01T00:00:00Z",
    )[-1]
    assert fold.boundaries[-2] == pd.Timestamp("2026-01-01", tz="UTC")
    with pytest.raises(ValueError, match="berurutan"):
        make_folds(tune_start="2026-07-01T00:00:00Z")
    with pytest.raises(ValueError, match="berurutan"):
        Config(final_test_start="2025-01-01T00:00:00Z")


def test_direction_outcomes_keep_abstentions_neutral_and_costs_separate() -> None:
    """A tiny correct move can lose after costs; every abstention is still neutral."""
    rows = pd.DataFrame(
        {
            "prediction_timestamp": pd.date_range("2026-01-01", periods=8, freq="5min", tz="UTC"),
            "target_timestamp": pd.date_range("2026-01-01 00:15", periods=8, freq="5min", tz="UTC"),
            "predicted_signal": [
                "BUY",
                "SELL",
                "BUY",
                "SELL",
                "NO_TRADE",
                "NO_TRADE",
                "BUY",
                "SELL",
            ],
            "close": [100.0] * 8,
            "actual_future_close": [100.01, 99, 99, 101, 110, 90, 100, np.nan],
            "realized_net_move": [-0.01, 0.98, -1.02, -1.02, np.nan, np.nan, -0.02, np.nan],
        }
    )
    out = attach_direction_outcomes(rows)
    assert out.outcome_status.tolist() == [
        "SUCCESS",
        "SUCCESS",
        "FAILURE",
        "FAILURE",
        "NEUTRAL",
        "NEUTRAL",
        "FLAT",
        "PENDING",
    ]
    assert out.net_outcome.iloc[0] == "LOSS"
    assert out.prediction_correct.iloc[4:6].isna().all()
    assert out.prediction_correct.iloc[:4].tolist() == [True, True, False, False]


def test_slippage_enters_labels_exactly_once_and_gaps_are_not_pending() -> None:
    """Costs consistent with labels; closed-history gaps and future targets differ."""
    frame = pd.DataFrame(
        {
            "prediction_timestamp": pd.date_range("2025-01-01", periods=6, freq="5min", tz="UTC"),
            "close": [100.0, 100.0, 100.0, 100.05, 100.05, 100.05],
            "spread_price": 0.01,
        }
    )
    labels = make_labels(frame, slippage_bps=10)
    assert labels.target_class.iloc[0] == 1
    np.testing.assert_allclose(decision_costs(labels, 10), labels.threshold_return)
    gap = make_labels(frame.drop(index=2))
    assert gap.actual_status.iloc[0] == "GAP_INVALID"
    assert gap.actual_status.iloc[-1] == "PENDING"


def test_execution_replay_delays_fills_and_rejects_overlapping_signals() -> None:
    """Replay uses next delayed bar prices, blocks overlap, and pays bid/ask spread."""
    times = pd.date_range("2026-01-01", periods=8, freq="5min", tz="UTC")
    candles = pd.DataFrame(
        {
            "open_time_utc": times,
            "prediction_timestamp": times + pd.Timedelta(minutes=5),
            "open": np.arange(100.0, 108.0),
            "close": np.arange(101.0, 109.0),
            "spread_price": 0.2,
        }
    )
    signals = pd.DataFrame(
        {
            "prediction_timestamp": [times[1], times[2]],
            "target_timestamp": [times[4], times[5]],
            "predicted_signal": ["BUY", "SELL"],
        }
    )
    result = replay_single_position(signals, candles, delay_bars=1, commission_bps=1)
    assert result["closed_trades"] == 1
    assert result["records"][0]["entry_price"] == 102.2
    assert result["records"][1]["status"] == "SKIPPED_OVERLAP"
    assert result["net_price_move"] == pytest.approx(104 - 102.2 - 102 / 10000)


@pytest.mark.parametrize("side,exit_bid,gross", [("SELL", 4100.0, 200.0), ("BUY", 4100.0, -200.0)])
def test_profit_breakdown_charges_spread_once(side: str, exit_bid: float, gross: float) -> None:
    """A 4300-to-4100 move is +200 for SELL, -200 for BUY, before explicit costs."""
    times = pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC")
    candles = pd.DataFrame(
        {
            "open_time_utc": times,
            "prediction_timestamp": times + pd.Timedelta(minutes=5),
            "open": 4300.0,
            "close": exit_bid,
            "spread_price": 0.5,
        }
    )
    signals = pd.DataFrame(
        {
            "prediction_timestamp": [times[1]],
            "target_timestamp": [times[4]],
            "predicted_signal": [side],
        }
    )
    result = replay_single_position(signals, candles, slippage_bps=1, commission_bps=2)
    trade = result["records"][0]
    assert trade["gross_price_move"] == gross
    assert trade["spread_cost"] == 0.5
    assert trade["slippage_cost"] == pytest.approx(0.43)
    assert trade["commission_cost"] == pytest.approx(0.86)
    assert trade["net_price_move"] == pytest.approx(gross - 0.5 - 0.43 - 0.86)
    assert result["by_side"][side]["closed_trades"] == 1
    assert result["by_side"][side]["net_price_move"] == result["net_price_move"]
