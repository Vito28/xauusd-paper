"""Tests for the consolidated single-file Matplotlib analytics module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from analytics.analytics import (
    FORECAST_START_UTC,
    align_actual_and_forecast,
    build_lagged_training_data,
    build_parser,
    calculate_forecast_metrics,
    create_interactive_dashboard,
    forecast_dates_for_actual,
    forecast_recursively,
    paired_values,
    plot_actual_vs_predicted,
    plot_confusion_matrix,
    plot_correlation_matrix,
    plot_error_distribution,
    plot_feature_histogram,
    plot_histogram,
    plot_historical_forecast_actual,
    plot_missing_values,
    plot_monthly_comparison,
    plot_residuals,
    plot_scatter,
    plot_time_series,
    split_training_and_actual,
    train_xgboost_forecaster,
    validate_evaluation_split,
    validate_feature_selection,
)


def market_frame() -> pd.DataFrame:
    """Return small market data with NaN and infinity edge cases."""

    start = datetime(2026, 1, 1, tzinfo=UTC)
    return pd.DataFrame(
        {
            "open_time_utc": [start + timedelta(minutes=5 * index) for index in range(4)],
            "close": [2600.0, 2601.0, np.nan, np.inf],
            "tick_volume": [100.0, np.nan, 120.0, 130.0],
            "spread_price": [0.2, 0.3, 0.2, 0.4],
        }
    )


def prediction_frame() -> pd.DataFrame:
    """Return one test split containing all three target classes."""

    return pd.DataFrame(
        {
            "decision_time": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z", "2026-01-01T00:10:00Z"]
            ),
            "evaluation_split": ["test", "test", "test"],
            "target_class": ["SELL", "NO_TRADE", "BUY"],
            "predicted_class": ["SELL", "BUY", "BUY"],
            "p_sell": [0.8, 0.1, 0.1],
            "p_no_trade": [0.1, 0.3, 0.1],
            "p_buy": [0.1, 0.6, 0.8],
        }
    )


def assert_png(path: Path) -> None:
    """Assert an analytics output is a non-empty PNG artifact."""

    assert path.suffix == ".png"
    assert path.exists()
    assert path.stat().st_size > 0
    assert plt.get_fignums() == []


def test_univariate_plots_create_non_empty_png(tmp_path: Path) -> None:
    """Histogram, time series, and missing-value plots save and close figures."""

    frame = market_frame()
    assert_png(plot_histogram(frame, "close", tmp_path / "histogram.png"))
    assert_png(
        plot_time_series(
            frame,
            "open_time_utc",
            "close",
            tmp_path / "time_series.png",
        )
    )
    assert_png(
        plot_missing_values(
            frame,
            ["close", "tick_volume"],
            tmp_path / "missing.png",
        )
    )


def test_bivariate_drops_nan_only_for_requested_pair(tmp_path: Path) -> None:
    """Scatter pairing does not drop rows or mutate unrelated source columns."""

    frame = market_frame()
    original = frame.copy(deep=True)
    pair = paired_values(frame, "tick_volume", "spread_price")
    assert len(pair) == 3
    pd.testing.assert_frame_equal(frame, original)
    assert_png(
        plot_scatter(
            frame,
            "tick_volume",
            "spread_price",
            tmp_path / "scatter.png",
        )
    )


def test_multivariate_correlation_creates_png(tmp_path: Path) -> None:
    """Explicit numeric columns produce a non-empty diagnostic matrix."""

    assert_png(
        plot_correlation_matrix(
            market_frame(),
            ["close", "tick_volume", "spread_price"],
            tmp_path / "correlation.png",
        )
    )


def test_feature_plot_preserves_frame_and_rejects_target(tmp_path: Path) -> None:
    """Feature EDA leaves source data unchanged and blocks target leakage."""

    frame = pd.DataFrame(
        {"rsi_14": [30.0, np.nan, 70.0], "target_class": ["SELL", "NO_TRADE", "BUY"]}
    )
    original = frame.copy(deep=True)
    assert_png(
        plot_feature_histogram(
            frame,
            "rsi_14",
            tmp_path / "feature.png",
        )
    )
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(ValueError, match="predictor"):
        validate_feature_selection(frame, ["target_class"])


def test_confusion_matrix_rejects_mixed_splits(tmp_path: Path) -> None:
    """Model evaluation creates PNG for one split and rejects train/test mixing."""

    frame = prediction_frame()
    assert_png(
        plot_confusion_matrix(
            frame,
            tmp_path / "confusion.png",
        )
    )
    mixed = pd.concat([frame, frame.assign(evaluation_split="train")], ignore_index=True)
    with pytest.raises(ValueError, match="mencampur"):
        validate_evaluation_split(mixed)


def test_errors_are_clear_for_empty_or_missing_columns(tmp_path: Path) -> None:
    """Empty numeric data and missing columns fail before writing artifacts."""

    with pytest.raises(ValueError, match="Kolom tidak ditemukan"):
        plot_histogram(market_frame(), "unknown", tmp_path / "missing.png")
    with pytest.raises(ValueError, match="Tidak ada nilai numerik"):
        plot_histogram(
            pd.DataFrame({"close": [np.nan, np.inf]}),
            "close",
            tmp_path / "empty.png",
        )


def test_reports_can_be_replaced_for_refresh(tmp_path: Path) -> None:
    """Repeated analytics runs replace derived PNGs instead of failing."""

    output = tmp_path / "refresh.png"
    frame = market_frame()
    assert_png(plot_histogram(frame, "close", output))
    assert_png(plot_histogram(frame.assign(close=frame["close"] * 2), "close", output))
    assert output.stat().st_size > 0


def test_eda_parser_accepts_watch_interval() -> None:
    """EDA CLI exposes an optional polling interval for automatic refresh."""

    arguments = build_parser().parse_args(
        [
            "univariate",
            "--source-id",
            "source",
            "--symbol",
            "XAUUSD",
            "--timeframe",
            "M5",
            "--start",
            "2026-01-01T00:00:00Z",
            "--end",
            "2026-02-01T00:00:00Z",
            "--watch-seconds",
            "60",
            "--columns",
            "close",
        ]
    )
    assert arguments.watch_seconds == 60


def daily_close_frame() -> pd.DataFrame:
    """Return deterministic daily closes spanning the training cutoff."""

    dates = pd.date_range("2025-10-01", "2026-01-16", freq="B", tz="UTC")
    return pd.DataFrame(
        {"timestamp_utc": dates, "close": 2500.0 + np.arange(len(dates), dtype=float)}
    )


class ConstantRegressor:
    """Minimal deterministic estimator for recursive forecasting tests."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return the prior forecast plus one without observing actual data."""

        return features["lag_1"].to_numpy() + 1.0


def test_xgboost_forecasting_excludes_2026_from_training_and_state() -> None:
    """Training rows and recursive state reject actual prices after the cutoff."""

    history, actual = split_training_and_actual(daily_close_frame())
    training, expected_features = build_lagged_training_data(history, lag_count=5)
    assert history["timestamp_utc"].max() < FORECAST_START_UTC
    assert actual["timestamp_utc"].min() >= FORECAST_START_UTC
    assert training["timestamp_utc"].max() < FORECAST_START_UTC
    model, features = train_xgboost_forecaster(
        history,
        lag_count=5,
        model_parameters={"n_estimators": 5, "max_depth": 2},
    )
    assert features == expected_features
    forecast = forecast_recursively(
        model,
        history,
        forecast_dates_for_actual(actual),
        lag_count=5,
    )
    assert forecast["timestamp_utc"].min() >= FORECAST_START_UTC
    with pytest.raises(ValueError, match="tidak boleh memuat harga mulai"):
        train_xgboost_forecaster(daily_close_frame(), lag_count=5)
    with pytest.raises(ValueError, match="tidak boleh memuat harga mulai"):
        forecast_recursively(
            ConstantRegressor(),
            daily_close_frame(),
            forecast_dates_for_actual(actual),
            lag_count=5,
        )


def test_xgboost_visualisation_reports_create_pngs(tmp_path: Path) -> None:
    """The reusable comparison visualisations create closed, non-empty PNGs."""

    history, actual = split_training_and_actual(daily_close_frame())
    forecast = forecast_recursively(
        ConstantRegressor(),
        history,
        forecast_dates_for_actual(actual),
        lag_count=5,
    )
    comparison = align_actual_and_forecast(actual, forecast)
    metrics = calculate_forecast_metrics(comparison)
    assert set(metrics) == {"mae", "mse", "rmse", "mape_percent", "observations"}
    assert_png(plot_historical_forecast_actual(history, forecast, actual, tmp_path / "main.png"))
    assert_png(plot_actual_vs_predicted(comparison, tmp_path / "actual_vs_predicted.png"))
    assert_png(plot_residuals(comparison, tmp_path / "residuals.png"))
    assert_png(plot_monthly_comparison(comparison, tmp_path / "monthly.png"))
    assert_png(plot_error_distribution(comparison, tmp_path / "errors.png"))
    dashboard = create_interactive_dashboard(
        history,
        forecast,
        actual,
        comparison,
        metrics,
        tmp_path / "dashboard.html",
    )
    assert dashboard.exists()
    assert dashboard.stat().st_size > 0
