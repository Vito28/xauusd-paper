"""Verify gap handling, immutable preparation and temporal analytics models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from analytics.analytics import evaluate_return_models, run_time_series_analysis
from analytics.data import monthly_returns, prepare_daily_summary, prepare_time_series


def native_history() -> pd.DataFrame:
    """Build varied native candles across pre-2026 history and a held-out period."""

    days = pd.date_range("2025-09-01", periods=180, freq="B", tz="UTC")
    times = pd.DatetimeIndex([day + pd.Timedelta(minutes=i * 5) for day in days for i in range(4)])
    rng = np.random.default_rng(7)
    closes = 2000 * np.exp(np.cumsum(rng.normal(0.0001, 0.001, len(times))))
    return pd.DataFrame(
        {
            "open_time_utc": times,
            "close_time_utc": times + pd.Timedelta(minutes=5),
            "source_id": "one-source",
            "actual_symbol": "XAUUSD",
            "timeframe": "M5",
            "open": closes,
            "high": closes + 2,
            "low": closes - 2,
            "close": closes,
            "tick_volume": rng.integers(1, 200, len(times)),
            "spread_price": rng.uniform(0.1, 0.4, len(times)),
        }
    )


def test_native_returns_do_not_bridge_gaps_invalid_prices_or_unclosed_candles() -> None:
    """Missing prices and timestamp gaps remain missing rather than becoming one-bar returns."""

    frame = native_history().iloc[:8].copy()
    original = frame.copy(deep=True)
    frame.loc[1, "close"] = np.inf
    prepared = prepare_time_series(frame, as_of=frame["close_time_utc"].iloc[-2])
    assert prepared.attrs["quality"]["unclosed_bars"] == 1
    assert prepared.attrs["quality"]["invalid_price_bars"] == 1
    assert prepared["log_return"].iloc[[0, 1, 2, 4]].isna().all()
    assert prepared["log_return"].iloc[3] == pytest.approx(
        np.log(frame["close"].iloc[3] / frame["close"].iloc[2])
    )
    # The input retains the injected infinity and all eight source rows.
    assert np.isinf(frame["close"].iloc[1]) and len(frame) == len(original)
    assert "log_return" not in frame


@pytest.mark.parametrize("case", ["mixed", "duplicate", "naive"])
def test_preparation_rejects_ambiguous_identity_or_times(case: str) -> None:
    """A single stream must have a consistent identity and explicit unique UTC times."""

    frame = native_history().iloc[:4].copy()
    if case == "mixed":
        frame.loc[1, "source_id"] = "other-source"
    elif case == "duplicate":
        frame.loc[1, "open_time_utc"] = frame["open_time_utc"].iloc[0]
    else:
        frame["open_time_utc"] = frame["open_time_utc"].dt.tz_localize(None)
    with pytest.raises(ValueError):
        prepare_time_series(frame)


def test_monthly_returns_keep_absent_months_unavailable() -> None:
    """A jump from January to March must not masquerade as a one-month return."""

    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2025-01-30", "2025-03-30", "2025-04-30"], utc=True),
            "close": [100.0, 130.0, 143.0],
        }
    )
    result = monthly_returns(frame)
    assert result.iloc[:3].isna().all()
    assert result.iloc[3] == pytest.approx(0.1)


def test_return_models_purge_boundaries_and_do_not_fit_holdout_targets() -> None:
    """Changing 2026 observations cannot change earlier validation metrics or training cutoffs."""

    daily = prepare_daily_summary(prepare_time_series(native_history()))
    original = daily.copy(deep=True)
    metrics = evaluate_return_models(daily)
    assert set(metrics["model"]) == {"UNI", "BI", "MULTI", "zero_return", "last_return"}
    assert (
        pd.to_datetime(metrics["train_target_end"], utc=True)
        < pd.to_datetime(metrics["test_decision_start"], utc=True)
    ).all()
    holdout = metrics.loc[metrics["split"] == "holdout"]
    assert not holdout.empty
    assert (
        pd.to_datetime(holdout["train_target_end"], utc=True) < pd.Timestamp("2026-01-01", tz="UTC")
    ).all()
    changed = daily.copy(deep=True)
    changed.loc[changed["timestamp_utc"].dt.year == 2026, "log_return"] *= -100
    altered = evaluate_return_models(changed)
    pd.testing.assert_frame_equal(
        metrics.loc[metrics["split"] != "holdout"].reset_index(drop=True),
        altered.loc[altered["split"] != "holdout"].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(daily, original)


def test_full_time_series_reports_create_figures_and_measured_summary(tmp_path: Path) -> None:
    """Every new report renders and closes its figure while retaining original native data."""

    source = native_history()
    original = source.copy(deep=True)
    paths = run_time_series_analysis(source, tmp_path, include_models=True)
    assert len(paths) == 10
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    assert len(plt.get_fignums()) == 0
    pd.testing.assert_frame_equal(source, original)
