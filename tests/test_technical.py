"""Prove causal features, purged boundaries, TRAIN-only fitting and pending outcomes."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from technical.config import LABELS, Config
from technical.data_preparation import (
    Fold,
    load_data,
    make_folds,
    make_labels,
    prepare_data,
    purged_cv,
)
from technical.xgboost_model import (
    PlattCalibration,
    build_features,
    evaluate,
    fit_development,
    load_frozen_models,
    load_model_bundle,
    predict_latest,
    quality_filter,
    run_pipeline,
    select_features,
    select_regression_features,
)


def market_frame(rows: int = 700, start: str = "2025-01-01") -> pd.DataFrame:
    """Create deterministic three-class M5 candles with observed spread and volume."""
    n = np.arange(rows)
    close = 2600 + np.cumsum(0.35 * np.sin(n / 3) + 0.16 * np.sin(n / 11))
    opening = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open_time_utc": pd.date_range(start, periods=rows, freq="5min", tz="UTC"),
            "open": opening,
            "high": np.maximum(opening, close) + 0.2,
            "low": np.minimum(opening, close) - 0.2,
            "close": close,
            "tick_volume": 100 + n % 37,
            "native_spread_price": 0.2,
        }
    )


def research_fixture() -> tuple[pd.DataFrame, Fold, Config]:
    """Create calendar-disjoint development/test blocks with a small real model config."""
    frame = pd.concat(
        [
            market_frame(700),
            market_frame(160, "2026-01-01"),
            market_frame(160, "2026-02-01"),
            market_frame(160, "2026-03-01"),
            market_frame(160, "2026-04-01"),
        ],
        ignore_index=True,
    )
    fold = Fold(
        "final_2026",
        tuple(
            pd.Timestamp(t, tz="UTC")
            for t in (
                "2025-01-01",
                "2026-01-01",
                "2026-02-01",
                "2026-03-01",
                "2026-04-01",
                "2026-11-01",
            )
        ),
    )
    config = Config(
        n_estimators=8,
        relieff_top_k=12,
        rfecv_min_features=4,
        rfecv_step=4,
        cv_splits=2,
        relieff_max_samples=150,
        relieff_anchors=16,
        relieff_neighbors=3,
        n_jobs=2,
    )
    return frame, fold, config


def test_preparation_preserves_missing_reports_gaps_and_rejects_bad_prices() -> None:
    """No interpolation occurs; OHLC, stable duplicates, source and spread are validated."""
    raw = market_frame(100)
    raw.loc[10, "high"] = 0
    raw.loc[20, "native_spread_price"] = -1
    raw.loc[21, "tick_volume"] = -1
    raw = pd.concat([raw.iloc[::-1], raw.iloc[[3]]], ignore_index=True)
    data, report = prepare_data(raw)
    assert len(data) == 99
    assert report["duplicate_rows"] == 1 and report["invalid_ohlc_rows"] == 1
    assert report["missing_rows"] == 1 and report["detected_gaps"] == 1
    assert data.spread_price.isna().sum() == 1 and data.tick_volume.isna().sum() == 1
    assert data.open_time_utc.is_monotonic_increasing and data.open_time_utc.is_unique
    mixed = market_frame(20).assign(source_key=["a"] * 10 + ["b"] * 10)
    with pytest.raises(ValueError, match="source_key"):
        prepare_data(mixed)
    with pytest.raises(ValueError, match="timezone"):
        prepare_data(
            market_frame(20).assign(open_time_utc=lambda d: d.open_time_utc.dt.tz_localize(None))
        )


def test_features_never_read_future_and_manifest_is_complete() -> None:
    """Mutating all future OHLC/volume/spread cannot change a past feature prefix."""
    config = Config()
    data, _ = prepare_data(market_frame())
    features, manifest = build_features(data, config)
    changed = data.copy()
    changed.loc[400:, ["open", "high", "low", "close"]] *= 1.05
    changed.loc[400:, ["spread_price", "tick_volume"]] *= 3
    other, _ = build_features(changed, config)
    assert_frame_equal(features.iloc[:400], other.iloc[:400])
    prefix, _ = build_features(data.iloc[:400], config)
    assert_frame_equal(features.iloc[:400], prefix)
    assert len(manifest) == len(features.columns)
    assert len(features.columns) >= 150
    assert {"macd_8_21", "macd_12_26", "macd_19_39", "rsi_14", "atr_pct_14"} <= set(features)
    assert not any("future" in name or "target" in name for name in features)
    assert not np.isinf(features.to_numpy()).any()


def test_labels_use_spread_gap_and_pending_boundaries() -> None:
    """Labels cover costs, preserve neutral class, and refuse gap-crossing targets."""
    data, _ = prepare_data(market_frame(50))
    labels = make_labels(data)
    assert labels.target_class.iloc[-3:].isna().all()
    assert labels.target_class.dropna().isin([0, 1, 2]).all()
    np.testing.assert_allclose(labels.threshold_return, 0.2 / data.close + 0.0001)
    assert labels.prediction_timestamp.iloc[0] == data.open_time_utc.iloc[0] + pd.Timedelta(
        minutes=5
    )
    gap, _ = prepare_data(market_frame(50).drop(index=20))
    g = make_labels(gap)
    assert g.future_close.iloc[17:20].isna().all()
    missing = make_labels(data.assign(spread_price=np.nan))
    assert missing.target_class.isna().all()


def test_loader_uses_bound_filters_in_read_only_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical loading enters READ ONLY before its parameterized SELECT, never writes SQL."""
    engine = MagicMock()
    connection = engine.connect.return_value.__enter__.return_value
    transaction = connection.begin.return_value
    source_key = "demo'; DROP TABLE market.bar; --"

    def read_query(statement: object, active_connection: object, *, params: dict) -> pd.DataFrame:
        """Check transaction mode and binding before returning local fixture bars."""
        assert active_connection is connection
        assert str(connection.execute.call_args.args[0]) == "SET TRANSACTION READ ONLY"
        assert "source_key = :source_key" in str(statement)
        assert "timeframe = 'M5'" in str(statement)
        assert "close_time_utc <= :now" in str(statement)
        assert source_key not in str(statement)
        assert params["source_key"] == source_key
        return market_frame(30)

    monkeypatch.setattr(pd, "read_sql_query", read_query)
    result = load_data(
        source_key, "XAUUSD", "2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z", engine=engine
    )
    assert len(result) == 30
    transaction.commit.assert_called_once()
    transaction.rollback.assert_not_called()
    assert connection.execute.call_count == 1


def test_unclosed_candles_and_missing_volume_do_not_enter_obv() -> None:
    """Decision data excludes unclosed candles and OBV remains unknown after missing volume."""
    raw = market_frame(400)
    cutoff = raw.open_time_utc.iloc[300]
    closed, report = prepare_data(raw, as_of=cutoff)
    assert len(closed) == 300 and report["unclosed_rows"] == 100
    raw.loc[100, "tick_volume"] = np.nan
    data, _ = prepare_data(raw)
    features, _ = build_features(data, Config())
    assert features.obv.iloc[100:].isna().all()


def test_purge_all_boundaries_and_rfecv_inner_boundaries() -> None:
    """No development label timestamp reaches the next block; CV also purges H bars."""
    raw, fold, config = research_fixture()
    data, _ = prepare_data(raw)
    labels = make_labels(data, config.horizon_bars)
    blocks = fold.split(labels)
    for i, name in enumerate(("train", "tune", "calibration", "policy")):
        assert labels.loc[blocks[name], "target_timestamp"].max() < fold.boundaries[i + 1]
    train = labels.loc[blocks["train"]]
    for a, b in purged_cv(train.prediction_timestamp, train.target_timestamp, 2):
        assert train.target_timestamp.iloc[a].max() < train.prediction_timestamp.iloc[b[0]]
    folds = make_folds()
    assert [f.name for f in folds] == [f"fold_{year}" for year in range(2020, 2026)] + [
        "final_2026"
    ]
    assert folds[-1].boundaries[-2] == pd.Timestamp("2026-01-01", tz="UTC")


@pytest.mark.parametrize("legacy_module", ["technical.modeling", "technical.data.cgboost_model"])
def test_legacy_calibrator_artifact_still_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy_module: str
) -> None:
    """The model reader resolves both earlier calibration paths without legacy source files."""
    path = tmp_path / "legacy_calibrator.joblib"
    with monkeypatch.context() as legacy:
        module = ModuleType(legacy_module)
        module.PlattCalibration = PlattCalibration
        legacy.setitem(sys.modules, legacy_module, module)
        legacy.setattr(PlattCalibration, "__module__", legacy_module)
        joblib.dump({"calibrator": PlattCalibration(models=[])}, path)
    restored = load_model_bundle(path)["calibrator"]
    assert type(restored) is PlattCalibration
    assert type(restored).__module__ == "technical.xgboost_model"
    assert restored.models == []


def test_quality_filter_only_observes_train() -> None:
    """All-null, constant, near-constant, infinity and exact duplicates are explained."""
    frame = pd.DataFrame(
        {
            "good": np.arange(100),
            "duplicate": np.arange(100),
            "constant": 1,
            "empty": np.nan,
            "inf": np.inf,
            "near": [1] * 99 + [2],
        }
    )
    kept, report = quality_filter(frame, Config(near_constant=0.99))
    assert kept == ["good"]
    assert report["removed"]["duplicate"] == "exact_duplicate:good"
    assert report["fit_block"] == "TRAIN"


def test_changing_test_actuals_cannot_change_any_fitted_state() -> None:
    """Real selection, tuning, calibration and policy are invariant to changed TEST data."""
    raw, fold, config = research_fixture()
    data, _ = prepare_data(raw)
    changed = data.copy()
    changed.loc[
        changed.prediction_timestamp >= fold.boundaries[-2], ["open", "high", "low", "close"]
    ] *= 1.2
    results = []
    for source in (data, changed):
        features, _ = build_features(source, config)
        labels = make_labels(source, config.horizon_bars, config.minimum_edge_bps)
        blocks = fold.split(labels)
        selected, quality, relief, rfe = select_features(features, labels, blocks["train"], config)
        development = {
            name: (features.loc[index, selected], labels.loc[index])
            for name, index in blocks.items()
            if name != "test"
        }
        regression_selected, regression_audit = select_regression_features(
            features, labels, blocks["train"], config
        )
        regression_development = {
            name: (features.loc[index, regression_selected], labels.loc[index])
            for name, index in blocks.items()
            if name != "test"
        }
        bundle, audit = fit_development(
            development, config, regression_blocks=regression_development
        )
        audit["regression_selection"] = regression_audit
        assert "test" not in development
        results.append((selected, quality, relief, rfe, audit, bundle))
    a, b = results
    assert a[0] == b[0] and a[1] == b[1] and a[4] == b[4]
    assert_frame_equal(a[2], b[2])
    assert_frame_equal(a[3], b[3])
    assert (
        a[5]["classifier"].get_booster().save_raw() == b[5]["classifier"].get_booster().save_raw()
    )
    assert a[5]["regressor"].get_booster().save_raw() == b[5]["regressor"].get_booster().save_raw()
    assert (
        a[5]["quantile_regressor"].get_booster().save_raw()
        == b[5]["quantile_regressor"].get_booster().save_raw()
    )
    with pytest.raises(ValueError, match="tanpa TEST"):
        fit_development({"test": (pd.DataFrame(), pd.DataFrame())}, config)


def test_inner_validation_labels_never_affect_screening() -> None:
    """Mutating RFECV validation labels cannot change frozen Quality/ReliefF candidates."""
    raw, fold, config = research_fixture()
    data, _ = prepare_data(raw)
    labels = make_labels(data)
    features, _ = build_features(data, config)
    train = fold.split(labels)["train"]
    cv = purged_cv(
        labels.loc[train, "prediction_timestamp"],
        labels.loc[train, "target_timestamp"],
        config.cv_splits,
    )
    altered = labels.copy()
    validation_start = cv[0][1][0]
    changed_rows = train[validation_start:]
    altered.loc[changed_rows, "target_class"] = (altered.loc[changed_rows, "target_class"] + 1) % 3
    _, original_quality, original_ranking, _ = select_features(features, labels, train, config)
    _, changed_quality, changed_ranking, _ = select_features(features, altered, train, config)
    assert_frame_equal(original_ranking, changed_ranking)
    assert original_quality["removed"] == changed_quality["removed"]
    assert (
        pd.Timestamp(original_quality["screening_max_target"])
        < labels.loc[train[validation_start], "prediction_timestamp"]
    )


def test_real_pipeline_outputs_and_dashboard_pending_metrics(tmp_path: Path) -> None:
    """A real miniature walk-forward run saves models, probabilities and offline HTML."""
    raw, fold, config = research_fixture()
    output = tmp_path / "walk_forward"
    path = run_pipeline("fixture", frame=raw, folds=[fold], config=config, output_dir=output)
    predictions = pd.read_csv(output / "final_2026" / "predictions.csv")
    assert predictions.predicted_signal.isin(LABELS).all()
    np.testing.assert_allclose(
        predictions[["probability_sell", "probability_no_trade", "probability_buy"]].sum(axis=1), 1
    )
    assert predictions.actual_signal.tail(3).isna().all()
    metrics = evaluate(predictions)
    assert metrics["classification"]["evaluated_rows"] == len(predictions) - 3
    assert metrics["regression"]["evaluated_rows"] == len(predictions) - 3
    assert "lot" in metrics["trading_proxy"]["note"]
    html = path.read_text(encoding="utf-8")
    assert "plotly.js" in html and "/*__PAYLOAD__*/null" not in html
    assert "cdn.plot.ly" not in html[: html.find("</head>")] or "<script src=" not in html
    assert "candlestick" in html and "scrollZoom:true" in html
    assert (output / "feature_stability.csv").exists()
    monthly = pd.read_csv(output / "final_2026" / "monthly_metrics.csv")
    assert monthly.evaluated_rows.sum() == metrics["classification"]["evaluated_rows"]
    assert monthly.buy_signals.sum() == metrics["selective"]["BUY"]["signals"]
    assert monthly.sell_signals.sum() == metrics["selective"]["SELL"]["signals"]
    assert (output / "final_2026" / "monthly_diagnostics.png").exists()
    saved = joblib.load(output / "final_2026" / "model_classifier.joblib")
    assert type(saved["calibrator"]) is PlattCalibration
    bundle = load_frozen_models(output / "final_2026")
    now = pd.to_datetime(raw.open_time_utc, utc=True).max() + pd.Timedelta(minutes=5)
    latest = predict_latest(bundle, raw, as_of=now)
    np.testing.assert_allclose(
        latest.predicted_future_return, predictions.predicted_future_return.tail(1)
    )
    np.testing.assert_allclose(latest.probability_buy, predictions.probability_buy.tail(1))
    assert latest.outcome_status.iloc[0] in {"NEUTRAL", "PENDING"}
    with pytest.raises(ValueError, match="stale"):
        predict_latest(bundle, raw, as_of=now + pd.Timedelta(hours=1))
    with pytest.raises(ValueError, match="origin"):
        predict_latest(bundle, raw.iloc[1:], as_of=now)
    model_path = output / "final_2026" / "model_regressor.joblib"
    model_path.write_bytes(model_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="tercampur"):
        load_frozen_models(output / "final_2026")
