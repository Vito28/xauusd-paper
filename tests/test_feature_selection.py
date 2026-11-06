"""Uji feature selection di dalam fold (technical/feature_selection.py)."""

import numpy as np
import pandas as pd
import pytest

from technical.config import TechnicalConfig, validate_config
from technical.feature_selection import (
    FeatureSelectedModel,
    filter_features,
    leakage_audit,
    summarize_selection,
)
from technical.run import build_dataset
from tests.test_technical_pipeline import prepared


def synthetic(count: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Satu fitur informatif, salinannya, dua noise, satu konstan, satu hampir kosong."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=count)
    labels = np.where(signal > 0.4, 2, np.where(signal < -0.4, 0, 1))
    mostly_missing = rng.normal(size=count)
    mostly_missing[rng.random(count) < 0.9] = np.nan
    features = pd.DataFrame(
        {
            "signal": signal,
            "signal_copy": 3 * signal + 1,
            "noise_a": rng.normal(size=count),
            "noise_b": rng.normal(size=count),
            "constant": np.ones(count),
            "mostly_missing": mostly_missing,
        }
    ).astype("float32")
    return features, labels


def small_config() -> TechnicalConfig:
    config = TechnicalConfig()
    config.xgboost.n_estimators = 40
    config.run.n_jobs = 1
    config.feature_selection.sample_rows = 2000
    config.feature_selection.min_features = 1
    return config


def test_filter_drops_missing_constant_and_redundant_features() -> None:
    features, labels = synthetic(3000, seed=0)
    result = filter_features(features, labels, small_config().feature_selection, seed=0)

    assert result["dropped_missing"] == ["mostly_missing"]
    assert result["dropped_constant"] == ["constant"]
    # signal dan signal_copy berperingkat identik: hanya satu yang dipertahankan.
    assert len(result["dropped_redundant"]) == 1
    assert {"signal", "signal_copy"} & set(result["kept"])
    assert len({"signal", "signal_copy"} & set(result["kept"])) == 1
    assert result["leakage_flags"] == []


def test_leakage_audit_flags_feature_that_copies_the_label() -> None:
    features, labels = synthetic(2000, seed=1)
    features = features[["signal", "noise_a", "noise_b"]].copy()
    features["leak"] = labels - 1
    flags = leakage_audit(features, labels, threshold=0.95)
    assert [flag["feature"] for flag in flags] == ["leak"]


def test_selected_model_keeps_informative_feature_and_reports_status() -> None:
    features, labels = synthetic(4000, seed=2)
    train_x, train_y = features.iloc[:3000], labels[:3000]
    eval_x, eval_y = features.iloc[3000:], labels[3000:]

    model = FeatureSelectedModel(small_config()).fit(train_x, train_y, eval_x, eval_y)
    report = model.selection_report()

    informative = {"signal", "signal_copy"} & set(report["selected"])
    assert len(informative) == 1
    best = max(
        report["selected"],
        key=lambda name: report["features"][name]["permutation_importance"],
    )
    assert best in informative
    assert report["features"]["constant"]["status"] == "dropped_constant"
    assert report["features"]["mostly_missing"]["status"] == "dropped_missing"

    probabilities = model.predict_proba(eval_x)
    assert probabilities.shape == (1000, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_pipeline_features_pass_leakage_audit() -> None:
    """Audit otomatis: tidak ada fitur pipeline yang hampir identik dengan label."""
    config = TechnicalConfig()
    candles = prepared(6000, seed=3)
    features, _, labels, _ = build_dataset(candles, config, config.features.groups)
    known = labels["label"].notna()
    flags = leakage_audit(features[known], labels.loc[known, "label"].to_numpy(), 0.95)
    assert flags == []


def test_summarize_selection_counts_folds() -> None:
    def report(status_a: str, status_b: str) -> dict:
        return {
            "features": {
                "a": {"status": status_a, "permutation_importance": 0.02},
                "b": {"status": status_b, "permutation_importance": -0.01},
            }
        }

    rows = summarize_selection(
        [report("selected", "dropped_permutation"), report("selected", "selected")],
        {"a": "trend", "b": "time"},
    )
    assert rows[0]["feature"] == "a"
    assert rows[0]["selected_folds"] == 2
    assert rows[1]["selected_folds"] == 1
    assert rows[0]["group"] == "trend"


def test_config_rejects_invalid_selection_settings() -> None:
    config = TechnicalConfig()
    config.feature_selection.redundancy_threshold = 1.5
    with pytest.raises(ValueError):
        validate_config(config)
