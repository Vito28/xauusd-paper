"""Uji ensemble soft voting berbobot (technical/ensemble.py)."""

import joblib
import numpy as np
import pandas as pd
import pytest

from technical.config import TechnicalConfig, validate_config
from technical.ensemble import blend, diversity, search_weights, weight_grid
from technical.models import create_model
from technical.run import build_dataset, run_fold
from technical.splits import Fold
from technical.xgboost_scratch import multiclass_log_loss
from tests.test_technical_pipeline import prepared


def test_weight_grid_covers_simplex_including_single_members() -> None:
    grid = weight_grid(3, 0.5)
    assert len(grid) == 6
    for weights in grid:
        assert sum(weights) == pytest.approx(1.0)
        assert min(weights) >= 0
    assert [1.0, 0.0, 0.0] in grid
    assert [0.0, 0.0, 1.0] in grid


def test_search_weights_prefers_informative_member_and_never_loses_to_it() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=400)
    informative = np.full((400, 3), 0.1)
    informative[np.arange(400), labels] = 0.8
    noise = rng.dirichlet([1, 1, 1], size=400)

    chosen = search_weights([noise, informative], labels, 0.05)
    assert chosen["weights"][1] > chosen["weights"][0]
    assert chosen["log_loss"] <= multiclass_log_loss(labels, informative) + 1e-12
    mixed = blend([noise, informative], chosen["weights"])
    assert np.allclose(mixed.sum(axis=1), 1.0)


def test_diversity_is_zero_for_identical_members() -> None:
    labels = np.array([0, 1, 2, 1])
    probabilities = np.eye(3)[[0, 1, 1, 1]] * 0.8 + 0.2 / 3
    report = diversity(["a", "b"], [probabilities, probabilities], labels)
    assert report["a|b"]["disagreement"] == 0.0
    assert report["a|b"]["error_correlation"] == pytest.approx(1.0)


def test_config_rejects_unknown_or_single_member() -> None:
    config = TechnicalConfig()
    config.ensemble.members = ["xgboost", "ensemble"]
    with pytest.raises(ValueError):
        validate_config(config)
    config.ensemble.members = ["xgboost"]
    with pytest.raises(ValueError):
        validate_config(config)


def small_config() -> TechnicalConfig:
    config = TechnicalConfig()
    config.xgboost.n_estimators = 20
    config.random_forest.n_estimators = 20
    config.decision.min_signals = 5
    config.run.n_jobs = 1
    return config


def test_ensemble_fold_reports_weights_and_saved_model_predicts_same(tmp_path) -> None:
    config = small_config()
    candles = prepared(6000, seed=11)
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
    result = run_fold("ensemble", fold, features, labels, baseline_inputs, config)

    report = result["ensemble"]
    weights = [member["weight"] for member in report["members"].values()]
    assert sum(weights) == pytest.approx(1.0)
    assert set(report["members"]) == {"xgboost", "random_forest", "logistic"}
    # Bobot dipilih pada CALIBRATION, dan kandidatnya memuat setiap anggota tunggal.
    assert report["ensemble_log_loss"] <= report["best_single_log_loss"] + 1e-12
    assert len(report["diversity"]) == 3

    # Bundle live disimpan dengan joblib: ensemble harus bisa disimpan dan dimuat ulang.
    model = result["_model"]
    evaluation_rows = features.loc[result["_blocks"]["evaluation"]]
    path = tmp_path / "ensemble.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    assert np.allclose(loaded.predict_proba(evaluation_rows), model.predict_proba(evaluation_rows))


def test_ensemble_requires_calibration_block() -> None:
    model = create_model("ensemble", small_config())
    with pytest.raises(ValueError):
        model.fit(np.zeros((10, 2)), np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]))
