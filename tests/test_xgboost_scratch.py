"""Verifikasi komponen matematis XGBoost from scratch (SOURCE_OF_TRUTH bagian E)."""

import numpy as np
import pytest
from xgboost import XGBClassifier

from technical.xgboost_scratch import (
    GradientTree,
    QuantileBinner,
    ScratchXGBoostClassifier,
    compute_initial_margin,
    leaf_weight,
    one_hot,
    softmax,
    softmax_gradient_hessian,
    split_gain,
)


def cross_entropy(margins: np.ndarray, label: int) -> float:
    probabilities = softmax(margins.reshape(1, -1))[0]
    return float(-np.log(probabilities[label]))


def test_softmax_rows_sum_to_one_and_is_stable_for_large_margins() -> None:
    margins = np.array([[1000.0, 1001.0, 999.0], [0.0, 0.0, 0.0]])
    probabilities = softmax(margins)
    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.allclose(probabilities[1], 1.0 / 3.0)


def test_gradient_and_hessian_match_finite_differences() -> None:
    margins = np.array([0.3, -0.2, 0.5])
    label = 2
    probabilities = softmax(margins.reshape(1, -1))
    targets = one_hot(np.array([label]), 3)
    step = 1e-5

    for class_index in range(3):
        gradient, hessian = softmax_gradient_hessian(probabilities, targets, class_index)

        plus = margins.copy()
        minus = margins.copy()
        plus[class_index] += step
        minus[class_index] -= step
        numeric_gradient = (cross_entropy(plus, label) - cross_entropy(minus, label)) / (2 * step)
        numeric_hessian = (
            cross_entropy(plus, label)
            - 2 * cross_entropy(margins, label)
            + cross_entropy(minus, label)
        ) / (step * step)

        assert gradient[0] == pytest.approx(numeric_gradient, abs=1e-6)
        # Hessian implementasi = 2 x Hessian diagonal eksak (konvensi XGBoost).
        assert hessian[0] == pytest.approx(2 * numeric_hessian, rel=1e-3)


def test_leaf_weight_is_regularized_newton_step() -> None:
    # w* = -T_alpha(G) / (H + lambda) = -(-6 + 1) / (3 + 2) = 1
    assert leaf_weight(-6.0, 3.0, reg_lambda=2.0, reg_alpha=1.0) == pytest.approx(1.0)
    # |G| <= alpha -> bobot nol (efek L1)
    assert leaf_weight(0.5, 3.0, reg_lambda=2.0, reg_alpha=1.0) == pytest.approx(0.0)


def test_leaf_weight_minimizes_second_order_objective() -> None:
    gradient_sum, hessian_sum, reg_lambda, reg_alpha = -4.0, 2.0, 1.0, 0.5
    weights = np.linspace(-5, 5, 200001)
    objective = (
        gradient_sum * weights
        + 0.5 * (hessian_sum + reg_lambda) * weights**2
        + reg_alpha * np.abs(weights)
    )
    best = weights[np.argmin(objective)]
    assert leaf_weight(gradient_sum, hessian_sum, reg_lambda, reg_alpha) == pytest.approx(
        best, abs=1e-4
    )


def test_split_gain_matches_hand_calculation() -> None:
    # G_L=-4, H_L=2, G_R=6, H_R=3, lambda=1, gamma=0.5
    # 0.5 * [16/3 + 36/4 - 4/6] - 0.5 = 6.3333...
    gain = split_gain(-4.0, 2.0, 6.0, 3.0, reg_lambda=1.0, reg_alpha=0.0, gamma=0.5)
    expected = 0.5 * (16 / 3 + 36 / 4 - 4 / 6) - 0.5
    assert gain == pytest.approx(expected)


def test_tree_finds_obvious_split_and_routes_missing_values() -> None:
    features = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [np.nan]])
    gradient = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    hessian = np.ones(6)
    binner = QuantileBinner(max_bin=16).fit(features)
    bins = binner.transform(features)

    tree = GradientTree(
        max_depth=1,
        min_child_weight=0.0,
        reg_lambda=0.0,
        reg_alpha=0.0,
        gamma=0.0,
        missing_bin=binner.missing_bin,
    )
    tree.build(bins, gradient, hessian)
    predictions = tree.predict(bins)

    assert tree.feature[0] == 0
    assert np.allclose(predictions[:3], 1.0)  # -G/H = 3/3
    assert np.allclose(predictions[3:], -1.0)  # NaN ikut kelompok kanan yang gradiennya sama


def test_gamma_prunes_weak_split() -> None:
    features = np.array([[1.0], [2.0], [3.0], [4.0]])
    gradient = np.array([-0.1, 0.1, -0.1, 0.1])
    bins = QuantileBinner(max_bin=8).fit(features).transform(features)
    tree = GradientTree(
        max_depth=3, min_child_weight=0.0, reg_lambda=1.0, reg_alpha=0.0, gamma=10.0, missing_bin=8
    )
    tree.build(bins, gradient, np.ones(4))
    assert tree.leaf_count() == 1


def test_initial_margin_reproduces_training_class_prior() -> None:
    labels = np.array([0, 1, 1, 2, 2, 2])
    margin = compute_initial_margin(labels, 3, "prior")
    assert softmax(margin.reshape(1, -1))[0] == pytest.approx([1 / 6, 2 / 6, 3 / 6])
    assert margin.mean() == pytest.approx(0.0)


def test_classifier_outputs_valid_probabilities_and_learns() -> None:
    rng = np.random.default_rng(3)
    features = rng.normal(size=(3000, 4))
    signal = features[:, 0] + rng.normal(0, 0.5, 3000)
    labels = np.where(signal > 0.7, 2, np.where(signal < -0.7, 0, 1))

    model = ScratchXGBoostClassifier(n_estimators=30, max_depth=3, learning_rate=0.3, max_bin=32)
    model.fit(features[:2000], labels[:2000])
    probabilities = model.predict_proba(features[2000:])

    assert probabilities.shape == (1000, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    accuracy = (probabilities.argmax(axis=1) == labels[2000:]).mean()
    majority = np.bincount(labels[2000:]).max() / 1000
    assert accuracy > majority + 0.1


def test_matches_official_xgboost_when_candidate_splits_are_identical() -> None:
    """Dengan fitur diskret (nilai unik < max_bin), kandidat split kedua implementasi sama,
    sehingga probabilitas harus identik hingga presisi float32."""
    rng = np.random.default_rng(0)
    features = rng.integers(0, 10, size=(600, 3)).astype(float)
    labels = np.where(
        features[:, 0] + rng.normal(0, 2, 600) > 6,
        2,
        np.where(features[:, 1] + rng.normal(0, 2, 600) > 6, 0, 1),
    )
    parameters = dict(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.3,
        min_child_weight=1.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        gamma=0.0,
    )

    library = XGBClassifier(
        objective="multi:softprob", tree_method="hist", max_bin=64, **parameters
    )
    library.fit(features, labels)
    scratch = ScratchXGBoostClassifier(max_bin=64, **parameters).fit(features, labels)

    difference = np.abs(library.predict_proba(features) - scratch.predict_proba(features))
    assert difference.max() < 1e-5


def test_early_stopping_keeps_best_round() -> None:
    rng = np.random.default_rng(5)
    features = rng.normal(size=(800, 3))
    labels = rng.integers(0, 3, size=800)  # tidak ada sinyal -> eval loss cepat memburuk
    model = ScratchXGBoostClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.5, early_stopping_rounds=3, max_bin=16
    )
    model.fit(features[:600], labels[:600], features[600:], labels[600:])
    assert len(model.rounds) == model.best_iteration + 1
    assert len(model.history) < 50


def test_gamma_convention_matches_library_with_half_factor() -> None:
    """gamma=g di library setara gamma=g/2 di sini (rumus paper memakai faktor 1/2)."""
    rng = np.random.default_rng(0)
    features = rng.integers(0, 10, size=(600, 3)).astype(float)
    labels = np.where(
        features[:, 0] + rng.normal(0, 2, 600) > 6,
        2,
        np.where(features[:, 1] + rng.normal(0, 2, 600) > 6, 0, 1),
    )
    parameters = dict(
        n_estimators=10, max_depth=3, learning_rate=0.3, min_child_weight=1.0, reg_lambda=1.0
    )
    library = XGBClassifier(
        objective="multi:softprob", tree_method="hist", max_bin=64, gamma=4.0, **parameters
    )
    library.fit(features, labels)
    scratch = ScratchXGBoostClassifier(max_bin=64, gamma=2.0, **parameters)
    scratch.fit(features, labels)
    difference = np.abs(library.predict_proba(features) - scratch.predict_proba(features))
    assert difference.max() < 1e-5
