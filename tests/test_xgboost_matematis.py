"""Check independent Newton math, missing routing, trading arithmetic and stdlib-only runtime."""

import ast
import math
import subprocess
import sys
from pathlib import Path

import pytest

from analisis_teknikall.xgboost_matematis import (
    FrozenTrees,
    ScratchClassifier,
    ScratchRegressor,
    TreeConfig,
    choose_signal,
    grow_tree,
    outcome,
    softmax,
)


def test_leaf_is_regularized_newton_step() -> None:
    """The unsplit leaf equals the analytic optimum, including L1 shrinkage."""
    tree = grow_tree(
        [[0.0], [1.0]], [-2.0, -4.0], [1.0, 2.0], TreeConfig(max_depth=0, reg_lambda=2, reg_alpha=1)
    )
    assert tree.value == pytest.approx(1.0)


def test_missing_only_group_can_be_separated() -> None:
    """Learn missing routing even when all observed feature values are identical."""
    x = [[math.nan], [math.nan], [1.0], [1.0]]
    tree = grow_tree(x, [-2.0, -2.0, 2.0, 2.0], [1.0] * 4, TreeConfig(max_depth=1, reg_lambda=0))
    assert tree.predict([math.nan]) == 2.0
    assert tree.predict([1.0]) == -2.0


@pytest.mark.parametrize("objective", ["squared", "huber"])
def test_scratch_regression_generalizes_simple_function(objective: str) -> None:
    """A held-out interleaved grid beats a constant baseline without borrowing trained trees."""
    train = [[i / 20] for i in range(-20, 21, 2)]
    test = [[i / 20] for i in range(-19, 20, 2)]
    model = ScratchRegressor(rounds=40, objective=objective).fit(train, [2 * r[0] for r in train])
    rmse = math.sqrt(
        sum((p - 2 * r[0]) ** 2 for p, r in zip(model.predict(test), test, strict=True)) / len(test)
    )
    assert rmse < 0.3


def test_classifier_and_regressor_are_independent() -> None:
    """Classifier learns three labels from scratch; its output is a normalized distribution."""
    x = [[float(i)] for i in range(60)]
    y = [i // 20 for i in range(60)]
    model = ScratchClassifier(rounds=25).fit(x, y)
    probabilities = model.predict_proba([[4.5], [30.5], [54.5]])
    assert model.predict([[4.5], [30.5], [54.5]]) == [0, 1, 2]
    for row in probabilities:
        assert sum(row) == pytest.approx(1)
    assert softmax([1000.0, 1000.0, 1000.0]) == pytest.approx([1 / 3] * 3)


def test_policy_can_buy_sell_and_abstain_without_future_labels() -> None:
    """Exercise enabled rules so all-NO_TRADE real models cannot conceal decision bugs."""
    policy = {
        "require_interval_agreement": True,
        "classes": {
            name: {"enabled": True, "threshold": 0.8, "probability_margin": 0.1, "edge_multiple": 1}
            for name in ("BUY", "SELL")
        },
    }
    assert choose_signal([0.9, 0.05, 0.05], -0.01, -0.02, -0.005, 0.001, policy)[0] == "SELL"
    assert choose_signal([0.05, 0.05, 0.9], 0.01, 0.005, 0.02, 0.001, policy)[0] == "BUY"
    assert choose_signal([0.05, 0.05, 0.9], -0.01, -0.02, -0.005, 0.001, policy)[0] == "NO_TRADE"
    assert (
        choose_signal([0.05, 0.05, 0.9], 0.01, -0.005, 0.02, 0.001, policy)[1]
        == "regression_uncertain"
    )


def test_direction_and_profit_match_requested_semantics() -> None:
    """SELL 4300 to 4100 earns 200 gross; abstention has no outcome or profit."""
    sell = outcome("SELL", 4300, 4100, 0.5)
    assert sell == {
        "outcome_status": "SUCCESS",
        "realized_gross_move": 200,
        "realized_net_move": 199.5,
    }
    assert outcome("NO_TRADE", 4300, 4100, 0.5)["outcome_status"] == "NEUTRAL"
    assert outcome("BUY", 4300, None, 0.5)["realized_net_move"] is None


def test_only_standard_library_and_no_site_packages() -> None:
    """Inspect imports and execute the training demo with site-packages disabled."""
    source = Path("analisis_teknikall/xgboost_matematis.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] in sys.stdlib_module_names for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module.split(".")[0] in sys.stdlib_module_names
    subprocess.run(
        [sys.executable, "-S", "-B", str(source), "demo"], check=True, capture_output=True
    )


def test_frozen_tree_threshold_missing_and_leaf_values() -> None:
    """Threshold equality routes right, missing follows default, leaf eta is not reapplied."""
    model = FrozenTrees(
        {
            "learner": {
                "objective": {"name": "reg:squarederror"},
                "feature_names": ["x"],
                "learner_model_param": {
                    "num_class": "0",
                    "num_target": "1",
                    "num_feature": "1",
                    "base_score": "[2.0]",
                },
                "gradient_booster": {
                    "name": "gbtree",
                    "model": {
                        "tree_info": [0],
                        "trees": [
                            {
                                "tree_param": {"size_leaf_vector": "1"},
                                "split_type": [0, 0, 0],
                                "left_children": [1, -1, -1],
                                "right_children": [2, -1, -1],
                                "default_left": [True, False, False],
                                "split_indices": [0, 0, 0],
                                "split_conditions": [1.0, 0.25, -0.5],
                            }
                        ],
                    },
                },
            }
        }
    )
    assert model.predict({"x": 0.5}) == [2.25]
    assert model.predict({"x": math.nan}) == [2.25]
    assert model.predict({"x": 1.0}) == [1.5]
    with pytest.raises(ValueError, match="Fitur wajib"):
        model.predict({})
