"""XGBoost matematis dalam satu file, hanya standard library Python.

Section 01-04: training Newton gradient boosting dari nol (exact split), dengan
classifier dan regressor terpisah. Section 05-08: inference manual pohon JSON
hasil technical, kalibrasi, policy, evaluasi, dan CLI pembanding angka.

PENTING: training exact-split di sini bukan replika training histogram XGBoost.
Mode pembanding memakai BOBOT TERLATIH YANG SAMA dan INPUT FITUR YANG SAMA.
Ia tidak memanggil XGBoost, NumPy, pandas, sklearn, technical, atau plugin.
Feature engineering 223 kolom, ReliefF/RFECV, pencarian parameter dan fitting
kalibrasi technical tidak diulang di sini. Input model adalah CSV fitur numerik,
bukan OHLC mentah. Pohon JSON/reference CSV hanya data, bukan library.

Jalankan tanpa site-packages untuk membuktikan runtime mandiri:
  python -S analisis_teknikall/xgboost_matematis.py demo
  python -S analisis_teknikall/xgboost_matematis.py verify
  python -S analisis_teknikall/xgboost_matematis.py predict --input fitur.csv

Rumus: https://xgboost.readthedocs.io/en/stable/tutorials/model.html
Format: https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html
"""

# %% 01. STANDARD LIBRARY DAN MATEMATIKA DASAR
# math = exp/log/sqrt; struct = pembulatan IEEE-754 float32; csv/json = baca data.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LABELS = ("SELL", "NO_TRADE", "BUY")
ROOT = Path(__file__).resolve().parent


def f32(value: float) -> float:
    """Round one value to IEEE-754 binary32, matching tree input/accumulation precision."""
    return struct.unpack("f", struct.pack("f", value))[0]


def softmax(margins: list[float], *, single: bool = False) -> list[float]:
    """Convert additive class margins into probabilities using stable exp(z-max(z))."""
    cast = f32 if single else float
    peak = max(margins)
    weights = [cast(math.exp(cast(value - peak))) for value in margins]
    total = 0.0
    for weight in weights:
        total = cast(total + weight)
    return [cast(weight / total) for weight in weights]


def sigmoid(value: float) -> float:
    """Compute logistic probability without overflowing large negative margins."""
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def soft_threshold(gradient: float, alpha: float) -> float:
    """Apply L1 regularization: sign(G) * max(abs(G)-alpha, 0)."""
    return math.copysign(max(abs(gradient) - alpha, 0.0), gradient)


def validate_matrix(x: list[list[float]], y: list[float | int]) -> None:
    """Require aligned finite targets and a rectangular matrix; NaN features are allowed."""
    if not x or not x[0] or len(x) != len(y):
        raise ValueError("X/y kosong atau panjang tidak sama")
    if any(len(row) != len(x[0]) for row in x):
        raise ValueError("Jumlah fitur setiap baris harus sama")
    if any(math.isinf(v) for row in x for v in row) or not all(map(math.isfinite, y)):
        raise ValueError("Infinity fitur atau target non-finite tidak diperbolehkan")


# %% 02. POHON NEWTON DARI NOL: GRADIENT, HESSIAN, LEAF, SPLIT GAIN
# Aproksimasi loss: L ≈ Σ[g_i*w + 1/2*h_i*w²] + 1/2*lambda*w² + alpha*|w|.
# G=Σg_i, H=Σh_i. Leaf optimal = -soft_threshold(G, alpha)/(H+lambda).
# Gain = 1/2*(score_left + score_right - score_parent) - gamma.
# Missing value diuji masuk kiri/kanan; arah terbaik disimpan dalam node.


@dataclass
class TreeConfig:
    """Controls for educational exact-split training, independently of technical tuning."""

    max_depth: int = 3
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    gamma: float = 0.0

    def __post_init__(self) -> None:
        """Reject invalid Hessian constraints or regularization before growing trees."""
        numbers = (self.min_child_weight, self.reg_lambda, self.reg_alpha, self.gamma)
        if self.max_depth < 0 or any(not math.isfinite(v) or v < 0 for v in numbers):
            raise ValueError("Parameter pohon harus finite dan nonnegatif")


@dataclass
class Node:
    """A learned scalar leaf or numeric decision node with explicit missing-value routing."""

    value: float
    feature: int = -1
    threshold: float = 0.0
    missing_left: bool = True
    left: Node | None = None
    right: Node | None = None

    def predict(self, row: list[float]) -> float:
        """Traverse stored splits until a leaf; equality follows the right branch."""
        node = self
        while node.feature >= 0:
            value = row[node.feature]
            left = node.missing_left if math.isnan(value) else value < node.threshold
            child = node.left if left else node.right
            if child is None:
                raise ValueError("Pohon rusak: child tidak tersedia")
            node = child
        return node.value


def grow_tree(
    x: list[list[float]], gradients: list[float], hessians: list[float], config: TreeConfig
) -> Node:
    """Fit an exact greedy Newton tree with prefix sums and no external ML implementation.

    Splits enumerate distinct observed values. This deliberately does not reproduce
    histogram quantile sketches, parallel reductions, or random subsampling.
    """
    validate_matrix(x, gradients)
    if len(hessians) != len(x) or any(not math.isfinite(h) or h < 0 for h in hessians):
        raise ValueError("Hessian harus finite, nonnegatif dan sejajar X")

    def score(g: float, h: float) -> float:
        """Return regularized gain numerator divided by curvature."""
        return (
            soft_threshold(g, config.reg_alpha) ** 2 / (h + config.reg_lambda)
            if h + config.reg_lambda > 0
            else 0.0
        )

    def build(indices: list[int], depth: int) -> Node:
        """Choose the largest positive gain, partition rows and recurse with depth limit."""
        g, h = sum(gradients[i] for i in indices), sum(hessians[i] for i in indices)
        node = Node(
            -soft_threshold(g, config.reg_alpha) / (h + config.reg_lambda)
            if h + config.reg_lambda > 0
            else 0.0
        )
        if depth >= config.max_depth or len(indices) < 2:
            return node
        best_gain = 0.0
        best: tuple[int, float, bool] | None = None
        for feature in range(len(x[0])):
            known = sorted((x[i][feature], i) for i in indices if not math.isnan(x[i][feature]))
            missing = [i for i in indices if math.isnan(x[i][feature])]
            mg, mh = sum(gradients[i] for i in missing), sum(hessians[i] for i in missing)
            lg = lh = 0.0
            # Include missing-only partitions at either extreme as well as interior splits.
            candidates = [(0, known[0][0])] if known else []
            candidates += [
                (j, known[j][0]) for j in range(1, len(known)) if known[j - 1][0] < known[j][0]
            ]
            if known and missing:
                candidates.append((len(known), math.nextafter(known[-1][0], math.inf)))
            consumed = 0
            for cut, threshold in candidates:
                while consumed < cut:
                    i = known[consumed][1]
                    lg += gradients[i]
                    lh += hessians[i]
                    consumed += 1
                for missing_left in (True, False):
                    count = cut + (len(missing) if missing_left else 0)
                    if count == 0 or count == len(indices):
                        continue
                    gl, hl = lg + (mg if missing_left else 0), lh + (mh if missing_left else 0)
                    gr, hr = g - gl, max(0.0, h - hl)
                    if min(hl, hr) < config.min_child_weight:
                        continue
                    gain = 0.5 * (score(gl, hl) + score(gr, hr) - score(g, h)) - config.gamma
                    if gain > best_gain:
                        best_gain, best = gain, (feature, threshold, missing_left)
        if best is not None:
            feature, threshold, missing_left = best
            left = [
                i
                for i in indices
                if (missing_left if math.isnan(x[i][feature]) else x[i][feature] < threshold)
            ]
            left_set = set(left)
            right = [i for i in indices if i not in left_set]
            node.feature, node.threshold, node.missing_left = best
            node.left, node.right = build(left, depth + 1), build(right, depth + 1)
        return node

    return build(list(range(len(x))), 0)


# %% 03. REGRESSION — TARGET ANGKA RETURN, BUKAN LABEL BUY/SELL
# Squared loss: L=1/2*(pred-y)², g=pred-y, h=1.
# Pseudo-Huber: g=r/sqrt(1+(r/delta)²), h=(1+(r/delta)²)^(-3/2).
# Boosting: F_(t+1)(x)=F_t(x)+eta*tree_t(x).
# technical memakai target return*10000 (bps); konversi kembali dilakukan terpisah.


class ScratchRegressor:
    """Train squared or pseudo-Huber Newton boosting manually using exact numeric splits."""

    def __init__(
        self,
        rounds: int = 30,
        learning_rate: float = 0.1,
        config: TreeConfig | None = None,
        objective: str = "squared",
        delta: float = 10.0,
    ) -> None:
        """Configure scalar boosting; no parameters or weights are borrowed from technical."""
        if (
            rounds < 1
            or not 0 < learning_rate <= 1
            or objective not in {"squared", "huber"}
            or not math.isfinite(delta)
            or delta <= 0
        ):
            raise ValueError("Parameter regressor tidak valid")
        self.rounds, self.learning_rate = rounds, learning_rate
        self.config, self.objective, self.delta = config or TreeConfig(), objective, delta
        self.base, self.trees, self.width = 0.0, [], 0

    def fit(self, x: list[list[float]], y: list[float]) -> ScratchRegressor:
        """Learn from supplied TRAIN rows only; reset previous state on every fit."""
        validate_matrix(x, y)
        self.base, self.trees, self.width = sum(y) / len(y), [], len(x[0])
        predictions = [self.base] * len(y)
        for _ in range(self.rounds):
            residual = [p - t for p, t in zip(predictions, y, strict=True)]
            if self.objective == "squared":
                gradients, hessians = residual, [1.0] * len(y)
            else:
                scale = [1 + (r / self.delta) ** 2 for r in residual]
                gradients = [r / math.sqrt(s) for r, s in zip(residual, scale, strict=True)]
                hessians = [s ** (-1.5) for s in scale]
            tree = grow_tree(x, gradients, hessians, self.config)
            self.trees.append(tree)
            predictions = [
                p + self.learning_rate * tree.predict(row)
                for p, row in zip(predictions, x, strict=True)
            ]
        return self

    def predict(self, x: list[list[float]]) -> list[float]:
        """Sum learned scalar leaves with learning-rate shrinkage to obtain numeric predictions."""
        if not self.trees or any(len(row) != self.width for row in x):
            raise ValueError("Model belum fit atau jumlah fitur berbeda")
        return [
            self.base + self.learning_rate * sum(tree.predict(row) for tree in self.trees)
            for row in x
        ]


# %% 04. CLASSIFIER — TARGET SELL=0, NO_TRADE=1, BUY=2
# Cross entropy: L=-log(p_y), p=softmax(margin).
# Gradient kelas k: g_k=p_k-1[y=k].
# Hessian diagonal upper bound XGBoost: h_k=max(2*p_k*(1-p_k), 1e-16).
# Satu pohon per kelas per ronde; semua gradient dihitung SEBELUM update ronde.


class ScratchClassifier:
    """Train three-class softmax Newton boosting independently from scalar regression."""

    def __init__(
        self, rounds: int = 30, learning_rate: float = 0.1, config: TreeConfig | None = None
    ) -> None:
        """Set training controls for three separate class trees in each boosting round."""
        if rounds < 1 or not 0 < learning_rate <= 1:
            raise ValueError("Parameter classifier tidak valid")
        self.rounds, self.learning_rate, self.config = rounds, learning_rate, config or TreeConfig()
        self.trees: list[list[Node]] = []
        self.width = 0

    def fit(self, x: list[list[float]], y: list[int]) -> ScratchClassifier:
        """Fit multiclass cross entropy from supplied TRAIN data; require all three classes."""
        validate_matrix(x, y)
        if set(y) != {0, 1, 2}:
            raise ValueError("Classifier membutuhkan SELL=0, NO_TRADE=1, BUY=2")
        self.trees, self.width = [], len(x[0])
        margins = [[0.0] * 3 for _ in y]
        for _ in range(self.rounds):
            probabilities = [softmax(row) for row in margins]
            trees = []
            for label in range(3):
                gradients = [
                    p[label] - int(target == label)
                    for p, target in zip(probabilities, y, strict=True)
                ]
                hessians = [max(2 * p[label] * (1 - p[label]), 1e-16) for p in probabilities]
                trees.append(grow_tree(x, gradients, hessians, self.config))
            self.trees.append(trees)
            for margin, row in zip(margins, x, strict=True):
                for label, tree in enumerate(trees):
                    margin[label] += self.learning_rate * tree.predict(row)
        return self

    def predict_proba(self, x: list[list[float]]) -> list[list[float]]:
        """Produce uncalibrated SELL/NO_TRADE/BUY probabilities from learned margins."""
        if not self.trees or any(len(row) != self.width for row in x):
            raise ValueError("Model belum fit atau jumlah fitur berbeda")
        return [
            softmax(
                [
                    self.learning_rate
                    * sum(round_trees[k].predict(row) for round_trees in self.trees)
                    for k in range(3)
                ]
            )
            for row in x
        ]

    def predict(self, x: list[list[float]]) -> list[int]:
        """Return first argmax class; this raw classifier is not a validated trading policy."""
        return [max(range(3), key=p.__getitem__) for p in self.predict_proba(x)]


# %% 05. INFERENCE POHON TECHNICAL SECARA MANUAL — BUKAN TRAINING ULANG
# JSON menyimpan threshold, arah missing dan nilai daun. Leaf SUDAH dikali eta.
# Margin_k = base_score_k + Σ leaf_t(x) untuk tree_info[t]=k.
# Gunakan split_conditions pada leaf, bukan base_weights node internal.


class FrozenTrees:
    """Read native XGBoost JSON and evaluate numeric scalar-leaf gbtree ensembles manually."""

    def __init__(self, document: dict[str, Any]) -> None:
        """Validate objectives and compile float32 trees; reject categorical/vector leaves."""
        learner = document["learner"]
        booster = learner["gradient_booster"]
        self.objective = learner["objective"]["name"]
        if booster["name"] != "gbtree" or self.objective not in {
            "multi:softprob",
            "reg:squarederror",
            "reg:pseudohubererror",
            "reg:quantileerror",
        }:
            raise ValueError("Hanya numeric gbtree technical yang didukung")
        self.names = learner["feature_names"]
        params = learner["learner_model_param"]
        groups = max(int(params["num_class"]), int(params["num_target"]))
        base = json.loads(params["base_score"])
        self.base = [f32(v) for v in (base if isinstance(base, list) else [base] * groups)]
        if len(self.base) != groups or len(self.names) != int(params["num_feature"]):
            raise ValueError("Schema base_score/feature tidak cocok")
        self.trees = []
        model = booster["model"]
        for tree, group in zip(model["trees"], model["tree_info"], strict=True):
            if (
                any(tree["split_type"])
                or int(tree["tree_param"]["size_leaf_vector"]) > 1
                or not 0 <= group < groups
            ):
                raise ValueError("Categorical/vector leaf tidak didukung")
            self.trees.append(
                (
                    group,
                    tree["left_children"],
                    tree["right_children"],
                    tree["default_left"],
                    tree["split_indices"],
                    [f32(v) for v in tree["split_conditions"]],
                )
            )

    def margins(self, features: dict[str, float]) -> list[float]:
        """Sum leaves; reject absent columns and route explicit NaN through default branches."""
        if not set(self.names) <= features.keys():
            raise ValueError(
                "Fitur wajib hilang: " + ", ".join(sorted(set(self.names) - features.keys()))
            )
        row = [f32(features[name]) for name in self.names]
        if any(math.isinf(v) for v in row):
            raise ValueError("Infinity tidak boleh masuk model")
        result = self.base.copy()
        for group, left, right, default, indices, values in self.trees:
            node = 0
            while left[node] != -1:
                value = row[indices[node]]
                go_left = default[node] if math.isnan(value) else value < values[node]
                node = left[node] if go_left else right[node]
            result[group] = f32(result[group] + values[node])
        return result

    def predict(self, features: dict[str, float]) -> list[float]:
        """Apply softmax only to the classifier; regression/quantile outputs remain numeric."""
        margins = self.margins(features)
        return softmax(margins, single=True) if self.objective == "multi:softprob" else margins


# %% 06. KALIBRASI CLASSIFIER, RETURN REGRESSOR, DAN POLICY TERPISAH
# Kalibrasi: logit=log(p/(1-p)), q=sigmoid(a*logit+b), p_cal=q/Σq.
# Return: raw_bps/10000*shrinkage; quantile lower/upper diperluas residual CAL.
# Policy tidak menerima harga masa depan. BUY/SELL wajib searah dengan regressor.


def choose_signal(
    probabilities: list[float],
    point: float,
    lower: float,
    upper: float,
    cost: float,
    policy: dict[str, Any],
) -> tuple[str, str]:
    """Reproduce frozen technical abstention rules without using any actual outcome."""
    if not all(map(math.isfinite, [*probabilities, point, lower, upper, cost])) or cost < 0:
        return "NO_TRADE", "unavailable_model_or_cost"
    label = max(range(3), key=probabilities.__getitem__)
    if label == 1:
        return "NO_TRADE", "classifier_no_trade"
    name = LABELS[label]
    rule = policy["classes"][name]
    if not rule["enabled"]:
        return "NO_TRADE", "class_precision_unconfirmed"
    margin = probabilities[label] - max(p for i, p in enumerate(probabilities) if i != label)
    if probabilities[label] < rule["threshold"] or margin < rule["probability_margin"]:
        return "NO_TRADE", "confidence_abstain"
    direction = 1 if label == 2 else -1
    if direction * point <= cost * rule["edge_multiple"]:
        return "NO_TRADE", "regression_edge_disagreement"
    if policy["require_interval_agreement"] and (lower if label == 2 else -upper) <= cost:
        return "NO_TRADE", "regression_uncertain"
    return name, "confirmed_model_agreement"


class TechnicalMath:
    """Combine separate classifier, regressor and quantile predictions with frozen metadata."""

    def __init__(self, folder: Path) -> None:
        """Load JSON data only, checking exported model hashes before use."""
        self.meta = json.loads((folder / "contract.json").read_text(encoding="utf-8"))
        if self.meta["schema"] != "technical-math-v1":
            raise ValueError("Contract matematika tidak didukung")
        for name, expected in self.meta["sha256"].items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Hash artifact tidak cocok: " + name)
        self.classifier = FrozenTrees(
            json.loads((folder / "classifier.json").read_text(encoding="utf-8"))
        )
        self.regressor = FrozenTrees(
            json.loads((folder / "regressor.json").read_text(encoding="utf-8"))
        )
        self.quantile = FrozenTrees(
            json.loads((folder / "quantile.json").read_text(encoding="utf-8"))
        )
        if (
            self.classifier.names != self.meta["classifier_features"]
            or self.regressor.names != self.meta["regression_features"]
            or self.quantile.names != self.regressor.names
        ):
            raise ValueError("Urutan fitur contract berbeda dari pohon")

    def predict(self, features: dict[str, float], close: float, spread: float) -> dict[str, Any]:
        """Return calibrated probabilities, return/price forecasts and one policy decision."""
        if (
            not math.isfinite(close)
            or close <= 0
            or (not math.isnan(spread) and (not math.isfinite(spread) or spread < 0))
        ):
            raise ValueError("Harga/spread tidak valid")
        raw = self.classifier.predict(features)
        calibrated = []
        for p, a, b in zip(
            raw,
            self.meta["calibration_coefficients"],
            self.meta["calibration_intercepts"],
            strict=True,
        ):
            clipped = min(max(p, 1e-6), 1 - 1e-6)
            calibrated.append(sigmoid(a * math.log(clipped / (1 - clipped)) + b))
        total = sum(calibrated)
        probabilities = [p / total for p in calibrated]
        # Float32 membulatkan tiap tahap array operasi technical, termasuk 1+return.
        point = f32(
            f32(self.regressor.predict(features)[0] / 10000)
            * f32(self.meta["regression_shrinkage"])
        )
        bounds = sorted(f32(v / 10000) for v in self.quantile.predict(features))
        expansion = f32(self.meta["interval_expansion"])
        lower, upper = f32(bounds[0] - expansion), f32(bounds[1] + expansion)
        cost = (
            spread / close
            + self.meta["minimum_edge_bps"] / 10000
            + self.meta["slippage_bps"] / 10000
        )
        signal, reason = choose_signal(
            probabilities, point, lower, upper, cost, self.meta["policy"]
        )
        if math.isnan(spread):
            signal, reason = "NO_TRADE", "spread_unavailable"
        raw_signal = LABELS[max(range(3), key=probabilities.__getitem__)]
        return {
            "probability_sell": probabilities[0],
            "probability_no_trade": probabilities[1],
            "probability_buy": probabilities[2],
            "confidence": max(probabilities),
            "raw_predicted_signal": raw_signal,
            "predicted_signal": signal,
            "policy_reason": reason,
            "predicted_future_return": point,
            "return_interval_lower": lower,
            "return_interval_upper": upper,
            "predicted_future_close": close * f32(1 + point),
        }


# %% 07. EVALUASI ARAH DAN LABA — TERPISAH DARI MODEL DAN TANPA LABEL LEAKAGE
# BUY gross=actual-close; SELL gross=close-actual. Net=gross-spread-slippage.
# NO_TRADE selalu netral. Harga tetap bukan sukses; actual hilang tidak dinilai.


def outcome(
    signal: str, close: float, actual: float | None, spread: float, slippage_bps: float = 0.0
) -> dict[str, Any]:
    """Evaluate actual after inference in price units, independently of account P/L."""
    if signal not in LABELS:
        raise ValueError("Label sinyal tidak dikenal")
    if signal == "NO_TRADE":
        return {"outcome_status": "NEUTRAL", "realized_gross_move": None, "realized_net_move": None}
    if actual is None or not math.isfinite(actual):
        return {
            "outcome_status": "UNAVAILABLE",
            "realized_gross_move": None,
            "realized_net_move": None,
        }
    gross = (1 if signal == "BUY" else -1) * (actual - close)
    net = gross - spread - close * slippage_bps / 10000
    return {
        "outcome_status": "SUCCESS" if gross > 0 else "FAILURE" if gross < 0 else "FLAT",
        "realized_gross_move": gross,
        "realized_net_move": net if math.isfinite(net) else None,
    }


def read_feature_rows(path: Path) -> list[dict[str, str]]:
    """Read named feature rows; empty numeric cells are explicit missing observations."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def predict_csv(model: TechnicalMath, path: Path) -> list[dict[str, Any]]:
    """Predict CSV features; labels are excluded from input and indicators are not recomputed."""
    names = set(model.classifier.names + model.regressor.names)
    result = []
    for row in read_feature_rows(path):
        if not names <= row.keys() or not {"close", "spread_price"} <= row.keys():
            raise ValueError("CSV membutuhkan fitur model, close, spread_price")
        features = {name: float(row[name]) if row[name] else math.nan for name in names}
        close, spread = (
            float(row["close"]),
            float(row["spread_price"]) if row["spread_price"] else math.nan,
        )
        prediction = model.predict(features, close, spread)
        actual = float(row["actual_future_close"]) if row.get("actual_future_close") else None
        result.append(
            {
                "prediction_timestamp": row.get("prediction_timestamp", ""),
                "close": close,
                "spread_price": spread,
                **prediction,
                **outcome(
                    prediction["predicted_signal"],
                    close,
                    actual,
                    spread,
                    model.meta["slippage_bps"],
                ),
            }
        )
    return result


def verify(model: TechnicalMath, inputs: Path, reference: Path) -> dict[str, Any]:
    """Compare manual outputs with independent technical reference, failing on any tolerance breach.

    Floating probabilities allow 2e-6 absolute error; return/interval 2e-9;
    reconstructed prices 0.002. Discrete class and policy reason must match exactly.
    Reference labels never enter the prediction function.
    """
    predictions, expected = predict_csv(model, inputs), read_feature_rows(reference)
    if not predictions or len(predictions) != len(expected):
        raise ValueError("Reference kosong atau jumlah baris berbeda")
    tolerances = {
        k: 2e-6
        for k in ("probability_sell", "probability_no_trade", "probability_buy", "confidence")
    }
    tolerances.update(
        {
            k: 2e-9
            for k in ("predicted_future_return", "return_interval_lower", "return_interval_upper")
        }
    )
    tolerances["predicted_future_close"] = 0.002
    maxima = dict.fromkeys(tolerances, 0.0)
    mismatches = dict.fromkeys(("raw_predicted_signal", "predicted_signal", "policy_reason"), 0)
    for prediction, truth in zip(predictions, expected, strict=True):
        if prediction["prediction_timestamp"] != truth["prediction_timestamp"]:
            raise ValueError("Timestamp reference tidak sejajar")
        for key in maxima:
            error = abs(prediction[key] - float(truth[key]))
            if not math.isfinite(error):
                raise ValueError("Output numerik non-finite pada parity")
            maxima[key] = max(maxima[key], error)
        for key in mismatches:
            mismatches[key] += prediction[key] != truth[key]
    passed = all(maxima[k] <= tolerances[k] for k in maxima) and not any(mismatches.values())
    return {
        "passed": passed,
        "rows": len(predictions),
        "scope": "same frozen trees and same input features; not retraining equivalence",
        "absolute_tolerances": tolerances,
        "max_absolute_error": maxima,
        "discrete_mismatches": mismatches,
        "signals": {
            label: sum(p["predicted_signal"] == label for p in predictions) for label in LABELS
        },
        "input_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
    }


# %% 08. DEMO TRAINING MURNI DAN CLI INFERENCE/PERBANDINGAN


def demo() -> dict[str, Any]:
    """Train both scratch models on synthetic data, explicitly separate from market evaluation."""
    x = [[i / 30] for i in range(-30, 31)]
    y_reg = [2 * row[0] + 0.25 for row in x]
    y_class = [0 if row[0] < -0.3 else 2 if row[0] > 0.3 else 1 for row in x]
    regression = ScratchRegressor().fit(x, y_reg)
    classifier = ScratchClassifier().fit(x, y_class)
    predictions = regression.predict(x)
    return {
        "scope": "synthetic TRAIN sanity check, not holdout or technical parity",
        "regression_train_rmse": math.sqrt(
            sum((p - y) ** 2 for p, y in zip(predictions, y_reg, strict=True)) / len(x)
        ),
        "classification_train_accuracy": sum(
            p == y for p, y in zip(classifier.predict(x), y_class, strict=True)
        )
        / len(x),
    }


def main() -> None:
    """Run pure-Python training demo, manual inference or strict reference verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("demo", "predict", "verify"))
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts" / "final_2026")
    parser.add_argument(
        "--input", type=Path, default=ROOT / "artifacts" / "final_2026" / "features.csv"
    )
    parser.add_argument(
        "--reference", type=Path, default=ROOT / "artifacts" / "final_2026" / "reference.csv"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "demo":
        print(json.dumps(demo(), indent=2))
        return
    model = TechnicalMath(args.model)
    if args.command == "verify":
        report = verify(model, args.input, args.reference)
        rendered = json.dumps(report, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered)
        if not report["passed"]:
            raise SystemExit(1)
    else:
        rows = predict_csv(model, args.input)
        if not rows:
            raise ValueError("CSV fitur kosong")
        output = args.output or ROOT / "artifacts" / "predictions_manual.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{len(rows)} prediksi ditulis ke {output}")


if __name__ == "__main__":
    main()
