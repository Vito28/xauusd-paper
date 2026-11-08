"""Uji XGBoost library vs from scratch pada data XAU/USD (laporan bab 8.3 dan 10.5).

Langkahnya sama dengan `technical.run compare-scratch`: hyperparameter disamakan dan
subsampling dimatikan. Ditambah satu varian, yaitu scratch yang memakai batas bin buatan
XGBoost hist dan fitur float32. Bila varian ini identik dengan library, semua selisih
berasal dari kandidat split (batas bin), bukan dari rumus gradien, Hessian, bobot daun,
gain, atau base score.

Jalankan dari root repo xauusd-paper:

    uv run python <dir>/uji_scratch_bin.py --year 2025 --out <dir>/uji_scratch_bin_2025.json \
        --candles-dir runs/<walk-forward>/backtest/candles --history-start 2021-01-01T00:00:00Z

Tanpa --candles-dir, candle dibaca dari PostgreSQL seperti perintah compare-scratch.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost  # noqa: E402

import technical.xgboost_scratch as scratch_module  # noqa: E402
from technical.config import load_config  # noqa: E402
from technical.decision import apply_temperature, decide  # noqa: E402
from technical.run import build_dataset, load_candles, run_fold  # noqa: E402
from technical.splits import make_validation_folds  # noqa: E402

NO_TRADE = 1
MARGIN_LEVELS = [0.01, 0.02]  # selisih dua probabilitas teratas untuk memisahkan baris "ragu"

ORIGINAL_FIT = scratch_module.QuantileBinner.fit
ORIGINAL_TRANSFORM = scratch_module.QuantileBinner.transform
CUT_REPORT = {}
LIBRARY_THREADS = {"n_jobs": 1}


def library_cut_fit(self, train_features):
    """Pakai batas bin yang dibuat XGBoost hist pada data TRAINING yang sama."""
    # nthread sama dengan n_jobs XGBClassifier agar sketch kuantilnya sama.
    matrix = xgboost.QuantileDMatrix(
        train_features, max_bin=self.max_bin, nthread=LIBRARY_THREADS["n_jobs"]
    )
    indptr, values = matrix.get_quantile_cut()
    plain_cuts = ORIGINAL_FIT(scratch_module.QuantileBinner(self.max_bin), train_features)
    self.cut_points = []
    identical = 0
    for column in range(train_features.shape[1]):
        cuts = np.asarray(values[indptr[column] : indptr[column + 1]], dtype=float)
        # XGBoost 3.x: nilai pertama -inf dan nilai terakhir penjaga di atas maksimum.
        cuts = cuts[1:-1]
        self.cut_points.append(cuts)
        plain = plain_cuts.cut_points[column]
        if len(plain) == len(cuts) and np.allclose(plain, cuts, rtol=0, atol=1e-6):
            identical += 1
    CUT_REPORT["features_with_identical_cuts"] = identical
    CUT_REPORT["feature_count"] = train_features.shape[1]
    return self


def float32_transform(self, values):
    """XGBoost menyimpan fitur sebagai float32; samakan sebelum dibandingkan dengan batas bin."""
    values = np.asarray(values, dtype=np.float32).astype(float)
    return ORIGINAL_TRANSFORM(self, values)


def equalize_hyperparameters(config) -> None:
    """Sama persis dengan command_compare_scratch di technical/run.py."""
    config.xgboost.subsample = 1.0
    config.xgboost.colsample_bytree = 1.0
    config.xgboost.n_estimators = config.scratch.n_estimators
    config.xgboost.max_depth = config.scratch.max_depth
    config.xgboost.learning_rate = config.scratch.learning_rate
    config.xgboost.min_child_weight = config.scratch.min_child_weight
    config.xgboost.reg_lambda = config.scratch.reg_lambda
    config.xgboost.reg_alpha = config.scratch.reg_alpha
    config.xgboost.gamma = 2 * config.scratch.gamma
    config.xgboost.max_bin = config.scratch.max_bin


def read_candles_dir(config, candles_dir: Path):
    """Gabungkan backtest/candles/validation_*.csv dari satu run walk-forward."""
    frames = []
    for path in sorted(candles_dir.glob("validation_*.csv")):
        frames.append(pd.read_csv(path))
    raw = pd.concat(frames).drop_duplicates("open_time_utc").sort_values("open_time_utc")
    with tempfile.TemporaryDirectory() as folder:
        csv_path = Path(folder) / "candles.csv"
        raw.to_csv(csv_path, index=False)
        return load_candles(config, str(csv_path), config.data.development_end)


def fit_variant(name, fold, features, labels, baseline_inputs, config) -> dict:
    use_library_cuts = name == "scratch_library_cuts"
    if use_library_cuts:
        scratch_module.QuantileBinner.fit = library_cut_fit
        scratch_module.QuantileBinner.transform = float32_transform
    try:
        model_name = "xgboost" if name == "library" else "xgboost_scratch"
        result = run_fold(model_name, fold, features, labels, baseline_inputs, config)
        rows = features.loc[result["_blocks"]["evaluation"]]
        raw = result["_model"].predict_proba(rows)
    finally:
        scratch_module.QuantileBinner.fit = ORIGINAL_FIT
        scratch_module.QuantileBinner.transform = ORIGINAL_TRANSFORM
    probabilities = apply_temperature(raw, result["_temperature"])
    decisions, _ = decide(probabilities, result["_thresholds"])
    print(f"{name}: {result['training_seconds']:.1f} detik", flush=True)
    return {"result": result, "probabilities": probabilities, "decisions": decisions}


def describe(variant: dict, labels_true: np.ndarray) -> dict:
    result = variant["result"]
    evaluation = result["evaluation"]
    thresholds = {}
    for direction, rule in result["thresholds"].items():
        thresholds[direction] = rule["threshold"] if rule["enabled"] else None
    return {
        "trees": result["best_iteration"] + 1,
        "training_seconds": result["training_seconds"],
        "temperature": result["calibration"]["temperature"],
        "thresholds": thresholds,
        "signals": {d: evaluation["per_class"][d]["predicted"] for d in ["BUY", "SELL"]},
        "precision": {d: evaluation["per_class"][d]["precision"] for d in ["BUY", "SELL"]},
        "coverage": evaluation["signal_coverage"],
        "argmax_macro_f1": result["evaluation_without_threshold"]["macro_f1"],
        "argmax_accuracy": float((variant["probabilities"].argmax(axis=1) == labels_true).mean()),
        "log_loss": evaluation["probability"]["log_loss"],
    }


def agreement(library: dict, other: dict) -> dict:
    reference = library["probabilities"]
    probabilities = other["probabilities"]
    ordered = np.sort(reference, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    same_argmax = reference.argmax(axis=1) == probabilities.argmax(axis=1)
    with_library_thresholds, _ = decide(probabilities, library["result"]["_thresholds"])
    difference = np.abs(reference - probabilities)
    report = {
        "argmax_agreement": float(same_argmax.mean()),
        "mean_abs_probability_difference": float(difference.mean()),
        "max_abs_probability_difference": float(difference.max()),
        "pearson_sell_notrade_buy": [
            float(np.corrcoef(reference[:, k], probabilities[:, k])[0, 1]) for k in range(3)
        ],
        "decision_agreement_own_thresholds": float(
            (library["decisions"] == other["decisions"]).mean()
        ),
        "decision_agreement_library_thresholds": float(
            (library["decisions"] == with_library_thresholds).mean()
        ),
    }
    for level in MARGIN_LEVELS:
        confident = margin >= level
        report[f"share_rows_margin_ge_{level}"] = float(confident.mean())
        report[f"argmax_agreement_margin_ge_{level}"] = float(same_argmax[confident].mean())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--out", required=True)
    parser.add_argument("--candles-dir", default=None)
    parser.add_argument("--history-start", default=None)
    arguments = parser.parse_args()

    config = load_config()
    if arguments.history_start:
        config.data.history_start = arguments.history_start
    equalize_hyperparameters(config)
    LIBRARY_THREADS["n_jobs"] = config.run.n_jobs
    if arguments.candles_dir:
        candles, data_report = read_candles_dir(config, Path(arguments.candles_dir))
        source = f"{arguments.candles_dir}/validation_*.csv"
    else:
        candles, data_report = load_candles(config, None, config.data.development_end)
        source = data_report["source"]
    features, _, labels, baseline_inputs = build_dataset(candles, config, config.features.groups)
    fold = make_validation_folds(config.data.history_start, [arguments.year])[0]

    variants = {}
    for name in ["library", "scratch", "scratch_library_cuts"]:
        variants[name] = fit_variant(name, fold, features, labels, baseline_inputs, config)

    evaluation_rows = variants["library"]["result"]["_blocks"]["evaluation"]
    labels_true = labels["label"].loc[evaluation_rows].astype(int).to_numpy()
    report = {
        "source": source,
        "history_start": config.data.history_start,
        "rows_used": data_report["rows_used"],
        "fold": fold.describe(),
        "block_sizes": variants["library"]["result"]["block_sizes"],
        "packages": {"xgboost": xgboost.__version__, "numpy": np.__version__},
        "cuts": CUT_REPORT,
        "models": {name: describe(variant, labels_true) for name, variant in variants.items()},
        "vs_library": {
            name: agreement(variants["library"], variants[name])
            for name in ["scratch", "scratch_library_cuts"]
        },
    }
    Path(arguments.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["vs_library"], indent=2))


if __name__ == "__main__":
    main()
