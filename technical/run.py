"""Runner eksperimen Technical Agent (command line).

Perintah:
  label-scan      Distribusi label untuk beberapa horizon dan multiple ATR,
                  hanya pada data pengembangan (< development_end). Tanpa model.
  walk-forward    Validasi walk-forward 2021-2025 untuk model yang dipilih.
  ablation        Walk-forward XGBoost dengan menghapus satu kelompok fitur per run.
  compare-scratch Bandingkan XGBoost resmi vs from scratch pada satu fold validasi.
  final-test      Latih model final (data <= 2025) lalu uji di 2026. Wajib --confirm.

Contoh:
  uv run python -m technical.run walk-forward --models logistic,xgboost
  uv run python -m technical.run walk-forward --input-csv data/xauusd_m5.csv

Setiap run menulis folder runs/<waktu>_<perintah>_<hash>/ berisi config,
laporan data, metrik, dan (untuk final-test) prediksi per candle.
"""

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from technical.config import (
    CLASS_NAMES,
    PROJECT_ROOT,
    TechnicalConfig,
    load_config,
)
from technical.data import load_from_csv, load_from_postgres, prepare_candles
from technical.decision import (
    apply_temperature,
    build_signal,
    decide,
    fit_temperature,
    select_thresholds,
)
from technical.evaluation import evaluate_predictions
from technical.features import build_features
from technical.labels import label_distribution, make_labels
from technical.models import create_model, is_rule_baseline
from technical.splits import (
    assert_no_overlap,
    make_final_fold,
    make_validation_folds,
    split_fold,
)

SNAPSHOT_FEATURES = [
    "rsi_14",
    "adx_14",
    "macd_histogram_atr",
    "atr_14_pct",
    "close_vs_ema_48_atr",
    "spread_atr",
]


# ---------------------------------------------------------------------------
# Utilitas pencatatan eksperimen
# ---------------------------------------------------------------------------


def config_hash(config: TechnicalConfig) -> str:
    text = json.dumps(config.to_dict(), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def git_commit() -> str:
    try:
        output = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return output.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def package_versions() -> dict:
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }


def peak_memory_mb() -> float:
    # Linux: ru_maxrss dalam kilobyte. Ini puncak memori proses sejak start.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def create_run_directory(config: TechnicalConfig, command: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}_{command}_{config_hash(config)}"
    directory = PROJECT_ROOT / config.run.output_dir / run_id
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def write_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str, allow_nan=False)


def clean_for_json(value):
    """Ubah NaN/inf menjadi None agar JSON valid."""
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            cleaned[key] = clean_for_json(item)
        return cleaned
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.floating):
        return clean_for_json(float(value))
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_run_metadata(
    directory: Path, config: TechnicalConfig, command: str, data_report: dict, extra: dict
) -> None:
    metadata = {
        "command": command,
        "created_utc": datetime.now(UTC).isoformat(),
        "config_hash": config_hash(config),
        "git_commit": git_commit(),
        "packages": package_versions(),
        "config": config.to_dict(),
        "data_report": data_report,
    }
    metadata.update(extra)
    write_json(directory / "run.json", clean_for_json(metadata))


# ---------------------------------------------------------------------------
# Persiapan dataset
# ---------------------------------------------------------------------------


def load_candles(
    config: TechnicalConfig, input_csv: str | None, end: str
) -> tuple[pd.DataFrame, dict]:
    if input_csv:
        raw = load_from_csv(input_csv)
        source = f"csv:{input_csv}"
    else:
        raw = load_from_postgres(
            config.data.source_key, config.data.symbol, config.data.history_start, end
        )
        source = f"postgres:market.processed_m5:{config.data.source_key}"

    raw = raw[
        pd.to_datetime(raw["open_time_utc"], utc=True) >= pd.Timestamp(config.data.history_start)
    ]
    candles, report = prepare_candles(raw)
    candles = candles[candles["prediction_time"] <= pd.Timestamp(end)].reset_index(drop=True)
    report["source"] = source
    report["rows_used"] = len(candles)
    return candles, report


def build_dataset(candles: pd.DataFrame, config: TechnicalConfig, groups: list[str]):
    features, feature_group = build_features(candles, groups)
    labels = make_labels(
        candles,
        config.labels.method,
        config.labels.horizon_bars,
        config.labels.atr_period,
        config.labels.barrier_atr_multiple,
    )
    # Input khusus baseline persistence: gerak H candle terakhir dan barrier di waktu t.
    horizon = config.labels.horizon_bars
    baseline_inputs = pd.DataFrame(index=candles.index)
    baseline_inputs["past_move"] = candles["close"] - candles["close"].shift(horizon)
    baseline_inputs["barrier"] = labels["barrier"]
    return features, feature_group, labels, baseline_inputs


# ---------------------------------------------------------------------------
# Satu fold: fit -> kalibrasi -> threshold -> evaluasi
# ---------------------------------------------------------------------------


def run_fold(model_name, fold, features, labels, baseline_inputs, config) -> dict:
    blocks = split_fold(labels, fold, config.splits.embargo_bars)
    assert_no_overlap(labels, blocks)
    for block_name, rows in blocks.items():
        if len(rows) == 0:
            raise ValueError(f"Blok {block_name} kosong pada {fold.name}")

    model_inputs = baseline_inputs if is_rule_baseline(model_name) else features
    y = labels["label"]

    train_x = model_inputs.loc[blocks["train"]]
    train_y = y.loc[blocks["train"]].astype(int).to_numpy()
    calibration_x = model_inputs.loc[blocks["calibration"]]
    calibration_y = y.loc[blocks["calibration"]].astype(int).to_numpy()
    threshold_x = model_inputs.loc[blocks["threshold"]]
    threshold_y = y.loc[blocks["threshold"]].astype(int).to_numpy()
    evaluation_x = model_inputs.loc[blocks["evaluation"]]
    evaluation_y = y.loc[blocks["evaluation"]].astype(int).to_numpy()

    model = create_model(model_name, config)
    started = time.perf_counter()
    model.fit(train_x, train_y, calibration_x, calibration_y)
    training_seconds = time.perf_counter() - started

    result = {
        "fold": fold.describe(),
        "model": model_name,
        "block_sizes": {name: len(rows) for name, rows in blocks.items()},
        "label_distribution": {
            "train": label_distribution(y.loc[blocks["train"]]),
            "evaluation": label_distribution(y.loc[blocks["evaluation"]]),
        },
        "training_seconds": training_seconds,
    }
    if hasattr(model, "best_iteration"):
        result["best_iteration"] = model.best_iteration()

    evaluation_raw = model.predict_proba(evaluation_x)

    if is_rule_baseline(model_name):
        # Baseline aturan langsung menghasilkan keputusan; tidak ada kalibrasi/threshold.
        decisions = evaluation_raw.argmax(axis=1)
        result["evaluation"] = evaluate_predictions(evaluation_y, decisions)
        return result

    calibration = fit_temperature(model.predict_proba(calibration_x), calibration_y)
    temperature = calibration["temperature"]
    threshold_probabilities = apply_temperature(model.predict_proba(threshold_x), temperature)
    thresholds = select_thresholds(threshold_probabilities, threshold_y, config.decision)

    evaluation_probabilities = apply_temperature(evaluation_raw, temperature)
    decisions, _ = decide(evaluation_probabilities, thresholds)
    argmax_decisions = evaluation_probabilities.argmax(axis=1)

    result["calibration"] = calibration
    result["thresholds"] = summarize_thresholds(thresholds)
    result["evaluation"] = evaluate_predictions(evaluation_y, decisions, evaluation_probabilities)
    result["evaluation_without_threshold"] = evaluate_predictions(evaluation_y, argmax_decisions)
    result["_model"] = model
    result["_thresholds"] = thresholds
    result["_temperature"] = temperature
    result["_blocks"] = blocks
    return result


def summarize_thresholds(thresholds: dict) -> dict:
    summary = {}
    for name, rule in thresholds.items():
        summary[name] = {
            "enabled": rule["enabled"],
            "threshold": rule["threshold"],
            "prevalence": rule["prevalence"],
            "required_precision": rule["required_precision"],
            "chosen_on_threshold_block": rule["chosen"],
        }
    return summary


def public_result(result: dict) -> dict:
    """Buang objek internal (diawali '_') sebelum ditulis ke JSON."""
    cleaned = {}
    for key, value in result.items():
        if not key.startswith("_"):
            cleaned[key] = value
    return cleaned


def summarize_folds(fold_results: list[dict]) -> dict:
    """Rata-rata dan simpangan baku metrik utama antar fold."""
    keys = {
        "macro_f1": lambda r: r["evaluation"]["macro_f1"],
        "directional_macro_f1": lambda r: r["evaluation"]["directional_macro_f1"],
        "signal_coverage": lambda r: r["evaluation"]["signal_coverage"],
        "buy_f1": lambda r: r["evaluation"]["per_class"]["BUY"]["f1"],
        "sell_f1": lambda r: r["evaluation"]["per_class"]["SELL"]["f1"],
        "buy_precision": lambda r: r["evaluation"]["per_class"]["BUY"]["precision"],
        "sell_precision": lambda r: r["evaluation"]["per_class"]["SELL"]["precision"],
    }
    summary = {}
    for name, getter in keys.items():
        values = []
        for result in fold_results:
            value = getter(result)
            if value is not None:
                values.append(value)
        if len(values) == 0:
            summary[name] = {"mean": None, "std": None, "folds_with_value": 0}
        else:
            summary[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "folds_with_value": len(values),
            }
    return summary


# ---------------------------------------------------------------------------
# Perintah
# ---------------------------------------------------------------------------


def command_label_scan(config: TechnicalConfig, input_csv: str | None) -> Path:
    candles, report = load_candles(config, input_csv, config.data.development_end)
    directory = create_run_directory(config, "label-scan")
    rows = []
    for horizon in [3, 6, 12]:
        for multiple in [0.5, 1.0, 1.5, 2.0]:
            for method in ["triple_barrier", "fixed_horizon"]:
                labels = make_labels(candles, method, horizon, config.labels.atr_period, multiple)
                distribution = label_distribution(labels["label"])
                rows.append(
                    {
                        "method": method,
                        "horizon_bars": horizon,
                        "barrier_atr_multiple": multiple,
                        "labeled_rows": distribution["total"],
                        "sell_share": distribution["SELL"]["share"],
                        "no_trade_share": distribution["NO_TRADE"]["share"],
                        "buy_share": distribution["BUY"]["share"],
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(directory / "label_scan.csv", index=False)
    write_run_metadata(
        directory, config, "label-scan", report, {"note": "Data < development_end saja"}
    )
    print(table.to_string(index=False))
    return directory


def command_walk_forward(config, input_csv, model_names, groups=None) -> dict:
    if groups is None:
        groups = config.features.groups
    candles, report = load_candles(config, input_csv, config.data.development_end)
    features, feature_group, labels, baseline_inputs = build_dataset(candles, config, groups)
    folds = make_validation_folds(config.data.history_start, config.splits.validation_years)

    results = {}
    for model_name in model_names:
        fold_results = []
        for fold in folds:
            print(f"[{model_name}] {fold.name}")
            fold_result = run_fold(model_name, fold, features, labels, baseline_inputs, config)
            fold_results.append(fold_result)
        results[model_name] = {
            "folds": [public_result(r) for r in fold_results],
            "summary": summarize_folds(fold_results),
        }
    return {
        "data_report": report,
        "feature_groups": groups,
        "features": list(features.columns),
        "feature_group_map": feature_group,
        "results": results,
    }


def command_walk_forward_and_save(config, input_csv, model_names) -> Path:
    output = command_walk_forward(config, input_csv, model_names)
    directory = create_run_directory(config, "walk-forward")
    write_json(directory / "metrics.json", clean_for_json(output["results"]))
    write_run_metadata(
        directory,
        config,
        "walk-forward",
        output["data_report"],
        {
            "models": model_names,
            "features": output["features"],
            "feature_group_map": output["feature_group_map"],
            "peak_memory_mb": peak_memory_mb(),
        },
    )
    print_summary(output["results"])
    return directory


def command_ablation(config: TechnicalConfig, input_csv: str | None) -> Path:
    directory = create_run_directory(config, "ablation")
    ablation = {}
    variants = [("all", list(config.features.groups))]
    for group in config.features.groups:
        remaining = []
        for name in config.features.groups:
            if name != group:
                remaining.append(name)
        variants.append((f"without_{group}", remaining))

    report = {}
    for variant_name, groups in variants:
        print(f"== Ablation: {variant_name}")
        output = command_walk_forward(config, input_csv, ["xgboost"], groups)
        report = output["data_report"]
        ablation[variant_name] = {
            "groups": groups,
            "summary": output["results"]["xgboost"]["summary"],
        }
    write_json(directory / "ablation.json", clean_for_json(ablation))
    write_run_metadata(directory, config, "ablation", report, {"peak_memory_mb": peak_memory_mb()})
    return directory


def command_compare_scratch(config: TechnicalConfig, input_csv: str | None, year: int) -> Path:
    """XGBoost resmi vs from scratch dengan hyperparameter yang disamakan."""
    # Samakan konfigurasi: tanpa subsampling, parameter pohon sama.
    config.xgboost.subsample = 1.0
    config.xgboost.colsample_bytree = 1.0
    config.xgboost.n_estimators = config.scratch.n_estimators
    config.xgboost.max_depth = config.scratch.max_depth
    config.xgboost.learning_rate = config.scratch.learning_rate
    config.xgboost.min_child_weight = config.scratch.min_child_weight
    config.xgboost.reg_lambda = config.scratch.reg_lambda
    config.xgboost.reg_alpha = config.scratch.reg_alpha
    # Library membandingkan gain tanpa faktor 1/2 (SOURCE_OF_TRUTH E-5), jadi gamma x 2.
    config.xgboost.gamma = 2 * config.scratch.gamma
    config.xgboost.max_bin = config.scratch.max_bin

    candles, report = load_candles(config, input_csv, config.data.development_end)
    features, _, labels, baseline_inputs = build_dataset(candles, config, config.features.groups)
    fold = make_validation_folds(config.data.history_start, [year])[0]

    comparison = {}
    probabilities = {}
    for model_name in ["xgboost", "xgboost_scratch"]:
        memory_before = peak_memory_mb()
        result = run_fold(model_name, fold, features, labels, baseline_inputs, config)
        evaluation_x = features.loc[result["_blocks"]["evaluation"]]
        probabilities[model_name] = apply_temperature(
            result["_model"].predict_proba(evaluation_x), result["_temperature"]
        )
        public = public_result(result)
        public["peak_memory_mb_after"] = peak_memory_mb()
        public["peak_memory_mb_before"] = memory_before
        comparison[model_name] = public

    library = probabilities["xgboost"]
    scratch = probabilities["xgboost_scratch"]
    comparison["agreement"] = {
        "argmax_agreement_rate": float((library.argmax(axis=1) == scratch.argmax(axis=1)).mean()),
        "mean_absolute_probability_difference": float(np.abs(library - scratch).mean()),
        "max_absolute_probability_difference": float(np.abs(library - scratch).max()),
        "note": "Peak memory adalah puncak proses (ru_maxrss), bukan memori per model.",
    }

    directory = create_run_directory(config, "compare-scratch")
    write_json(directory / "comparison.json", clean_for_json(comparison))
    write_run_metadata(directory, config, "compare-scratch", report, {"fold": fold.describe()})
    return directory


def command_final_test(config: TechnicalConfig, input_csv: str | None, model_name: str) -> Path:
    candles, report = load_candles(config, input_csv, config.data.test_end)
    features, feature_group, labels, baseline_inputs = build_dataset(
        candles, config, config.features.groups
    )
    fold = make_final_fold(
        config.data.history_start, config.data.development_end, config.data.test_end
    )

    # Semua fitting di run_fold memakai blok train/calibration/threshold (< development_end).
    result = run_fold(model_name, fold, features, labels, baseline_inputs, config)
    model = result["_model"]

    # Prediksi untuk SEMUA candle 2026 (termasuk yang labelnya belum lengkap).
    test_start = pd.Timestamp(config.data.development_end)
    test_end = pd.Timestamp(config.data.test_end)
    in_test = (labels["prediction_time"] >= test_start) & (labels["prediction_time"] < test_end)
    test_rows = labels.index[in_test]

    raw = model.predict_proba(features.loc[test_rows])
    calibrated = apply_temperature(raw, result["_temperature"])
    decisions, reasons = decide(calibrated, result["_thresholds"])

    predictions = pd.DataFrame(index=test_rows)
    predictions["prediction_time"] = labels.loc[test_rows, "prediction_time"]
    predictions["label_end_time"] = labels.loc[test_rows, "label_end_time"]
    predictions["close"] = candles.loc[test_rows, "close"]
    predictions["barrier"] = labels.loc[test_rows, "barrier"]
    predictions["probability_sell"] = calibrated[:, 0]
    predictions["probability_no_trade"] = calibrated[:, 1]
    predictions["probability_buy"] = calibrated[:, 2]
    predictions["signal"] = [CLASS_NAMES[value] for value in decisions]
    predictions["reason_code"] = reasons
    predictions["label_status"] = labels.loc[test_rows, "label_status"]
    actual = []
    for value in labels.loc[test_rows, "label"]:
        actual.append(CLASS_NAMES[int(value)] if pd.notna(value) else "")
    predictions["actual_label"] = actual

    evaluated = result["_blocks"]["evaluation"]
    period = {
        "first_decision": labels.loc[test_rows, "prediction_time"].min().isoformat(),
        "last_decision": labels.loc[test_rows, "prediction_time"].max().isoformat(),
        "evaluated_rows": len(evaluated),
        "last_evaluated_decision": labels.loc[evaluated, "prediction_time"].max().isoformat(),
        "unlabeled_rows": int((labels.loc[test_rows, "label"].isna()).sum()),
        "status_counts": labels.loc[test_rows, "label_status"].value_counts().to_dict(),
    }

    model_version = f"technical-{model_name}-{config_hash(config)}"
    last_row = test_rows[-1]
    latest_signal = build_signal(
        timestamp=labels.loc[last_row, "prediction_time"].isoformat(),
        symbol=config.data.symbol,
        model_version=model_version,
        horizon_bars=config.labels.horizon_bars,
        calibrated_probabilities=calibrated[-1],
        decision=int(decisions[-1]),
        reason=str(reasons[-1]),
        feature_row=features.loc[last_row].to_dict(),
        snapshot_features=SNAPSHOT_FEATURES,
    )

    directory = create_run_directory(config, "final-test")
    predictions.to_csv(directory / "predictions_2026.csv", index=False)
    write_json(directory / "metrics.json", clean_for_json(public_result(result)))
    write_json(directory / "test_period.json", clean_for_json(period))
    write_json(directory / "latest_signal.json", clean_for_json(latest_signal.to_dict()))
    write_run_metadata(
        directory,
        config,
        "final-test",
        report,
        {
            "model": model_name,
            "model_version": model_version,
            "features": list(features.columns),
            "peak_memory_mb": peak_memory_mb(),
            "warning": (
                "Hasil 2026 bersifat retrospective out-of-sample; "
                "lihat SOURCE_OF_TRUTH.md bagian A."
            ),
        },
    )
    print_summary({model_name: {"summary": summarize_folds([result])}})
    print(json.dumps(clean_for_json(period), indent=2))
    return directory


def print_summary(results: dict) -> None:
    for model_name, output in results.items():
        summary = output["summary"]
        line = [f"{model_name:16s}"]
        for key in [
            "macro_f1",
            "directional_macro_f1",
            "buy_precision",
            "sell_precision",
            "signal_coverage",
        ]:
            mean = summary[key]["mean"]
            text = "n/a" if mean is None else f"{mean:.4f}"
            line.append(f"{key}={text}")
        print("  ".join(line))


def main() -> None:
    parser = argparse.ArgumentParser(description="Eksperimen Technical Agent XAUUSD M5")
    parser.add_argument(
        "command",
        choices=["label-scan", "walk-forward", "ablation", "compare-scratch", "final-test"],
    )
    parser.add_argument("--config", default=None, help="Path TOML (default configs/technical.toml)")
    parser.add_argument("--input-csv", default=None, help="Ekspor market.processed_m5 (opsional)")
    parser.add_argument("--models", default="always_no_trade,persistence,logistic,xgboost")
    parser.add_argument("--model", default="xgboost", help="Model untuk final-test")
    parser.add_argument(
        "--year", type=int, default=2025, help="Fold validasi untuk compare-scratch"
    )
    parser.add_argument(
        "--confirm", action="store_true", help="Wajib untuk final-test (memakai data 2026)"
    )
    arguments = parser.parse_args()

    if arguments.config:
        config = load_config(arguments.config)
    else:
        config = load_config()

    if arguments.command == "label-scan":
        directory = command_label_scan(config, arguments.input_csv)
    elif arguments.command == "walk-forward":
        model_names = arguments.models.split(",")
        directory = command_walk_forward_and_save(config, arguments.input_csv, model_names)
    elif arguments.command == "ablation":
        directory = command_ablation(config, arguments.input_csv)
    elif arguments.command == "compare-scratch":
        directory = command_compare_scratch(config, arguments.input_csv, arguments.year)
    else:
        if not arguments.confirm:
            raise SystemExit(
                "final-test memakai data 2026. Jalankan hanya setelah desain dikunci "
                "berdasarkan walk-forward, lalu tambahkan --confirm."
            )
        directory = command_final_test(config, arguments.input_csv, arguments.model)

    print(f"Hasil tersimpan di: {directory}")


if __name__ == "__main__":
    main()
