"""Ringkas folder runs/<id>/ hasil walk-forward atau final-test menjadi tabel Markdown.

Semua angka dibaca dari file run (run.json, metrics.json, backtest/*/predictions.csv),
tidak ada angka yang diketik manual. Jalankan dari root repo dengan environment proyek:

    uv run python ringkas_runs.py runs/<id_walk_forward> [runs/<id_final_test>] > hasil.md
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))

from technical.backtest import read_candles, read_predictions, summarize_period  # noqa: E402
from technical.simulation import InstrumentSpec  # noqa: E402

BASELINES = ["always_no_trade", "persistence"]
DISPLAY_ORDER = [
    "always_no_trade",
    "persistence",
    "logistic",
    "xgboost",
    "xgboost_balanced",
    "random_forest",
    "ensemble",
    "xgboost_selected",
    "xgboost_scratch",
]


def number(value, digits=4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    return f"{value:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def percent(value, digits=2) -> str:
    if value is None:
        return "–"
    return number(100 * value, digits) + "%"


def mean_std(summary: dict, key: str, as_percent: bool = False) -> str:
    entry = summary.get(key) or {}
    if entry.get("mean") is None:
        return "–"
    if as_percent:
        return f"{percent(entry['mean'])} ± {percent(entry['std'])}"
    return f"{number(entry['mean'])} ± {number(entry['std'])}"


def ordered(models: dict) -> list[str]:
    names = [name for name in DISPLAY_ORDER if name in models]
    return names + [name for name in models if name not in names]


def load_run(directory: Path) -> tuple[dict, dict]:
    with open(directory / "run.json", encoding="utf-8") as file:
        run = json.load(file)
    with open(directory / "metrics.json", encoding="utf-8") as file:
        metrics = json.load(file)
    return run, metrics


def run_header(directory: Path, run: dict) -> list[str]:
    report = run.get("data_report", {})
    lines = [
        f"Sumber: `{directory.name}/run.json` (perintah {run.get('command')}, "
        f"config hash `{run.get('config_hash')}`, commit `{str(run.get('git_commit'))[:10]}`, "
        f"dibuat {run.get('created_utc', '')[:19]} UTC).",
        "",
        "Tabel 10.2 Data yang Dipakai Run",
        "<!-- widths: 45,55 -->",
        "| Atribut | Nilai |",
        "|---|---|",
        f"| Sumber data | `{report.get('source')}` |",
        f"| Candle masuk / valid / dipakai | {report.get('input_rows'):,} / {report.get('valid_rows'):,} / "
        f"{report.get('rows_used'):,} |".replace(",", "."),
        f"| Candle pertama – terakhir | {report.get('first_open_time', '')[:16]} – "
        f"{report.get('last_close_time', '')[:16]} UTC |",
        f"| Gap waktu | {report.get('gap_count')} |",
        f"| Jumlah fitur | {len(run.get('features', []))} |",
        f"| Versi paket | {', '.join(f'{k} {v}' for k, v in run.get('packages', {}).items())} |",
        "",
    ]
    return lines


def summary_table(metrics: dict) -> list[str]:
    lines = [
        "Tabel 10.3 Ringkasan Walk-Forward per Model (rata-rata ± simpangan baku antarfold)",
        "<!-- widths: 20,16,16,16,16,16 -->",
        "| Model | Macro F1 | Directional F1 | Precision BUY | Precision SELL | Coverage sinyal |",
        "|---|---|---|---|---|---|",
    ]
    for name in ordered(metrics):
        summary = metrics[name]["summary"]
        lines.append(
            f"| {name} | {mean_std(summary, 'macro_f1')} | {mean_std(summary, 'directional_macro_f1')} | "
            f"{mean_std(summary, 'buy_precision', True)} | {mean_std(summary, 'sell_precision', True)} | "
            f"{mean_std(summary, 'signal_coverage', True)} |"
        )
    lines.append("")
    return lines


def fold_table(metrics: dict, name: str, table_number: str) -> list[str]:
    lines = [
        f"Tabel {table_number} Hasil per Fold Model {name}",
        "<!-- widths: 16,11,17,17,11,16,12 -->",
        "| Fold | Macro F1 | BUY: precision (sinyal) | SELL: precision (sinyal) | Coverage | Threshold BUY / SELL | Temperatur |",
        "|---|---|---|---|---|---|---|",
    ]
    for fold in metrics[name]["folds"]:
        evaluation = fold["evaluation"]
        classes = evaluation["per_class"]
        thresholds = fold.get("thresholds", {})

        def threshold_text(direction):
            rule = thresholds.get(direction)
            if rule is None:
                return "–"
            if not rule["enabled"]:
                return "nonaktif"
            return number(rule["threshold"], 2)

        temperature = fold.get("calibration", {}).get("temperature")
        lines.append(
            f"| {fold['fold']['name']} | {number(evaluation['macro_f1'])} | "
            f"{percent(classes['BUY']['precision'])} ({classes['BUY']['predicted']}) | "
            f"{percent(classes['SELL']['precision'])} ({classes['SELL']['predicted']}) | "
            f"{percent(evaluation['signal_coverage'])} | {threshold_text('BUY')} / {threshold_text('SELL')} | "
            f"{number(temperature, 3)} |"
        )
    lines.append("")
    return lines


def lift_table(metrics: dict, names: list[str], table_number: str) -> list[str]:
    """Precision dibagi prevalensi kelas pada blok EVALUATION (lift > 1 = lebih baik dari acak)."""
    lines = [
        f"Tabel {table_number} Lift Precision terhadap Prevalensi Kelas (blok EVALUATION)",
        "<!-- widths: 22,13,13,13,13,13,13 -->",
        "| Model / fold | Prevalensi BUY | Precision BUY | Lift BUY | Prevalensi SELL | Precision SELL | Lift SELL |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in names:
        for fold in metrics[name]["folds"]:
            evaluation = fold["evaluation"]
            total = evaluation["samples"]
            row = [f"{name} / {fold['fold']['name'].replace('validation_', '')}"]
            for direction in ["BUY", "SELL"]:
                prevalence = evaluation["actual_distribution"][direction] / total if total else None
                precision = evaluation["per_class"][direction]["precision"]
                lift = precision / prevalence if precision is not None and prevalence else None
                row += [percent(prevalence), percent(precision), number(lift, 2)]
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def label_table(metrics: dict, name: str, table_number: str) -> list[str]:
    lines = [
        f"Tabel {table_number} Distribusi Label per Fold (TRAIN dan EVALUATION)",
        "<!-- widths: 18,14,14,14,14,13,13 -->",
        "| Fold | TRAIN SELL | TRAIN NO_TRADE | TRAIN BUY | EVAL SELL | EVAL NO_TRADE | EVAL BUY |",
        "|---|---|---|---|---|---|---|",
    ]
    for fold in metrics[name]["folds"]:
        train = fold["label_distribution"]["train"]
        evaluation = fold["label_distribution"]["evaluation"]
        row = [fold["fold"]["name"]]
        for block in [train, evaluation]:
            for label in ["SELL", "NO_TRADE", "BUY"]:
                row.append(percent(block[label]["share"], 1))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def confusion_total(metrics: dict, name: str, table_number: str) -> list[str]:
    total = np.zeros((3, 3), dtype=int)
    for fold in metrics[name]["folds"]:
        total += np.array(fold["evaluation"]["confusion_matrix"]["values"])
    labels = ["SELL", "NO_TRADE", "BUY"]
    lines = [
        f"Tabel {table_number} Confusion Matrix Gabungan Semua Fold Model {name} (baris aktual, kolom prediksi)",
        "| Aktual \\ Prediksi | SELL | NO_TRADE | BUY |",
        "|---|---|---|---|",
    ]
    for index, label in enumerate(labels):
        lines.append(f"| {label} | " + " | ".join(f"{value:,}".replace(",", ".") for value in total[index]) + " |")
    lines.append("")
    return lines


def ensemble_table(metrics: dict, table_number: str, diversity_number: str) -> list[str]:
    if "ensemble" not in metrics:
        return []
    folds = metrics["ensemble"]["folds"]
    members = list(folds[0]["ensemble"]["members"])
    header = " | ".join(f"Bobot {member}" for member in members)
    lines = [
        f"Tabel {table_number} Bobot dan Log Loss Ensemble per Fold (blok CALIBRATION)",
        "| Fold | " + header + " | Log loss ensemble | Anggota terbaik (log loss) |",
        "|" + "---|" * (len(members) + 3),
    ]
    for fold in folds:
        report = fold["ensemble"]
        weights = " | ".join(number(report["members"][member]["weight"], 2) for member in members)
        lines.append(
            f"| {fold['fold']['name']} | {weights} | {number(report['ensemble_log_loss'])} | "
            f"{report['best_single_member']} ({number(report['best_single_log_loss'])}) |"
        )
    lines.append("")
    pairs = list(folds[0]["ensemble"]["diversity"])
    lines += [
        f"Tabel {diversity_number} Keragaman Kesalahan Antaranggota (rata-rata semua fold)",
        "| Pasangan anggota | Disagreement | Korelasi kesalahan |",
        "|---|---|---|",
    ]
    for pair in pairs:
        disagreement = [fold["ensemble"]["diversity"][pair]["disagreement"] for fold in folds]
        correlation = [
            fold["ensemble"]["diversity"][pair]["error_correlation"]
            for fold in folds
            if fold["ensemble"]["diversity"][pair]["error_correlation"] is not None
        ]
        lines.append(
            f"| {pair.replace('|', ' – ')} | {percent(float(np.mean(disagreement)))} | "
            f"{number(float(np.mean(correlation)) if correlation else None, 3)} |"
        )
    lines.append("")
    return lines


def selection_tables(metrics: dict, table_number: str, stability_number: str) -> list[str]:
    if "xgboost_selected" not in metrics:
        return []
    result = metrics["xgboost_selected"]
    lines = [
        f"Tabel {table_number} Hasil Feature Selection per Fold",
        "| Fold | Fitur awal | Lolos filter | Terpilih | Dugaan bocor | Korelasi peringkat gain vs permutation |",
        "|---|---|---|---|---|---|",
    ]
    for fold in result["folds"]:
        report = fold["feature_selection"]
        lines.append(
            f"| {fold['fold']['name']} | {report['total_features']} | {report['after_filter']} | "
            f"{report['selected_count']} | {len(report['leakage_flags'])} | "
            f"{number(report['gain_permutation_rank_correlation'], 3)} |"
        )
    lines += [
        "",
        f"Tabel {stability_number} Fitur Paling Stabil (terpilih di paling banyak fold)",
        "| Fitur | Kelompok | Terpilih (fold) | Rata-rata permutation importance |",
        "|---|---|---|---|",
    ]
    for row in result.get("feature_selection_stability", [])[:15]:
        lines.append(
            f"| `{row['feature']}` | {row['group']} | {row['selected_folds']}/{row['total_folds']} | "
            f"{number(row['mean_permutation_importance'], 5)} |"
        )
    lines.append("")
    return lines


def backtest_tables(directory: Path, table_number: str, spec: InstrumentSpec) -> list[str]:
    backtest = directory / "backtest"
    if not backtest.is_dir():
        return []
    with open(directory / "run.json", encoding="utf-8") as file:
        horizon = int(json.load(file)["config"]["labels"]["horizon_bars"])
    lines = [
        f"Tabel {table_number} Simulasi Transaksi per Periode (lot {number(spec.lot, 2)}, setelah spread)",
        "<!-- widths: 22,9,9,9,9,10,10,11,11 -->",
        "| Model / periode | Transaksi | Win rate | TP | SL | Waktu habis | Profit factor | Net P&L USD | Biaya spread USD |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model_directory in sorted(path for path in backtest.iterdir() if path.name != "candles"):
        for period_directory in sorted(model_directory.iterdir()):
            predictions_path = period_directory / "predictions.csv"
            if not predictions_path.exists():
                continue
            candles_path = backtest / "candles" / f"{period_directory.name}.csv"
            candles = read_candles(candles_path) if candles_path.exists() else None
            summary = summarize_period(read_predictions(predictions_path), horizon, spec, candles)
            simulation = summary["simulation"]
            period_name = period_directory.name.replace("validation_", "")
            if not simulation.get("trades"):
                lines.append(f"| {model_directory.name} / {period_name} | 0 | – | – | – | – | – | – | – |")
                continue
            shares = simulation["exit_reason_share"]
            lines.append(
                f"| {model_directory.name} / {period_name} | "
                f"{simulation['trades']:,} | ".replace(",", ".")
                + f"{percent(simulation['win_rate'], 1)} | {percent(shares.get('TP', 0), 1)} | "
                f"{percent(shares.get('SL', 0) + shares.get('BOTH_SAME_CANDLE', 0), 1)} | "
                f"{percent(shares.get('TIME', 0), 1)} | {number(simulation['profit_factor'], 2)} | "
                f"{number(simulation['net_pnl'], 2)} | {number(simulation['spread']['total_cost_usd'], 2)} |"
            )
    lines.append("")
    return lines


def main() -> None:
    directories = [Path(argument) for argument in sys.argv[1:]]
    spec = InstrumentSpec()
    output = []
    for directory in directories:
        run, metrics = load_run(directory)
        output += [f"<!-- ringkasan otomatis dari {directory.name} -->"]
        output += run_header(directory, run)
        if run.get("command") == "walk-forward":
            output += summary_table(metrics)
            main_models = [name for name in ["xgboost", "ensemble"] if name in metrics]
            for position, name in enumerate(main_models):
                output += fold_table(metrics, name, f"10.{4 + position}")
            output += lift_table(metrics, main_models, "10.6")
            first = ordered(metrics)[0]
            output += label_table(metrics, first, "10.7")
            if "xgboost" in metrics:
                output += confusion_total(metrics, "xgboost", "10.8")
            output += ensemble_table(metrics, "10.9", "10.10")
            output += selection_tables(metrics, "10.11", "10.12")
            output += backtest_tables(directory, "10.13", spec)
        else:
            # final-test: metrics.json adalah satu fold.
            fake = {run.get("model", "model"): {"folds": [metrics], "summary": {}}}
            output += fold_table(fake, run.get("model", "model"), "10.14")
            output += backtest_tables(directory, "10.15", spec)
    print("\n".join(output))


if __name__ == "__main__":
    main()
