# ruff: noqa: E501  (string tabel Markdown sengaja ditulis utuh per baris)
"""Ringkas folder runs/<id>/ menjadi tabel Markdown untuk LAPORAN UTS MILESTONE 1.

Semua angka dibaca dari file run (run.json, metrics.json, label_scan.csv,
test_period.json, backtest/*/predictions.csv); tidak ada angka yang diketik manual.
Jalankan dari root repo branch yang memuat technical/simulation.py terbaru:

    uv run python <dir>/ringkas_runs.py --out <dir> runs/<id_1> runs/<id_2> ...

Menulis tiga berkas ke --out:
    hasil_label_scan.md     tabel bab 3 (distribusi label per horizon dan k)
    hasil_walk_forward.md   tabel bab 10 (gabungan semua run walk-forward)
    hasil_final_test.md     tabel bab 10 (uji 2026)
Model yang muncul di lebih dari satu run walk-forward hanya diambil sekali, setelah
dicek apakah ringkasannya identik.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))

from technical.backtest import read_candles, read_predictions, summarize_period  # noqa: E402
from technical.simulation import InstrumentSpec  # noqa: E402

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


# ---------------------------------------------------------------------------
# Format angka gaya Indonesia (koma desimal, titik ribuan)
# ---------------------------------------------------------------------------


def number(value, digits=4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    text = f"{value:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def integer(value) -> str:
    return f"{int(value):,}".replace(",", ".")


def percent(value, digits=2) -> str:
    if value is None:
        return "–"
    return number(100 * value, digits) + "%"


def mean_std(summary: dict, key: str, as_percent: bool = False) -> str:
    entry = summary.get(key) or {}
    if entry.get("mean") is None:
        return "–"
    folds = entry.get("folds_with_value")
    if as_percent:
        text = f"{percent(entry['mean'])} ± {percent(entry['std'])}"
    else:
        text = f"{number(entry['mean'])} ± {number(entry['std'])}"
    if folds is not None and folds < 5:
        text += f" ({folds} fold)"
    return text


class Numbering:
    """Nomor tabel berurutan per bab, misalnya 10.2, 10.3, ..."""

    def __init__(self, chapter: int, start: int):
        self.chapter = chapter
        self.next_number = start

    def take(self) -> str:
        label = f"{self.chapter}.{self.next_number}"
        self.next_number += 1
        return label


def ordered(names) -> list[str]:
    names = list(names)
    result = [name for name in DISPLAY_ORDER if name in names]
    return result + [name for name in names if name not in result]


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------------------------
# Label-scan (bab 3)
# ---------------------------------------------------------------------------


def label_scan_tables(directory: Path, numbering: Numbering) -> list[str]:
    with open(directory / "label_scan.csv", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    lines = [
        f"Tabel {numbering.take()} Distribusi Label 2016–2025 per Horizon dan Pengali Barrier (label-scan)",
        "<!-- widths: 16,12,8,16,16,16,16 -->",
        "| Metode | Horizon (candle) | k | Baris berlabel | SELL | NO_TRADE | BUY |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        method = "triple-barrier" if row["method"] == "triple_barrier" else "fixed horizon"
        chosen = (
            row["method"] == "triple_barrier"
            and row["horizon_bars"] == "6"
            and row["barrier_atr_multiple"] == "1.0"
        )
        marker = "**" if chosen else ""
        lines.append(
            f"| {marker}{method}{marker} | {row['horizon_bars']} | "
            f"{number(float(row['barrier_atr_multiple']), 1)} | {integer(row['labeled_rows'])} | "
            f"{percent(float(row['sell_share']), 1)} | {percent(float(row['no_trade_share']), 1)} | "
            f"{percent(float(row['buy_share']), 1)} |"
        )
    lines += [
        "",
        f"Sumber: `{directory.name}/label_scan.csv`. Baris tebal adalah konfigurasi yang dipakai "
        "(triple-barrier, 6 candle, k = 1,0).",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Walk-forward (bab 10)
# ---------------------------------------------------------------------------


def merge_walk_forward(directories: list[Path]) -> tuple[dict, list[str]]:
    """Gabungkan model dari beberapa run; kembalikan (metrics, catatan reproduksi)."""
    merged = {}
    notes = []
    hashes = set()
    for directory in directories:
        hashes.add(load_json(directory / "run.json")["config_hash"])
        metrics = load_json(directory / "metrics.json")
        for name, result in metrics.items():
            if name not in merged:
                merged[name] = result
                continue
            same = json.dumps(merged[name]["summary"], sort_keys=True) == json.dumps(
                result["summary"], sort_keys=True
            )
            status = (
                "identik (hasil dapat direproduksi)" if same else "BERBEDA, periksa konfigurasi"
            )
            notes.append(f"model {name} juga ada di `{directory.name}` dan ringkasannya {status}.")
    if len(hashes) > 1:
        notes.append(f"config hash berbeda antarrun: {', '.join(sorted(hashes))}.")
    return merged, notes


def data_table(directories: list[Path], numbering: Numbering) -> list[str]:
    run = load_json(directories[0] / "run.json")
    report = run.get("data_report", {})
    run_list = "<br>".join(f"`{directory.name}`" for directory in directories)
    packages = ", ".join(f"{key} {value}" for key, value in run.get("packages", {}).items())
    return [
        f"Tabel {numbering.take()} Data dan Run Walk-Forward yang Dipakai",
        "<!-- widths: 34,66 -->",
        "| Atribut | Nilai |",
        "|---|---|",
        f"| Run | {run_list} |",
        f"| Sumber data | `{report.get('source')}` |",
        f"| Candle masuk / valid / dipakai | {integer(report.get('input_rows'))} / "
        f"{integer(report.get('valid_rows'))} / {integer(report.get('rows_used'))} |",
        f"| Rentang | {report.get('first_open_time', '')[:16]} – {report.get('last_close_time', '')[:16]} UTC |",
        f"| Transisi gap | {integer(report.get('gap_count'))} |",
        f"| Fitur | {len(run.get('features', []))} fitur, 7 kelompok |",
        f"| Commit kode / config hash | `{str(run.get('git_commit'))[:7]}` / `{run.get('config_hash')}` |",
        f"| Versi paket | {packages} |",
        "",
    ]


def summary_table(metrics: dict, numbering: Numbering) -> list[str]:
    lines = [
        f"Tabel {numbering.take()} Ringkasan Walk-Forward 2021–2025 per Model "
        "(rata-rata ± simpangan baku lima fold)",
        "<!-- widths: 22,14,14,18,18,14 -->",
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
    lines += [
        "",
        'Keterangan: "(n fold)" berarti precision hanya terdefinisi pada n fold karena pada fold lain '
        "arah tersebut dinonaktifkan atau tidak menghasilkan sinyal.",
        "",
    ]
    return lines


def per_fold_matrix(metrics, key, title, numbering, as_percent=False) -> list[str]:
    first = next(iter(metrics.values()))
    folds = [fold["fold"]["name"].replace("validation_", "") for fold in first["folds"]]
    lines = [
        f"Tabel {numbering.take()} {title}",
        "| Model | " + " | ".join(folds) + " |",
        "|" + "---|" * (len(folds) + 1),
    ]
    for name in ordered(metrics):
        values = []
        for fold in metrics[name]["folds"]:
            value = fold["evaluation"][key]
            values.append(percent(value) if as_percent else number(value))
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines.append("")
    return lines


def threshold_text(thresholds: dict, direction: str) -> str:
    rule = thresholds.get(direction)
    if rule is None:
        return "–"
    if not rule["enabled"]:
        return "nonaktif"
    return number(rule["threshold"], 2)


def fold_table(metrics: dict, name: str, numbering: Numbering) -> list[str]:
    lines = [
        f"Tabel {numbering.take()} Hasil per Tahun Evaluasi Model {name}",
        "<!-- widths: 9,10,18,18,11,14,10,10 -->",
        "| Tahun | Macro F1 | BUY: precision (sinyal) | SELL: precision (sinyal) | Coverage | "
        "Threshold BUY / SELL | Temperatur | Pohon |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for fold in metrics[name]["folds"]:
        evaluation = fold["evaluation"]
        classes = evaluation["per_class"]
        thresholds = fold.get("thresholds", {})

        trees = fold.get("best_iteration")
        lines.append(
            f"| {fold['fold']['name'].replace('validation_', '')} | {number(evaluation['macro_f1'])} | "
            f"{percent(classes['BUY']['precision'])} ({integer(classes['BUY']['predicted'])}) | "
            f"{percent(classes['SELL']['precision'])} ({integer(classes['SELL']['predicted'])}) | "
            f"{percent(evaluation['signal_coverage'])} | {threshold_text(thresholds, 'BUY')} / {threshold_text(thresholds, 'SELL')} | "
            f"{number(fold.get('calibration', {}).get('temperature'), 3)} | "
            f"{'–' if trees is None else integer(trees + 1)} |"
        )
    lines += [
        "",
        "Pohon = best_iteration + 1, yaitu jumlah ronde yang dipakai setelah early stopping (maksimal 300).",
        "",
    ]
    return lines


def lift_table(metrics: dict, names: list[str], numbering: Numbering) -> list[str]:
    lines = [
        f"Tabel {numbering.take()} Lift Precision terhadap Prevalensi Kelas pada Blok EVALUATION",
        "<!-- widths: 22,13,13,13,13,13,13 -->",
        "| Model / tahun | Prevalensi BUY | Precision BUY | Lift BUY | Prevalensi SELL | Precision SELL | Lift SELL |",
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
                row += [percent(prevalence, 1), percent(precision, 1), number(lift, 2)]
            lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "Lift = precision ÷ prevalensi. Lift 1 setara tebakan acak; kriteria keberhasilan proyek lift ≥ 1,2.",
        "",
    ]
    return lines


def label_table(metrics: dict, name: str, numbering: Numbering) -> list[str]:
    lines = [
        f"Tabel {numbering.take()} Distribusi Label per Fold (blok TRAIN dan EVALUATION)",
        "<!-- widths: 10,15,15,15,15,15,15 -->",
        "| Tahun | TRAIN SELL | TRAIN NO_TRADE | TRAIN BUY | EVAL SELL | EVAL NO_TRADE | EVAL BUY |",
        "|---|---|---|---|---|---|---|",
    ]
    for fold in metrics[name]["folds"]:
        row = [fold["fold"]["name"].replace("validation_", "")]
        for block in [
            fold["label_distribution"]["train"],
            fold["label_distribution"]["evaluation"],
        ]:
            for label in ["SELL", "NO_TRADE", "BUY"]:
                row.append(percent(block[label]["share"], 1))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def confusion_total(metrics: dict, name: str, numbering: Numbering) -> list[str]:
    total = np.zeros((3, 3), dtype=int)
    opposite = 0
    for fold in metrics[name]["folds"]:
        total += np.array(fold["evaluation"]["confusion_matrix"]["values"])
        errors = fold["evaluation"]["opposite_direction_errors"]
        opposite += errors["BUY_when_SELL"] + errors["SELL_when_BUY"]
    labels = ["SELL", "NO_TRADE", "BUY"]
    lines = [
        f"Tabel {numbering.take()} Confusion Matrix Gabungan Lima Fold Model {name} "
        "(baris aktual, kolom prediksi)",
        "| Aktual \\ Prediksi | SELL | NO_TRADE | BUY |",
        "|---|---|---|---|",
    ]
    for index, label in enumerate(labels):
        lines.append(f"| {label} | " + " | ".join(integer(value) for value in total[index]) + " |")
    signals = int(total[:, 0].sum() + total[:, 2].sum())
    share = opposite / signals if signals else None
    lines += [
        "",
        f"Total sinyal BUY/SELL {integer(signals)}; kesalahan arah berlawanan (BUY saat label SELL atau "
        f"sebaliknya) {integer(opposite)}, atau {percent(share, 1)} dari sinyal.",
        "",
    ]
    return lines


def ensemble_tables(metrics: dict, numbering: Numbering) -> list[str]:
    if "ensemble" not in metrics:
        return []
    folds = metrics["ensemble"]["folds"]
    members = list(folds[0]["ensemble"]["members"])
    lines = [
        f"Tabel {numbering.take()} Bobot dan Log Loss Ensemble per Fold (dipilih pada blok CALIBRATION)",
        "| Tahun | "
        + " | ".join(f"Bobot {member}" for member in members)
        + " | Log loss ensemble | Anggota terbaik (log loss) |",
        "|" + "---|" * (len(members) + 3),
    ]
    for fold in folds:
        report = fold["ensemble"]
        weights = " | ".join(number(report["members"][member]["weight"], 2) for member in members)
        lines.append(
            f"| {fold['fold']['name'].replace('validation_', '')} | {weights} | "
            f"{number(report['ensemble_log_loss'])} | {report['best_single_member']} "
            f"({number(report['best_single_log_loss'])}) |"
        )
    lines += [
        "",
        f"Tabel {numbering.take()} Keragaman Kesalahan Antaranggota Ensemble "
        "(rata-rata lima fold, blok CALIBRATION)",
        "| Pasangan anggota | Disagreement | Korelasi kesalahan |",
        "|---|---|---|",
    ]
    for pair in folds[0]["ensemble"]["diversity"]:
        disagreement = [fold["ensemble"]["diversity"][pair]["disagreement"] for fold in folds]
        correlation = []
        for fold in folds:
            value = fold["ensemble"]["diversity"][pair]["error_correlation"]
            if value is not None:
                correlation.append(value)
        mean_correlation = float(np.mean(correlation)) if correlation else None
        lines.append(
            f"| {pair.replace('|', ' – ')} | {percent(float(np.mean(disagreement)), 1)} | "
            f"{number(mean_correlation, 3)} |"
        )
    lines.append("")
    return lines


def selection_tables(metrics: dict, numbering: Numbering) -> list[str]:
    if "xgboost_selected" not in metrics:
        return []
    result = metrics["xgboost_selected"]
    lines = [
        f"Tabel {numbering.take()} Hasil Feature Selection per Fold",
        "| Tahun | Fitur awal | Lolos filter | Terpilih | Dugaan bocor | Korelasi gain vs permutation |",
        "|---|---|---|---|---|---|",
    ]
    for fold in result["folds"]:
        report = fold["feature_selection"]
        lines.append(
            f"| {fold['fold']['name'].replace('validation_', '')} | {report['total_features']} | "
            f"{report['after_filter']} | {report['selected_count']} | {len(report['leakage_flags'])} | "
            f"{number(report['gain_permutation_rank_correlation'], 3)} |"
        )
    lines += [
        "",
        f"Tabel {numbering.take()} Fitur Paling Stabil (terpilih di paling banyak fold)",
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


# ---------------------------------------------------------------------------
# Backtest dan simulasi transaksi
# ---------------------------------------------------------------------------


def period_summaries(directories: list[Path], spec: InstrumentSpec) -> dict:
    """{(model, periode): ringkasan summarize_period} dari semua run, tanpa duplikat."""
    found = {}
    for directory in directories:
        backtest = directory / "backtest"
        if not backtest.is_dir():
            continue
        horizon = int(load_json(directory / "run.json")["config"]["labels"]["horizon_bars"])
        for model_directory in sorted(
            path for path in backtest.iterdir() if path.name != "candles"
        ):
            for period_directory in sorted(model_directory.iterdir()):
                key = (model_directory.name, period_directory.name)
                predictions_path = period_directory / "predictions.csv"
                if key in found or not predictions_path.exists():
                    continue
                candles_path = backtest / "candles" / f"{period_directory.name}.csv"
                candles = read_candles(candles_path) if candles_path.exists() else None
                predictions = read_predictions(predictions_path)
                found[key] = summarize_period(predictions, horizon, spec, candles)
    return found


def aggregate(summaries: list[dict]) -> dict:
    trades = 0
    wins = 0.0
    gross_profit = gross_loss = net = spread_cost = gross_before = 0.0
    exits = {}
    spread_price = []
    spread_barrier = []
    for summary in summaries:
        simulation = summary["simulation"]
        count = simulation.get("trades", 0)
        if not count:
            continue
        trades += count
        wins += simulation["win_rate"] * count
        gross_profit += simulation["gross_profit"]
        gross_loss += simulation["gross_loss"]
        net += simulation["net_pnl"]
        spread = simulation["spread"]
        spread_cost += spread["total_cost_usd"]
        gross_before += spread["gross_pnl_before_spread"]
        spread_price.append((spread["average_to_price"], count))
        spread_barrier.append((spread["average_to_barrier"], count))
        for reason, value in simulation["exit_reasons"].items():
            exits[reason] = exits.get(reason, 0) + value
    if trades == 0:
        return {"trades": 0}

    def weighted(pairs):
        return sum(value * count for value, count in pairs) / sum(count for _, count in pairs)

    return {
        "trades": trades,
        "win_rate": wins / trades,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        "net_pnl": net,
        "gross_before": gross_before,
        "spread_cost": spread_cost,
        "spread_to_price": weighted(spread_price),
        "spread_to_barrier": weighted(spread_barrier),
        "tp": exits.get("TP", 0) / trades,
        "sl": (exits.get("SL", 0) + exits.get("BOTH_SAME_CANDLE", 0)) / trades,
        "time": exits.get("TIME", 0) / trades,
    }


def simulation_row(label: str, result: dict) -> str:
    if not result.get("trades"):
        return f"| {label} | 0 | – | – | – | – | – | – |"
    return (
        f"| {label} | {integer(result['trades'])} | {percent(result['win_rate'], 1)} | "
        f"{percent(result['tp'], 1)} | {percent(result['sl'], 1)} | {percent(result['time'], 1)} | "
        f"{number(result['profit_factor'], 2)} | {number(result['net_pnl'], 2)} |"
    )


SIMULATION_HEADER = [
    "<!-- widths: 22,11,10,9,9,11,11,17 -->",
    "| Model / periode | Transaksi | Win rate | TP | SL | Waktu habis | Profit factor | Net P&L USD |",
    "|---|---|---|---|---|---|---|---|",
]


def simulation_tables(summaries: dict, spec: InstrumentSpec, numbering: Numbering) -> list[str]:
    if not summaries:
        return []
    models = ordered({model for model, period in summaries if period.startswith("validation_")})
    lines = [
        f"Tabel {numbering.take()} Simulasi Transaksi Gabungan 2021–2025 per Model "
        f"(lot {number(spec.lot, 2)}, setelah spread)",
        *SIMULATION_HEADER,
    ]
    totals = {}
    for model in models:
        periods = []
        for (name, period), summary in sorted(summaries.items()):
            if name == model and period.startswith("validation_"):
                periods.append(summary)
        totals[model] = aggregate(periods)
        lines.append(simulation_row(model, totals[model]))
    lines += [
        "",
        f"Tabel {numbering.take()} Laporan Spread Transaksi 2021–2025 per Model",
        "<!-- widths: 20,13,16,17,15,19 -->",
        "| Model | Transaksi | Spread / harga buka | Spread / jarak TP | Biaya spread USD | P&L kotor → bersih USD |",
        "|---|---|---|---|---|---|",
    ]
    for model in models:
        result = totals[model]
        if not result.get("trades"):
            lines.append(f"| {model} | 0 | – | – | – | – |")
            continue
        lines.append(
            f"| {model} | {integer(result['trades'])} | {percent(result['spread_to_price'], 4)} | "
            f"{percent(result['spread_to_barrier'], 1)} | {number(result['spread_cost'], 2)} | "
            f"{number(result['gross_before'], 2)} → {number(result['net_pnl'], 2)} |"
        )
    if "xgboost" in models:
        lines += [
            "",
            f"Tabel {numbering.take()} Simulasi Transaksi XGBoost per Tahun",
            *SIMULATION_HEADER,
        ]
        for (model, period), summary in sorted(summaries.items()):
            if model == "xgboost" and period.startswith("validation_"):
                lines.append(
                    simulation_row(period.replace("validation_", ""), aggregate([summary]))
                )
    lines += [
        "",
        "Simulasi tingkat candle: buka di close candle keputusan, tutup pada TP/SL atau close candle ke-6, "
        "spread tutup = spread buka, posisi boleh tumpang tindih, tanpa slippage/komisi/swap. Angka mengukur "
        "kualitas sinyal setelah biaya, bukan proyeksi saldo akun.",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Final test 2026
# ---------------------------------------------------------------------------


def final_test_tables(directory: Path, spec: InstrumentSpec, numbering: Numbering) -> list[str]:
    run = load_json(directory / "run.json")
    metrics = load_json(directory / "metrics.json")
    period = load_json(directory / "test_period.json")
    evaluation = metrics["evaluation"]
    classes = evaluation["per_class"]
    thresholds = metrics["thresholds"]
    total = evaluation["samples"]
    model = run.get("model")
    blocks = metrics["block_sizes"]
    calibration = metrics["calibration"]
    probability = evaluation["probability"]

    def rule_text(direction):
        rule = thresholds[direction]
        chosen = rule["chosen_on_threshold_block"]
        if not rule["enabled"]:
            return (
                f"nonaktif (prevalensi blok THRESHOLD {percent(rule['prevalence'], 1)}; syarat precision "
                f"{percent(rule['required_precision'], 1)} tidak tercapai)"
            )
        return (
            f"{number(rule['threshold'], 2)} ({chosen['signals']} sinyal, precision "
            f"{percent(chosen['precision'], 1)} pada Okt–Des 2025; syarat {percent(rule['required_precision'], 1)})"
        )

    def precision_text(direction):
        entry = classes[direction]
        if entry["precision"] is None:
            return "– (tidak ada sinyal)"
        low, high = entry["precision_wilson95"]
        return f"{percent(entry['precision'])} (Wilson 95%: {percent(low, 1)}–{percent(high, 1)})"

    counts = period["status_counts"]
    lines = [
        f"Sumber: `{directory.name}` (model {model}, dibuat {run['created_utc'][:16]} UTC, commit "
        f"`{run['git_commit'][:7]}`, config hash `{run['config_hash']}`).",
        "",
        f"Tabel {numbering.take()} Hasil Uji Out-of-Sample Retrospektif 2026 Model {model}",
        "<!-- widths: 36,64 -->",
        "| Atribut | Nilai |",
        "|---|---|",
        f"| Periode keputusan | {period['first_decision'][:16]} – {period['last_decision'][:16]} UTC |",
        f"| Candle dinilai (label lengkap) | {integer(period['evaluated_rows'])} (GAP "
        f"{integer(counts.get('GAP', 0))}, PENDING {integer(counts.get('PENDING', 0))}) |",
        f"| Candle TRAIN / CALIBRATION / THRESHOLD | {integer(blocks['train'])} / "
        f"{integer(blocks['calibration'])} / {integer(blocks['threshold'])} |",
        f"| Pohon setelah early stopping | {integer(metrics['best_iteration'] + 1)} dari maksimal 300 |",
        f"| Temperatur kalibrasi | {number(calibration['temperature'], 3)} (log loss "
        f"{number(calibration['log_loss_before'])} → {number(calibration['log_loss_after'])}) |",
        f"| Threshold BUY | {rule_text('BUY')} |",
        f"| Threshold SELL | {rule_text('SELL')} |",
        f"| Sinyal BUY / SELL | {integer(classes['BUY']['predicted'])} / {integer(classes['SELL']['predicted'])} "
        f"(coverage {percent(evaluation['signal_coverage'])}) |",
        f"| Precision BUY | {precision_text('BUY')} |",
        f"| Precision SELL | {precision_text('SELL')} |",
        f"| Prevalensi BUY / SELL 2026 | {percent(evaluation['actual_distribution']['BUY'] / total, 1)} / "
        f"{percent(evaluation['actual_distribution']['SELL'] / total, 1)} |",
        f"| Macro F1 / directional F1 / accuracy | {number(evaluation['macro_f1'])} / "
        f"{number(evaluation['directional_macro_f1'])} / {percent(evaluation['accuracy'])} |",
        f"| Log loss / Brier / ECE | {number(probability['log_loss'])} / "
        f"{number(probability['brier_multiclass'])} / {number(probability['ece_top_label'])} |",
        "",
    ]
    summaries = period_summaries([directory], spec)
    if summaries:
        lines += [
            f"Tabel {numbering.take()} Simulasi Transaksi 2026 (lot {number(spec.lot, 2)}, setelah spread)",
            *SIMULATION_HEADER,
        ]
        for (name, _), summary in sorted(summaries.items()):
            lines.append(simulation_row(f"{name} / 2026", aggregate([summary])))
        lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=int, default=2, help="nomor tabel pertama bab 10")
    parser.add_argument("runs", nargs="+")
    arguments = parser.parse_args()
    out = Path(arguments.out)
    spec = InstrumentSpec()

    groups = {"walk-forward": [], "final-test": [], "label-scan": []}
    for path in arguments.runs:
        directory = Path(path)
        command = load_json(directory / "run.json")["command"]
        groups.setdefault(command, []).append(directory)

    if groups["label-scan"]:
        lines = label_scan_tables(groups["label-scan"][-1], Numbering(3, 2))
        (out / "hasil_label_scan.md").write_text("\n".join(lines), encoding="utf-8")

    numbering = Numbering(10, arguments.start)
    if groups["walk-forward"]:
        directories = groups["walk-forward"]
        metrics, notes = merge_walk_forward(directories)
        lines = data_table(directories, numbering)
        lines += summary_table(metrics, numbering)
        for note in notes:
            lines += [f"Catatan reproduksi: {note}", ""]
        lines += per_fold_matrix(
            metrics, "directional_macro_f1", "Directional F1 per Tahun Evaluasi", numbering
        )
        lines += per_fold_matrix(
            metrics, "signal_coverage", "Coverage Sinyal per Tahun Evaluasi", numbering, True
        )
        for name in ["xgboost", "ensemble"]:
            if name in metrics:
                lines += fold_table(metrics, name, numbering)
        lift_models = [name for name in ["persistence", "xgboost", "ensemble"] if name in metrics]
        lines += lift_table(metrics, lift_models, numbering)
        lines += label_table(metrics, ordered(metrics)[0], numbering)
        if "xgboost" in metrics:
            lines += confusion_total(metrics, "xgboost", numbering)
        lines += ensemble_tables(metrics, numbering)
        lines += selection_tables(metrics, numbering)
        lines += simulation_tables(period_summaries(directories, spec), spec, numbering)
        (out / "hasil_walk_forward.md").write_text("\n".join(lines), encoding="utf-8")

    if groups["final-test"]:
        lines = final_test_tables(groups["final-test"][-1], spec, numbering)
        (out / "hasil_final_test.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"tabel bab 10 terakhir: 10.{numbering.next_number - 1}")


if __name__ == "__main__":
    main()
