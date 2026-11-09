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
    hasil_uji_scratch.md      tabel bab 10.5 dari --uji-scratch <uji_scratch_bin_2025.json>
    hasil_compare_scratch.md  tabel bab 10 (library vs from scratch), bila ada run compare-scratch
    hasil_ablation.md         tabel bab 10 (ablation kelompok fitur), bila ada run ablation
Model yang muncul di lebih dari satu run walk-forward hanya diambil sekali, setelah
dicek apakah ringkasannya identik.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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


SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def probability_difference(value) -> str:
    """Selisih kecil ditulis ilmiah (2,4 × 10⁻⁶) agar tidak tampil sebagai 0,0000."""
    if value >= 0.001:
        return number(value)
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa.replace('.', ',')} × 10{str(int(exponent)).translate(SUPERSCRIPT)}"


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


# Riwayat git ditulis ulang pada 8 November 2026; run.json masih menyimpan hash lama.
COMMIT_MAP_PATH = Path(__file__).with_name("peta_commit.json")
COMMIT_MAP = {}
if COMMIT_MAP_PATH.exists():
    for row in load_json(COMMIT_MAP_PATH)["peta"]:
        COMMIT_MAP[row["lama"]] = row["baru"]


def commit_hash(run: dict) -> str:
    """Hash commit pendek pada riwayat sekarang (diterjemahkan dari hash lama bila perlu)."""
    commit = str(run.get("git_commit"))
    return COMMIT_MAP.get(commit, commit)[:7]


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
        f"| Commit kode / config hash | `{commit_hash(run)}` / `{run.get('config_hash')}` |",
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
        f"`{commit_hash(run)}`, config hash `{run['config_hash']}`).",
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


WIB_OFFSET_HOURS = 7  # WIB = UTC+7, untuk menulis jam sinyal dalam waktu Indonesia
# Timestamp tersimpan sebenarnya jam server MetaQuotes (EET/EEST), bukan UTC (laporan bab 3.4).
SERVER_TIMEZONE = "Europe/Helsinki"


def server_to_utc(times: pd.Series) -> pd.Series:
    naive = times.dt.tz_localize(None)
    local = naive.dt.tz_localize(SERVER_TIMEZONE, ambiguous="NaT", nonexistent="shift_forward")
    return local.dt.tz_convert("UTC")


def interval_text(minutes: float) -> str:
    if minutes < 120:
        return f"{number(minutes, 0)} menit"
    return f"{number(minutes / 60, 1)} jam"


def frequency_row(label: str, path: Path) -> str:
    """Seberapa sering model mengeluarkan BUY/SELL pada satu periode prediksi."""
    frame = pd.read_csv(path, usecols=["candle_open_time", "signal"])
    times = pd.to_datetime(frame["candle_open_time"], utc=True)
    is_signal = frame["signal"] != "NO_TRADE"
    signal_times = times[is_signal]
    buy = int((frame["signal"] == "BUY").sum())
    sell = int((frame["signal"] == "SELL").sum())
    total = buy + sell
    days = times.dt.date.nunique()
    if total == 0:
        return f"| {label} | {integer(len(frame))} | 0 | 0 | – | 0 / {days} | – | – | – |"
    per_day = signal_times.groupby(signal_times.dt.date).size()
    gaps = signal_times.diff().dt.total_seconds().div(60).dropna()
    consecutive = float((gaps == 5).mean()) if len(gaps) else 0.0
    hours = server_to_utc(signal_times).dt.hour.value_counts()
    busiest = []
    for hour in hours.index[:2]:
        busiest.append(f"{hour:02d} ({(hour + WIB_OFFSET_HOURS) % 24:02d})")
    return (
        f"| {label} | {integer(len(frame))} | {integer(buy)} | {integer(sell)} | "
        f"{interval_text(len(frame) / total * 5)} | {len(per_day)} / {days} | "
        f"{number(per_day.median(), 0)} | {percent(consecutive, 0)} | {', '.join(busiest)} |"
    )


def frequency_table(walk_directories, final_directory, numbering, model="xgboost") -> list[str]:
    periods = []
    for directory in walk_directories:
        model_directory = directory / "backtest" / model
        if not model_directory.is_dir():
            continue
        for period_directory in sorted(model_directory.iterdir()):
            label = period_directory.name.replace("validation_", "")
            if label not in [name for name, _ in periods]:
                periods.append((label, period_directory / "predictions.csv"))
    if final_directory is not None:
        path = final_directory / "backtest" / model / "final_test" / "predictions.csv"
        if path.exists():
            periods.append(("2026 (final-test)", path))
    if not periods:
        return []
    lines = [
        f"Tabel {numbering.take()} Frekuensi Sinyal Model {model} per Periode",
        "<!-- widths: 10,10,10,8,10,11,10,13,18 -->",
        "| Periode | Candle | BUY | SELL | Rata-rata 1 sinyal tiap | Hari ada sinyal | Median sinyal per hari aktif | Sinyal berurutan | Jam tersibuk UTC (WIB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, path in periods:
        if path.exists():
            lines.append(frequency_row(label, path))
    lines += [
        "",
        "Model menilai setiap candle M5 yang sudah tutup. Kolom candle mencakup candle yang labelnya belum "
        "lengkap (GAP/PENDING), sehingga jumlah sinyal di sini dapat sedikit lebih besar daripada jumlah sinyal "
        "yang dinilai. Sinyal berurutan adalah proporsi sinyal yang berjarak tepat 5 menit dari sinyal "
        "sebelumnya. Jam ditulis dalam UTC setelah koreksi zona waktu server (bab 3.4), dengan WIB (UTC+7) "
        "di dalam kurung.",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Ablation kelompok fitur (bab 10, feature selection wrapper)
# ---------------------------------------------------------------------------

ABLATION_KEYS = [
    ("directional_macro_f1", False),
    ("buy_precision", True),
    ("sell_precision", True),
    ("signal_coverage", True),
]


def ablation_tables(directory: Path, numbering: Numbering) -> list[str]:
    """Walk-forward XGBoost tanpa satu kelompok fitur; selisih dibanding semua fitur."""
    ablation = load_json(directory / "ablation.json")
    run = load_json(directory / "run.json")
    reference = ablation["all"]["summary"]
    lines = [
        f"Sumber: `{directory.name}` (dibuat {run['created_utc'][:16]} UTC, config hash `{run['config_hash']}`).",
        "",
        f"Tabel {numbering.take()} Ablation Kelompok Fitur: Walk-Forward XGBoost 2021–2025 tanpa Satu Kelompok",
        "<!-- widths: 22,22,12,16,16,12 -->",
        "| Varian | Directional F1 | Δ vs semua | Precision BUY | Precision SELL | Coverage |",
        "|---|---|---|---|---|---|",
    ]
    for variant, entry in ablation.items():
        summary = entry["summary"]
        cells = [mean_std(summary, key, as_percent) for key, as_percent in ABLATION_KEYS]
        delta = "–"
        if variant != "all":
            difference = (
                summary["directional_macro_f1"]["mean"] - reference["directional_macro_f1"]["mean"]
            )
            delta = ("+" if difference >= 0 else "−") + number(abs(difference))
        label = "semua kelompok" if variant == "all" else variant.replace("without_", "tanpa ")
        lines.append(f"| {label} | {cells[0]} | {delta} | {cells[1]} | {cells[2]} | {cells[3]} |")
    lines += [
        "",
        "Δ negatif berarti kinerja turun ketika kelompok dihapus (kelompok berguna); Δ mendekati nol atau "
        "positif berarti kelompok tidak menambah informasi pada konfigurasi ini.",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# compare-scratch: XGBoost library vs from scratch pada satu fold
# ---------------------------------------------------------------------------


def compare_scratch_tables(directory: Path, numbering: Numbering) -> list[str]:
    comparison = load_json(directory / "comparison.json")
    run = load_json(directory / "run.json")
    library = comparison["xgboost"]
    scratch = comparison["xgboost_scratch"]
    agreement = comparison["agreement"]

    def threshold(result, direction):
        rule = result["thresholds"][direction]
        return number(rule["threshold"], 2) if rule["enabled"] else "nonaktif"

    def precision(result, direction):
        return percent(result["evaluation"]["per_class"][direction]["precision"])

    def signals(result):
        per_class = result["evaluation"]["per_class"]
        return (
            f"{integer(per_class['BUY']['predicted'])} / {integer(per_class['SELL']['predicted'])}"
        )

    rows = [
        ("Pohon per kelas (early stopping)", lambda r: integer(r["best_iteration"] + 1)),
        ("Waktu training (detik)", lambda r: number(r["training_seconds"], 1)),
        ("Temperatur kalibrasi", lambda r: number(r["calibration"]["temperature"], 3)),
        ("Threshold BUY / SELL", lambda r: f"{threshold(r, 'BUY')} / {threshold(r, 'SELL')}"),
        ("Sinyal BUY / SELL", signals),
        ("Precision BUY", lambda r: precision(r, "BUY")),
        ("Precision SELL", lambda r: precision(r, "SELL")),
        ("Coverage", lambda r: percent(r["evaluation"]["signal_coverage"])),
        ("Directional F1", lambda r: number(r["evaluation"]["directional_macro_f1"])),
        (
            "Macro F1 tanpa threshold (argmax)",
            lambda r: number(r["evaluation_without_threshold"]["macro_f1"]),
        ),
        ("Log loss EVALUATION", lambda r: number(r["evaluation"]["probability"]["log_loss"])),
    ]
    fold = run.get("fold", {})
    lines = [
        f"Sumber: `{directory.name}` (fold {fold.get('name', '')}, dibuat {run['created_utc'][:16]} UTC, "
        f"config hash `{run['config_hash']}`).",
        "",
        f"Tabel {numbering.take()} XGBoost Library vs From Scratch dengan Hyperparameter yang Disamakan",
        "<!-- widths: 40,30,30 -->",
        "| Ukuran | Library (xgboost) | From scratch |",
        "|---|---|---|",
    ]
    for label, getter in rows:
        lines.append(f"| {label} | {getter(library)} | {getter(scratch)} |")
    lines += [
        "",
        f"Tabel {numbering.take()} Kesesuaian Probabilitas Library dan From Scratch pada Blok EVALUATION",
        "<!-- widths: 60,40 -->",
        "| Ukuran | Nilai |",
        "|---|---|",
        f"| Kelas argmax sama | {percent(agreement['argmax_agreement_rate'])} |",
        f"| Rata-rata selisih absolut probabilitas | {number(agreement['mean_absolute_probability_difference'])} |",
        f"| Selisih absolut probabilitas maksimum | {number(agreement['max_absolute_probability_difference'])} |",
        "",
    ]
    return lines


def scratch_bin_tables(path: Path, numbering: Numbering) -> list[str]:
    """Tabel dari uji_scratch_bin.py: library, scratch, dan scratch dengan batas bin library."""
    test = load_json(path)
    models = test["models"]
    names = ["library", "scratch", "scratch_library_cuts"]
    fold = test["fold"]
    blocks = test["block_sizes"]

    def thresholds(model):
        texts = []
        for direction in ["BUY", "SELL"]:
            value = model["thresholds"][direction]
            texts.append("nonaktif" if value is None else number(value, 2))
        return " / ".join(texts)

    rows = [
        ("Pohon per kelas (early stopping)", lambda m: integer(m["trees"])),
        ("Waktu training (detik)", lambda m: number(m["training_seconds"], 1)),
        ("Temperatur kalibrasi", lambda m: number(m["temperature"], 3)),
        ("Threshold BUY / SELL", thresholds),
        (
            "Sinyal BUY / SELL",
            lambda m: f"{integer(m['signals']['BUY'])} / {integer(m['signals']['SELL'])}",
        ),
        ("Precision BUY", lambda m: percent(m["precision"]["BUY"])),
        ("Precision SELL", lambda m: percent(m["precision"]["SELL"])),
        ("Coverage", lambda m: percent(m["coverage"])),
        ("Accuracy argmax (tanpa threshold)", lambda m: percent(m["argmax_accuracy"])),
        ("Macro F1 argmax (tanpa threshold)", lambda m: number(m["argmax_macro_f1"])),
        ("Log loss EVALUATION", lambda m: number(m["log_loss"])),
    ]
    lines = [
        f"Sumber: `documents/laporan_uts/uji_scratch_bin_2025.json` dari `documents/laporan_uts/uji_scratch_bin.py` "
        f"(XGBoost {test['packages']['xgboost']}). Data: {integer(test['rows_used'])} candle "
        f"{test['history_start'][:10]} s.d. 2025 dari `{test['source']}`. TRAIN "
        f"{fold['train']['start'][:7]} s.d. {fold['train']['end'][:7]} ({integer(blocks['train'])} candle), "
        f"CALIBRATION {integer(blocks['calibration'])}, THRESHOLD {integer(blocks['threshold'])}, "
        f"EVALUATION 2025 {integer(blocks['evaluation'])} candle.",
        "",
        f"Tabel {numbering.take()} XGBoost Library vs From Scratch dengan Hyperparameter yang Disamakan (fold 2025)",
        "<!-- widths: 34,22,22,22 -->",
        "| Ukuran | Library | Scratch (kuantil sendiri) | Scratch (batas bin library) |",
        "|---|---|---|---|",
    ]
    for label, getter in rows:
        cells = [getter(models[name]) for name in names]
        lines.append(f"| {label} | {' | '.join(cells)} |")

    plain = test["vs_library"]["scratch"]
    same_cuts = test["vs_library"]["scratch_library_cuts"]

    def pearson(entry):
        return " / ".join(number(value, 4) for value in entry["pearson_sell_notrade_buy"])

    agreement_rows = [
        ("Kelas argmax sama, semua baris", lambda e: percent(e["argmax_agreement"])),
        (
            f"Kelas argmax sama, baris dengan selisih dua probabilitas teratas ≥ 0,01 "
            f"({percent(plain['share_rows_margin_ge_0.01'], 1)} baris)",
            lambda e: percent(e["argmax_agreement_margin_ge_0.01"]),
        ),
        (
            f"Kelas argmax sama, selisih ≥ 0,02 ({percent(plain['share_rows_margin_ge_0.02'], 1)} baris)",
            lambda e: percent(e["argmax_agreement_margin_ge_0.02"]),
        ),
        (
            "Rata-rata selisih absolut probabilitas",
            lambda e: probability_difference(e["mean_abs_probability_difference"]),
        ),
        ("Korelasi Pearson P(SELL) / P(NO_TRADE) / P(BUY)", pearson),
        (
            "Keputusan akhir sama (threshold masing-masing)",
            lambda e: percent(e["decision_agreement_own_thresholds"]),
        ),
        (
            "Keputusan akhir sama bila memakai threshold library",
            lambda e: percent(e["decision_agreement_library_thresholds"]),
        ),
    ]
    lines += [
        "",
        f"Tabel {numbering.take()} Kesesuaian From Scratch dengan Library pada Blok EVALUATION 2025",
        "<!-- widths: 50,25,25 -->",
        "| Ukuran | Scratch (kuantil sendiri) | Scratch (batas bin library) |",
        "|---|---|---|",
    ]
    for label, getter in agreement_rows:
        lines.append(f"| {label} | {getter(plain)} | {getter(same_cuts)} |")
    cuts = test["cuts"]
    lines += [
        "",
        f"Batas bin hasil `np.quantile` sama dengan batas bin XGBoost hanya pada "
        f"{cuts['features_with_identical_cuts']} dari {cuts['feature_count']} fitur. Varian batas bin "
        "library memakai batas dari `xgboost.QuantileDMatrix(...).get_quantile_cut()` pada data TRAINING "
        "yang sama dan membulatkan fitur ke float32 seperti XGBoost; rumus gradien, Hessian, bobot daun, "
        "gain, dan base score tidak diubah.",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Tabel bab 6 (daftar fitur) dan bab 9 (ukuran blok fold)
# ---------------------------------------------------------------------------

FEATURE_DESCRIPTIONS = {
    "log_return_1": "ln(close[t] / close[t−1])",
    "log_return_3": "ln(close[t] / close[t−3])",
    "log_return_6": "ln(close[t] / close[t−6])",
    "log_return_12": "ln(close[t] / close[t−12]), return 1 jam",
    "return_6_atr": "(close[t] − close[t−6]) ÷ ATR14",
    "return_24_atr": "(close[t] − close[t−24]) ÷ ATR14, perubahan 2 jam",
    "candle_body_atr": "(close − open) ÷ ATR14, badan candle bertanda",
    "candle_range_atr": "(high − low) ÷ ATR14",
    "upper_wick_ratio": "Sumbu atas ÷ rentang candle",
    "lower_wick_ratio": "Sumbu bawah ÷ rentang candle",
    "close_position_in_range": "(close − low) ÷ (high − low)",
    "close_vs_ema_12_atr": "(close − EMA12) ÷ ATR14",
    "close_vs_ema_48_atr": "(close − EMA48) ÷ ATR14",
    "close_vs_ema_288_atr": "(close − EMA288) ÷ ATR14, jarak ke tren 24 jam",
    "ema_12_vs_48_atr": "(EMA12 − EMA48) ÷ ATR14",
    "ema_48_slope_atr": "(EMA48[t] − EMA48[t−6]) ÷ ATR14",
    "adx_14": "Average Directional Index 14 (kekuatan tren)",
    "di_difference_14": "+DI14 − −DI14 (arah tren)",
    "rsi_14": "Relative Strength Index 14",
    "rsi_14_change_3": "RSI14[t] − RSI14[t−3]",
    "macd_histogram_atr": "Histogram MACD (12, 26, 9) ÷ ATR14",
    "stochastic_k_14": "Stochastic %K 14",
    "donchian_position_48": "Posisi close dalam kanal high/low 48 candle",
    "atr_14_pct": "ATR14 ÷ close",
    "atr_14_vs_daily_mean": "ATR14 ÷ rata-rata ATR14 288 candle (rezim volatilitas)",
    "realized_vol_12": "Simpangan baku log return 12 candle",
    "realized_vol_48": "Simpangan baku log return 48 candle",
    "bollinger_width_20": "Lebar Bollinger Bands (20, 2) ÷ middle band",
    "bollinger_position_20": "Posisi close di dalam Bollinger Bands",
    "tick_volume_z_48": "Z-score ln(1 + tick volume) terhadap 48 candle",
    "tick_volume_ratio_12_288": "Rata-rata tick volume 12 ÷ 288 candle",
    "spread_atr": "Spread ÷ ATR14 (porsi biaya)",
    "spread_z_288": "Z-score spread terhadap 288 candle",
    "hour_sin": "sin(2π · jam / 24)",
    "hour_cos": "cos(2π · jam / 24)",
    "day_of_week": "Hari dalam minggu (0 = Senin)",
    "session_asia": "1 bila jam 09–18 waktu Tokyo",
    "session_london": "1 bila jam 08–17 waktu London",
    "session_newyork": "1 bila jam 08–17 waktu New York",
    "session_london_newyork_overlap": "1 bila sesi London dan New York sama-sama buka",
    "bars_since_gap": "Jumlah candle sejak gap terakhir (maksimum 288)",
}


def feature_table(directory: Path) -> list[str]:
    """Tabel 6.3: 41 fitur, kelompok, deskripsi, dan berapa fold terpilih oleh xgboost_selected."""
    run = load_json(directory / "run.json")
    groups = run["feature_group_map"]
    selected = {}
    stability = directory / "feature_selection_xgboost_selected.csv"
    if stability.exists():
        with open(stability, encoding="utf-8") as file:
            for row in csv.DictReader(file):
                selected[row["feature"]] = f"{row['selected_folds']}/{row['total_folds']}"
    lines = [
        "Tabel 6.3 Daftar Fitur, Rumus, dan Stabilitas Seleksi",
        "<!-- widths: 6,30,14,38,12 -->",
        "| No. | Fitur | Kelompok | Rumus atau deskripsi | Terpilih (fold) |",
        "|---|---|---|---|---|",
    ]
    for index, name in enumerate(run["features"], start=1):
        description = FEATURE_DESCRIPTIONS.get(name, "")
        lines.append(
            f"| {index} | `{name}` | {groups[name]} | {description} | {selected.get(name, '–')} |"
        )
    lines += [
        "",
        f"Sumber: `{directory.name}/run.json` dan `feature_selection_xgboost_selected.csv`. Kolom terakhir "
        "adalah jumlah fold walk-forward yang mempertahankan fitur tersebut pada xgboost_selected (bab 8.6). "
        "Jam dan sesi dihitung dari timestamp tersimpan, yang sebenarnya jam server broker (bab 3.4).",
        "",
    ]
    return lines


def block_size_table(metrics: dict) -> list[str]:
    """Tabel 9.3: jumlah candle berlabel per blok setiap fold (setelah purging dan embargo)."""
    folds = metrics["xgboost"]["folds"]
    lines = [
        "Tabel 9.3 Jumlah Candle Berlabel per Blok Setiap Fold (setelah purging dan embargo)",
        "<!-- widths: 20,20,20,20,20 -->",
        "| Fold | TRAIN | CALIBRATION | THRESHOLD | EVALUATION |",
        "|---|---|---|---|---|",
    ]
    for fold in folds:
        sizes = fold["block_sizes"]
        lines.append(
            f"| {fold['fold']['name']} | {integer(sizes['train'])} | {integer(sizes['calibration'])} | "
            f"{integer(sizes['threshold'])} | {integer(sizes['evaluation'])} |"
        )
    lines += ["", "Sumber: `block_sizes` model xgboost pada metrics.json run walk-forward.", ""]
    return lines


def selection_comparison(metrics: dict, summaries: dict, numbering: Numbering) -> list[str]:
    """xgboost (41 fitur) vs xgboost_selected: rata-rata lima fold dan profit factor 2021–2025."""
    base = metrics["xgboost"]
    chosen = metrics["xgboost_selected"]
    counts = [len(fold["feature_selection"]["selected"]) for fold in chosen["folds"]]
    total = chosen["folds"][0]["feature_selection"]["total_features"]

    def profit_factor(model):
        periods = [
            v
            for (name, period), v in summaries.items()
            if name == model and period.startswith("validation_")
        ]
        return aggregate(periods).get("profit_factor")

    rows = []
    for key, label, as_percent in [
        ("macro_f1", "Macro F1", False),
        ("directional_macro_f1", "Directional F1", False),
        ("buy_precision", "Precision BUY", True),
        ("sell_precision", "Precision SELL", True),
        ("signal_coverage", "Coverage sinyal", True),
    ]:
        first = base["summary"][key]["mean"]
        second = chosen["summary"][key]["mean"]
        if as_percent:
            difference = (
                f"{'+' if second >= first else '−'}{number(abs(second - first) * 100, 2)} poin"
            )
        else:
            difference = f"{'+' if second >= first else '−'}{number(abs(second - first))}"
        rows.append(
            f"| {label} | {mean_std(base['summary'], key, as_percent)} | "
            f"{mean_std(chosen['summary'], key, as_percent)} | {difference} |"
        )
    pf_base = profit_factor("xgboost")
    pf_chosen = profit_factor("xgboost_selected")
    return [
        f"Tabel {numbering.take()} xgboost (41 Fitur) dibanding xgboost_selected (Rata-rata ± Simpangan Baku Lima Fold)",
        "<!-- widths: 28,26,26,20 -->",
        "| Ukuran | xgboost | xgboost_selected | Selisih |",
        "|---|---|---|---|",
        f"| Jumlah fitur per fold | {total} | {min(counts)}–{max(counts)} | −{total - max(counts)} sampai −{total - min(counts)} |",
        *rows,
        f"| Profit factor 2021–2025 | {number(pf_base, 2)} | {number(pf_chosen, 2)} | "
        f"{'+' if pf_chosen >= pf_base else '−'}{number(abs(pf_chosen - pf_base), 2)} |",
        "",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=int, default=2, help="nomor tabel pertama bab 10")
    parser.add_argument("--uji-scratch", default=None, help="JSON dari uji_scratch_bin.py")
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
        summaries = period_summaries(directories, spec)
        lines += simulation_tables(summaries, spec, numbering)
        (out / "hasil_walk_forward.md").write_text("\n".join(lines), encoding="utf-8")
        if "xgboost" in metrics:
            blocks = block_size_table(metrics)
            (out / "hasil_blok_fold.md").write_text("\n".join(blocks), encoding="utf-8")
        for directory in directories:
            if (directory / "feature_selection_xgboost_selected.csv").exists():
                features = feature_table(directory)
                (out / "tabel_fitur.md").write_text("\n".join(features), encoding="utf-8")

    if groups["final-test"]:
        lines = final_test_tables(groups["final-test"][-1], spec, numbering)
        lines += frequency_table(groups["walk-forward"], groups["final-test"][-1], numbering)
        (out / "hasil_final_test.md").write_text("\n".join(lines), encoding="utf-8")

    if arguments.uji_scratch:
        lines = scratch_bin_tables(Path(arguments.uji_scratch), numbering)
        (out / "hasil_uji_scratch.md").write_text("\n".join(lines), encoding="utf-8")

    if groups["walk-forward"] and "xgboost_selected" in metrics and "xgboost" in metrics:
        lines = selection_comparison(metrics, summaries, numbering)
        (out / "hasil_seleksi.md").write_text("\n".join(lines), encoding="utf-8")

    if groups.get("compare-scratch"):
        lines = compare_scratch_tables(groups["compare-scratch"][-1], numbering)
        (out / "hasil_compare_scratch.md").write_text("\n".join(lines), encoding="utf-8")

    if groups.get("ablation"):
        lines = ablation_tables(groups["ablation"][-1], numbering)
        (out / "hasil_ablation.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"tabel bab 10 terakhir: 10.{numbering.next_number - 1}")


if __name__ == "__main__":
    main()
