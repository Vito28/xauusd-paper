"""Notebook-style UNI, BI, MULTI, time-series and model analysis.

Read-only inputs and causal data transformations live in analytics.data.
This file owns figures, descriptive relationships, frozen-model evaluation,
and the existing pre-2026 XGBoost experiment. Run --help for the unified CLI.
See docs/analytics.md for assumptions, paper references and interpretation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)

from analytics.data import (
    calculate_return,
    load_feature_importance,
    load_features,
    load_market_data,
    load_predictions,
    monthly_returns,
    prepare_daily_close,
    prepare_daily_summary,
    prepare_time_series,
    prepare_univariate_data,
    require_data,
)

# %% [markdown] 01 | Konfigurasi laporan dan kontrak variabel
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "analytics"
FORECAST_REPORT_ROOT = REPORT_ROOT / "xgboost_forecast"
EVALUATION_REPORT_ROOT = REPORT_ROOT / "model_evaluation"
FORECAST_START_UTC = pd.Timestamp("2026-01-01T00:00:00Z")
DEFAULT_HISTORY_START = "2016-01-01T00:00:00Z"


CLASS_LABELS = ("SELL", "NO_TRADE", "BUY")
UNIVARIATE_REQUIRED_COLUMNS = ("open_time_utc", "close", "tick_volume", "spread_price")
UNIVARIATE_OUTPUT_NAMES = frozenset(
    {
        "missing_values.png",
        "close_time_series.png",
        "close_histogram.png",
        "close_boxplot.png",
        "return_histogram.png",
        "return_boxplot.png",
        "tick_volume_histogram.png",
        "tick_volume_boxplot.png",
        "spread_price_histogram.png",
        "spread_price_boxplot.png",
    }
)
FEATURE_METADATA_COLUMNS = frozenset(
    {
        "timestamp_utc",
        "decision_time",
        "source_id",
        "symbol",
        "actual_symbol",
        "feature_version",
        "quality_status",
        "target_class",
        "predicted_class",
    }
)


# %% 02 | Utilitas plot: salinan data, nilai finite, PNG


def validate_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Return explicit columns after rejecting missing or unsafe names."""

    if not columns:
        raise ValueError("Minimal satu kolom harus dipilih")
    validated: list[str] = []
    for column in columns:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", column):
            raise ValueError(f"Nama kolom tidak valid: {column!r}")
        if column not in frame.columns:
            raise ValueError(f"Kolom tidak ditemukan: {column}")
        validated.append(column)
    return validated


def validate_feature_selection(frame: pd.DataFrame, features: Sequence[str]) -> list[str]:
    """Reject metadata, targets, IDs, and future information as predictors."""

    validated = validate_columns(frame, features)
    forbidden = [
        feature
        for feature in validated
        if feature in FEATURE_METADATA_COLUMNS
        or feature.startswith("future_")
        or feature.endswith("_id")
    ]
    if forbidden:
        raise ValueError(f"Kolom bukan predictor yang valid: {forbidden}")
    return validated


def safe_output_name(value: str) -> str:
    """Convert a column label into a safe filename stem."""

    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_").lower()
    if not cleaned:
        raise ValueError("Nama output tidak dapat dibuat")
    return cleaned


def save_plot(output_path: Path) -> Path:
    """Save the current figure as a non-empty PNG, replacing a prior report.

    Analytics reports are derived artifacts. Replacing the previous PNG makes
    repeated runs and ``--watch-seconds`` refreshes reflect the latest committed
    PostgreSQL data instead of failing because an older report exists.
    """

    path = Path(output_path)
    if path.suffix.lower() != ".png":
        plt.close()
        raise ValueError("Output analytics wajib berupa file PNG")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"PNG gagal dibuat: {path}")
    return path


def numeric_values(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a finite numeric copy for one plot without mutating source data."""

    validate_columns(frame, [column])
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        raise ValueError(f"Tidak ada nilai numerik finite untuk kolom {column}")
    return values


def paired_values(frame: pd.DataFrame, x_column: str, y_column: str) -> pd.DataFrame:
    """Return finite X/Y pairs, dropping NaN only for the requested pair."""

    validate_columns(frame, [x_column, y_column])
    pair = frame[[x_column, y_column]].copy()
    pair[x_column] = pd.to_numeric(pair[x_column], errors="coerce")
    pair[y_column] = pd.to_numeric(pair[y_column], errors="coerce")
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_column, y_column])
    if pair.empty:
        raise ValueError(f"Tidak ada pasangan valid untuk {x_column} dan {y_column}")
    return pair


# %% 03 | UNI / UNIVARIAT: satu variabel, distribusi dan missing values


def plot_time_series(
    frame: pd.DataFrame,
    timestamp_column: str,
    value_column: str,
    output_path: Path,
    *,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    resample_rule: str | None = None,
) -> Path:
    """Plot one numeric variable against UTC time without filling missing rows.

    ``resample_rule`` only changes the visual trace. The caller's DataFrame is
    never mutated or replaced with the resampled values.
    """

    validate_columns(frame, [timestamp_column, value_column])
    local = frame[[timestamp_column, value_column]].copy()
    local[timestamp_column] = pd.to_datetime(local[timestamp_column], utc=True, errors="coerce")
    local[value_column] = pd.to_numeric(local[value_column], errors="coerce")
    local = local.replace([np.inf, -np.inf], np.nan).dropna()
    require_data(local, f"time series {value_column}")
    local = local.sort_values(timestamp_column)
    if resample_rule:
        local = (
            local.set_index(timestamp_column)[value_column]
            .resample(resample_rule)
            .last()
            .dropna()
            .reset_index()
        )
        require_data(local, f"resampled time series {value_column}")
    plt.figure(figsize=(10, 5))
    plt.plot(local[timestamp_column], local[value_column])
    plt.title(title or f"{value_column} Time Series")
    plt.xlabel(x_label or timestamp_column)
    plt.ylabel(y_label or value_column)
    return save_plot(output_path)


def plot_histogram(
    frame: pd.DataFrame,
    column: str,
    output_path: Path,
    bins: int = 30,
    *,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str = "Frekuensi",
) -> Path:
    """Plot a finite-value histogram while leaving the source frame intact."""

    if bins <= 0:
        raise ValueError("bins harus lebih besar dari nol")
    values = numeric_values(frame, column)
    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=bins)
    plt.title(title or f"Distribusi {column}")
    plt.xlabel(x_label or column)
    plt.ylabel(y_label)
    return save_plot(output_path)


def plot_boxplot(
    frame: pd.DataFrame,
    column: str,
    output_path: Path,
    *,
    title: str | None = None,
    y_label: str | None = None,
) -> Path:
    """Plot a numeric boxplot without clipping outliers or filling NaN."""

    values = numeric_values(frame, column)
    plt.figure(figsize=(8, 5))
    plt.boxplot(values, tick_labels=[column])
    plt.title(title or f"Boxplot {column}")
    plt.ylabel(y_label or column)
    return save_plot(output_path)


def plot_missing_values(frame: pd.DataFrame, columns: Sequence[str], output_path: Path) -> Path:
    """Plot missing counts and percentages for explicit columns without imputation."""

    selected = validate_columns(frame, columns)
    counts = frame[selected].isna().sum()
    percentages = counts.div(len(frame)).mul(100) if len(frame) else counts.astype(float)
    plt.figure(figsize=(max(10, len(selected) * 0.65), 5))
    bars = plt.bar(selected, counts.to_numpy())
    for bar, count, percentage in zip(bars, counts, percentages, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.title("Missing Values per Kolom")
    plt.xlabel("Kolom")
    plt.ylabel("Jumlah Missing")
    plt.xticks(rotation=45, ha="right")
    return save_plot(output_path)


def cleanup_legacy_univariate_outputs(output_dir: Path) -> None:
    """Remove obsolete charts so the report directory holds only the ten current PNGs.

    This folder is owned by the univariate report. Removing PNGs outside the
    declared output contract clears prior open/high/low/real-volume and generic
    time-series reports without touching source data or other report folders.
    """

    directory = Path(output_dir)
    if not directory.exists():
        return
    for candidate in directory.glob("*.png"):
        if candidate.name not in UNIVARIATE_OUTPUT_NAMES:
            candidate.unlink()


def report_uninformative_real_volume(frame: pd.DataFrame) -> None:
    """Log when broker real volume is absent or almost entirely zero.

    Real volume stays in the source frame but is intentionally not charted by
    this fixed report because zero-valued broker placeholders are not useful
    transaction-volume evidence.
    """

    if "real_volume" not in frame.columns:
        return
    values = pd.to_numeric(frame["real_volume"], errors="coerce")
    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty or finite.eq(0).mean() >= 0.99:
        print(
            "real_volume tidak divisualisasikan karena broker tidak menyediakan "
            "data volume transaksi riil yang informatif.",
            file=sys.stderr,
            flush=True,
        )


def plot_close_time_series(frame: pd.DataFrame, output_path: Path) -> Path:
    """Plot daily close points for a legible view of the M5 price history."""

    return plot_time_series(
        frame,
        "open_time_utc",
        "close",
        output_path,
        title="Pergerakan Harga Penutupan XAU/USD",
        x_label="Waktu",
        y_label="Harga Close",
        resample_rule="1D",
    )


def plot_close_histogram(frame: pd.DataFrame, output_path: Path) -> Path:
    """Plot the unscaled distribution of closing prices."""

    return plot_histogram(
        frame,
        "close",
        output_path,
        bins=100,
        title="Distribusi Harga Penutupan XAU/USD",
        x_label="Harga Close",
    )


def plot_close_boxplot(frame: pd.DataFrame, output_path: Path) -> Path:
    """Plot closing-price quartiles without treating outliers as invalid data."""

    return plot_boxplot(
        frame,
        "close",
        output_path,
        title="Boxplot Harga Penutupan XAU/USD",
        y_label="Harga Close",
    )


def plot_return_histogram(frame: pd.DataFrame, output_path: Path) -> Path:
    """Plot percentage-return distribution without clipping extreme movements."""

    return plot_histogram(
        frame,
        "return",
        output_path,
        bins=100,
        title="Distribusi Return XAU/USD",
        x_label="Return",
    )


def plot_return_boxplot(frame: pd.DataFrame, output_path: Path) -> Path:
    """Plot percentage-return quartiles without automatic outlier removal."""

    return plot_boxplot(
        frame,
        "return",
        output_path,
        title="Boxplot Return XAU/USD",
        y_label="Return",
    )


def plot_tick_volume_histogram(frame: pd.DataFrame, output_path: Path) -> Path:
    """Plot the distribution of broker-reported tick activity."""

    return plot_histogram(
        frame,
        "tick_volume",
        output_path,
        bins=100,
        title="Distribusi Tick Volume XAU/USD",
        x_label="Tick Volume",
    )


def plot_tick_volume_boxplot(frame: pd.DataFrame, output_path: Path) -> Path:
    """Plot tick-volume quartiles without removing active-market observations."""

    return plot_boxplot(
        frame,
        "tick_volume",
        output_path,
        title="Boxplot Tick Volume XAU/USD",
        y_label="Tick Volume",
    )


def plot_spread_price_histogram(frame: pd.DataFrame, output_path: Path) -> Path:
    """Plot the raw distribution of bid-ask spread prices."""

    return plot_histogram(
        frame,
        "spread_price",
        output_path,
        bins=100,
        title="Distribusi Spread XAU/USD",
        x_label="Spread Price",
    )


def plot_spread_price_boxplot(frame: pd.DataFrame, output_path: Path) -> Path:
    """Plot spread-price quartiles without treating high spreads as invalid."""

    return plot_boxplot(
        frame,
        "spread_price",
        output_path,
        title="Boxplot Spread XAU/USD",
        y_label="Spread Price",
    )


def run_univariate_analysis(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Create the fixed ten-chart XAU/USD M5 univariate EDA report.

    Analysis is limited to data quality, close prices, percentage returns, tick
    volume, and spread price. It neither mutates the supplied frame nor writes
    to PostgreSQL, trains a model, imputes values, or removes outliers.
    """

    prepared = prepare_univariate_data(frame)
    output = Path(output_dir)
    cleanup_legacy_univariate_outputs(output)
    report_uninformative_real_volume(prepared)
    analysis = calculate_return(prepared)
    paths = [
        plot_missing_values(prepared, list(prepared.columns), output / "missing_values.png"),
        plot_close_time_series(analysis, output / "close_time_series.png"),
        plot_close_histogram(analysis, output / "close_histogram.png"),
        plot_close_boxplot(analysis, output / "close_boxplot.png"),
        plot_return_histogram(analysis, output / "return_histogram.png"),
        plot_return_boxplot(analysis, output / "return_boxplot.png"),
        plot_tick_volume_histogram(analysis, output / "tick_volume_histogram.png"),
        plot_tick_volume_boxplot(analysis, output / "tick_volume_boxplot.png"),
        plot_spread_price_histogram(analysis, output / "spread_price_histogram.png"),
        plot_spread_price_boxplot(analysis, output / "spread_price_boxplot.png"),
    ]
    if {path.name for path in paths} != UNIVARIATE_OUTPUT_NAMES:
        raise RuntimeError("Output univariate tidak sesuai kontrak sepuluh visualisasi")
    return paths


# %% 04 | BI / BIVARIAT: hubungan dua variabel, korelasi bukan kausalitas


def plot_scatter(frame: pd.DataFrame, x_column: str, y_column: str, output_path: Path) -> Path:
    """Plot valid numeric pairs as a diagnostic, not a causal conclusion."""

    pair = paired_values(frame, x_column, y_column)
    plt.figure(figsize=(8, 6))
    plt.scatter(pair[x_column], pair[y_column], alpha=0.5)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f"{x_column} vs {y_column}")
    return save_plot(output_path)


def plot_grouped_boxplot(
    frame: pd.DataFrame,
    value_column: str,
    group_column: str,
    output_path: Path,
) -> Path:
    """Compare a numeric value across non-empty categories using boxplots."""

    validate_columns(frame, [value_column, group_column])
    local = frame[[value_column, group_column]].copy()
    local[value_column] = pd.to_numeric(local[value_column], errors="coerce")
    local = local.replace([np.inf, -np.inf], np.nan).dropna()
    require_data(local, f"grouped boxplot {value_column}")
    groups = sorted(local[group_column].astype(str).unique())
    values = [local.loc[local[group_column].astype(str) == group, value_column] for group in groups]
    plt.figure(figsize=(10, 6))
    plt.boxplot(values, tick_labels=groups)
    plt.xlabel(group_column)
    plt.ylabel(value_column)
    plt.title(f"{value_column} by {group_column}")
    plt.xticks(rotation=45, ha="right")
    return save_plot(output_path)


# %% 05 | MULTI / MULTIVARIAT: hubungan beberapa variabel


def plot_correlation_matrix(frame: pd.DataFrame, columns: Sequence[str], output_path: Path) -> Path:
    """Plot a diagnostic Pearson correlation matrix for explicit columns."""

    selected = validate_columns(frame, columns)
    numeric = frame[selected].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    if numeric.dropna(how="all").empty:
        raise ValueError("Tidak ada data numerik untuk correlation matrix")
    correlation = numeric.corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation, vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(selected)), selected, rotation=90)
    plt.yticks(range(len(selected)), selected)
    if len(selected) <= 15:
        for row in range(len(selected)):
            for column in range(len(selected)):
                value = correlation.iloc[row, column]
                if pd.notna(value):
                    plt.text(column, row, f"{value:.2f}", ha="center", va="center")
    plt.title("Correlation Matrix")
    return save_plot(output_path)


# %% 06 | Feature EDA: distribusi, target dan pergeseran periode


def plot_feature_histogram(frame: pd.DataFrame, feature: str, output_path: Path) -> Path:
    """Plot one explicitly approved feature distribution."""

    validate_feature_selection(frame, [feature])
    return plot_histogram(frame, feature, output_path)


def plot_feature_boxplot(frame: pd.DataFrame, feature: str, output_path: Path) -> Path:
    """Plot one explicitly approved feature boxplot."""

    validate_feature_selection(frame, [feature])
    return plot_boxplot(frame, feature, output_path)


def plot_feature_missing_values(
    frame: pd.DataFrame, features: Sequence[str], output_path: Path
) -> Path:
    """Plot missing counts for explicit predictor features without filling."""

    selected = validate_feature_selection(frame, features)
    return plot_missing_values(frame, selected, output_path)


def plot_feature_correlation(
    frame: pd.DataFrame, features: Sequence[str], output_path: Path
) -> Path:
    """Plot feature correlations without automatic feature removal."""

    selected = validate_feature_selection(frame, features)
    return plot_correlation_matrix(frame, selected, output_path)


def plot_feature_by_target(
    frame: pd.DataFrame,
    feature: str,
    target_column: str,
    output_path: Path,
) -> Path:
    """Compare a feature across SELL, NO_TRADE, and BUY target classes."""

    validate_feature_selection(frame, [feature])
    validate_columns(frame, [target_column])
    local = frame[[feature, target_column]].copy()
    local[feature] = pd.to_numeric(local[feature], errors="coerce")
    local = local.replace([np.inf, -np.inf], np.nan).dropna()
    data = [local.loc[local[target_column] == label, feature] for label in CLASS_LABELS]
    if any(values.empty for values in data):
        raise ValueError("Target harus memiliki data SELL, NO_TRADE, dan BUY")
    plt.figure(figsize=(9, 6))
    plt.boxplot(data, tick_labels=CLASS_LABELS)
    plt.xlabel(target_column)
    plt.ylabel(feature)
    plt.title(f"{feature} by {target_column}")
    return save_plot(output_path)


def plot_feature_drift(
    reference_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    feature: str,
    output_path: Path,
) -> Path:
    """Compare reference and current distributions without mutation."""

    validate_feature_selection(reference_frame, [feature])
    validate_feature_selection(current_frame, [feature])
    reference_values = numeric_values(reference_frame, feature)
    current_values = numeric_values(current_frame, feature)
    plt.figure(figsize=(10, 5))
    plt.hist(reference_values, bins=30, alpha=0.5, label="Reference")
    plt.hist(current_values, bins=30, alpha=0.5, label="Current")
    plt.legend()
    plt.title(f"Feature Drift: {feature}")
    plt.xlabel(feature)
    plt.ylabel("Jumlah")
    return save_plot(output_path)


# %% 07 | Evaluasi model beku: satu split per laporan


def validate_evaluation_split(frame: pd.DataFrame) -> str:
    """Require predictions to contain exactly one evaluation split."""

    validate_columns(frame, ["evaluation_split"])
    splits = frame["evaluation_split"].dropna().astype(str).unique().tolist()
    if len(splits) != 1:
        raise ValueError(f"Data mencampur evaluation split: {splits}")
    return splits[0]


def plot_confusion_matrix(frame: pd.DataFrame, output_path: Path) -> Path:
    """Plot target versus prediction counts for one evaluation split."""

    require_data(frame, "confusion matrix")
    validate_evaluation_split(frame)
    validate_columns(frame, ["target_class", "predicted_class"])
    local = frame[["target_class", "predicted_class"]].dropna().copy()
    require_data(local, "confusion matrix labels")
    _, axis = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        local["target_class"],
        local["predicted_class"],
        labels=list(CLASS_LABELS),
        ax=axis,
    )
    axis.set_title("Confusion Matrix")
    return save_plot(output_path)


def plot_classification_metrics(frame: pd.DataFrame, output_path: Path) -> Path:
    """Plot per-class precision, recall, and F1 for one split."""

    require_data(frame, "classification metrics")
    validate_evaluation_split(frame)
    validate_columns(frame, ["target_class", "predicted_class"])
    local = frame[["target_class", "predicted_class"]].dropna().copy()
    require_data(local, "classification metric labels")
    precision, recall, f1, _ = precision_recall_fscore_support(
        local["target_class"],
        local["predicted_class"],
        labels=list(CLASS_LABELS),
        zero_division=0,
    )
    positions = np.arange(len(CLASS_LABELS))
    width = 0.25
    plt.figure(figsize=(10, 6))
    plt.bar(positions - width, precision, width, label="Precision")
    plt.bar(positions, recall, width, label="Recall")
    plt.bar(positions + width, f1, width, label="F1")
    plt.xticks(positions, CLASS_LABELS)
    plt.ylim(0, 1)
    plt.title("Classification Metrics")
    plt.legend()
    return save_plot(output_path)


def plot_probability_histograms(frame: pd.DataFrame, output_path: Path) -> Path:
    """Plot SELL, NO_TRADE, and BUY probability distributions."""

    validate_evaluation_split(frame)
    probability_columns = ("p_sell", "p_no_trade", "p_buy")
    validate_columns(frame, probability_columns)
    plt.figure(figsize=(10, 6))
    for column in probability_columns:
        plt.hist(numeric_values(frame, column), bins=30, alpha=0.5, label=column)
    plt.xlabel("Probability")
    plt.ylabel("Jumlah")
    plt.title("Prediction Probability Histograms")
    plt.legend()
    return save_plot(output_path)


def plot_feature_importance(
    importance_frame: pd.DataFrame, output_path: Path, top_n: int = 20
) -> Path:
    """Plot top frozen-model feature importance values as horizontal bars."""

    if top_n <= 0:
        raise ValueError("top_n harus lebih besar dari nol")
    validate_columns(importance_frame, ["feature", "importance"])
    local = importance_frame[["feature", "importance"]].copy()
    local["importance"] = pd.to_numeric(local["importance"], errors="coerce")
    local = local.replace([np.inf, -np.inf], np.nan).dropna()
    local = local.nlargest(top_n, "importance")
    require_data(local, "feature importance")
    plt.figure(figsize=(10, 7))
    plt.barh(local["feature"].astype(str), local["importance"])
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()
    return save_plot(output_path)


def plot_accuracy_by_month(frame: pd.DataFrame, output_path: Path) -> Path:
    """Plot accuracy by UTC month for one split; accuracy is not profit."""

    validate_evaluation_split(frame)
    validate_columns(frame, ["decision_time", "target_class", "predicted_class"])
    local = frame[["decision_time", "target_class", "predicted_class"]].copy()
    local["decision_time"] = pd.to_datetime(local["decision_time"], utc=True, errors="coerce")
    local = local.dropna()
    require_data(local, "monthly accuracy")
    local["month"] = local["decision_time"].dt.strftime("%Y-%m")
    local["correct"] = local["target_class"] == local["predicted_class"]
    accuracy = local.groupby("month", sort=True)["correct"].mean()
    plt.figure(figsize=(10, 5))
    plt.plot(accuracy.index, accuracy.to_numpy(), marker="o")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("UTC Month")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Month")
    return save_plot(output_path)


# %% 08 | Forecast XGBoost: histori < 2026, aktual hanya untuk evaluasi


def split_training_and_actual(
    daily_close: pd.DataFrame,
    *,
    forecast_start: pd.Timestamp = FORECAST_START_UTC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split model-visible history from ground truth with a strict UTC cutoff.

    Args:
        daily_close: Daily close data from prepare_daily_close.
        forecast_start: First instant excluded from training; by default
            2026-01-01 UTC, retaining history only through 31 December 2025.

    Returns:
        The pair (history, actual). Actual is intended only for visualisation
        and metrics, never for model fitting or recursive inference.
    """

    cutoff = _as_utc_timestamp(forecast_start, "forecast_start")
    validated = _validated_daily_close(daily_close)
    history = validated.loc[validated["timestamp_utc"] < cutoff].copy()
    actual = validated.loc[validated["timestamp_utc"] >= cutoff].copy()
    if history.empty:
        raise ValueError("Data historis sebelum cut-off forecast tidak tersedia")
    if actual.empty:
        raise ValueError("Data aktual pada/selepas cut-off forecast tidak tersedia")
    return history.reset_index(drop=True), actual.reset_index(drop=True)


def build_lagged_training_data(
    history: pd.DataFrame,
    lag_count: int,
    *,
    forecast_start: pd.Timestamp = FORECAST_START_UTC,
) -> tuple[pd.DataFrame, list[str]]:
    """Create prior-close lag features from history strictly before 2026.

    Args:
        history: The history half from split_training_and_actual.
        lag_count: Number of earlier daily closes for each model row.

    Returns:
        Model-ready target_close rows and the ordered lag feature names.
    """

    if lag_count < 1:
        raise ValueError("lag_count harus lebih besar dari nol")
    _ensure_history_precedes_cutoff(history, forecast_start)
    validated = _validated_daily_close(history)
    if len(validated) < lag_count + 2:
        raise ValueError("Data historis tidak cukup untuk lag dan target forecasting")
    data = validated.rename(columns={"close": "target_close"}).copy()
    features = [f"lag_{number}" for number in range(1, lag_count + 1)]
    for number, feature in enumerate(features, start=1):
        data[feature] = data["target_close"].shift(number)
    return data.dropna().reset_index(drop=True), features


def train_xgboost_forecaster(
    history: pd.DataFrame,
    *,
    lag_count: int = 30,
    random_state: int = 42,
    model_parameters: Mapping[str, Any] | None = None,
    forecast_start: pd.Timestamp = FORECAST_START_UTC,
) -> tuple[Any, list[str]]:
    """Fit XGBoost only from pre-2026 daily prices.

    Args:
        history: Daily close frame whose timestamps must all precede 2026.
        lag_count: Number of prior closes supplied to XGBoost.
        random_state: Reproducible estimator seed.
        model_parameters: Optional non-safety-critical XGBoost overrides.

    Returns:
        A fitted XGBRegressor and its required ordered feature names.

    Raises:
        ValueError: If history contains 2026 prices or protected model
            parameters are overridden.
    """

    _ensure_history_precedes_cutoff(history, forecast_start)
    training, features = build_lagged_training_data(
        history, lag_count, forecast_start=forecast_start
    )
    from xgboost import XGBRegressor

    parameters: dict[str, Any] = {
        "objective": "reg:squarederror",
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": random_state,
        "n_jobs": 1,
    }
    if model_parameters:
        forbidden = {"objective", "random_state", "n_jobs"}.intersection(model_parameters)
        if forbidden:
            raise ValueError(f"Parameter XGBoost tidak boleh dioverride: {sorted(forbidden)}")
        parameters.update(model_parameters)
    model = XGBRegressor(**parameters)
    model.fit(training[features], training["target_close"])
    return model, features


def forecast_recursively(
    model: Any,
    history: pd.DataFrame,
    forecast_dates: Sequence[pd.Timestamp],
    *,
    lag_count: int,
    forecast_start: pd.Timestamp = FORECAST_START_UTC,
) -> pd.DataFrame:
    """Produce 2026 prices using only prior history and prior predictions.

    Actual 2026 values are absent from this interface. After the first
    forecast, each newly predicted close is appended to the next lag window.
    """

    cutoff = _as_utc_timestamp(forecast_start, "forecast_start")
    _ensure_history_precedes_cutoff(history, cutoff)
    if lag_count < 1:
        raise ValueError("lag_count harus lebih besar dari nol")
    values = _validated_daily_close(history)["close"].astype(float).tolist()
    if len(values) < lag_count:
        raise ValueError("Data historis tidak cukup untuk recursive forecast")
    dates = pd.DatetimeIndex(pd.to_datetime(list(forecast_dates), utc=True, errors="raise"))
    if dates.empty:
        raise ValueError("Minimal satu tanggal forecast diperlukan")
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("Tanggal forecast harus unik dan berurutan naik")
    if (dates < cutoff).any():
        raise ValueError(f"Tanggal forecast tidak boleh sebelum {cutoff.isoformat()}")
    features = [f"lag_{number}" for number in range(1, lag_count + 1)]
    predictions: list[float] = []
    for _ in dates:
        row = pd.DataFrame([list(reversed(values[-lag_count:]))], columns=features)
        prediction = float(model.predict(row)[0])
        if not np.isfinite(prediction) or prediction <= 0:
            raise RuntimeError("Model menghasilkan forecast close yang tidak valid")
        predictions.append(prediction)
        values.append(prediction)
    return pd.DataFrame({"timestamp_utc": dates, "forecast_close": predictions})


def forecast_dates_for_actual(actual: pd.DataFrame) -> pd.DatetimeIndex:
    """Return business-day prediction dates through the latest actual date.

    Only an availability date is used; no actual close value enters forecasting.
    """

    end = _validated_daily_close(actual)["timestamp_utc"].max().normalize()
    return pd.date_range(FORECAST_START_UTC, end, freq="B", tz="UTC")


def align_actual_and_forecast(actual: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Join actual and forecast prices on exact dates and compute residuals."""

    observed = _validated_daily_close(actual).rename(columns={"close": "actual_close"})
    predicted = _validated_forecast(forecast)
    comparison = observed.merge(predicted, on="timestamp_utc", how="inner", validate="one_to_one")
    if comparison.empty:
        raise ValueError("Tidak ada tanggal yang sama antara actual dan forecast")
    comparison["residual"] = comparison["actual_close"] - comparison["forecast_close"]
    return comparison


def calculate_forecast_metrics(comparison: pd.DataFrame) -> dict[str, float]:
    """Calculate MAE, RMSE, and MAPE for aligned actual and forecast prices."""

    aligned = _validated_comparison(comparison)
    valid = aligned.loc[aligned["actual_close"] != 0]
    if valid.empty:
        raise ValueError("Tidak ada pasangan actual/forecast valid untuk metrik")
    return {
        "mae": float(mean_absolute_error(valid["actual_close"], valid["forecast_close"])),
        "mse": float(mean_squared_error(valid["actual_close"], valid["forecast_close"])),
        "rmse": float(np.sqrt(mean_squared_error(valid["actual_close"], valid["forecast_close"]))),
        "mape_percent": float(
            (
                np.abs(
                    (valid["actual_close"] - valid["forecast_close"]) / valid["actual_close"]
                ).mean()
            )
            * 100
        ),
        "observations": float(len(valid)),
    }


def create_interactive_dashboard(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    actual: pd.DataFrame,
    comparison: pd.DataFrame,
    metrics: Mapping[str, float],
    output_path: Path,
    *,
    forecast_start: pd.Timestamp = FORECAST_START_UTC,
) -> Path:
    """Write a self-contained Plotly dashboard with zoom, pan, and hover.

    The dashboard contains the same blue historical, red forecast, and green
    actual series as the static report, plus residuals and evaluation metrics.
    Plotly JavaScript is embedded so the HTML remains interactive offline.
    """

    from plotly import graph_objects as go
    from plotly.subplots import make_subplots

    historical = _validated_daily_close(history)
    predicted = _validated_forecast(forecast)
    observed = _validated_daily_close(actual)
    aligned = _validated_comparison(comparison)
    cutoff = _as_utc_timestamp(forecast_start, "forecast_start")
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
        subplot_titles=("Price forecast", "Residual: actual - forecast"),
    )
    figure.add_trace(
        go.Scatter(
            x=historical["timestamp_utc"],
            y=historical["close"],
            name="Historical used by model",
            line={"color": "#2563eb", "width": 1.5},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=predicted["timestamp_utc"],
            y=predicted["forecast_close"],
            name="XGBoost forecast",
            line={"color": "#dc2626", "width": 2.2},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=observed["timestamp_utc"],
            y=observed["close"],
            name="Actual comparison only",
            line={"color": "#16a34a", "width": 1.8},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=aligned["timestamp_utc"],
            y=aligned["residual"],
            name="Residual",
            marker_color=np.where(aligned["residual"] >= 0, "#16a34a", "#dc2626"),
        ),
        row=2,
        col=1,
    )
    figure.add_vline(x=cutoff.timestamp() * 1000, line_dash="dash", line_color="#111827")
    figure.update_xaxes(rangeslider={"visible": True}, row=2, col=1)
    figure.update_yaxes(title_text="Close price", row=1, col=1)
    figure.update_yaxes(title_text="Price error", row=2, col=1)
    figure.update_layout(
        title=(
            "XAUUSD XGBoost Forecast"
            f" | MAE {metrics['mae']:.4f}"
            f" | RMSE {metrics['rmse']:.4f}"
            f" | MAPE {metrics['mape_percent']:.2f}%"
        ),
        template="plotly_white",
        hovermode="x unified",
        height=850,
        dragmode="zoom",
        legend={"orientation": "h", "y": 1.06},
    )
    path = Path(output_path)
    if path.suffix.lower() != ".html":
        raise ValueError("Dashboard interaktif wajib berupa file HTML")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs=True, full_html=True)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Dashboard HTML gagal dibuat: {path}")
    return path


def plot_historical_forecast_actual(
    history: pd.DataFrame, forecast: pd.DataFrame, actual: pd.DataFrame, output_path: Path
) -> Path:
    """Save the principal blue-history, red-forecast, green-actual time series."""

    historical = _validated_daily_close(history)
    predicted = _validated_forecast(forecast)
    observed = _validated_daily_close(actual)
    plt.figure(figsize=(15, 7))
    plt.plot(
        historical["timestamp_utc"],
        historical["close"],
        color="tab:blue",
        label="Historical used by model (through 2025)",
        linewidth=1.1,
    )
    plt.plot(
        predicted["timestamp_utc"],
        predicted["forecast_close"],
        color="tab:red",
        label="XGBoost recursive forecast (2026)",
        linewidth=1.4,
    )
    plt.plot(
        observed["timestamp_utc"],
        observed["close"],
        color="tab:green",
        label="Actual XAUUSD (comparison only)",
        linewidth=1.2,
    )
    plt.axvline(
        FORECAST_START_UTC, color="black", linestyle="--", linewidth=0.8, label="Training cut-off"
    )
    plt.title("XAUUSD: Historical Data, XGBoost Forecast, and Actual Price")
    plt.xlabel("UTC date")
    plt.ylabel("Close price")
    plt.legend(loc="best")
    plt.grid(alpha=0.2)
    return _save_plot(output_path)


def plot_actual_vs_predicted(comparison: pd.DataFrame, output_path: Path) -> Path:
    """Save an aligned 2026 actual-versus-XGBoost-forecast line chart."""

    aligned = _validated_comparison(comparison)
    plt.figure(figsize=(13, 6))
    plt.plot(aligned["timestamp_utc"], aligned["actual_close"], color="tab:green", label="Actual")
    plt.plot(
        aligned["timestamp_utc"],
        aligned["forecast_close"],
        color="tab:red",
        label="XGBoost forecast",
    )
    plt.title("XAUUSD 2026: Actual vs Predicted")
    plt.xlabel("UTC date")
    plt.ylabel("Close price")
    plt.legend()
    plt.grid(alpha=0.2)
    return _save_plot(output_path)


def plot_residuals(comparison: pd.DataFrame, output_path: Path) -> Path:
    """Save actual-minus-forecast residuals over the reporting period."""

    aligned = _validated_comparison(comparison)
    plt.figure(figsize=(13, 5))
    plt.axhline(0, color="black", linewidth=0.8)
    plt.plot(aligned["timestamp_utc"], aligned["residual"], color="tab:purple")
    plt.title("XAUUSD 2026: Prediction Residuals (Actual - Forecast)")
    plt.xlabel("UTC date")
    plt.ylabel("Price residual")
    plt.grid(alpha=0.2)
    return _save_plot(output_path)


def plot_monthly_comparison(comparison: pd.DataFrame, output_path: Path) -> Path:
    """Save monthly mean actual and forecast closes for 2026."""

    aligned = _validated_comparison(comparison)
    monthly = (
        aligned.set_index("timestamp_utc")[["actual_close", "forecast_close"]].resample("MS").mean()
    )
    positions = np.arange(len(monthly))
    plt.figure(figsize=(12, 6))
    plt.bar(positions - 0.19, monthly["actual_close"], 0.38, color="tab:green", label="Actual")
    plt.bar(
        positions + 0.19, monthly["forecast_close"], 0.38, color="tab:red", label="XGBoost forecast"
    )
    plt.xticks(positions, monthly.index.strftime("%Y-%m"), rotation=45, ha="right")
    plt.title("XAUUSD 2026: Monthly Mean Actual vs Forecast")
    plt.xlabel("UTC month")
    plt.ylabel("Mean close price")
    plt.legend()
    return _save_plot(output_path)


def plot_error_distribution(comparison: pd.DataFrame, output_path: Path, bins: int = 30) -> Path:
    """Save the distribution of actual-minus-forecast residuals."""

    if bins < 1:
        raise ValueError("bins harus lebih besar dari nol")
    aligned = _validated_comparison(comparison)
    plt.figure(figsize=(10, 5))
    plt.hist(aligned["residual"], bins=bins, color="tab:purple", edgecolor="white")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("XAUUSD 2026: Prediction Error Distribution")
    plt.xlabel("Residual (actual - forecast)")
    plt.ylabel("Number of days")
    return _save_plot(output_path)


def generate_forecast_reports(
    source_id: str,
    symbol: str,
    timeframe: str,
    *,
    history_start: str = DEFAULT_HISTORY_START,
    actual_end: str,
    lag_count: int = 30,
    output_dir: Path = FORECAST_REPORT_ROOT,
) -> tuple[dict[str, Path], dict[str, float]]:
    """Load data, forecast without leakage, and write PNG/metric report artifacts.

    The history query ends precisely at 2026-01-01 UTC. The actual query starts
    then and is intentionally kept separate from all model arguments.
    """

    _as_utc_timestamp(history_start, "history_start")
    end = _as_utc_timestamp(actual_end, "actual_end")
    if end <= FORECAST_START_UTC:
        raise ValueError("actual_end harus sesudah 2026-01-01 UTC")
    cutoff = FORECAST_START_UTC.isoformat().replace("+00:00", "Z")
    history = prepare_daily_close(
        load_market_data(source_id, symbol, timeframe, history_start, cutoff)
    )
    actual = prepare_daily_close(load_market_data(source_id, symbol, timeframe, cutoff, actual_end))
    _ensure_history_precedes_cutoff(history)
    model, _ = train_xgboost_forecaster(history, lag_count=lag_count)
    forecast = forecast_recursively(
        model, history, forecast_dates_for_actual(actual), lag_count=lag_count
    )
    comparison = align_actual_and_forecast(actual, forecast)
    metrics = calculate_forecast_metrics(comparison)
    destination = Path(output_dir)
    paths = {
        "historical_forecast_actual": plot_historical_forecast_actual(
            history, forecast, actual, destination / "historical_forecast_actual.png"
        ),
        "actual_vs_predicted": plot_actual_vs_predicted(
            comparison, destination / "actual_vs_predicted.png"
        ),
        "residuals": plot_residuals(comparison, destination / "residuals.png"),
        "monthly_comparison": plot_monthly_comparison(
            comparison, destination / "monthly_comparison.png"
        ),
        "error_distribution": plot_error_distribution(
            comparison, destination / "error_distribution.png"
        ),
        "interactive_dashboard": create_interactive_dashboard(
            history,
            forecast,
            actual,
            comparison,
            metrics,
            destination / "dashboard.html",
        ),
    }
    metrics_path = destination / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    paths["metrics"] = metrics_path
    return paths, metrics


def _as_utc_timestamp(value: pd.Timestamp | str, name: str) -> pd.Timestamp:
    """Parse one timezone-aware UTC timestamp."""

    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        raise ValueError(f"{name} wajib memiliki timezone")
    return parsed.tz_convert("UTC")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Reject missing input columns with a readable error."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")


def _validated_daily_close(frame: pd.DataFrame) -> pd.DataFrame:
    """Return finite, positive, unique, chronologically ordered daily closes."""

    _require_columns(frame, ("timestamp_utc", "close"))
    local = frame[["timestamp_utc", "close"]].copy()
    local["timestamp_utc"] = pd.to_datetime(local["timestamp_utc"], utc=True, errors="coerce")
    local["close"] = pd.to_numeric(local["close"], errors="coerce")
    local = local.replace([np.inf, -np.inf], np.nan).dropna()
    local = local.loc[local["close"] > 0].sort_values("timestamp_utc")
    local = local.drop_duplicates("timestamp_utc", keep="last").reset_index(drop=True)
    if local.empty:
        raise ValueError("Tidak ada daily close valid")
    return local


def _validated_forecast(frame: pd.DataFrame) -> pd.DataFrame:
    """Return finite, positive, unique, chronologically ordered forecast closes."""

    _require_columns(frame, ("timestamp_utc", "forecast_close"))
    local = frame[["timestamp_utc", "forecast_close"]].copy()
    local["timestamp_utc"] = pd.to_datetime(local["timestamp_utc"], utc=True, errors="coerce")
    local["forecast_close"] = pd.to_numeric(local["forecast_close"], errors="coerce")
    local = local.replace([np.inf, -np.inf], np.nan).dropna()
    local = local.loc[local["forecast_close"] > 0].sort_values("timestamp_utc")
    local = local.drop_duplicates("timestamp_utc", keep="last").reset_index(drop=True)
    if local.empty:
        raise ValueError("Tidak ada forecast close valid")
    return local


def _validated_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Return valid aligned actual and forecast data with a residual column."""

    _require_columns(frame, ("timestamp_utc", "actual_close", "forecast_close"))
    local = frame[["timestamp_utc", "actual_close", "forecast_close"]].copy()
    local["timestamp_utc"] = pd.to_datetime(local["timestamp_utc"], utc=True, errors="coerce")
    for column in ("actual_close", "forecast_close"):
        local[column] = pd.to_numeric(local[column], errors="coerce")
    local = local.replace([np.inf, -np.inf], np.nan).dropna().sort_values("timestamp_utc")
    if local.empty:
        raise ValueError("Tidak ada data comparison valid")
    local["residual"] = local["actual_close"] - local["forecast_close"]
    return local


def _ensure_history_precedes_cutoff(
    history: pd.DataFrame,
    forecast_start: pd.Timestamp = FORECAST_START_UTC,
) -> None:
    """Reject values on/after the configured cutoff before model inference."""

    cutoff = _as_utc_timestamp(forecast_start, "forecast_start")
    if (_validated_daily_close(history)["timestamp_utc"] >= cutoff).any():
        raise ValueError(
            f"Data training/recursive state tidak boleh memuat harga mulai {cutoff.isoformat()}"
        )


def _save_plot(output_path: Path) -> Path:
    """Save and close the active Matplotlib figure as a non-empty PNG."""

    path = Path(output_path)
    if path.suffix.lower() != ".png":
        plt.close()
        raise ValueError("Output analytics wajib berupa file PNG")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"PNG gagal dibuat: {path}")
    return path


# %% 09A | UNI TIME SERIES: tren, drawdown, distribusi return dan volatilitas
def plot_trend_drawdown(daily: pd.DataFrame, output_path: Path) -> Path:
    """Compare price levels, relative growth and observed drawdown over UTC sessions."""

    usable = daily.loc[daily["close"].notna()]
    require_data(usable, "trend")
    _, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    dates = usable["timestamp_utc"]
    axes[0].plot(dates, usable["close"], linewidth=0.8)
    axes[0].set_ylabel("Close / USD")
    axes[0].set_title("Observed UTC session close; absent sessions are not filled")
    axes[1].plot(dates, usable["close"] / usable["close"].iloc[0] * 100)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Index 100 / log scale")
    axes[2].fill_between(dates, usable["drawdown"] * 100, 0, alpha=0.6)
    axes[2].set_ylabel("Drawdown / %")
    axes[2].set_xlabel("UTC date")
    for axis in axes:
        axis.grid(alpha=0.2)
    return save_plot(output_path)


def plot_return_volatility(daily: pd.DataFrame, output_path: Path) -> Path:
    """Show session returns and trailing observed-return volatility without annualising."""

    _, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    dates = daily["timestamp_utc"]
    axes[0].plot(dates, daily["log_return"] * 100, linewidth=0.7)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_ylabel("Session log return / %")
    axes[1].plot(dates, daily["rolling_volatility"] * 100)
    axes[1].set_ylabel("Rolling std / %")
    axes[1].set_xlabel("UTC date")
    axes[0].set_title(
        f"Returns between observed sessions; rolling window = {daily.attrs['window']} observations"
    )
    return save_plot(output_path)


def plot_monthly_return_heatmap(daily: pd.DataFrame, output_path: Path) -> Path:
    """Plot month-close returns with absent months blank and an explicit sample-end label."""

    returns = monthly_returns(daily)
    matrix = (
        pd.DataFrame(
            {
                "year": returns.index.year,
                "month": returns.index.month,
                "return": returns.to_numpy() * 100,
            }
        )
        .pivot(index="year", columns="month", values="return")
        .reindex(columns=range(1, 13))
    )
    values = matrix.to_numpy()
    finite = np.abs(values[np.isfinite(values)])
    limit = max(float(finite.max()) if finite.size else 1.0, 1.0)
    plt.figure(figsize=(14, max(4, len(matrix) * 0.45)))
    plt.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    plt.colorbar(label="Month-close return / %")
    plt.xticks(range(12), range(1, 13))
    plt.yticks(range(len(matrix)), matrix.index)
    for row in range(len(matrix)):
        for column in range(12):
            if np.isfinite(values[row, column]):
                plt.text(
                    column, row, f"{values[row, column]:.1f}", ha="center", va="center", fontsize=8
                )
    end = daily["timestamp_utc"].max().strftime("%Y-%m-%d")
    plt.title(f"Monthly returns through {end}; final month may be partial; blanks = unavailable")
    return save_plot(output_path)


def plot_yearly_return_distribution(daily: pd.DataFrame, output_path: Path) -> Path:
    """Compare return distributions across years with observed sample counts."""

    groups = [
        (year, group["log_return"].dropna() * 100)
        for year, group in daily.groupby(daily["timestamp_utc"].dt.year)
    ]
    groups = [(year, values) for year, values in groups if not values.empty]
    require_data(pd.DataFrame(groups), "yearly returns")
    plt.figure(figsize=(14, 6))
    plt.boxplot(
        [values for _, values in groups],
        tick_labels=[f"{year}\nn={len(values)}" for year, values in groups],
    )
    plt.axhline(0, color="black", linewidth=0.5)
    plt.ylabel("Observed session log return / %")
    plt.title("Year-to-year distribution; final year is only the observed portion")
    return save_plot(output_path)


def plot_return_dependence(daily: pd.DataFrame, output_path: Path, max_lag: int = 30) -> Path:
    """Diagnose return memory and volatility clustering over observed-session lags.

    This descriptive autocorrelation chart has no significance bands and does
    not establish stationarity. Lags count observations rather than calendar days.
    """

    if max_lag < 1:
        raise ValueError("max_lag minimal satu")
    _, axes = plt.subplots(2, 1, figsize=(12, 7))
    for axis, column in zip(axes, ("log_return", "absolute_return"), strict=True):
        values = daily[column]
        lags = range(1, min(max_lag, max(1, len(values) // 4)) + 1)
        correlations = [
            values.autocorr(lag) if values.notna().sum() > lag + 2 else np.nan for lag in lags
        ]
        axis.bar(list(lags), correlations)
        axis.axhline(0, color="black", linewidth=0.5)
        axis.set_ylabel(f"ACF {column}")
        axis.set_ylim(-1, 1)
    axes[0].set_title("Descriptive dependence; no stationarity or causal claim")
    axes[1].set_xlabel("Lag / observed UTC sessions")
    return save_plot(output_path)


# %% 09B | BI TIME SERIES: aktivitas tick dan spread versus besarnya gerakan
def plot_activity_relationships(native: pd.DataFrame, output_path: Path) -> Path:
    """Use density plots for native activity/spread versus absolute log returns.

    All finite pairs enter the hexagonal bins. Tick volume represents broker
    activity, not exchange-wide traded volume; correlation is descriptive.
    """

    columns = [column for column in ("log_tick_volume", "relative_spread") if column in native]
    if not columns:
        raise ValueError("Kolom aktivitas atau spread diperlukan untuk analisis BI")
    _, axes = plt.subplots(1, len(columns), figsize=(7 * len(columns), 5), squeeze=False)
    for axis, column in zip(axes[0], columns, strict=True):
        pairs = paired_values(native, column, "absolute_return")
        image = axis.hexbin(
            pairs[column], pairs["absolute_return"] * 100, gridsize=45, bins="log", mincnt=1
        )
        plt.colorbar(image, ax=axis, label="Pair count / log colour")
        axis.set_xlabel(column)
        axis.set_ylabel("Absolute native log return / %")
        axis.set_title(f"Aligned finite pairs: n={len(pairs):,}")
    return save_plot(output_path)


# %% 09C | MULTI TIME SERIES: korelasi fitur terukur dan audit cakupan
def plot_return_feature_correlation(daily: pd.DataFrame, output_path: Path) -> Path:
    """Plot pairwise Spearman correlations and counts for scale-relative daily features."""

    columns = [
        name
        for name in (
            "log_return",
            "absolute_return",
            "relative_range",
            "relative_spread",
            "tick_volume",
        )
        if name in daily
    ]
    numeric = daily[columns].replace([np.inf, -np.inf], np.nan)
    correlation = numeric.corr(method="spearman", min_periods=3)
    counts = numeric.notna().astype(int).T.dot(numeric.notna().astype(int))
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Spearman rho")
    plt.xticks(range(len(columns)), columns, rotation=40, ha="right")
    plt.yticks(range(len(columns)), columns)
    for row in range(len(columns)):
        for column in range(len(columns)):
            value = correlation.iloc[row, column]
            if np.isfinite(value):
                plt.text(
                    column,
                    row,
                    f"{value:.2f}\nn={counts.iloc[row, column]}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    plt.title("Daily relationships; pairwise samples; correlation does not establish causality")
    return save_plot(output_path)


def plot_observed_coverage(native: pd.DataFrame, output_path: Path) -> Path:
    """Show observed candle counts by UTC month without assuming a trading calendar."""

    counts = native.set_index("open_time_utc").resample("MS").size()
    matrix = (
        pd.DataFrame(
            {"year": counts.index.year, "month": counts.index.month, "bars": counts.to_numpy()}
        )
        .pivot(index="year", columns="month", values="bars")
        .reindex(columns=range(1, 13))
    )
    plt.figure(figsize=(14, max(4, len(matrix) * 0.45)))
    plt.imshow(matrix, cmap="Blues", aspect="auto")
    plt.colorbar(label="Observed native candles")
    plt.xticks(range(12), range(1, 13))
    plt.yticks(range(len(matrix)), matrix.index)
    plt.title(
        "Observed counts; zero/blank months need investigation; closures are not missing candles"
    )
    return save_plot(output_path)


def summarize_time_series(native: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
    """Report measured changes, declining sessions, drawdowns and observed yearly counts."""

    prices = daily["close"].dropna()
    require_data(prices.to_frame(), "summary prices")
    yearly: list[dict[str, Any]] = []
    for year, group in daily.groupby(daily["timestamp_utc"].dt.year):
        observed = group["close"].dropna()
        yearly.append(
            {
                "year": int(year),
                "sessions": len(group),
                "native_bars": int(group["bars"].sum()),
                "first_session": group["timestamp_utc"].min().isoformat(),
                "last_session": group["timestamp_utc"].max().isoformat(),
                "within_observed_year_change_percent": float(
                    (observed.iloc[-1] / observed.iloc[0] - 1) * 100
                )
                if len(observed)
                else None,
                "negative_return_sessions": int((group["log_return"] < 0).sum()),
            }
        )
    return {
        "source_id": str(native["source_id"].iloc[0]) if "source_id" in native else None,
        "first_bar_utc": native["open_time_utc"].min().isoformat(),
        "last_bar_utc": native["open_time_utc"].max().isoformat(),
        "observed_price_change_percent": float((prices.iloc[-1] / prices.iloc[0] - 1) * 100),
        "max_observed_drawdown_percent": float(daily["drawdown"].min() * 100),
        "negative_return_sessions": int((daily["log_return"] < 0).sum()),
        "quality": native.attrs.get("quality", {}),
        "years": yearly,
        "interpretation": (
            "Perubahan berbasis cakupan observasi, bukan return tahun penuh atau hasil strategi. "
            "Tahun/bulan terakhir dapat parsial."
        ),
    }


# %% 09D | MODEL UNI, BI, MULTI: regresi return dan baseline dengan validasi waktu
def evaluate_return_models(
    daily: pd.DataFrame,
    *,
    folds: int = 5,
    holdout_start: pd.Timestamp = FORECAST_START_UTC,
) -> pd.DataFrame:
    """Compare fixed Ridge models and baselines on future observed-session returns.

    UNI uses current return, BI adds relative range, MULTI adds relative spread
    and log tick activity. StandardScaler is fitted inside each training fold.
    Expanding pre-holdout folds purge labels overlapping the test decision time.
    The final holdout model sees only earlier data. No hyperparameter search or
    feature selection observes validation/holdout targets. Model groups use the
    same complete-case sample; no prices are filled to increase sample size.
    The last observed date is conservatively omitted because ingestion may
    cover only part of that UTC session; plots still display that observation.
    """

    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    require_columns = [
        "timestamp_utc",
        "log_return",
        "relative_range",
        "relative_spread",
        "tick_volume",
    ]
    validate_columns(daily, require_columns)
    if folds < 1:
        raise ValueError("folds minimal satu")
    cutoff = _as_utc_timestamp(holdout_start, "holdout_start")
    data = daily.iloc[:-1].copy(deep=True)
    data["log_tick_volume"] = np.log1p(data["tick_volume"])
    # MODEL-UNI/BI/MULTI | Keputusan setelah akhir sesi t, target dari sesi t+1.
    data["decision_time"] = data["timestamp_utc"] + pd.Timedelta(days=1)
    data["target_time"] = data["decision_time"].shift(-1)
    data["target"] = data["log_return"].shift(-1)
    groups = {
        # MODEL-UNI | Satu prediktor: log return sesi saat ini.
        "UNI": ["log_return"],
        # MODEL-BI | Dua prediktor: return dan range relatif.
        "BI": ["log_return", "relative_range"],
        # MODEL-MULTI | Empat prediktor: tambah spread relatif dan aktivitas tick.
        "MULTI": ["log_return", "relative_range", "relative_spread", "log_tick_volume"],
    }
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=groups["MULTI"] + ["target", "target_time"]
    )
    before = data.loc[data["target_time"] < cutoff].reset_index(drop=True)
    holdout = data.loc[(data["decision_time"] >= cutoff) & (data["target_time"] >= cutoff)]
    if len(before) < max(20, folds * 4):
        raise ValueError("Observasi sebelum holdout tidak cukup untuk validasi model return")
    first_test = len(before) // 2
    boundaries = np.linspace(first_test, len(before), folds + 1, dtype=int)
    splits = [
        (f"walk_forward_{i + 1}", before.iloc[:start], before.iloc[start:end])
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True))
    ]
    if not holdout.empty:
        splits.append(("holdout", before, holdout))
    rows: list[dict[str, Any]] = []
    for split, available, test in splits:
        training = available.loc[available["target_time"] < test["decision_time"].min()]
        if len(training) < 5 or test.empty:
            continue
        predictions = {
            "zero_return": np.zeros(len(test)),
            "last_return": test["log_return"].to_numpy(),
        }
        for name, features in groups.items():
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(training[features], training["target"])
            predictions[name] = model.predict(test[features])
        for name, prediction in predictions.items():
            rows.append(
                {
                    "split": split,
                    "model": name,
                    "train_n": len(training),
                    "test_n": len(test),
                    "train_target_end": training["target_time"].max().isoformat(),
                    "test_decision_start": test["decision_time"].min().isoformat(),
                    "test_target_end": test["target_time"].max().isoformat(),
                    "mae": float(mean_absolute_error(test["target"], prediction)),
                    "rmse": float(np.sqrt(mean_squared_error(test["target"], prediction))),
                }
            )
    return pd.DataFrame(rows)


def plot_return_model_comparison(metrics: pd.DataFrame, output_path: Path) -> Path:
    """Plot each temporal split's RMSE, preserving the distinction from the final holdout."""

    require_data(metrics, "model return metrics")
    matrix = metrics.pivot(index="split", columns="model", values="rmse")
    order = sorted(name for name in matrix.index if name != "holdout")
    if "holdout" in matrix.index:
        order.append("holdout")
    matrix = matrix.reindex(order)
    axis = matrix.plot(kind="bar", figsize=(14, 6))
    axis.set_ylabel("Next observed-session log return RMSE")
    axis.set_title("Fixed UNI / BI / MULTI models versus baselines; lower is better")
    plt.xticks(rotation=30, ha="right")
    return save_plot(output_path)


def run_time_series_analysis(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    timeframe: str = "M5",
    include_models: bool = False,
) -> list[Path]:
    """Generate eight scientific figures and measured JSON insights from isolated source copies."""

    native = prepare_time_series(frame, timeframe)
    daily = prepare_daily_summary(native)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_trend_drawdown(daily, output_dir / "uni_trend_drawdown.png"),
        plot_return_volatility(daily, output_dir / "uni_return_volatility.png"),
        plot_monthly_return_heatmap(daily, output_dir / "uni_monthly_returns.png"),
        plot_yearly_return_distribution(daily, output_dir / "uni_yearly_returns.png"),
        plot_return_dependence(daily, output_dir / "uni_return_dependence.png"),
        plot_activity_relationships(native, output_dir / "bi_activity_return.png"),
        plot_return_feature_correlation(daily, output_dir / "multi_return_features.png"),
        plot_observed_coverage(native, output_dir / "data_observed_coverage.png"),
    ]
    summary = summarize_time_series(native, daily)
    if include_models:
        metrics = evaluate_return_models(daily)
        paths.append(
            plot_return_model_comparison(metrics, output_dir / "models_return_comparison.png")
        )
        summary["return_model_metrics"] = metrics.to_dict(orient="records")
    summary_path = output_dir / "insights.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    paths.append(summary_path)
    return paths


def run_time_series(arguments: argparse.Namespace) -> list[Path]:
    """Read one source and report observed time-series behaviour in a source-specific folder."""

    frame = load_market_data(
        arguments.source_id, arguments.symbol, arguments.timeframe, arguments.start, arguments.end
    )
    output = (
        arguments.output_dir
        or REPORT_ROOT / "time_series" / safe_output_name(arguments.source_id) / arguments.timeframe
    )
    return run_time_series_analysis(
        frame, output, timeframe=arguments.timeframe, include_models=arguments.include_models
    )


def run_evaluation(arguments: argparse.Namespace) -> list[Path]:
    """Load one frozen prediction split and generate its evaluation reports."""

    frame = load_predictions(
        arguments.source_id,
        arguments.model_version,
        arguments.split,
        arguments.start,
        arguments.end,
    )
    paths = [
        plot_confusion_matrix(frame, EVALUATION_REPORT_ROOT / "confusion_matrix.png"),
        plot_classification_metrics(frame, EVALUATION_REPORT_ROOT / "classification_metrics.png"),
        plot_probability_histograms(frame, EVALUATION_REPORT_ROOT / "probability_histograms.png"),
        plot_accuracy_by_month(frame, EVALUATION_REPORT_ROOT / "accuracy_by_month.png"),
    ]
    if arguments.include_feature_importance:
        importance = load_feature_importance(arguments.source_id, arguments.model_version)
        paths.append(
            plot_feature_importance(
                importance, EVALUATION_REPORT_ROOT / "feature_importance.png", arguments.top_n
            )
        )
    return paths


def run_forecast(arguments: argparse.Namespace) -> list[Path]:
    """Execute the retained pre-2026 recursive XGBoost experiment and display its metrics."""

    paths, metrics = generate_forecast_reports(
        arguments.source_id,
        arguments.symbol,
        arguments.timeframe,
        history_start=arguments.history_start,
        actual_end=arguments.actual_end,
        lag_count=arguments.lag_count,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(metrics, indent=2), flush=True)
    return list(paths.values())


# %% 10 | CLI: pemuatan sumber dan eksekusi laporan


def add_market_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach shared required market-query arguments to a subcommand."""

    parser.add_argument("--source-id", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)


def add_refresh_argument(parser: argparse.ArgumentParser) -> None:
    """Add optional polling refresh support to an analytics subcommand."""

    parser.add_argument(
        "--watch-seconds",
        type=float,
        metavar="SECONDS",
        help="Ulangi query dan timpa PNG setiap interval ini; Ctrl+C untuk berhenti.",
    )


def run_univariate(arguments: argparse.Namespace) -> list[Path]:
    """Load market bars and create the fixed XAU/USD M5 univariate reports."""

    frame = load_market_data(
        arguments.source_id,
        arguments.symbol,
        arguments.timeframe,
        arguments.start,
        arguments.end,
    )
    return run_univariate_analysis(frame, REPORT_ROOT / "univariate")


def run_bivariate(arguments: argparse.Namespace) -> list[Path]:
    """Load market bars and create a two-variable scatter report."""

    frame = load_market_data(
        arguments.source_id,
        arguments.symbol,
        arguments.timeframe,
        arguments.start,
        arguments.end,
    )
    frame = prepare_time_series(frame, arguments.timeframe)
    name = f"{safe_output_name(arguments.x_column)}_vs_{safe_output_name(arguments.y_column)}"
    return [
        plot_scatter(
            frame,
            arguments.x_column,
            arguments.y_column,
            REPORT_ROOT / "bivariate" / f"{name}.png",
        )
    ]


def run_multivariate(arguments: argparse.Namespace) -> list[Path]:
    """Load market bars and create one explicit correlation matrix."""

    frame = load_market_data(
        arguments.source_id,
        arguments.symbol,
        arguments.timeframe,
        arguments.start,
        arguments.end,
    )
    frame = prepare_time_series(frame, arguments.timeframe)
    return [
        plot_correlation_matrix(
            frame,
            arguments.columns,
            REPORT_ROOT / "multivariate" / "correlation_matrix.png",
        )
    ]


def run_feature_eda(arguments: argparse.Namespace) -> list[Path]:
    """Load one feature version and create explicit feature EDA reports."""

    frame = load_features(
        arguments.source_id,
        arguments.feature_version,
        arguments.start,
        arguments.end,
    )
    features = validate_feature_selection(frame, arguments.features)
    output = REPORT_ROOT / "feature_eda"
    paths: list[Path] = []
    for feature in features:
        name = safe_output_name(feature)
        paths.extend(
            [
                plot_feature_histogram(frame, feature, output / f"{name}_histogram.png"),
                plot_feature_boxplot(frame, feature, output / f"{name}_boxplot.png"),
            ]
        )
        if arguments.target_column:
            paths.append(
                plot_feature_by_target(
                    frame,
                    feature,
                    arguments.target_column,
                    output / f"{name}_by_target.png",
                )
            )
    paths.extend(
        [
            plot_feature_missing_values(frame, features, output / "feature_missing_values.png"),
            plot_feature_correlation(frame, features, output / "feature_correlation.png"),
        ]
    )
    return paths


def build_parser() -> argparse.ArgumentParser:
    """Build the unified UNI, BI, MULTI, time-series, evaluation and forecast CLI."""

    parser = argparse.ArgumentParser(
        description="UNI, BI, MULTI, time-series insights and temporal model evaluation."
    )
    commands = parser.add_subparsers(dest="analysis", required=True)

    univariate = commands.add_parser("univariate", help="Analisis satu variabel")
    add_market_arguments(univariate)
    add_refresh_argument(univariate)
    univariate.add_argument(
        "--columns",
        nargs="+",
        help=(
            "Parameter lama yang dipertahankan untuk kompatibilitas. "
            "Analisis univariat selalu memakai close, tick_volume, dan spread_price."
        ),
    )
    univariate.set_defaults(handler=run_univariate)

    bivariate = commands.add_parser("bivariate", help="Analisis dua variabel")
    add_market_arguments(bivariate)
    add_refresh_argument(bivariate)
    bivariate.add_argument("--x-column", default="log_tick_volume")
    bivariate.add_argument("--y-column", default="absolute_return")
    bivariate.set_defaults(handler=run_bivariate)

    multivariate = commands.add_parser("multivariate", help="Analisis banyak variabel")
    add_market_arguments(multivariate)
    add_refresh_argument(multivariate)
    multivariate.add_argument(
        "--columns",
        nargs="+",
        default=[
            "log_return",
            "absolute_return",
            "relative_range",
            "relative_spread",
            "log_tick_volume",
        ],
    )
    multivariate.set_defaults(handler=run_multivariate)

    feature_eda = commands.add_parser("feature-eda", help="EDA feature hasil pipeline")
    feature_eda.add_argument("--source-id", required=True)
    feature_eda.add_argument("--feature-version", required=True)
    feature_eda.add_argument("--start", required=True)
    feature_eda.add_argument("--end", required=True)
    feature_eda.add_argument("--features", nargs="+", required=True)
    feature_eda.add_argument("--target-column")
    add_refresh_argument(feature_eda)
    feature_eda.set_defaults(handler=run_feature_eda)

    series = commands.add_parser(
        "time-series", help="Tren, return, volatilitas, drawdown dan insight"
    )
    add_market_arguments(series)
    add_refresh_argument(series)
    series.add_argument("--output-dir", type=Path)
    series.add_argument(
        "--include-models",
        action="store_true",
        help="Bandingkan model return UNI/BI/MULTI dengan baseline menggunakan validasi waktu",
    )
    series.set_defaults(handler=run_time_series)

    evaluation = commands.add_parser("evaluate", help="Evaluasi satu split model beku")
    for name in ("source-id", "model-version", "split", "start", "end"):
        evaluation.add_argument(f"--{name}", required=True)
    add_refresh_argument(evaluation)
    evaluation.add_argument("--include-feature-importance", action="store_true")
    evaluation.add_argument("--top-n", type=int, default=20)
    evaluation.set_defaults(handler=run_evaluation)

    forecast = commands.add_parser("forecast", help="Eksperimen recursive XGBoost 2026")
    forecast.add_argument("--source-id", required=True)
    forecast.add_argument("--symbol", default="XAUUSD")
    forecast.add_argument("--timeframe", default="M5")
    forecast.add_argument("--history-start", default=DEFAULT_HISTORY_START)
    forecast.add_argument("--actual-end", required=True)
    forecast.add_argument("--lag-count", type=int, default=30)
    forecast.add_argument("--output-dir", type=Path, default=FORECAST_REPORT_ROOT)
    forecast.set_defaults(handler=run_forecast, watch_seconds=None)
    return parser


def main() -> None:
    """Run EDA once or poll PostgreSQL and refresh PNG reports continuously."""

    argv = sys.argv[1:]
    # CLI-01 | Pertahankan pemanggilan forecast lama yang langsung memakai flags.
    if argv and argv[0].startswith("--") and argv[0] not in {"--help", "-h"}:
        argv = ["forecast", *argv]
    arguments = build_parser().parse_args(argv)
    if arguments.watch_seconds is not None and arguments.watch_seconds <= 0:
        raise ValueError("--watch-seconds harus lebih besar dari nol")
    while True:
        try:
            paths = arguments.handler(arguments)
            for path in paths:
                print(path, flush=True)
        except Exception as exc:
            if arguments.watch_seconds is None:
                raise
            plt.close("all")
            print(f"Refresh analytics gagal: {exc}", file=sys.stderr, flush=True)
        if arguments.watch_seconds is None:
            return
        try:
            time.sleep(arguments.watch_seconds)
        except KeyboardInterrupt:
            print("Analytics watcher dihentikan.", file=sys.stderr)
            return


if __name__ == "__main__":
    main()
