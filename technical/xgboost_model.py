"""Run the complete XGBoost M5 workflow in notebook-style sections.

Sections cover causal features, TRAIN-only Quality/ReliefF/RFECV selection,
TUNE fitting, CALIBRATION, POLICY, frozen TEST evaluation, artifacts and CLI.
Use the # %% markers to navigate or run cells in a compatible notebook editor.
Importing this module defines helpers; training starts only through run_pipeline
or the CLI. Data preparation and temporal folds live in data_preparation.py.

Run: uv run python -B -m technical.xgboost_model --source-key metaquotes-demo-xauusd
All writes are local derived artifacts; PostgreSQL access remains read-only.
"""

# %% 01. Import, konfigurasi, dan dependensi pipeline

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    make_scorer,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from analytics.data import parse_utc
from technical.config import LABELS, ROOT, Config
from technical.dashboard import write_dashboard
from technical.data_preparation import (
    Fold,
    load_data,
    make_folds,
    make_labels,
    prepare_data,
    purged_cv,
)

# %% 02. Feature engineering: indikator causal dan manifest
# Tahap ini mengubah OHLC, tick volume, spread, dan waktu menjadi input numerik model.
# Alur: data candle M5 yang sudah tutup -> kandidat fitur X -> manifest audit.
# Seluruh fitur memakai candle saat ini/masa lalu; future_close dan label tidak masuk X.
# Penanda FE-01 sampai FE-14 menjelaskan kelompok fitur di dalam build_features().


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide aligned numeric series, mapping zero denominators/infinity to NaN."""
    return (numerator / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def build_features(data: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Return float32 candidates and a formula/parameter/provenance manifest.

    Args: prepared, sorted, closed M5 OHLC/spread/volume and indicator settings.
    Returns: feature-only matrix with unchanged row indices; causal audit manifest.
    Missing volume/spread stays unavailable; TRAIN quality filtering decides use.
    EMA/RSI/ATR lookbacks are minimum warm-up, not a finite history cutoff.
    """
    # Input fitur: harga OHLC, jumlah tick, dan spread yang diketahui pada candle close.
    c, o, h, low = (data[name].astype(float) for name in ("close", "open", "high", "low"))
    v, s = data.tick_volume.astype(float), data.spread_price.astype(float)
    values: dict[str, pd.Series] = {}
    manifest: list[dict[str, Any]] = []

    # Registry fitur: simpan nilai dan metadata rumus, lookback, lag, serta asal data.
    def add(
        name: str,
        series: pd.Series,
        group: str,
        formula: str,
        lookback: int = 1,
        *,
        parameters: dict[str, Any] | None = None,
        recursive: bool = False,
        lag: int = 0,
    ) -> None:
        """Register numeric candidates with the exact formula and causal dependencies."""
        values[name] = series
        manifest.append(
            {
                "feature_name": name,
                "feature_group": group,
                "base_indicator": group,
                "parameters": parameters or {},
                "transformation": formula,
                "lag": lag,
                "lookback_bars": lookback,
                "recursive_past_history": recursive,
                "requires_volume": group == "volume",
                "requires_spread": group == "spread",
                "availability": "candle_close",
            }
        )

    # FE-01 | Return: perubahan harga relatif, log return, dan jumlah return rolling.
    # Contoh return_3 = close saat ini / close 3 candle sebelumnya - 1.
    # Periode return: 1/2/3/6/12/24/48 bar; rolling sum: 3/6/12/24 bar.
    returns = c.pct_change(fill_method=None)
    for n in (1, 2, 3, 6, 12, 24, 48):
        add(
            f"return_{n}",
            c.pct_change(n, fill_method=None),
            "return",
            "close/close.shift(n)-1",
            n + 1,
            parameters={"n": n},
        )
        add(
            f"log_return_{n}",
            np.log(c / c.shift(n)),
            "return",
            "log(close/close.shift(n))",
            n + 1,
            parameters={"n": n},
        )
    for n in (3, 6, 12, 24):
        add(
            f"rolling_return_{n}",
            returns.rolling(n).sum(),
            "return",
            "sum(return_1,n)",
            n + 1,
            parameters={"n": n},
        )
    # FE-02 | Bentuk candle: range, body, wick atas/bawah, arah, dan posisi close.
    # Rasio terhadap harga/range membuat ukuran candle bisa dibandingkan antarperiode.
    # Termasuk perbandingan OHLC saat ini dengan harga candle sebelumnya.
    span, body = h - low, (c - o).abs()
    upper, lower = (
        h - pd.concat([o, c], axis=1).max(axis=1),
        pd.concat([o, c], axis=1).min(axis=1) - low,
    )
    for name, a, b in (
        ("range_pct", span, c),
        ("body_pct", body, c),
        ("upper_wick_pct", upper, c),
        ("lower_wick_pct", lower, c),
        ("body_to_range", body, span),
        ("upper_to_range", upper, span),
        ("lower_to_range", lower, span),
        ("close_position_in_range", c - low, span),
    ):
        add(name, safe_divide(a, b), "candle", name)
    add("candle_direction", np.sign(c - o), "candle", "sign(close-open)")
    for name, a, b in (
        ("open_to_previous_close", o, c.shift()),
        ("high_to_previous_close", h, c.shift()),
        ("low_to_previous_close", low, c.shift()),
        ("close_to_previous_high", c, h.shift()),
        ("close_to_previous_low", c, low.shift()),
    ):
        add(name, safe_divide(a, b) - 1, "candle", name, 2)
    # FE-03 | Tren SMA/EMA: jarak close dari moving average, slope, dan perubahan MA.
    # Default lookback: 5/8/12/24/48/96/144/288 bar; EMA memakai riwayat rekursif.
    for n in config.ma_windows:
        for kind, ma in (
            ("sma", c.rolling(n).mean()),
            ("ema", c.ewm(span=n, adjust=False, min_periods=n).mean()),
        ):
            args = {"parameters": {"n": n}, "recursive": kind == "ema"}
            add(f"close_vs_{kind}_{n}", safe_divide(c, ma) - 1, kind, "close/MA-1", n, **args)
            add(
                f"{kind}_slope_{n}",
                safe_divide(ma.diff(), c),
                kind,
                "MA.diff()/close",
                n + 1,
                **args,
            )
            add(
                f"{kind}_change_{n}",
                ma.pct_change(fill_method=None),
                kind,
                "MA/MA.shift(1)-1",
                n + 1,
                **args,
            )
    # FE-04 | MACD: selisih EMA cepat/lambat, signal, histogram, dan delta histogram.
    # Default (fast, slow, signal): (8,21,5), (12,26,9), (19,39,9).
    # Nilai dibagi close agar menjadi ukuran relatif terhadap harga.
    for fast, slow, signal in config.macd_configs:
        macd = (
            c.ewm(span=fast, adjust=False, min_periods=fast).mean()
            - c.ewm(span=slow, adjust=False, min_periods=slow).mean()
        )
        smoothed = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
        params = {"fast": fast, "slow": slow, "signal": signal}
        for name, series, formula in (
            (f"macd_{fast}_{slow}", macd, "(EMA_fast-EMA_slow)/close"),
            (f"macd_signal_{fast}_{slow}_{signal}", smoothed, "EMA_signal(MACD)/close"),
            (f"macd_hist_{fast}_{slow}_{signal}", macd - smoothed, "(MACD-signal)/close"),
            (
                f"macd_hist_delta_{fast}_{slow}_{signal}",
                (macd - smoothed).diff(),
                "hist.diff()/close",
            ),
        ):
            add(
                name,
                safe_divide(series, c),
                "macd",
                formula,
                slow + signal,
                parameters=params,
                recursive=True,
            )
    # FE-05 | RSI dan ATR dengan periode default 7/14/21/28 bar.
    # RSI menggambarkan momentum: nilai RSI, jarak dari 50, dan perubahan RSI.
    # ATR menggambarkan besar gerak harga: ATR/close, perubahan, dan rasio ke median ATR.
    delta = c.diff()
    tr = pd.concat([span, (h - c.shift()).abs(), (low - c.shift()).abs()], axis=1).max(axis=1)
    atrs = {}
    for n in config.oscillator_windows:
        gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
        rsi = 100 - 100 / (1 + safe_divide(gain, loss))
        rsi = rsi.mask((loss == 0) & (gain > 0), 100).mask((loss == 0) & (gain == 0), 50)
        for name, series, formula in (
            (f"rsi_{n}", rsi, "Wilder EWM RSI"),
            (f"rsi_distance_50_{n}", rsi - 50, "RSI-50"),
            (f"rsi_delta_{n}", rsi.diff(), "RSI.diff()"),
        ):
            add(name, series, "rsi", formula, n + 1, parameters={"n": n}, recursive=True)
        atr = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
        atrs[n] = atr
        for name, series, formula in (
            (f"atr_pct_{n}", safe_divide(atr, c), "ATR/close"),
            (f"atr_change_{n}", atr.pct_change(fill_method=None), "ATR.pct_change()"),
            (
                f"atr_vs_median_{n}",
                safe_divide(atr, atr.rolling(n).median()) - 1,
                "ATR/median(ATR,n)-1",
            ),
        ):
            add(name, series, "atr", formula, 2 * n, parameters={"n": n}, recursive=True)
    # FE-06 | Stochastic: posisi close dalam rentang high-low (%K) dan smoothing (%D).
    # Pasangan (lookback, smoothing): (9,3), (14,3), dan (21,5).
    for n, smooth in ((9, 3), (14, 3), (21, 5)):
        minimum, maximum = low.rolling(n).min(), h.rolling(n).max()
        k = 100 * safe_divide(c - minimum, maximum - minimum)
        add(
            f"stoch_k_{n}",
            k,
            "stochastic",
            "100*(close-min(low,n))/range(n)",
            n,
            parameters={"n": n},
        )
        add(
            f"stoch_d_{n}_{smooth}",
            k.rolling(smooth).mean(),
            "stochastic",
            "mean(K,smooth)",
            n + smooth,
            parameters={"n": n, "smooth": smooth},
        )
    # FE-07 | Momentum ROC: persentase perubahan close selama 3/6/12/24/48 bar.
    for n in (3, 6, 12, 24, 48):
        add(
            f"roc_{n}",
            100 * c.pct_change(n, fill_method=None),
            "momentum",
            "100*return_n",
            n + 1,
            parameters={"n": n},
        )
    # FE-08 | Volatilitas: standar deviasi return dan estimator Parkinson high-low.
    # Default lookback: 6/12/24/48/96/288 bar untuk membandingkan beberapa horizon.
    for n in config.volatility_windows:
        add(
            f"rolling_std_{n}",
            returns.rolling(n).std(),
            "volatility",
            "std(return_1,n)",
            n + 1,
            parameters={"n": n},
        )
        add(
            f"high_low_volatility_{n}",
            np.sqrt(np.log(h / low).pow(2).rolling(n).mean() / (4 * np.log(2))),
            "volatility",
            "Parkinson high-low estimator",
            n,
            parameters={"n": n},
        )
    # FE-09 | Bollinger Bands: lebar band, posisi close dalam band, dan jarak ke mid.
    # Menggunakan lookback 20/40/60 bar dan batas band +/- 2 standar deviasi.
    for n in (20, 40, 60):
        mid, std = c.rolling(n).mean(), c.rolling(n).std()
        add(
            f"bb_width_{n}_2",
            safe_divide(4 * std, mid),
            "bollinger",
            "4*std/mid",
            n,
            parameters={"n": n, "std": 2},
        )
        add(
            f"bb_position_{n}_2",
            safe_divide(c - (mid - 2 * std), 4 * std),
            "bollinger",
            "(close-lower)/(upper-lower)",
            n,
            parameters={"n": n, "std": 2},
        )
        add(
            f"close_vs_bb_mid_{n}",
            safe_divide(c, mid) - 1,
            "bollinger",
            "close/mid-1",
            n,
            parameters={"n": n},
        )
    # FE-10 | Donchian: posisi dalam channel, jarak ke high/low, dan arah breakout.
    # Lookback 12/24/48/96 bar; breakout dibandingkan dengan channel candle sebelumnya.
    for n in (12, 24, 48, 96):
        hi, lo = h.rolling(n).max(), low.rolling(n).min()
        for name, series, formula in (
            (f"donchian_position_{n}", safe_divide(c - lo, hi - lo), "(close-low_n)/range_n"),
            (f"distance_to_high_{n}", safe_divide(c - hi, c), "(close-high_n)/close"),
            (f"distance_to_low_{n}", safe_divide(c - lo, c), "(close-low_n)/close"),
            (
                f"range_breakout_{n}",
                (c > hi.shift()).astype(float) - (c < lo.shift()).astype(float),
                "breakout vs prior channel",
            ),
        ):
            add(name, series, "donchian", formula, n + 1, parameters={"n": n})
    # FE-11 | Aktivitas tick: perubahan volume, OBV, volume z-score, dan rasio ke MA.
    # Tick volume adalah jumlah tick, bukan volume kontrak transaksi.
    # Data volume yang hilang tetap NaN; OBV berikutnya tidak direkonstruksi.
    add("tick_volume_change", v.diff(), "volume", "volume.diff()", 2)
    add("tick_volume_return", v.pct_change(fill_method=None), "volume", "volume.pct_change()", 2)
    # Once volume is missing, cumulative OBV is unidentified for subsequent rows.
    obv = (np.sign(delta) * v).cumsum().where(~v.isna().cummax())
    add("obv", obv, "volume", "cumsum(sign(close.diff())*volume)", recursive=True)
    add("obv_change", obv.diff(), "volume", "OBV.diff()", 2, recursive=True)
    for n in (12, 24, 48, 96):
        mean, std = v.rolling(n).mean(), v.rolling(n).std()
        add(
            f"volume_zscore_{n}",
            safe_divide(v - mean, std),
            "volume",
            "(volume-mean)/std",
            n,
            parameters={"n": n},
        )
        if n != 96:
            add(
                f"volume_vs_ma_{n}",
                safe_divide(v, mean) - 1,
                "volume",
                "volume/mean-1",
                n,
                parameters={"n": n},
            )
    # FE-12 | Biaya spread: nilai/harga relatif, perubahan, median, z-score, dan spread/ATR.
    # Spread berasal dari observasi candle yang sama; data hilang tidak diisi nilai palsu.
    add("spread_price", s, "spread", "known spread in price units")
    add("spread_pct", safe_divide(s, c), "spread", "spread/close")
    add("spread_change", s.diff(), "spread", "spread.diff()", 2)
    for n in (12, 24, 48):
        median = s.rolling(n).median()
        add(f"spread_median_{n}", median, "spread", "median(spread,n)", n, parameters={"n": n})
        add(
            f"spread_vs_median_{n}",
            safe_divide(s, median) - 1,
            "spread",
            "spread/median-1",
            n,
            parameters={"n": n},
        )
        add(
            f"spread_zscore_{n}",
            safe_divide(s - s.rolling(n).mean(), s.rolling(n).std()),
            "spread",
            "(spread-mean)/std",
            n,
            parameters={"n": n},
        )
    if 14 in atrs:
        add("spread_to_atr", safe_divide(s, atrs[14]), "spread", "spread/ATR14", 14, recursive=True)
    # FE-13 | Waktu dan sesi: encoding siklik jam/hari serta indikator sesi pasar.
    # Sesi Asia/London/New York memakai timezone lokal, termasuk penyesuaian DST.
    # Tambahkan indikator overlap London-New York jika kedua sesi tersedia.
    times = data.prediction_timestamp
    hour = times.dt.hour + times.dt.minute / 60
    for name, series in (
        ("hour_sin", np.sin(2 * np.pi * hour / 24)),
        ("hour_cos", np.cos(2 * np.pi * hour / 24)),
        ("weekday_sin", np.sin(2 * np.pi * times.dt.dayofweek / 7)),
        ("weekday_cos", np.cos(2 * np.pi * times.dt.dayofweek / 7)),
    ):
        add(name, series, "time", "cyclical UTC decision time")
    for name, (zone, start, end) in config.sessions.items():
        hours = times.dt.tz_convert(zone).dt.hour
        session = (
            (hours >= start) & (hours < end) if start < end else (hours >= start) | (hours < end)
        )
        add(
            name,
            session.astype(float),
            "session",
            "local session hours with DST",
            parameters={"timezone": zone, "start_hour": start, "end_hour": end},
        )
    if {"london_session", "new_york_session"} <= set(values):
        add(
            "london_newyork_overlap",
            values["london_session"] * values["new_york_session"],
            "session",
            "London AND New York",
        )
    # FE-14 | Lag fitur terpilih: salinan nilai 1/3/6 candle sebelumnya (default config).
    # Contoh rsi_14_lag_3 memberi konteks momentum 15 menit sebelum keputusan saat ini.
    # Lag dibuat jika fitur asal tersedia; nama dan ketergantungannya masuk manifest.
    lag_sources = (
        "return_1",
        "rsi_14",
        "macd_hist_12_26_9",
        "atr_pct_14",
        "rolling_std_24",
        "spread_pct",
        "volume_zscore_24",
        "close_vs_ema_24",
        "close_vs_sma_24",
    )
    original = {m["feature_name"]: m for m in manifest}
    for name in lag_sources:
        if name not in values:
            continue
        parent = original[name]
        for lag in config.lag_bars:
            add(
                f"{name}_lag_{lag}",
                values[name].shift(lag),
                parent["feature_group"],
                f"{name}.shift({lag})",
                parent["lookback_bars"] + lag,
                parameters=parent["parameters"],
                recursive=parent["recursive_past_history"],
                lag=lag,
            )
    # Output feature engineering: X float32 dan manifest, dengan indeks baris tetap.
    # Warm-up indikator/observasi hilang tetap NaN; XGBoost menangani missing values.
    frame = pd.DataFrame(values, index=data.index).replace([np.inf, -np.inf], np.nan)
    return frame.astype("float32"), manifest


# %% 03. Feature selection: Quality Filter, ReliefF, dan RFECV
# Setelah kandidat X dibuat, tahap ini memilih kolom yang dipakai model.
# Seluruh keputusan seleksi hanya memakai TRAIN; TUNE/CALIBRATION/POLICY/TEST tidak fit.


def classifier(config: Config, **parameters: Any) -> XGBClassifier:
    """Create a deterministic histogram XGBoost classifier with fixed three-class order."""
    defaults = {
        "objective": "multi:softprob",
        "num_class": 3,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "n_estimators": config.n_estimators,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.0,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
        "max_bin": 128,
        "random_state": config.random_state,
        "n_jobs": config.n_jobs,
    }
    defaults.update(parameters)
    return XGBClassifier(**defaults)


def quality_filter(train: pd.DataFrame, config: Config) -> tuple[list[str], dict[str, Any]]:
    """Drop numerical/missing/constant/duplicate problems using only provided TRAIN rows."""
    kept, removed = [], {}
    hashes: dict[int, list[str]] = {}
    for name in train:
        series = train[name]
        reason = None
        if not pd.api.types.is_numeric_dtype(series):
            reason = "non_numeric"
        elif np.isinf(series.to_numpy()).any():
            reason = "infinity"
        elif series.notna().sum() == 0:
            reason = "all_null"
        elif series.isna().mean() > config.max_missing:
            reason = "too_many_missing"
        elif series.nunique(dropna=True) <= 1:
            reason = "constant"
        elif series.value_counts(normalize=True).iloc[0] >= config.near_constant:
            reason = "near_constant"
        else:
            key = int(pd.util.hash_pandas_object(series, index=False).sum())
            duplicate = next((p for p in hashes.get(key, []) if series.equals(train[p])), None)
            if duplicate:
                reason = f"exact_duplicate:{duplicate}"
            else:
                hashes.setdefault(key, []).append(name)
        if reason:
            removed[name] = reason
        else:
            kept.append(name)
    if not kept:
        raise ValueError("Tidak ada fitur yang lolos quality filter TRAIN")
    return kept, {
        "fit_block": "TRAIN",
        "input_features": len(train.columns),
        "retained_features": len(kept),
        "removed": removed,
    }


def relieff_rank(train: pd.DataFrame, target: pd.Series, config: Config) -> pd.DataFrame:
    """Rank multiclass feature relevance with deterministic TRAIN-only reference/anchors.

    For each anchor, nearest hits subtract normalized absolute differences; misses
    add differences weighted by class priors/(1-prior(anchor class)). Neighbors use
    Manhattan distance. Self is explicitly excluded from hits. Sampling is bounded
    for memory/runtime and exported to the audit report by select_features.
    """
    positions = np.unique(
        np.linspace(0, len(train) - 1, min(len(train), config.relieff_max_samples), dtype=int)
    )
    sample = train.iloc[positions]
    labels = target.iloc[positions].to_numpy(dtype=int)
    if set(labels) != {0, 1, 2}:
        raise ValueError("ReliefF TRAIN sample harus mencakup SELL, NO_TRADE, BUY")
    values = sample.fillna(sample.median()).to_numpy(dtype=float)
    ranges = np.ptp(values, axis=0)
    ranges[ranges == 0] = 1
    values = (values - values.min(axis=0)) / ranges
    anchors = np.unique(
        np.linspace(0, len(sample) - 1, min(config.relieff_anchors, len(sample)), dtype=int)
    )
    priors = {label: float((labels == label).mean()) for label in (0, 1, 2)}
    scores = np.zeros(values.shape[1])
    for label in (0, 1, 2):
        class_positions = np.flatnonzero(labels == label)
        k = min(config.relieff_neighbors + 1, len(class_positions))
        neighbors = NearestNeighbors(
            n_neighbors=k, metric="manhattan", algorithm="brute", n_jobs=config.n_jobs
        ).fit(values[class_positions])
        indices = neighbors.kneighbors(values[anchors], return_distance=False)
        for i, anchor in enumerate(anchors):
            chosen = class_positions[indices[i]]
            chosen = chosen[chosen != anchor][: config.relieff_neighbors]
            if len(chosen) == 0:
                continue
            difference = np.abs(values[anchor] - values[chosen]).mean(axis=0)
            if labels[anchor] == label:
                scores -= difference / len(anchors)
            else:
                scores += difference * priors[label] / (1 - priors[labels[anchor]]) / len(anchors)
    ranking = (
        pd.DataFrame({"feature": train.columns, "score": scores})
        .sort_values(["score", "feature"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def select_features(
    features: pd.DataFrame, labels: pd.DataFrame, train_indices: pd.Index, config: Config
) -> tuple[list[str], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Screen on an early TRAIN prefix, then run purged RFECV with frozen candidates.

    Raises ValueError when a purged CV train lacks one of three target classes.
    Quality and ReliefF see only the first inner-training prefix; all inner
    validation labels are excluded from screening. The fixed screen deliberately
    favors isolation over refreshing candidates on later TRAIN observations.
    RFECV selection uses fixed-label macro F1 so missing validation classes do
    not silently change the scoring denominator.
    """
    train, target = (
        features.loc[train_indices],
        labels.loc[train_indices, "target_class"].astype(int),
    )
    cv = purged_cv(
        labels.loc[train_indices, "prediction_timestamp"],
        labels.loc[train_indices, "target_timestamp"],
        config.cv_splits,
    )
    # Prefix screening selesai sebelum validation internal pertama. Daftar kandidat
    # dibekukan agar label seluruh validation RFECV tidak ikut menentukan ReliefF.
    screen = cv[0][0]
    # SELECT-01 | Quality filter: buang fitur nonnumerik, infinity, terlalu banyak NaN,
    # konstan/nyaris konstan, dan duplikat persis berdasarkan statistik TRAIN.
    kept, report = quality_filter(train.iloc[screen], config)
    # SELECT-02 | ReliefF: beri peringkat relevansi terhadap kelas dari tetangga TRAIN.
    # Median/range untuk ReliefF hanya berasal dari sample TRAIN, bukan preprocessing XGBoost.
    ranking = relieff_rank(train.iloc[screen][kept], target.iloc[screen], config)
    top = ranking.feature.head(config.relieff_top_k).tolist()
    # SELECT-03 | Purged CV: hapus label TRAIN yang waktunya mencapai awal validation.
    for inner_train, _ in cv:
        if set(target.iloc[inner_train]) != {0, 1, 2}:
            raise ValueError("Purged RFECV TRAIN harus mencakup ketiga kelas")
    scorer = make_scorer(f1_score, labels=[0, 1, 2], average="macro", zero_division=0)
    # SELECT-04 | RFECV: eliminasi fitur bertahap dengan XGBoost dan skor macro F1.
    # Hasilnya adalah subset final dari kandidat teratas ReliefF.
    selector = RFECV(
        classifier(config),
        step=config.rfecv_step,
        min_features_to_select=min(config.rfecv_min_features, len(top)),
        cv=cv,
        scoring=scorer,
        n_jobs=1,
    )
    selector.fit(train[top], target)
    selected = list(np.asarray(top)[selector.support_])
    results = pd.DataFrame(
        {
            k: v
            for k, v in selector.cv_results_.items()
            if k in ("n_features", "mean_test_score", "std_test_score")
        }
    )
    report.update(
        {
            "train_rows": len(train),
            "screening_rows": len(screen),
            "screening_method": "frozen first purged inner-training prefix",
            "screening_max_target": labels.loc[train_indices, "target_timestamp"]
            .iloc[screen]
            .max()
            .isoformat(),
            "train_first": labels.loc[train_indices, "prediction_timestamp"].min().isoformat(),
            "train_last": labels.loc[train_indices, "prediction_timestamp"].max().isoformat(),
            "relieff_reference_rows": min(len(screen), config.relieff_max_samples),
            "relieff_anchors": min(len(screen), config.relieff_max_samples, config.relieff_anchors),
            "relieff_top_k": len(top),
            "rfecv_selected": len(selected),
            "rfecv_splits": [
                {
                    "max_train_target": labels.loc[train_indices, "target_timestamp"]
                    .iloc[a]
                    .max()
                    .isoformat(),
                    "validation_start": labels.loc[train_indices, "prediction_timestamp"]
                    .iloc[b[0]]
                    .isoformat(),
                }
                for a, b in cv
            ],
        }
    )
    return selected, report, ranking, results


# %% 04. XGBoost modeling, tuning, calibration, dan confidence policy
# Alur fitting: TRAIN -> pilih parameter di TUNE -> refit TRAIN+TUNE ->
# sigmoid/interval di CALIBRATION -> selective precision di POLICY -> frozen TEST.
# Classifier menghasilkan SELL/NO_TRADE/BUY; regressor memprediksi future return langsung.


def macro_f1(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> float:
    """Score all three fixed classes, including absent classes with zero F1."""
    return float(f1_score(actual, predicted, labels=[0, 1, 2], average="macro", zero_division=0))


@dataclass
class PlattCalibration:
    """One-vs-rest sigmoid calibration of log odds, normalized to three probabilities."""

    models: list[LogisticRegression]

    @staticmethod
    def logits(probabilities: np.ndarray) -> np.ndarray:
        """Clip raw probabilities and return finite one-vs-rest log odds."""
        p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    @classmethod
    def fit(cls, probabilities: np.ndarray, targets: pd.Series, seed: int) -> "PlattCalibration":
        """Fit sigmoid maps only on CALIBRATION; reject unidentifiable missing classes."""
        if set(targets) != {0, 1, 2}:
            raise ValueError("CALIBRATION harus mencakup ketiga kelas untuk sigmoid")
        logits = cls.logits(probabilities)
        models = []
        for label in (0, 1, 2):
            model = LogisticRegression(C=100, random_state=seed, max_iter=300)
            model.fit(logits[:, [label]], (targets.to_numpy() == label).astype(int))
            models.append(model)
        return cls(models)

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply frozen sigmoid maps and normalize class probabilities to sum to one."""
        logits = self.logits(probabilities)
        result = np.column_stack(
            [m.predict_proba(logits[:, [i]])[:, 1] for i, m in enumerate(self.models)]
        )
        return result / result.sum(axis=1, keepdims=True)


def apply_policy(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Return class argmax; abstain to NO_TRADE when maximum confidence is too low."""
    return np.where(probabilities.max(axis=1) < threshold, 1, probabilities.argmax(axis=1))


def load_model_bundle(path: Path) -> dict[str, Any]:
    """Load a locally generated model bundle, including earlier calibration paths.

    Args:
        path: Classifier/regressor artifact saved by this research pipeline.

    Returns:
        Frozen model state. Legacy calibration modules resolve to this module
        in memory; no compatibility source file or artifact rewrite is needed.
        New artifacts use the canonical technical.xgboost_model class path.
    """
    # MODEL-LOAD | Nama class dalam bundle lama tetap bisa dibaca setelah folder dipindah.
    for legacy_module in ("technical.modeling", "technical.data.cgboost_model"):
        sys.modules.setdefault(legacy_module, sys.modules[__name__])
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("Artifact model harus berupa bundle dictionary")
    return bundle


def regressor(config: Config, **parameters: Any) -> XGBRegressor:
    """Create a return regressor independently of classifier-only parameters."""
    defaults = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_estimators": config.n_estimators,
        "max_depth": 3,
        "learning_rate": 0.04,
        "min_child_weight": 16,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.3,
        "reg_lambda": 8.0,
        "max_bin": 128,
        "random_state": config.random_state,
        "n_jobs": config.n_jobs,
    }
    defaults.update(parameters)
    return XGBRegressor(**defaults)


def decision_costs(labels: pd.DataFrame, slippage_bps: float) -> np.ndarray:
    """Return spread/edge/slippage threshold without double counting labeled slippage."""
    included = labels.get("slippage_return", 0.0)
    return (labels.threshold_return + slippage_bps / 10000 - included).to_numpy(dtype=float)


def attach_direction_outcomes(predictions: pd.DataFrame) -> pd.DataFrame:
    """Judge accepted signals at the fixed target close; abstentions are always neutral.

    Strictly positive signed movement is SUCCESS; negative is FAILURE and zero
    is FLAT. These direction outcomes are separate from net-after-cost outcomes
    and three-class training labels. Missing/gap actuals never count as failures.
    Supports upgrading older saved prediction tables without altering signals.
    """
    result = predictions.copy()
    traded = result.predicted_signal.ne("NO_TRADE")
    actual_known = result.actual_future_close.notna()
    if "actual_status" not in result:
        target = pd.to_datetime(result.target_timestamp, utc=True)
        latest = pd.to_datetime(result.prediction_timestamp, utc=True).max()
        result["actual_status"] = np.select(
            [actual_known, target > latest], ["AVAILABLE", "PENDING"], default="GAP_INVALID"
        )
    signed_move = result.predicted_signal.map({"BUY": 1, "SELL": -1, "NO_TRADE": 0}) * (
        result.actual_future_close - result.close
    )
    result["prediction_correct"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    judged = traded & actual_known
    result.loc[judged, "prediction_correct"] = signed_move.loc[judged].gt(0)
    result["outcome_status"] = np.select(
        [~traded, ~actual_known, signed_move.gt(0), signed_move.lt(0)],
        ["NEUTRAL", result.actual_status, "SUCCESS", "FAILURE"],
        default="FLAT",
    )
    net = result.get("realized_net_move", pd.Series(np.nan, index=result.index))
    result["net_outcome"] = np.select(
        [~traded, net.isna(), net.gt(0), net.lt(0)],
        ["NEUTRAL", "UNAVAILABLE", "PROFIT", "LOSS"],
        default="BREAKEVEN",
    )
    return result


def load_frozen_models(directory: Path) -> dict[str, Any]:
    """Load a verified pair of locally trusted models and their immutable inference contract.

    Hash mismatches, mixed feature lists or unsupported contracts fail closed.
    Legacy files remain readable individually using load_model_bundle, but must
    not silently qualify as a validated live inference bundle.
    """
    contract = json.loads((directory / "inference_contract.json").read_text(encoding="utf-8"))
    if contract["schema"] != "direction-v3":
        raise ValueError("Schema inference tidak didukung")
    for name, expected in contract["sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Pasangan model berubah atau tercampur")
    classification = load_model_bundle(directory / "model_classifier.joblib")
    regression = load_model_bundle(directory / "model_regressor.joblib")
    if (
        classification["features"] != contract["classifier_features"]
        or regression["features"] != contract["regression_features"]
    ):
        raise ValueError("Schema fitur model tidak cocok")
    bundle = {**classification, **{k: v for k, v in regression.items() if k != "features"}}
    for key, names in (
        ("classifier", contract["classifier_features"]),
        ("regressor", contract["regression_features"]),
        ("quantile_regressor", contract["regression_features"]),
    ):
        if list(bundle[key].feature_names_in_) != names:
            raise ValueError("Urutan fitur estimator tidak cocok")
    bundle["inference_contract"] = contract
    return bundle


def predict_latest(
    bundle: dict[str, Any],
    history: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    max_staleness_minutes: float = 10.0,
) -> pd.DataFrame:
    """Predict one closed candle from actual history with a frozen validated model pair.

    Requires the same history origin as training to preserve cumulative/EMA
    features. Rejects stale data, source mismatch and in-development decisions.
    This read-only research endpoint places no orders and does not retrain.
    """
    if as_of.tzinfo is None or not np.isfinite(max_staleness_minutes) or max_staleness_minutes <= 0:
        raise ValueError("as_of harus UTC-aware dan batas stale harus positif")
    contract = bundle["inference_contract"]
    config = Config(**contract["config"])
    data, _ = prepare_data(history, as_of=as_of)
    if data.open_time_utc.min() != pd.Timestamp(contract["history_start"]):
        raise ValueError("History harus memakai origin yang sama dengan training")
    for key, value in contract["identity"].items():
        if key not in data or not data[key].eq(value).all():
            raise ValueError("Identitas sumber/simbol tidak cocok")
    latest = data.prediction_timestamp.iloc[-1]
    if as_of - latest > pd.Timedelta(minutes=max_staleness_minutes):
        raise ValueError("Candle terbaru stale; inference ditolak")
    if latest < pd.Timestamp(contract["decision_not_before"]):
        raise ValueError("Keputusan masih berada di development")
    features, _ = build_features(data, config)
    labels = make_labels(data, config.horizon_bars, config.minimum_edge_bps, config.slippage_bps)
    return predict_test(
        bundle,
        features,
        labels,
        data.index[-1:],
        fold=contract["fold"],
        evaluation_end=latest + pd.Timedelta(minutes=5 * config.horizon_bars + 1),
    )


def select_regression_features(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    train_indices: pd.Index,
    config: Config,
) -> tuple[list[str], dict[str, Any]]:
    """Rank regression features on TRAIN return targets, independently of class RFECV.

    Gain ranking is a screening heuristic, not causal importance. All retained
    rows are used; missing features are handled natively. TUNE tests whether the
    resulting regression is useful against the zero-return baseline.
    """
    train = features.loc[train_indices]
    kept, quality = quality_filter(train, config)
    screening = regressor(config, n_estimators=min(80, config.n_estimators))
    screening.fit(train[kept], labels.loc[train_indices, "future_return"] * 10000)
    ranking = sorted(
        zip(kept, screening.feature_importances_, strict=True), key=lambda pair: (-pair[1], pair[0])
    )
    selected = [name for name, _ in ranking[: config.regression_top_k]]
    return selected, {
        "fit_block": "TRAIN",
        "method": "return-regressor gain screening",
        "quality": quality,
        "selected": selected,
        "ranking": [{"feature": name, "gain": float(gain)} for name, gain in ranking],
    }


def precision_statistics(actual: np.ndarray, predicted: np.ndarray, label: int) -> dict[str, Any]:
    """Report per-direction precision, recall, F1, support and descriptive Wilson bounds.

    Precision is unavailable when no signals are emitted. Wilson bounds assume
    independent Bernoulli outcomes and are diagnostic, not a time-series guarantee.
    """
    selected = predicted == label
    count = int(selected.sum())
    correct = int(((actual == label) & selected).sum())
    support = int((actual == label).sum())
    precision = correct / count if count else None
    recall = correct / support if support else 0.0
    f1 = 2 * correct / (count + support) if count + support else 0.0
    lower = upper = None
    if count:
        z = 1.96
        center = (precision + z * z / (2 * count)) / (1 + z * z / count)
        margin = (
            z
            * np.sqrt(precision * (1 - precision) / count + z * z / (4 * count * count))
            / (1 + z * z / count)
        )
        lower, upper = max(0.0, float(center - margin)), min(1.0, float(center + margin))
    return {
        "signals": count,
        "correct": correct,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "actual_support": support,
        "wilson95_lower": lower,
        "wilson95_upper": upper,
    }


def non_overlapping_positions(labels: pd.DataFrame) -> np.ndarray:
    """Select fixed chronological evaluation anchors with nonoverlapping target horizons."""
    positions: list[int] = []
    available_at = None
    for position, row in enumerate(
        labels[["prediction_timestamp", "target_timestamp"]].itertuples(index=False)
    ):
        if available_at is None or row.prediction_timestamp >= available_at:
            positions.append(position)
            available_at = row.target_timestamp
    return np.asarray(positions, dtype=int)


def apply_selective_policy(
    probabilities: np.ndarray,
    returns: np.ndarray,
    costs: np.ndarray,
    policy: dict[str, Any],
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Abstain unless class confidence, regression direction, cost and uncertainty agree.

    Inputs contain only decision-time features or frozen model outputs. A
    disabled class cannot trade. This function never receives actual outcomes.
    """
    raw = probabilities.argmax(axis=1)
    result = np.full(len(raw), 1, dtype=int)
    reasons = np.full(len(raw), "classifier_no_trade", dtype=object)
    valid = (
        np.isfinite(probabilities).all(axis=1)
        & np.isfinite(returns)
        & np.isfinite(costs)
        & (costs >= 0)
        & np.isfinite(lower)
        & np.isfinite(upper)
    )
    for name, label, direction in (("SELL", 0, -1), ("BUY", 2, 1)):
        mask = (raw == label) & valid
        rule = policy["classes"][name]
        reasons[mask] = "class_precision_unconfirmed"
        if not rule["enabled"]:
            continue
        margin = probabilities[:, label] - np.max(np.delete(probabilities, label, axis=1), axis=1)
        confidence = (probabilities[:, label] >= rule["threshold"]) & (
            margin >= rule["probability_margin"]
        )
        edge = direction * returns > costs * rule["edge_multiple"]
        bound = lower if label == 2 else -upper
        certain = (
            bound > costs if policy["require_interval_agreement"] else np.ones(len(raw), dtype=bool)
        )
        reasons[mask] = "confidence_abstain"
        reasons[mask & confidence] = "regression_edge_disagreement"
        reasons[mask & confidence & edge] = "regression_uncertain"
        take = mask & confidence & edge & certain
        result[take] = label
        reasons[take] = "confirmed_model_agreement"
    reasons[~valid] = "unavailable_model_or_cost"
    return result, reasons


def select_precision_policy(
    probabilities: np.ndarray,
    returns: np.ndarray,
    labels: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    config: Config,
) -> dict[str, Any]:
    """Choose per-class rules by F1 under a precision target, then confirm on later POLICY data.

    The first half selects thresholds. Target-overlap rows are purged. The later
    half confirms the locked rule using nonoverlapping anchors; a failed side is
    disabled without retrying on confirmation outcomes. TEST is never consulted.
    """
    split = labels.prediction_timestamp.iloc[len(labels) // 2]
    selection = np.flatnonzero((labels.target_timestamp < split).to_numpy())
    confirmation = np.flatnonzero((labels.prediction_timestamp >= split).to_numpy())
    selection = selection[non_overlapping_positions(labels.iloc[selection])]
    confirmation = confirmation[non_overlapping_positions(labels.iloc[confirmation])]
    # Precision arah mengikuti permintaan: harga naik/turun dari anchor keputusan.
    actual = (
        np.select([labels.future_return.gt(0), labels.future_return.lt(0)], [2, 0], default=1)
        if "future_return" in labels
        else labels.target_class.to_numpy(dtype=int)
    )
    costs = decision_costs(labels, config.slippage_bps)
    policy: dict[str, Any] = {
        "fit_block": "POLICY",
        "metric": "directional per-class F1 subject to precision and support",
        "target_precision": config.target_precision,
        "min_signals_per_stage": config.min_policy_signals,
        "require_interval_agreement": config.require_interval_agreement,
        "confirmation_start": split.isoformat(),
        "selection_rows": len(selection),
        "confirmation_rows": len(confirmation),
        "evaluation_sampling": "nonoverlapping horizons",
        "classes": {},
    }
    raw = probabilities.argmax(axis=1)
    for name, label, direction in (("SELL", 0, -1), ("BUY", 2, 1)):
        trials = []
        margin = probabilities[:, label] - np.max(np.delete(probabilities, label, axis=1), axis=1)
        bound = lower if label == 2 else -upper
        valid = np.isfinite(returns) & np.isfinite(costs) & np.isfinite(bound)
        base = (raw == label) & valid
        gate_masks = {
            "raw_class": raw == label,
            "finite_inputs": base,
            "point_beyond_cost": base & (direction * returns > costs),
            "interval_beyond_cost": base & (bound > costs),
        }
        funnel = {
            stage: {key: int(mask[positions].sum()) for key, mask in gate_masks.items()}
            for stage, positions in (("selection", selection), ("confirmation", confirmation))
        }
        if config.require_interval_agreement:
            base &= bound > costs
        for threshold in (0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98):
            for probability_margin in (0.0, 0.1, 0.2):
                for edge_multiple in (1.0, 1.5, 2.0):
                    take = (
                        base
                        & (probabilities[:, label] >= threshold)
                        & (margin >= probability_margin)
                        & (direction * returns > costs * edge_multiple)
                    )
                    predicted = np.where(take, label, 1)
                    stats = precision_statistics(actual[selection], predicted[selection], label)
                    trials.append(
                        {
                            "threshold": threshold,
                            "probability_margin": probability_margin,
                            "edge_multiple": edge_multiple,
                            **stats,
                        }
                    )
        eligible = [
            trial
            for trial in trials
            if trial["signals"] >= config.min_policy_signals
            and trial["precision"] >= config.target_precision
        ]
        chosen = (
            max(eligible, key=lambda trial: (trial["f1"], trial["signals"], trial["precision"]))
            if eligible
            else None
        )
        rule: dict[str, Any] = {
            "enabled": False,
            "threshold": chosen["threshold"] if chosen else 1.0,
            "probability_margin": chosen["probability_margin"] if chosen else 0.0,
            "edge_multiple": chosen["edge_multiple"] if chosen else 1.0,
            "selection": chosen,
            "confirmation": None,
            "trials": trials,
            "status": "no_selection_rule_meets_target",
            "gate_funnel": funnel,
        }
        if chosen:
            take = (
                base
                & (probabilities[:, label] >= rule["threshold"])
                & (margin >= rule["probability_margin"])
                & (direction * returns > costs * rule["edge_multiple"])
            )
            predicted = np.where(take, label, 1)
            stats = precision_statistics(actual[confirmation], predicted[confirmation], label)
            rule["confirmation"] = stats
            rule["enabled"] = (
                stats["signals"] >= config.min_policy_signals
                and stats["precision"] >= config.target_precision
            )
            rule["status"] = "confirmed" if rule["enabled"] else "confirmation_failed"
        policy["classes"][name] = rule
    return policy


def predict_regression(
    bundle: dict[str, Any],
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return point and empirical interval forecasts in decimal-return units."""
    names = bundle.get("regression_features", bundle["features"])
    x = features[names]
    point = bundle["regressor"].predict(x) / 10000 * bundle.get("regression_shrinkage", 1.0)
    if "quantile_regressor" in bundle:
        interval = np.sort(bundle["quantile_regressor"].predict(x) / 10000, axis=1)
        expansion = bundle["interval_expansion"]
        return point, interval[:, 0] - expansion, interval[:, 1] + expansion
    return point, point.copy(), point.copy()


def fit_development(
    blocks: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    config: Config,
    *,
    regression_blocks: dict[str, tuple[pd.DataFrame, pd.DataFrame]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit class/return models and selective decisions using only four development blocks.

    Macro F1 chooses classifier candidates; return RMSE chooses squared/robust
    regression and shrinkage. Early stopping uses TUNE, then refits TRAIN+TUNE
    at the chosen tree count. CALIBRATION fits probabilities and interval width.
    POLICY selection/confirmation can disable either direction. No TEST data
    enters any fitted component.
    """
    required = {"train", "tune", "calibration", "policy"}
    if set(blocks) != required or (
        regression_blocks is not None and set(regression_blocks) != required
    ):
        raise ValueError("Fitting hanya menerima empat development blocks, tanpa TEST")
    for name, (x, y) in blocks.items():
        if len(x) < 30 or len(x) != len(y) or y.target_class.isna().any():
            raise ValueError(f"Block {name} membutuhkan >=30 labeled rows")
    regression_blocks = regression_blocks or blocks
    for name in required:
        rx, ry = regression_blocks[name]
        if not rx.index.equals(blocks[name][0].index) or not ry.equals(blocks[name][1]):
            raise ValueError(
                "Classifier dan regressor wajib memakai observasi development yang sama"
            )
    x_train, y_train = blocks["train"]
    x_tune, y_tune = blocks["tune"]
    if set(y_train.target_class) != {0, 1, 2}:
        raise ValueError("TRAIN membutuhkan SELL, NO_TRADE, BUY")
    candidates = [
        {"max_depth": 2, "learning_rate": 0.04, "min_child_weight": 24, "reg_lambda": 12.0},
        {"max_depth": 3, "learning_rate": 0.04, "min_child_weight": 16, "reg_lambda": 8.0},
        {"max_depth": 4, "learning_rate": 0.03, "min_child_weight": 24, "reg_lambda": 12.0},
    ]
    # MODEL-01 | Macro F1 di TUNE, early stopping dan pengaruh class weighting diuji eksplisit.
    trials, best_score, best_params, best_weighting, best_trees = [], -1.0, {}, False, 1
    for params in candidates:
        for balanced in (False, True):
            model = classifier(config, **params, early_stopping_rounds=config.early_stopping_rounds)
            weights = (
                compute_sample_weight("balanced", y_train.target_class.astype(int))
                if balanced
                else None
            )
            model.fit(
                x_train,
                y_train.target_class.astype(int),
                sample_weight=weights,
                eval_set=[(x_tune, y_tune.target_class.astype(int))],
                verbose=False,
            )
            score = macro_f1(y_tune.target_class.astype(int), model.predict(x_tune))
            trees = model.best_iteration + 1
            trials.append(
                {
                    "parameters": params,
                    "balanced_weights": balanced,
                    "trees": trees,
                    "tune_macro_f1": score,
                }
            )
            if score > best_score:
                best_score, best_params, best_weighting, best_trees = score, params, balanced, trees
    x_fit, y_fit = pd.concat([x_train, x_tune]), pd.concat([y_train, y_tune])
    model = classifier(config, **best_params, n_estimators=best_trees)
    weights = (
        compute_sample_weight("balanced", y_fit.target_class.astype(int))
        if best_weighting
        else None
    )
    model.fit(x_fit, y_fit.target_class.astype(int), sample_weight=weights)
    # MODEL-02 | Probabilitas dikalibrasi terpisah dari fitting dan keputusan trading.
    x_cal, y_cal = blocks["calibration"]
    before = model.predict_proba(x_cal)
    calibrator = PlattCalibration.fit(before, y_cal.target_class.astype(int), config.random_state)
    after = calibrator.predict(before)
    targets_onehot = np.eye(3)[y_cal.target_class.to_numpy(dtype=int)]
    calibration_report = {
        "fit_block": "CALIBRATION",
        "method": "OVR sigmoid on log odds, normalized",
        "pre_calibration_logloss": float(
            log_loss(y_cal.target_class.astype(int), before, labels=[0, 1, 2])
        ),
        "post_calibration_logloss": float(
            log_loss(y_cal.target_class.astype(int), after, labels=[0, 1, 2])
        ),
        "pre_brier_multiclass": float(np.square(before - targets_onehot).sum(axis=1).mean()),
        "post_brier_multiclass": float(np.square(after - targets_onehot).sum(axis=1).mean()),
        "coefficients": [m.coef_.tolist() for m in calibrator.models],
        "intercepts": [m.intercept_.tolist() for m in calibrator.models],
    }
    # MODEL-03 | Regresi memakai fitur tersendiri dan membandingkan squared loss dengan robust loss.
    rx_train, _ = regression_blocks["train"]
    rx_tune, _ = regression_blocks["tune"]
    rx_fit = pd.concat([rx_train, rx_tune])
    regression_trials, best_reg_score, reg_params, reg_trees, shrinkage = (
        [],
        float("inf"),
        {},
        1,
        1.0,
    )
    for params in candidates:
        for objective in ("reg:squarederror", "reg:pseudohubererror"):
            chosen_params = {**params, "objective": objective}
            if objective == "reg:pseudohubererror":
                chosen_params["huber_slope"] = 10.0
            reg = regressor(
                config, **chosen_params, early_stopping_rounds=config.early_stopping_rounds
            )
            reg.fit(
                rx_train,
                y_train.future_return.to_numpy() * 10000,
                eval_set=[(rx_tune, y_tune.future_return.to_numpy() * 10000)],
                verbose=False,
            )
            raw = reg.predict(rx_tune) / 10000
            for factor in (0.1, 0.25, 0.5, 1.0):
                rmse = float(
                    np.sqrt(np.mean((raw * factor - y_tune.future_return.to_numpy()) ** 2))
                )
                regression_trials.append(
                    {
                        "parameters": chosen_params,
                        "trees": reg.best_iteration + 1,
                        "shrinkage": factor,
                        "tune_return_rmse": rmse,
                    }
                )
                if rmse < best_reg_score:
                    best_reg_score, reg_params, reg_trees, shrinkage = (
                        rmse,
                        chosen_params,
                        reg.best_iteration + 1,
                        factor,
                    )
    reg = regressor(config, **reg_params, n_estimators=reg_trees)
    reg.fit(rx_fit, y_fit.future_return.to_numpy() * 10000)
    # MODEL-04 | Quantile interval menangkap ketidakpastian; kalibrasi tidak memakai TEST.
    quantiles = [config.interval_alpha / 2, 1 - config.interval_alpha / 2]
    quantile = regressor(
        config,
        **{
            **reg_params,
            "objective": "reg:quantileerror",
            "quantile_alpha": quantiles,
            "eval_metric": "quantile",
        },
    )
    quantile.set_params(huber_slope=None, early_stopping_rounds=config.early_stopping_rounds)
    quantile.fit(
        rx_train,
        y_train.future_return.to_numpy() * 10000,
        eval_set=[(rx_tune, y_tune.future_return.to_numpy() * 10000)],
        verbose=False,
    )
    quantile_trees = quantile.best_iteration + 1
    qparams = {key: value for key, value in reg_params.items() if key != "huber_slope"}
    quantile = regressor(
        config,
        **{
            **qparams,
            "n_estimators": quantile_trees,
            "objective": "reg:quantileerror",
            "quantile_alpha": quantiles,
            "eval_metric": "quantile",
        },
    )
    quantile.fit(rx_fit, y_fit.future_return.to_numpy() * 10000)
    raw_interval = quantile.predict(regression_blocks["calibration"][0]) / 10000
    interval = np.sort(raw_interval, axis=1)
    truth = y_cal.future_return.to_numpy()
    scores = np.maximum.reduce(
        [interval[:, 0] - truth, truth - interval[:, 1], np.zeros(len(truth))]
    )
    rank = min(len(scores), int(np.ceil((len(scores) + 1) * (1 - config.interval_alpha))))
    expansion = float(np.sort(scores)[rank - 1])
    calibration_report["return_interval"] = {
        "nominal_coverage": 1 - config.interval_alpha,
        "quantiles": quantiles,
        "expansion": expansion,
        "calibration_rows": len(scores),
        "raw_quantile_crossing_rate": float((raw_interval[:, 0] > raw_interval[:, 1]).mean()),
        "empirical_calibration_coverage": float(
            ((truth >= interval[:, 0] - expansion) & (truth <= interval[:, 1] + expansion)).mean()
        ),
        "note": (
            "Empirical interval calibration; market dependence and distribution shift "
            "prevent an unconditional future coverage guarantee."
        ),
    }
    bundle = {
        "classifier": model,
        "regressor": reg,
        "quantile_regressor": quantile,
        "calibrator": calibrator,
        "features": list(x_train.columns),
        "regression_features": list(rx_train.columns),
        "regression_shrinkage": shrinkage,
        "interval_expansion": expansion,
        "interval_alpha": config.interval_alpha,
        "slippage_bps": config.slippage_bps,
    }
    # MODEL-05 | Prioritaskan precision BUY/SELL, konfirmasi rule di bagian POLICY berikutnya.
    x_policy, y_policy = blocks["policy"]
    probabilities = calibrator.predict(model.predict_proba(x_policy))
    point, lower, upper = predict_regression(bundle, regression_blocks["policy"][0])
    policy = select_precision_policy(probabilities, point, y_policy, lower, upper, config)
    bundle["selective_policy"] = policy
    audit = {
        "hyperparameters": {
            "tuning_block": "TUNE",
            "refit_blocks": ["TRAIN", "TUNE"],
            "classifier": best_params,
            "classifier_trials": trials,
            "balanced_weights": best_weighting,
            "classifier_trees": best_trees,
            "regressor": reg_params,
            "regressor_trials": regression_trials,
            "regressor_trees": reg_trees,
            "regression_shrinkage": shrinkage,
            "quantile_trees": quantile_trees,
            "tune_zero_return_rmse": float(np.sqrt(np.mean(y_tune.future_return.to_numpy() ** 2))),
            "tune_selected_return_rmse": best_reg_score,
        },
        "calibration": calibration_report,
        "policy": policy,
    }
    return bundle, audit


# %% 05. Prediksi TEST dan evaluation
# Prediksi memakai model yang sudah dibekukan. Actual hanya dipasang sesudah inference.
# Metrik mengecualikan actual pending/gap; proxy menghitung spread dan slippage konfigurasi.


def predict_test(
    bundle: dict[str, Any],
    features: pd.DataFrame,
    labels: pd.DataFrame,
    test_indices: pd.Index,
    *,
    fold: str,
    evaluation_end: pd.Timestamp,
) -> pd.DataFrame:
    """Predict from completed M5 snapshots with frozen models; attach actuals afterwards.

    Future prices do not enter inference. Outcomes whose target is outside the
    test interval are OUTSIDE_EVALUATION; unknown spread forces a non-executable NO_TRADE
    with explicit reason and excludes classification/trading evaluation.
    """
    x = features.loc[test_indices, bundle["features"]]
    probabilities = bundle["calibrator"].predict(bundle["classifier"].predict_proba(x))
    source = labels.loc[test_indices]
    returns, lower, upper = predict_regression(bundle, features.loc[test_indices])
    costs = decision_costs(source, bundle.get("slippage_bps", 0.0))
    if "selective_policy" in bundle:
        predicted, reasons = apply_selective_policy(
            probabilities,
            returns,
            costs,
            bundle["selective_policy"],
            lower,
            upper,
        )
    else:
        predicted = apply_policy(probabilities, bundle["threshold"])
        reasons = np.where(
            probabilities.max(axis=1) < bundle["threshold"], "confidence_abstain", "legacy_model"
        )
    output = source[
        [
            "prediction_timestamp",
            "target_timestamp",
            "open_time_utc",
            "open",
            "high",
            "low",
            "close",
            "spread_price",
            "spread_source",
            "threshold_return",
        ]
    ].copy()
    output["raw_predicted_signal"] = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    output["predicted_signal"] = np.asarray(LABELS)[predicted]
    output["policy_reason"] = reasons
    missing_spread = output.spread_price.isna()
    output.loc[missing_spread, "predicted_signal"] = "NO_TRADE"
    output.loc[missing_spread, "policy_reason"] = "spread_unavailable"
    output[["probability_sell", "probability_no_trade", "probability_buy"]] = probabilities
    output["confidence"] = probabilities.max(axis=1)
    output["predicted_future_return"] = returns
    output["return_interval_lower"] = lower
    output["return_interval_upper"] = upper
    output["decision_cost_return"] = costs
    output["raw_regression_disagreement"] = (
        (probabilities.argmax(axis=1) == 2) & (returns <= 0)
    ) | ((probabilities.argmax(axis=1) == 0) & (returns >= 0))
    output["predicted_future_close"] = output.close * (1 + returns)
    # Pasang ground truth hanya untuk target di dalam batas TEST.
    in_window = source.target_timestamp.lt(evaluation_end)
    output["actual_future_return"] = source.future_return.where(in_window)
    output["actual_future_close"] = source.future_close.where(in_window)
    output["actual_signal"] = source.target_class.map(dict(enumerate(LABELS))).where(in_window)
    if "actual_status" in source:
        output["actual_status"] = source.actual_status.where(in_window, "OUTSIDE_EVALUATION")
    output["spread_pct"] = output.spread_price / output.close
    known = output.actual_signal.notna()
    direction = output.predicted_signal.map({"SELL": -1, "NO_TRADE": 0, "BUY": 1})
    traded = direction.ne(0) & known
    output["realized_gross_move"] = (direction * (output.actual_future_close - output.close)).where(
        traded
    )
    output["slippage_price"] = output.close * bundle.get("slippage_bps", 0.0) / 10000
    output["realized_net_move"] = (
        output.realized_gross_move - output.spread_price - output.slippage_price
    ).where(traded)
    output["spread_multiple"] = output.realized_net_move / output.spread_price.replace(0, np.nan)
    output["fold"] = fold
    output["model_version"] = (
        "m5-direction-xgb-v3" if "selective_policy" in bundle else "m5-direct-xgb-v1"
    )
    return attach_direction_outcomes(output.reset_index(drop=True))


def evaluate(predictions: pd.DataFrame) -> dict[str, Any]:
    """Report raw/action F1, selective precision, return baselines and cost-adjusted proxy.

    Classification requires known actual class and spread; price metrics require
    known actual price only. NO_TRADE has no transaction. Pending outcomes are
    excluded. Cumulative proxy does not model account cashflow or trade overlap.
    """
    predictions = attach_direction_outcomes(predictions)
    known = predictions.dropna(subset=["actual_signal"])
    prices = predictions.dropna(subset=["actual_future_close"])
    # EVAL-01 | Klasifikasi: macro/weighted F1, balanced accuracy, dan confusion matrix.
    classification: dict[str, Any] = {
        "evaluated_rows": len(known),
        "f1_macro": None,
        "f1_weighted": None,
        "balanced_accuracy": None,
        "confusion_matrix": [[0] * 3 for _ in range(3)],
    }
    if len(known):
        report = classification_report(
            known.actual_signal,
            known.predicted_signal,
            labels=list(LABELS),
            output_dict=True,
            zero_division=0,
        )
        supported = [report[label]["recall"] for label in LABELS if report[label]["support"] > 0]
        classification.update(
            {
                "f1_macro": float(
                    f1_score(
                        known.actual_signal,
                        known.predicted_signal,
                        labels=list(LABELS),
                        average="macro",
                        zero_division=0,
                    )
                ),
                "f1_weighted": float(
                    f1_score(
                        known.actual_signal,
                        known.predicted_signal,
                        labels=list(LABELS),
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "balanced_accuracy": float(np.mean(supported)),
                "per_class": {label: report[label] for label in LABELS},
                "confusion_matrix": confusion_matrix(
                    known.actual_signal, known.predicted_signal, labels=list(LABELS)
                ).tolist(),
            }
        )
        for label in LABELS:
            if not known.predicted_signal.eq(label).any():
                classification["per_class"][label]["precision"] = None
    # EVAL-02 | Harga: MAE, MSE, RMSE, dan MAPE dari actual future close yang tersedia.
    price: dict[str, Any] = {
        "unit": "XAUUSD price units",
        "evaluated_rows": len(prices),
        "mae": None,
        "mse": None,
        "rmse": None,
        "mape_percent": None,
    }
    if len(prices):
        error = prices.predicted_future_close - prices.actual_future_close
        price.update(
            {
                "mae": float(error.abs().mean()),
                "mse": float(error.pow(2).mean()),
                "rmse": float(np.sqrt(error.pow(2).mean())),
                "mape_percent": float((error / prices.actual_future_close).abs().mean() * 100),
            }
        )
        baseline = prices.close - prices.actual_future_close
        actual_return = prices.actual_future_return.to_numpy(dtype=float)
        predicted_return = prices.predicted_future_return.to_numpy(dtype=float)
        baseline_mse = float(np.mean(actual_return**2))
        return_mse = float(np.mean((predicted_return - actual_return) ** 2))
        price.update(
            {
                "return_rmse": float(np.sqrt(return_mse)),
                "return_mae": float(np.mean(np.abs(predicted_return - actual_return))),
                "zero_return_rmse": float(np.sqrt(baseline_mse)),
                "persistence_price_rmse": float(np.sqrt(baseline.pow(2).mean())),
                "return_skill_vs_zero": 1 - return_mse / baseline_mse if baseline_mse > 0 else None,
                "direction_accuracy": float(
                    (np.sign(predicted_return) == np.sign(actual_return)).mean()
                ),
            }
        )
        if "return_interval_lower" in prices:
            price["interval_coverage"] = float(
                (
                    (prices.actual_future_return >= prices.return_interval_lower)
                    & (prices.actual_future_return <= prices.return_interval_upper)
                ).mean()
            )
            price["mean_interval_width"] = float(
                (prices.return_interval_upper - prices.return_interval_lower).mean()
            )
    # EVAL-SELECTIVE | Precision tanpa sinyal = null; abstain tidak dianggap prediksi sukses.
    directional = predictions.dropna(subset=["actual_future_close"])
    delta = directional.actual_future_close - directional.close
    actual_codes = np.select([delta.gt(0), delta.lt(0)], [2, 0], default=1)
    action_codes = directional.predicted_signal.map(
        dict(zip(LABELS, range(3), strict=True))
    ).to_numpy(dtype=int)
    selective = {
        name: precision_statistics(actual_codes, action_codes, label)
        for name, label in (("SELL", 0), ("BUY", 2))
    }
    selective["no_trade_fraction"] = float(predictions.predicted_signal.eq("NO_TRADE").mean())
    selective["definition"] = (
        "strict direction from decision close to fixed-horizon actual close; NO_TRADE neutral"
    )
    selective["outcome_counts"] = predictions.outcome_status.value_counts().to_dict()
    selective["reason_counts"] = (
        predictions.policy_reason.value_counts().to_dict() if "policy_reason" in predictions else {}
    )
    selective["regression_direction_conflicts"] = int(
        (
            (predictions.predicted_signal.eq("BUY") & predictions.predicted_future_return.le(0))
            | (predictions.predicted_signal.eq("SELL") & predictions.predicted_future_return.ge(0))
        ).sum()
    )
    raw_classification = None
    if len(known) and "raw_predicted_signal" in known:
        raw_classification = classification_report(
            known.actual_signal,
            known.raw_predicted_signal,
            labels=list(LABELS),
            output_dict=True,
            zero_division=0,
        )
    # EVAL-03 | Trading proxy: coverage, win/loss, dan pergerakan net setelah spread.
    # Ini penjumlahan sinyal; belum mensimulasikan akun, posisi overlap, atau biaya lain.
    traded = known.loc[known.predicted_signal.ne("NO_TRADE")].dropna(subset=["realized_net_move"])
    counts = predictions.predicted_signal.value_counts().reindex(LABELS, fill_value=0)
    signals = {label: int(counts[label]) for label in LABELS}
    wins, losses = (
        int(traded.realized_net_move.gt(0).sum()),
        int(traded.realized_net_move.lt(0).sum()),
    )
    # Jumlah sinyal bukan jumlah posisi akun: proxy tetap menghitung horizon overlap.
    profit_by_side = {}
    for side in ("BUY", "SELL"):
        sample = traded.loc[traded.predicted_signal.eq(side)]
        direction = 1 if side == "BUY" else -1
        gross = direction * (sample.actual_future_close - sample.close)
        profit_by_side[side] = {
            "signals": signals[side],
            "evaluated": len(sample),
            "unevaluated": signals[side] - len(sample),
            "gross_price_move": float(gross.sum()),
            "spread_cost": float(sample.spread_price.sum()),
            "slippage_cost": float((gross - sample.spread_price - sample.realized_net_move).sum()),
            "net_price_move": float(sample.realized_net_move.sum()),
            "positive_net_total": float(sample.realized_net_move.clip(lower=0).sum()),
            "negative_net_total": float(sample.realized_net_move.clip(upper=0).sum()),
        }
    return {
        "total_predictions": len(predictions),
        "pending_rows": int(predictions.actual_status.eq("PENDING").sum()),
        "unavailable_actual_rows": int(predictions.actual_future_close.isna().sum()),
        "actual_status_counts": predictions.actual_status.value_counts().to_dict(),
        "classification": classification,
        "raw_classifier": raw_classification,
        "selective": selective,
        "regression": price,
        "signals": signals,
        "average_spread": float(predictions.spread_price.mean())
        if predictions.spread_price.notna().any()
        else None,
        "trading_proxy": {
            "unit": "XAUUSD price units; not account currency profit",
            "by_side": profit_by_side,
            "gross_price_move": sum(s["gross_price_move"] for s in profit_by_side.values()),
            "spread_cost": sum(s["spread_cost"] for s in profit_by_side.values()),
            "slippage_cost": sum(s["slippage_cost"] for s in profit_by_side.values()),
            "signal_count": int(counts["SELL"] + counts["BUY"]),
            "signal_coverage": float(predictions.predicted_signal.ne("NO_TRADE").mean()),
            "evaluated_trades": len(traded),
            "wins": wins,
            "losses": losses,
            "flat": len(traded) - wins - losses,
            "win_rate": wins / len(traded) if len(traded) else None,
            "mean_net_move": float(traded.realized_net_move.mean()) if len(traded) else None,
            "median_net_move": float(traded.realized_net_move.median()) if len(traded) else None,
            "mean_move_in_spread_multiple": float(traded.spread_multiple.mean())
            if traded.spread_multiple.notna().any()
            else None,
            "cumulative_proxy_PL": float(traded.realized_net_move.sum()),
            "note": (
                "Trading proxy only — bukan simulasi akun riil. Belum memperhitungkan "
                "lot, contract size, leverage, swap, commission, stop loss, "
                "take profit, atau overlapping positions."
                " Slippage hanya berupa asumsi bps yang dikonfigurasi (default nol)."
            ),
        },
    }


def replay_single_position(
    predictions: pd.DataFrame,
    candles: pd.DataFrame,
    *,
    delay_bars: int = 1,
    slippage_bps: float = 0.0,
    commission_bps: float = 0.0,
) -> dict[str, Any]:
    """Replay at most one position using delayed OHLC bid/ask fill assumptions.

    Signals are fixed before replay. Entry uses the bar opening delay_bars*5
    minutes after decision; exit is the originally specified target close.
    Assume OHLC is bid and ask=bid+observed bar spread. Total round-trip slippage
    and commission are charged on entry bid. No intrabar stop/TP, leverage or
    cash-account simulation is implied. Missing exits remain unresolved trades,
    never silently dropped losses; overlap is rejected before checking outcomes.
    """
    if (
        delay_bars < 0
        or not np.isfinite([slippage_bps, commission_bps]).all()
        or min(slippage_bps, commission_bps) < 0
    ):
        raise ValueError("Delay/biaya replay tidak valid")
    frame = candles.copy()
    for key in ("open_time_utc", "prediction_timestamp"):
        frame[key] = pd.to_datetime(frame[key], utc=True)
    opening = frame.set_index("open_time_utc")
    closing = frame.set_index("prediction_timestamp")
    available_at = pd.Timestamp.min.tz_localize("UTC")
    records = []
    for row in predictions.sort_values("prediction_timestamp").itertuples(index=False):
        if row.predicted_signal == "NO_TRADE":
            continue
        decision, target = (
            pd.Timestamp(row.prediction_timestamp),
            pd.Timestamp(row.target_timestamp),
        )
        item: dict[str, Any] = {
            "decision": decision.isoformat(),
            "target": target.isoformat(),
            "signal": row.predicted_signal,
        }
        entry_time = decision + pd.Timedelta(minutes=5 * delay_bars)
        if decision < available_at:
            item["status"] = "SKIPPED_OVERLAP"
        elif entry_time >= target:
            item["status"] = "DELAY_EXCEEDS_HORIZON"
        elif entry_time not in opening.index or pd.isna(opening.loc[entry_time, "spread_price"]):
            item["status"] = "ENTRY_UNAVAILABLE"
        else:
            available_at = target
            entry = opening.loc[entry_time]
            buy = row.predicted_signal == "BUY"
            entry_price = float(entry.open + (entry.spread_price if buy else 0))
            item.update({"entry_time": entry_time.isoformat(), "entry_price": entry_price})
            if target not in closing.index or pd.isna(closing.loc[target, "spread_price"]):
                item["status"] = "EXIT_UNAVAILABLE"
            else:
                exit_row = closing.loc[target]
                exit_price = float(exit_row.close + (0 if buy else exit_row.spread_price))
                gross = (1 if buy else -1) * float(exit_row.close - entry.open)
                spread = float(entry.spread_price if buy else exit_row.spread_price)
                slippage = float(entry.open) * slippage_bps / 10000
                commission = float(entry.open) * commission_bps / 10000
                net = gross - spread - slippage - commission
                item.update(
                    {
                        "status": "CLOSED",
                        "exit_price": exit_price,
                        "entry_bid": float(entry.open),
                        "exit_bid": float(exit_row.close),
                        "gross_price_move": gross,
                        "spread_cost": spread,
                        "slippage_cost": slippage,
                        "commission_cost": commission,
                        "net_price_move": net,
                    }
                )
        records.append(item)
    closed = [row for row in records if row["status"] == "CLOSED"]
    return {
        "assumptions": (
            "OHLC bid; ask=bid+bar spread; delayed entry; fixed target exit; "
            "one position; no stops/TP/account leverage"
        ),
        "delay_bars": delay_bars,
        "round_trip_slippage_bps": slippage_bps,
        "round_trip_commission_bps": commission_bps,
        "closed_trades": len(closed),
        "unresolved_trades": sum(r["status"] == "EXIT_UNAVAILABLE" for r in records),
        "win_rate": sum(r["net_price_move"] > 0 for r in closed) / len(closed) if closed else None,
        "net_price_move": sum(r["net_price_move"] for r in closed),
        "gross_price_move": sum(r["gross_price_move"] for r in closed),
        "spread_cost": sum(r["spread_cost"] for r in closed),
        "slippage_cost": sum(r["slippage_cost"] for r in closed),
        "commission_cost": sum(r["commission_cost"] for r in closed),
        "unit": "XAUUSD price units; not account currency profit",
        "by_side": {
            side: {
                "closed_trades": sum(r["signal"] == side for r in closed),
                **{
                    key: sum(r[key] for r in closed if r["signal"] == side)
                    for key in (
                        "gross_price_move",
                        "spread_cost",
                        "slippage_cost",
                        "commission_cost",
                        "net_price_move",
                    )
                },
            }
            for side in ("BUY", "SELL")
        },
        "records": records,
    }


def write_evaluation_diagnostics(predictions: pd.DataFrame, destination: Path) -> None:
    """Save monthly metrics and a static regime comparison without fitting any model.

    Monthly slices expose return-baseline skill, raw/action F1 and abstention.
    A separate nonoverlapping-horizon evaluation reduces repeated outcomes;
    it is still a diagnostic and does not simulate account execution or returns.
    """
    destination.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(predictions.prediction_timestamp, utc=True)
    rows = []
    for month, sample in predictions.groupby(dates.dt.strftime("%Y-%m"), sort=True):
        metrics = evaluate(sample)
        rows.append(
            {
                "month": month,
                "decisions": len(sample),
                "evaluated_rows": metrics["classification"]["evaluated_rows"],
                "return_rmse_bps": metrics["regression"].get("return_rmse", np.nan) * 10000,
                "zero_return_rmse_bps": metrics["regression"].get("zero_return_rmse", np.nan)
                * 10000,
                "return_skill": metrics["regression"].get("return_skill_vs_zero"),
                "raw_macro_f1": (metrics["raw_classifier"] or {})
                .get("macro avg", {})
                .get("f1-score"),
                "action_macro_f1": metrics["classification"]["f1_macro"],
                "coverage": metrics["trading_proxy"]["signal_coverage"],
                "interval_coverage": metrics["regression"].get("interval_coverage"),
                **{
                    f"{side.lower()}_{key}": metrics["selective"][side][key]
                    for side in ("BUY", "SELL")
                    for key in ("precision", "recall", "f1", "signals")
                },
            }
        )
    monthly = pd.DataFrame(rows)
    monthly.to_csv(destination / "monthly_metrics.csv", index=False)
    anchors = non_overlapping_positions(predictions)
    write_json(
        destination / "nonoverlapping_metrics.json",
        {
            "sampling": "fixed chronological anchors with nonoverlapping target horizons",
            "metrics": evaluate(predictions.iloc[anchors]),
        },
    )
    if monthly.empty:
        return
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for column, label in (("return_rmse_bps", "XGBoost"), ("zero_return_rmse_bps", "Zero return")):
        axes[0, 0].plot(
            monthly.month,
            monthly[column],
            marker="o",
            linestyle="--" if label == "Zero return" else "-",
            label=label,
        )
    axes[0, 0].set(title="Return RMSE: lower is better", ylabel="Basis points")
    for column, label in (
        ("raw_macro_f1", "Raw classifier"),
        ("action_macro_f1", "After abstention"),
    ):
        axes[0, 1].plot(monthly.month, monthly[column], marker="o", label=label)
    axes[0, 1].set(title="Three-class macro F1", ylim=(0, 1))
    for side in ("buy", "sell"):
        axes[1, 0].plot(monthly.month, monthly[f"{side}_signals"], marker="o", label=side.upper())
    axes[1, 0].set(title="Accepted signals with known outcomes", ylabel="Count", ylim=(0, None))
    if monthly[["buy_signals", "sell_signals"]].to_numpy().max() == 0:
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_yticks([0, 1])
        axes[1, 0].text(
            0.5, 0.5, "No BUY/SELL passed validation", ha="center", transform=axes[1, 0].transAxes
        )
    axes[1, 1].plot(
        monthly.month, monthly.return_skill * 100, marker="o", label="MSE skill vs zero"
    )
    axes[1, 1].axhline(0, color="gray", linestyle="--")
    axes[1, 1].set(title="Return skill: positive beats baseline", ylabel="Percent")
    for axis in axes.flat:
        axis.tick_params(axis="x", rotation=45)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Frozen XAUUSD M5 model — retrospective monthly diagnostics")
    figure.savefig(destination / "monthly_diagnostics.png", dpi=150)
    plt.close(figure)


# %% 06. Pipeline walk-forward, artifacts, dan dashboard
# Orkestrasi menghubungkan preparation -> features -> selection -> fitting -> evaluation.
# Model, audit, prediksi, dan dashboard disimpan lokal secara terpisah untuk setiap fold.

LOG = logging.getLogger("technical")


def write_json(path: Path, payload: Any) -> None:
    """Persist strict UTF-8 JSON, refusing non-finite metrics or hidden NaN values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def run_fold(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    manifest: list[dict[str, Any]],
    fold: Fold,
    config: Config,
    destination: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Execute isolated selection, development fitting and frozen TEST for one fold.

    Fit interfaces accept development blocks only. Persist models, selections,
    calibration/policy/tuning audits, outcome CSV and confusion matrix per fold.
    Returns: predictions, performance metadata and dashboard feature audit.
    """
    # FOLD-01 | Bagi candle ke TRAIN/TUNE/CALIBRATION/POLICY/TEST dan purge label batas.
    indices = fold.split(labels)
    for name, rows in indices.items():
        LOG.info("[%s] %s %s rows", fold.name, name.upper(), len(rows))
        if len(rows) < (30 if name != "test" else 1):
            raise ValueError(f"{fold.name}: block {name} tidak cukup data ({len(rows)})")
    destination.mkdir(parents=True, exist_ok=True)
    LOG.info(
        "[%s FEATURE SELECTION] %s candidates → Quality → ReliefF → RFECV",
        fold.name,
        len(features.columns),
    )
    selected, quality, ranking, rfecv = select_features(features, labels, indices["train"], config)
    LOG.info(
        "[%s FEATURE SELECTION] quality=%s ReliefF=%s selected=%s",
        fold.name,
        quality["retained_features"],
        quality["relieff_top_k"],
        len(selected),
    )
    write_json(destination / "quality_filter_report.json", quality)
    write_json(destination / "selected_features.json", selected)
    write_json(destination / "feature_manifest.json", manifest)
    ranking.to_csv(destination / "relieff_ranking.csv", index=False)
    rfecv.to_csv(destination / "rfecv_results.csv", index=False)
    # FOLD-02 | Kirim hanya empat blok development ke proses fitting, tanpa TEST.
    development = {
        name: (features.loc[rows, selected], labels.loc[rows])
        for name, rows in indices.items()
        if name != "test"
    }
    regression_selected, regression_selection = select_regression_features(
        features,
        labels,
        indices["train"],
        config,
    )
    write_json(destination / "regression_feature_selection.json", regression_selection)
    regression_development = {
        name: (features.loc[rows, regression_selected], labels.loc[rows])
        for name, rows in indices.items()
        if name != "test"
    }
    LOG.info("[%s MODEL] Tune classifier/regressor; refit TRAIN+TUNE; calibrate; policy", fold.name)
    bundle, audit = fit_development(development, config, regression_blocks=regression_development)
    LOG.info(
        "[%s MODEL] Best classifier %s; direction status %s",
        fold.name,
        audit["hyperparameters"]["classifier"],
        {name: rule["status"] for name, rule in audit["policy"]["classes"].items()},
    )
    # Save frozen state before any TEST outcome evaluation.
    joblib.dump(
        {
            key: value
            for key, value in bundle.items()
            if key not in {"regressor", "quantile_regressor"}
        },
        destination / "model_classifier.joblib",
    )
    joblib.dump(
        {
            key: bundle[key]
            for key in (
                "regressor",
                "quantile_regressor",
                "regression_features",
                "regression_shrinkage",
                "interval_expansion",
                "interval_alpha",
            )
        }
        | {"features": regression_selected, "training_target_scale": 10000},
        destination / "model_regressor.joblib",
    )
    write_json(
        destination / "inference_contract.json",
        {
            "schema": "direction-v3",
            "fold": fold.name,
            "config": asdict(config),
            "classifier_features": selected,
            "regression_features": regression_selected,
            "history_start": labels.open_time_utc.min().isoformat(),
            "decision_not_before": fold.boundaries[-2].isoformat(),
            "identity": {
                key: str(labels[key].iloc[0])
                for key in ("source_key", "actual_symbol")
                if key in labels
            },
            "sha256": {
                name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
                for name in ("model_classifier.joblib", "model_regressor.joblib")
            },
        },
    )
    for name, report in audit.items():
        write_json(destination / f"{name}.json", report)
    # FOLD-03 | Jalankan inference frozen pada TEST, lalu hitung metrik actual tersedia.
    predictions = predict_test(
        bundle,
        features,
        labels,
        indices["test"],
        fold=fold.name,
        evaluation_end=fold.boundaries[-1],
    )
    metrics = evaluate(predictions)
    replay = replay_single_position(predictions, labels, slippage_bps=config.slippage_bps)
    write_json(destination / "execution_replay.json", replay)
    names = ("train", "tune", "calibration", "policy", "test")
    ranges = {
        name: f"{fold.boundaries[i].date()} ≤ t < {fold.boundaries[i + 1].date()}"
        for i, name in enumerate(names)
    }
    counts = {name: len(rows) for name, rows in indices.items()}
    summary = {
        "fold": fold.name,
        "status": "completed",
        "ranges": ranges,
        "rows": counts,
        "number_features": len(selected),
        "metrics": metrics,
    }
    write_json(destination / "metrics.json", summary)
    predictions.to_csv(destination / "predictions.csv", index=False)
    write_evaluation_diagnostics(predictions, destination)
    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        np.asarray(metrics["classification"]["confusion_matrix"]), display_labels=LABELS
    ).plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title(f"{fold.name} · Out-of-sample")
    figure.tight_layout()
    figure.savefig(destination / "confusion_matrix.png", dpi=130)
    plt.close(figure)
    feature_audit = {
        "fold": fold.name,
        "candidate_count": len(features.columns),
        "quality": quality,
        "selected": selected,
        "regression_selection": regression_selection,
        "ranking": ranking.to_dict(orient="records"),
        "importance": [
            {"feature": name, "importance": float(value)}
            for name, value in zip(selected, bundle["classifier"].feature_importances_, strict=True)
        ],
        **audit,
    }
    LOG.info(
        "[%s EVALUATION] Macro F1=%s RMSE=%s coverage=%.3f",
        fold.name,
        metrics["classification"]["f1_macro"],
        metrics["regression"]["rmse"],
        metrics["trading_proxy"]["signal_coverage"],
    )
    return predictions, summary, feature_audit


def run_pipeline(
    source_key: str,
    symbol: str = "XAUUSD",
    *,
    start: str = "2016-01-01T00:00:00Z",
    evaluation_end: str = "2026-10-01T00:00:00Z",
    config: Config | None = None,
    output_dir: Path = ROOT / "artifacts" / "direction_v3",
    frame: pd.DataFrame | None = None,
    folds: list[Fold] | None = None,
    fold_names: list[str] | None = None,
) -> Path:
    """Run all seven research folds on PostgreSQL data (or injected test fixture).

    Returns: final offline dashboard path. Fails explicitly on missing blocks;
    successful folds are checkpointed individually for audit and later inspection.
    Actual evaluation ends at available data; no future actual is fabricated.
    """
    active = config or Config()
    active_folds = (
        folds
        if folds is not None
        else make_folds(
            start,
            evaluation_end,
            tune_start=active.final_tune_start,
            calibration_start=active.final_calibration_start,
            policy_start=active.final_policy_start,
            test_start=active.final_test_start,
        )
    )
    if fold_names is not None:
        available = {fold.name for fold in active_folds}
        if not set(fold_names) <= available or "final_2026" not in fold_names:
            raise ValueError("Pilih fold yang tersedia dan sertakan final_2026")
        active_folds = [fold for fold in active_folds if fold.name in fold_names]
    output_dir = Path(output_dir)
    LOG.info("[DATA] Loading canonical M5 %s %s", source_key, symbol)
    # PIPE-01 | Ambil data M5 dan bersihkan candle; frame dapat diinjeksi untuk pengujian.
    raw = frame if frame is not None else load_data(source_key, symbol, start, evaluation_end)
    prepared, quality = prepare_data(raw)
    LOG.info(
        "[DATA] Loaded=%s validated=%s gaps=%s spread_missing=%s",
        len(raw),
        len(prepared),
        quality["detected_gaps"],
        quality["spread_missing"],
    )
    write_json(output_dir / "data_quality.json", quality)
    write_json(output_dir / "config.json", asdict(active))
    LOG.info("[FEATURES] Building causal candidate pool")
    # PIPE-02 | Buat kandidat input X dan target y melalui dua fungsi yang terpisah.
    features, manifest = build_features(prepared, active)
    labels = make_labels(
        prepared, active.horizon_bars, active.minimum_edge_bps, active.slippage_bps
    )
    write_json(output_dir / "feature_manifest.json", manifest)
    summaries, audits, by_fold = [], [], []
    final = None
    # PIPE-03 | Ulangi selection/fitting/evaluation secara independen untuk setiap fold.
    for fold in active_folds:
        predictions, summary, audit = run_fold(
            features, labels, manifest, fold, active, output_dir / fold.name
        )
        summaries.append(summary)
        audits.append(audit)
        by_fold.extend({"fold": fold.name, "feature": name} for name in audit["selected"])
        write_json(output_dir / "walk_forward_metrics.json", summaries)
        write_json(output_dir / "feature_audits.json", audits)
        if fold.name == "final_2026":
            final = predictions
    pd.DataFrame(by_fold).to_csv(output_dir / "feature_selection_by_fold.csv", index=False)
    # PIPE-04 | Ringkas seberapa sering setiap fitur dipilih pada seluruh fold.
    stability = pd.DataFrame(by_fold).groupby("feature").size().sort_values(ascending=False)
    stability.rename("fold_count").to_csv(output_dir / "feature_stability.csv")
    flattened = []
    for summary in summaries:
        m = summary["metrics"]
        flattened.append(
            {
                "fold": summary["fold"],
                "number_features": summary["number_features"],
                **{
                    k: v
                    for k, v in m["classification"].items()
                    if k in ("f1_macro", "f1_weighted", "balanced_accuracy")
                },
                "price_rmse": m["regression"]["rmse"],
                "win_rate": m["trading_proxy"]["win_rate"],
                "signal_coverage": m["trading_proxy"]["signal_coverage"],
            }
        )
    pd.DataFrame(flattened).to_csv(output_dir / "walk_forward_metrics.csv", index=False)
    if final is None:
        raise ValueError("Final 2026 fold wajib untuk dashboard penelitian")
    metrics = {
        **evaluate(final),
        "execution_replay": json.loads(
            (output_dir / "final_2026" / "execution_replay.json").read_text(encoding="utf-8")
        ),
        "data_quality": quality,
        "metadata": {
            "source_key": source_key,
            "symbol": symbol,
            "base_timeframe": "M5",
            "horizon_bars": active.horizon_bars,
            "requested_start": start,
            "evaluation_end_exclusive": evaluation_end,
            "latest_available_actual": prepared.prediction_timestamp.max().isoformat(),
            "actual_eval_end": min(
                parse_utc(evaluation_end), prepared.prediction_timestamp.max()
            ).isoformat(),
            "development_cutoff_exclusive": next(
                fold.boundaries[-2].isoformat()
                for fold in active_folds
                if fold.name == "final_2026"
            ),
            "inference_protocol": (
                "frozen models; actual closed candles update features at every decision"
            ),
            "config": asdict(active),
        },
    }
    write_json(output_dir / "final_2026" / "dashboard_metrics.json", metrics)
    final_candles = prepared.loc[
        (prepared.prediction_timestamp >= final.prediction_timestamp.min())
        & (prepared.prediction_timestamp < parse_utc(evaluation_end))
    ]
    # PIPE-05 | Gabungkan prediksi final, candle, metrik, dan audit ke dashboard offline.
    path = write_dashboard(
        final,
        final_candles,
        metrics,
        summaries,
        audits,
        output_dir / "final_2026" / "dashboard_2026.html",
        rolling_window=active.rolling_window,
    )
    LOG.info("[COMPLETE] %s", path)
    return path


# %% 07. CLI: menjalankan seluruh pipeline


def main() -> None:
    """Expose temporal development boundaries, model and selective precision settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["walk-forward-m5"], default="walk-forward-m5")
    parser.add_argument("--source-key", required=True)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--start", default="2016-01-01T00:00:00Z")
    parser.add_argument("--evaluation-end", default="2026-10-01T00:00:00Z")
    for name in (
        "final-tune-start",
        "final-calibration-start",
        "final-policy-start",
        "final-test-start",
    ):
        parser.add_argument(f"--{name}", default=getattr(Config(), name.replace("-", "_")))
    parser.add_argument("--folds", nargs="+", help="Misalnya fold_2024 fold_2025 final_2026")
    for name, default in (
        ("horizon-bars", 3),
        ("relieff-top-k", 40),
        ("rfecv-min-features", 10),
        ("relieff-max-samples", 10000),
        ("relieff-anchors", 512),
        ("rfecv-step", 5),
        ("cv-splits", 3),
        ("n-estimators", 160),
        ("n-jobs", 4),
        ("random-state", 42),
        ("rolling-window", 288),
        ("early-stopping-rounds", 20),
        ("regression-top-k", 40),
        ("min-policy-signals", 50),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--minimum-edge-bps", type=float, default=1.0)
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--interval-alpha", type=float, default=0.2)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument(
        "--require-interval-agreement", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "direction_v3")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    fields = set(Config.__dataclass_fields__) & set(vars(args))
    config = Config(**{name: getattr(args, name) for name in fields})
    run_pipeline(
        args.source_key,
        args.symbol,
        start=args.start,
        evaluation_end=args.evaluation_end,
        config=config,
        output_dir=args.output_dir,
        fold_names=args.folds,
    )


if __name__ == "__main__":
    # Use the importable module so joblib stores a stable PlattCalibration path.
    from technical.xgboost_model import main as cli_main

    cli_main()
