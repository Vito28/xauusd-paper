"""Feature engineering teknikal untuk candle M5 XAUUSD.

Aturan utama:
  1. Fitur pada baris t hanya memakai candle t dan candle sebelumnya
     (rolling, ewm, dan shift positif). Tidak ada shift negatif di file ini.
  2. Sebagian besar fitur dinormalisasi dengan ATR atau harga, supaya nilai
     tahun 2016 (emas sekitar 1.100 USD) dan 2026 (emas jauh lebih mahal)
     dapat dibandingkan oleh model.
  3. Nilai yang belum bisa dihitung (warm-up indikator) tetap NaN.
     XGBoost menangani NaN secara native; Logistic Regression memakai imputasi
     median yang dihitung dari data training saja (lihat models.py).

Penjelasan rumus dan alasan setiap fitur ada di SOURCE_OF_TRUTH.md bagian D.
"""

import numpy as np
import pandas as pd

from technical.config import ALL_FEATURE_GROUPS

ATR_PERIOD = 14
SESSIONS = {
    # nama: (timezone IANA, jam buka lokal, jam tutup lokal)
    "session_asia": ("Asia/Tokyo", 9, 18),
    "session_london": ("Europe/London", 8, 17),
    "session_newyork": ("America/New_York", 8, 17),
}


# ---------------------------------------------------------------------------
# Indikator dasar
# ---------------------------------------------------------------------------


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Pembagian yang mengubah pembagi nol atau hasil tak hingga menjadi NaN."""
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average dengan alpha = 2 / (span + 1)."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def wilder_average(series: pd.Series, period: int) -> pd.Series:
    """Rata-rata Wilder (RMA): EMA dengan alpha = 1 / period."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """TR = max(high - low, |high - close_sebelumnya|, |low - close_sebelumnya|)."""
    previous_close = close.shift(1)
    range_now = high - low
    range_up = (high - previous_close).abs()
    range_down = (low - previous_close).abs()
    combined = pd.concat([range_now, range_up, range_down], axis=1)
    return combined.max(axis=1)


def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """ATR Wilder."""
    return wilder_average(true_range(high, low, close), period)


def relative_strength_index(close: pd.Series, period: int) -> pd.Series:
    """RSI Wilder: 100 - 100 / (1 + RS), RS = rata-rata gain / rata-rata loss."""
    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    average_gain = wilder_average(gain, period)
    average_loss = wilder_average(loss, period)
    relative_strength = safe_divide(average_gain, average_loss)
    rsi = 100 - 100 / (1 + relative_strength)
    # Jika tidak ada loss sama sekali, RSI = 100 (semua gerakan naik).
    no_loss = (average_loss == 0) & (average_gain > 0)
    rsi = rsi.mask(no_loss, 100.0)
    # Jika harga datar total, RSI didefinisikan netral 50.
    flat = (average_loss == 0) & (average_gain == 0)
    rsi = rsi.mask(flat, 50.0)
    return rsi


def directional_index(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Hitung +DI, -DI, dan ADX menurut definisi Wilder."""
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr = average_true_range(high, low, close, period)
    plus_di = 100 * safe_divide(wilder_average(plus_dm, period), atr)
    minus_di = 100 * safe_divide(wilder_average(minus_dm, period), atr)

    dx = 100 * safe_divide((plus_di - minus_di).abs(), plus_di + minus_di)
    adx = wilder_average(dx, period)
    return plus_di, minus_di, adx


# ---------------------------------------------------------------------------
# Kelompok fitur
# ---------------------------------------------------------------------------


def price_action_features(data: pd.DataFrame, atr: pd.Series) -> dict[str, pd.Series]:
    close = data["close"]
    open_ = data["open"]
    high = data["high"]
    low = data["low"]

    features = {}
    for bars in [1, 3, 6, 12]:
        features[f"log_return_{bars}"] = np.log(close / close.shift(bars))
    for bars in [6, 24]:
        features[f"return_{bars}_atr"] = safe_divide(close - close.shift(bars), atr)

    candle_range = high - low
    body_top = pd.concat([open_, close], axis=1).max(axis=1)
    body_bottom = pd.concat([open_, close], axis=1).min(axis=1)

    features["candle_body_atr"] = safe_divide(close - open_, atr)
    features["candle_range_atr"] = safe_divide(candle_range, atr)
    features["upper_wick_ratio"] = safe_divide(high - body_top, candle_range)
    features["lower_wick_ratio"] = safe_divide(body_bottom - low, candle_range)
    features["close_position_in_range"] = safe_divide(close - low, candle_range)
    return features


def trend_features(data: pd.DataFrame, atr: pd.Series) -> dict[str, pd.Series]:
    close = data["close"]
    ema_12 = ema(close, 12)
    ema_48 = ema(close, 48)
    ema_288 = ema(close, 288)  # 288 x 5 menit = 24 jam

    plus_di, minus_di, adx = directional_index(data["high"], data["low"], close, ATR_PERIOD)

    features = {}
    features["close_vs_ema_12_atr"] = safe_divide(close - ema_12, atr)
    features["close_vs_ema_48_atr"] = safe_divide(close - ema_48, atr)
    features["close_vs_ema_288_atr"] = safe_divide(close - ema_288, atr)
    features["ema_12_vs_48_atr"] = safe_divide(ema_12 - ema_48, atr)
    features["ema_48_slope_atr"] = safe_divide(ema_48 - ema_48.shift(6), atr)
    features["adx_14"] = adx
    features["di_difference_14"] = plus_di - minus_di
    return features


def momentum_features(data: pd.DataFrame, atr: pd.Series) -> dict[str, pd.Series]:
    close = data["close"]
    high = data["high"]
    low = data["low"]

    rsi = relative_strength_index(close, 14)
    macd_line = ema(close, 12) - ema(close, 26)
    macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()

    lowest_14 = low.rolling(14).min()
    highest_14 = high.rolling(14).max()
    lowest_48 = low.rolling(48).min()
    highest_48 = high.rolling(48).max()

    features = {}
    features["rsi_14"] = rsi
    features["rsi_14_change_3"] = rsi - rsi.shift(3)
    features["macd_histogram_atr"] = safe_divide(macd_line - macd_signal, atr)
    features["stochastic_k_14"] = 100 * safe_divide(close - lowest_14, highest_14 - lowest_14)
    features["donchian_position_48"] = safe_divide(close - lowest_48, highest_48 - lowest_48)
    return features


def volatility_features(data: pd.DataFrame, atr: pd.Series) -> dict[str, pd.Series]:
    close = data["close"]
    log_return = np.log(close / close.shift(1))

    middle_band = close.rolling(20).mean()
    band_std = close.rolling(20).std()
    lower_band = middle_band - 2 * band_std
    upper_band = middle_band + 2 * band_std

    features = {}
    features["atr_14_pct"] = safe_divide(atr, close)
    features["atr_14_vs_daily_mean"] = safe_divide(atr, atr.rolling(288).mean())
    features["realized_vol_12"] = log_return.rolling(12).std()
    features["realized_vol_48"] = log_return.rolling(48).std()
    features["bollinger_width_20"] = safe_divide(upper_band - lower_band, middle_band)
    features["bollinger_position_20"] = safe_divide(close - lower_band, upper_band - lower_band)
    return features


def volume_features(data: pd.DataFrame) -> dict[str, pd.Series]:
    # tick_volume = jumlah perubahan harga (tick) di candle, bukan volume kontrak.
    log_volume = np.log1p(data["tick_volume"])
    mean_48 = log_volume.rolling(48).mean()
    std_48 = log_volume.rolling(48).std()

    features = {}
    features["tick_volume_z_48"] = safe_divide(log_volume - mean_48, std_48)
    features["tick_volume_ratio_12_288"] = safe_divide(
        data["tick_volume"].rolling(12).mean(), data["tick_volume"].rolling(288).mean()
    )
    return features


def spread_features(data: pd.DataFrame, atr: pd.Series) -> dict[str, pd.Series]:
    spread = data["spread"]
    mean_288 = spread.rolling(288).mean()
    std_288 = spread.rolling(288).std()

    features = {}
    features["spread_atr"] = safe_divide(spread, atr)
    features["spread_z_288"] = safe_divide(spread - mean_288, std_288)
    return features


def time_features(data: pd.DataFrame) -> dict[str, pd.Series]:
    decision_time = data["prediction_time"]
    hour = decision_time.dt.hour + decision_time.dt.minute / 60.0

    features = {}
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["day_of_week"] = decision_time.dt.dayofweek.astype(float)

    for name, (timezone, open_hour, close_hour) in SESSIONS.items():
        local_hour = decision_time.dt.tz_convert(timezone).dt.hour
        inside = (local_hour >= open_hour) & (local_hour < close_hour)
        features[name] = inside.astype(float)
    features["session_london_newyork_overlap"] = (
        features["session_london"] * features["session_newyork"]
    )

    # Jumlah candle sejak gap terakhir (misalnya pembukaan pasar setelah akhir pekan).
    gap_group = data["gap_before"].cumsum()
    bars_since_gap = data.groupby(gap_group).cumcount()
    features["bars_since_gap"] = bars_since_gap.clip(upper=288).astype(float)
    return features


# ---------------------------------------------------------------------------
# Fungsi utama
# ---------------------------------------------------------------------------


def build_features(data: pd.DataFrame, groups: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Bangun matriks fitur untuk kelompok yang dipilih.

    Mengembalikan:
      features      : DataFrame float32 dengan indeks sama seperti data.
      feature_group : dict nama_fitur -> nama kelompok (untuk ablation dan dokumentasi).
    """
    for group in groups:
        if group not in ALL_FEATURE_GROUPS:
            raise ValueError(f"Kelompok fitur tidak dikenal: {group}")

    atr = average_true_range(data["high"], data["low"], data["close"], ATR_PERIOD)

    builders = {
        "price_action": lambda: price_action_features(data, atr),
        "trend": lambda: trend_features(data, atr),
        "momentum": lambda: momentum_features(data, atr),
        "volatility": lambda: volatility_features(data, atr),
        "volume": lambda: volume_features(data),
        "spread": lambda: spread_features(data, atr),
        "time": lambda: time_features(data),
    }

    columns = {}
    feature_group = {}
    for group in groups:
        group_features = builders[group]()
        for name, values in group_features.items():
            columns[name] = values
            feature_group[name] = group

    features = pd.DataFrame(columns, index=data.index)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.astype("float32")
    return features, feature_group
