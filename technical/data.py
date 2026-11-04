"""Membaca candle M5 dari PostgreSQL (read-only) atau CSV, lalu memvalidasinya.

Modul ini tidak mengambil data dari broker. Pengambilan data tetap dilakukan
oleh collector di folder market/. Di sini hanya ada pembacaan dan validasi.
Harga tidak pernah diisi (fill) atau diinterpolasi.
"""

import numpy as np
import pandas as pd

M5_MINUTES = 5

REQUIRED_COLUMNS = ["open_time_utc", "open", "high", "low", "close"]

M5_QUERY = """
SELECT source_key, dataset_symbol, actual_symbol, timeframe,
       open_time_utc, close_time_utc, open, high, low, close,
       tick_volume, native_spread_price, spread_mean
FROM market.processed_m5
WHERE source_key = :source_key
  AND (dataset_symbol = :symbol OR actual_symbol = :symbol)
  AND timeframe = 'M5'
  AND open_time_utc >= :start
  AND close_time_utc <= :end
ORDER BY open_time_utc, bar_observation_id
"""


def load_from_postgres(source_key: str, symbol: str, start: str, end: str) -> pd.DataFrame:
    """Baca candle M5 untuk satu sumber dan simbol dalam transaksi read-only."""
    # Import di dalam fungsi agar unit test tidak membutuhkan koneksi database.
    from analytics.data import read_postgres

    parameters = {
        "source_key": source_key,
        "symbol": symbol,
        "start": pd.Timestamp(start).to_pydatetime(),
        "end": pd.Timestamp(end).to_pydatetime(),
    }
    frame = read_postgres(M5_QUERY, parameters, ["open_time_utc", "close_time_utc"])
    if frame.empty:
        raise ValueError("Tidak ada candle M5 untuk source/symbol/rentang tersebut")
    return frame


def load_from_csv(path: str) -> pd.DataFrame:
    """Baca hasil ekspor market.processed_m5 dalam format CSV."""
    frame = pd.read_csv(path)
    frame["open_time_utc"] = pd.to_datetime(frame["open_time_utc"], utc=True)
    if "close_time_utc" in frame.columns:
        frame["close_time_utc"] = pd.to_datetime(frame["close_time_utc"], utc=True)
    return frame


def prepare_candles(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validasi OHLC, pilih spread, tandai gap, dan kembalikan laporan kualitas data.

    Kolom hasil:
      prediction_time : waktu keputusan = waktu close candle (fitur hanya boleh
                        memakai informasi sampai titik ini).
      spread          : spread dalam satuan harga (native bar, atau rata-rata tick
                        pada candle yang sama jika native tidak tersedia).
      gap_before      : True jika candle sebelumnya bukan tepat 5 menit sebelumnya.
    """
    for column in REQUIRED_COLUMNS:
        if column not in raw.columns:
            raise ValueError(f"Kolom wajib tidak ada: {column}")

    data = raw.copy()
    data["open_time_utc"] = pd.to_datetime(data["open_time_utc"], utc=True)
    if "close_time_utc" in data.columns:
        data["close_time_utc"] = pd.to_datetime(data["close_time_utc"], utc=True)
    else:
        data["close_time_utc"] = data["open_time_utc"] + pd.Timedelta(minutes=M5_MINUTES)

    for column in [
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "native_spread_price",
        "spread_mean",
    ]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    # Pilih spread: native bar lebih dulu, lalu rata-rata spread tick di candle yang sama.
    spread = pd.Series(np.nan, index=data.index)
    if "native_spread_price" in data.columns:
        spread = data["native_spread_price"].where(data["native_spread_price"] >= 0)
    if "spread_mean" in data.columns:
        tick_spread = data["spread_mean"].where(data["spread_mean"] >= 0)
        spread = spread.fillna(tick_spread)
    if "spread" in data.columns:
        spread = spread.fillna(pd.to_numeric(data["spread"], errors="coerce"))
    data["spread"] = spread

    if "tick_volume" in data.columns:
        data["tick_volume"] = data["tick_volume"].where(data["tick_volume"] >= 0)
    else:
        data["tick_volume"] = np.nan

    total_rows = len(data)

    # Validasi OHLC: semua positif, high tertinggi, low terendah.
    prices = data[["open", "high", "low", "close"]]
    all_present = prices.notna().all(axis=1)
    all_positive = (prices > 0).all(axis=1)
    high_ok = data["high"] >= prices.max(axis=1)
    low_ok = data["low"] <= prices.min(axis=1)
    valid_ohlc = all_present & all_positive & high_ok & low_ok

    # Validasi waktu: durasi tepat 5 menit dan open time berada di grid 5 menit.
    duration = data["close_time_utc"] - data["open_time_utc"]
    on_grid = (data["open_time_utc"].dt.minute % M5_MINUTES == 0) & (
        data["open_time_utc"].dt.second == 0
    )
    valid_time = (duration == pd.Timedelta(minutes=M5_MINUTES)) & on_grid

    data = data[valid_ohlc & valid_time]
    data = data.sort_values("open_time_utc", kind="stable")
    duplicate_count = int(data["open_time_utc"].duplicated(keep="last").sum())
    data = data.drop_duplicates("open_time_utc", keep="last")
    data = data.reset_index(drop=True)
    if data.empty:
        raise ValueError("Tidak ada candle M5 valid setelah validasi")

    data["prediction_time"] = data["close_time_utc"]
    step = data["open_time_utc"].diff()
    data["gap_before"] = step != pd.Timedelta(minutes=M5_MINUTES)
    data.loc[0, "gap_before"] = True

    report = {
        "input_rows": total_rows,
        "valid_rows": len(data),
        "invalid_ohlc_rows": int((~valid_ohlc).sum()),
        "invalid_time_rows": int((~valid_time).sum()),
        "duplicate_rows_removed": duplicate_count,
        "gap_count": int(data["gap_before"].sum()) - 1,
        "spread_missing_rows": int(data["spread"].isna().sum()),
        "tick_volume_missing_rows": int(data["tick_volume"].isna().sum()),
        "first_open_time": data["open_time_utc"].iloc[0].isoformat(),
        "last_close_time": data["close_time_utc"].iloc[-1].isoformat(),
    }
    return data, report
