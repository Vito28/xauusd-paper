"""Pembuatan label BUY / SELL / NO_TRADE.

Label boleh memakai harga masa depan karena label adalah ground truth untuk
supervised learning. Label TIDAK PERNAH masuk ke matriks fitur.

Definisi bersama (lihat SOURCE_OF_TRUTH.md bagian A dan G):
  entry       = close candle t (harga bid MT5) pada waktu keputusan.
  ATR_t       = ATR Wilder periode atr_period, dihitung sampai candle t.
  spread_t    = spread candle t (satuan harga).
  barrier     = k * ATR_t + spread_t
                Spread ditambahkan agar pergerakan yang menyentuh barrier
                masih memberi hasil positif sebesar k * ATR setelah biaya spread.
  horizon     = H candle M5 setelah t.

Metode "triple_barrier" (utama):
  BUY      jika high menyentuh entry + barrier lebih dulu dalam H candle.
  SELL     jika low menyentuh entry - barrier lebih dulu dalam H candle.
  NO_TRADE jika tidak ada barrier yang tersentuh sampai candle ke-H,
           atau kedua barrier tersentuh pada candle yang sama (urutan di dalam
           candle tidak diketahui, jadi dipilih label konservatif).

Metode "fixed_horizon" (pembanding):
  BUY      jika close[t+H] - entry >  barrier
  SELL     jika close[t+H] - entry < -barrier
  NO_TRADE selain itu.

Label dianggap TIDAK DIKETAHUI (NaN) jika:
  - H candle berikutnya melewati gap (pasar tutup / data hilang),
  - data masa depan belum tersedia,
  - ATR atau spread pada candle t tidak tersedia.
"""

import numpy as np
import pandas as pd

from technical.config import BUY, NO_TRADE, SELL
from technical.features import average_true_range

M5_MINUTES = 5


def future_window_is_contiguous(data: pd.DataFrame, horizon_bars: int) -> np.ndarray:
    """True jika candle t+1 ... t+H semuanya ada dan berurutan tanpa gap."""
    times = data["prediction_time"].to_numpy()
    count = len(data)
    contiguous = np.zeros(count, dtype=bool)
    expected = np.timedelta64(horizon_bars * M5_MINUTES, "m")
    for index in range(count - horizon_bars):
        elapsed = times[index + horizon_bars] - times[index]
        contiguous[index] = elapsed == expected
    return contiguous


def make_labels(
    data: pd.DataFrame,
    method: str,
    horizon_bars: int,
    atr_period: int,
    barrier_atr_multiple: float,
) -> pd.DataFrame:
    """Hitung label dan kolom pendukung audit untuk setiap candle.

    Kolom hasil:
      label            : 0=SELL, 1=NO_TRADE, 2=BUY, atau NaN jika tidak diketahui.
      barrier          : lebar barrier dalam satuan harga.
      label_end_time   : waktu paling akhir yang informasinya dipakai label
                         (= prediction_time + H x 5 menit). Dipakai untuk purging.
      label_status     : AVAILABLE, PENDING, GAP, atau INPUT_MISSING.
    """
    if method not in ("triple_barrier", "fixed_horizon"):
        raise ValueError(f"Metode label tidak dikenal: {method}")

    close = data["close"].to_numpy(dtype=float)
    high = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)
    atr = average_true_range(data["high"], data["low"], data["close"], atr_period).to_numpy()
    spread = data["spread"].to_numpy(dtype=float)
    count = len(data)

    barrier = barrier_atr_multiple * atr + spread
    upper = close + barrier
    lower = close - barrier

    # Untuk setiap langkah ke depan, catat langkah pertama barrier atas/bawah tersentuh.
    never = horizon_bars + 1
    first_up_step = np.full(count, never)
    first_down_step = np.full(count, never)
    for step in range(1, horizon_bars + 1):
        future_high = np.full(count, np.nan)
        future_low = np.full(count, np.nan)
        future_high[: count - step] = high[step:]
        future_low[: count - step] = low[step:]

        hit_up = (future_high >= upper) & (first_up_step == never)
        hit_down = (future_low <= lower) & (first_down_step == never)
        first_up_step[hit_up] = step
        first_down_step[hit_down] = step

    future_close = np.full(count, np.nan)
    future_close[: count - horizon_bars] = close[horizon_bars:]

    label = np.full(count, float(NO_TRADE))
    if method == "triple_barrier":
        up_first = first_up_step < first_down_step
        down_first = first_down_step < first_up_step
        label[up_first] = BUY
        label[down_first] = SELL
        # first_up_step == first_down_step < never: sentuh dua sisi di candle sama -> NO_TRADE.
    else:
        move = future_close - close
        label[move > barrier] = BUY
        label[move < -barrier] = SELL

    contiguous = future_window_is_contiguous(data, horizon_bars)
    inputs_known = np.isfinite(barrier)

    status = np.full(count, "AVAILABLE", dtype=object)
    status[~inputs_known] = "INPUT_MISSING"
    status[~contiguous] = "GAP"
    pending = np.arange(count) >= count - horizon_bars
    status[pending] = "PENDING"

    known = status == "AVAILABLE"
    label[~known] = np.nan

    result = pd.DataFrame(index=data.index)
    result["prediction_time"] = data["prediction_time"]
    result["label"] = label
    result["barrier"] = barrier
    result["future_close"] = future_close
    result["label_end_time"] = data["prediction_time"] + pd.Timedelta(
        minutes=horizon_bars * M5_MINUTES
    )
    result["label_status"] = status
    return result


def label_distribution(labels: pd.Series) -> dict:
    """Jumlah dan proporsi setiap kelas (label NaN diabaikan)."""
    known = labels.dropna().astype(int)
    total = len(known)
    result = {"total": total}
    for value, name in [(SELL, "SELL"), (NO_TRADE, "NO_TRADE"), (BUY, "BUY")]:
        count = int((known == value).sum())
        share = count / total if total > 0 else 0.0
        result[name] = {"count": count, "share": share}
    return result
