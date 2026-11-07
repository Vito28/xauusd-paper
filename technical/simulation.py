"""Simulasi transaksi offline (paper trading) untuk sinyal BUY / SELL.

Modul ini TIDAK mengirim order ke broker dan tidak dipakai saat training.
Tujuannya memisahkan "hasil simulasi trading" dari "akurasi klasifikasi"
(technical/evaluation.py), karena keduanya adalah ukuran yang berbeda.

Asumsi harga (data historis M5 MT5):
  - OHLC candle MT5 adalah harga BID.
  - ASK = BID + spread candle saat keputusan.
  - Spread saat keluar dianggap sama dengan spread saat masuk, karena spread
    pada titik sentuh barrier di dalam candle tidak diketahui.
  - Ini simulasi tingkat candle, BUKAN eksekusi tingkat tick: slippage, requote,
    swap, dan komisi tidak dimodelkan.

Aturan keluar mengikuti label triple-barrier (technical/labels.py):
  barrier B = k * ATR + spread, entry referensi c = close candle keputusan (bid).
  BUY : masuk di ASK = c + s.  TP jika bid >= c + B, SL jika bid <= c - B.
  SELL: masuk di BID = c.      TP jika bid <= c - B, SL jika bid >= c + B.
  Jika kedua sisi tersentuh di candle yang sama -> dianggap SL (konservatif).
  Jika tidak ada yang tersentuh sampai H candle -> keluar di close candle t+H.

Dengan aturan ini, TP menghasilkan +k*ATR bersih dan SL menghasilkan
-(k*ATR + 2*spread) bersih, sesuai rancangan barrier.

Waktu buka dan tutup (dipakai dashboard, lihat trade_details):
  buka  = prediction_time = close candle keputusan t.
  tutup = close candle t+exit_step. Untuk TP/SL, barrier tersentuh di DALAM candle
          tersebut; waktu persisnya di dalam candle tidak diketahui dari data M5.
          Untuk TIME, posisi ditutup tepat di close candle t+H.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

M5_MINUTES = 5
SECONDS_PER_MINUTE = 60
UNIX_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


@dataclass
class InstrumentSpec:
    """Spesifikasi kontrak yang dipakai simulasi. Sesuaikan dengan symbol_info MT5."""

    symbol: str = "XAUUSD"
    contract_size: float = 100.0  # troy ounce per 1 lot (umum untuk XAUUSD)
    point: float = 0.01
    lot: float = 0.01
    leverage: float = 100.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimulatedTrade:
    direction: str
    entry_price: float
    exit_price: float
    exit_reason: str
    exit_step: int
    spread_cost: float
    gross_move: float
    net_move: float
    pnl: float
    margin: float
    notional: float


def exit_from_touches(up_step, down_step, direction: str) -> tuple[str, float]:
    """Tentukan alasan keluar dari langkah sentuh barrier.

    Return (alasan, langkah). Alasan: TP, SL, BOTH_SAME_CANDLE, TIME.
    """
    up_hit = up_step is not None and np.isfinite(up_step)
    down_hit = down_step is not None and np.isfinite(down_step)

    if up_hit and down_hit and up_step == down_step:
        return "BOTH_SAME_CANDLE", float(up_step)

    if direction == "BUY":
        favorable_hit = up_hit
        favorable_step = up_step
        adverse_hit = down_hit
        adverse_step = down_step
    else:
        favorable_hit = down_hit
        favorable_step = down_step
        adverse_hit = up_hit
        adverse_step = up_step

    if favorable_hit and (not adverse_hit or favorable_step < adverse_step):
        return "TP", float(favorable_step)
    if adverse_hit:
        return "SL", float(adverse_step)
    return "TIME", float("nan")


def simulate_trade(
    direction: str,
    close: float,
    spread: float,
    barrier: float,
    future_close: float,
    up_step,
    down_step,
    horizon_bars: int,
    spec: InstrumentSpec,
) -> SimulatedTrade:
    """Simulasikan satu transaksi dari satu sinyal BUY atau SELL."""
    if direction not in ("BUY", "SELL"):
        raise ValueError(f"Arah transaksi tidak dikenal: {direction}")

    reason, step = exit_from_touches(up_step, down_step, direction)

    # Harga bid saat keluar.
    if reason == "TP":
        if direction == "BUY":
            exit_bid = close + barrier
        else:
            exit_bid = close - barrier
    elif reason == "SL" or reason == "BOTH_SAME_CANDLE":
        if direction == "BUY":
            exit_bid = close - barrier
        else:
            exit_bid = close + barrier
    else:
        exit_bid = future_close
        step = float(horizon_bars)

    if direction == "BUY":
        entry_price = close + spread  # ASK
        exit_price = exit_bid  # jual di BID
        gross_move = exit_bid - close
        net_move = exit_price - entry_price
    else:
        entry_price = close  # BID
        exit_price = exit_bid + spread  # beli kembali di ASK
        gross_move = close - exit_bid
        net_move = entry_price - exit_price

    units = spec.lot * spec.contract_size
    notional = entry_price * units
    return SimulatedTrade(
        direction=direction,
        entry_price=float(entry_price),
        exit_price=float(exit_price),
        exit_reason=reason,
        exit_step=int(step) if np.isfinite(step) else horizon_bars,
        spread_cost=float(spread),
        gross_move=float(gross_move),
        net_move=float(net_move),
        pnl=float(net_move * units),
        margin=float(notional / spec.leverage),
        notional=float(notional),
    )


def simulate_predictions(
    predictions: pd.DataFrame, horizon_bars: int, spec: InstrumentSpec
) -> pd.DataFrame:
    """Simulasikan semua baris BUY/SELL yang horizon labelnya sudah lengkap.

    Hasilnya sama dengan memanggil simulate_trade() untuk setiap baris, tetapi
    dihitung per kolom (NumPy) agar cepat untuk puluhan ribu sinyal.

    Kolom wajib: signal, close, spread, barrier, future_close, up_touch_step,
    down_touch_step, label_status, prediction_time.

    Indeks hasil sama dengan indeks baris predictions yang disimulasikan, sehingga
    hasilnya bisa digabung kembali per baris (lihat trade_details). Selain kolom
    SimulatedTrade, hasil juga memuat barrier, gross_pnl (USD sebelum spread), dan
    spread_cost_usd untuk laporan spread.
    """
    selected = predictions[
        predictions["signal"].isin(["BUY", "SELL"]) & (predictions["label_status"] == "AVAILABLE")
    ]
    columns = list(asdict(SimulatedTrade("BUY", 0, 0, "", 0, 0, 0, 0, 0, 0, 0)).keys())
    if selected.empty:
        return pd.DataFrame(
            columns=[*columns, "barrier", "gross_pnl", "spread_cost_usd", "prediction_time"]
        )

    is_buy = (selected["signal"] == "BUY").to_numpy()
    close = selected["close"].to_numpy(dtype=float)
    spread = selected["spread"].to_numpy(dtype=float)
    barrier = selected["barrier"].to_numpy(dtype=float)
    future_close = selected["future_close"].to_numpy(dtype=float)
    up_step = selected["up_touch_step"].to_numpy(dtype=float)
    down_step = selected["down_touch_step"].to_numpy(dtype=float)

    # Langkah sentuh barrier yang menguntungkan / merugikan untuk arah masing-masing.
    never = np.inf
    up = np.where(np.isfinite(up_step), up_step, never)
    down = np.where(np.isfinite(down_step), down_step, never)
    favorable = np.where(is_buy, up, down)
    adverse = np.where(is_buy, down, up)

    both = (up == down) & np.isfinite(up)
    take_profit = ~both & (favorable < adverse)
    stop_loss = ~both & ~take_profit & np.isfinite(adverse)

    # Sisanya (tidak TP, tidak SL) keluar karena waktu habis di close candle t+H.
    reason = np.full(len(selected), "TIME", dtype=object)
    reason[take_profit] = "TP"
    reason[stop_loss] = "SL"
    reason[both] = "BOTH_SAME_CANDLE"

    step = np.full(len(selected), float(horizon_bars))
    step[take_profit] = favorable[take_profit]
    step[stop_loss] = adverse[stop_loss]
    step[both] = up[both]

    direction_sign = np.where(is_buy, 1.0, -1.0)
    exit_bid = future_close.copy()
    exit_bid[take_profit] = close[take_profit] + direction_sign[take_profit] * barrier[take_profit]
    loss = stop_loss | both
    exit_bid[loss] = close[loss] - direction_sign[loss] * barrier[loss]

    entry_price = np.where(is_buy, close + spread, close)
    exit_price = np.where(is_buy, exit_bid, exit_bid + spread)
    gross_move = direction_sign * (exit_bid - close)
    net_move = np.where(is_buy, exit_price - entry_price, entry_price - exit_price)

    units = spec.lot * spec.contract_size
    notional = entry_price * units
    trades = pd.DataFrame(
        {
            "direction": np.where(is_buy, "BUY", "SELL"),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_reason": reason,
            "exit_step": step.astype(int),
            "spread_cost": spread,
            "gross_move": gross_move,
            "net_move": net_move,
            "pnl": net_move * units,
            "margin": notional / spec.leverage,
            "notional": notional,
            # Tambahan untuk laporan spread (tidak ada di SimulatedTrade).
            "barrier": barrier,
            "gross_pnl": gross_move * units,
            "spread_cost_usd": spread * units,
        },
        index=selected.index,
    )
    trades["prediction_time"] = selected["prediction_time"].to_numpy()
    return trades


# ---------------------------------------------------------------------------
# Waktu buka/tutup dan laporan spread (untuk dashboard backtest)
# ---------------------------------------------------------------------------


def epoch_seconds(times) -> np.ndarray:
    """Waktu UTC -> detik sejak epoch, tidak bergantung pada resolusi datetime pandas."""
    values = pd.to_datetime(pd.Series(times), utc=True)
    return ((values - UNIX_EPOCH) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)


def open_close_seconds(trades: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(detik buka, detik tutup) setiap transaksi.

    Buka = close candle keputusan. Tutup = close candle ke-exit_step setelahnya.
    """
    opened = epoch_seconds(trades["prediction_time"])
    steps = trades["exit_step"].to_numpy(dtype=np.int64)
    closed = opened + steps * M5_MINUTES * SECONDS_PER_MINUTE
    return opened, closed


def candle_spread_at(open_seconds: np.ndarray, candles: pd.DataFrame | None) -> np.ndarray:
    """Spread tercatat pada candle yang dibuka di open_seconds (NaN jika tidak ada)."""
    if candles is None or len(candles) == 0 or "spread" not in candles.columns:
        return np.full(len(open_seconds), np.nan)
    keys = epoch_seconds(candles["open_time_utc"])
    lookup = pd.Series(candles["spread"].to_numpy(dtype=float), index=keys)
    lookup = lookup[~lookup.index.duplicated(keep="last")]
    return lookup.reindex(open_seconds).to_numpy(dtype=float)


def trade_details(
    predictions: pd.DataFrame,
    horizon_bars: int,
    spec: InstrumentSpec,
    candles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rincian buka/tutup posisi untuk setiap baris predictions (indeks sama).

    Baris yang tidak disimulasikan (NO_TRADE, PENDING, UNAVAILABLE) bernilai kosong.
    Kolom rasio (net_return, spread_to_price, spread_to_barrier) berupa pecahan, bukan persen.
    exit_candle_spread hanya informasi: simulasi tetap memakai spread saat buka untuk
    keluar, karena spread pada titik sentuh barrier di dalam candle tidak diketahui.
    """
    trades = simulate_predictions(predictions, horizon_bars, spec)
    columns = [
        "exit_reason",
        "exit_step",
        "holding_minutes",
        "exit_candle_open_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "gross_move",
        "net_move",
        "pnl",
        "net_return",
        "spread_to_price",
        "spread_to_barrier",
        "exit_candle_spread",
    ]
    if trades.empty:
        return pd.DataFrame(index=predictions.index, columns=columns, dtype=object)

    _, closed = open_close_seconds(trades)
    exit_candle_open = closed - M5_MINUTES * SECONDS_PER_MINUTE

    detail = pd.DataFrame(index=trades.index)
    detail["exit_reason"] = trades["exit_reason"]
    detail["exit_step"] = trades["exit_step"]
    detail["holding_minutes"] = trades["exit_step"] * M5_MINUTES
    detail["exit_candle_open_time"] = pd.to_datetime(exit_candle_open, unit="s", utc=True)
    detail["exit_time"] = pd.to_datetime(closed, unit="s", utc=True)
    detail["entry_price"] = trades["entry_price"]
    detail["exit_price"] = trades["exit_price"]
    detail["gross_move"] = trades["gross_move"]
    detail["net_move"] = trades["net_move"]
    detail["pnl"] = trades["pnl"]
    detail["net_return"] = trades["net_move"] / trades["entry_price"]
    detail["spread_to_price"] = trades["spread_cost"] / trades["entry_price"]
    detail["spread_to_barrier"] = trades["spread_cost"] / trades["barrier"]
    detail["exit_candle_spread"] = candle_spread_at(exit_candle_open, candles)
    return detail.reindex(predictions.index)


def count_overlapping(trades: pd.DataFrame) -> int:
    """Jumlah transaksi yang dibuka saat transaksi sebelumnya masih terbuka."""
    if len(trades) < 2:
        return 0
    opened, closed = open_close_seconds(trades)
    order = np.argsort(opened, kind="stable")
    opened = opened[order]
    closed = closed[order]
    latest_close = np.maximum.accumulate(closed)
    # Posisi ke-i tumpang tindih jika dibuka sebelum semua posisi sebelumnya tutup.
    return int((opened[1:] < latest_close[:-1]).sum())


def spread_report(trades: pd.DataFrame, candles: pd.DataFrame | None = None) -> dict:
    """Laporan biaya spread: nilai absolut, persentase, dan pengaruhnya pada P&L.

    Persentase berupa pecahan (0.01 = 1%):
      to_price   : spread / harga buka, biaya relatif terhadap harga.
      to_barrier : spread / barrier, porsi jarak TP yang hanya menutup biaya spread.
    """
    if trades.empty:
        return {}
    spread = trades["spread_cost"].to_numpy(dtype=float)
    entry = trades["entry_price"].to_numpy(dtype=float)
    barrier = trades["barrier"].to_numpy(dtype=float)

    _, closed = open_close_seconds(trades)
    exit_spread = candle_spread_at(closed - M5_MINUTES * SECONDS_PER_MINUTE, candles)
    known_exit_spread = exit_spread[np.isfinite(exit_spread)]
    if len(known_exit_spread) > 0:
        average_exit_spread = float(known_exit_spread.mean())
    else:
        average_exit_spread = None

    return {
        "average": float(spread.mean()),
        "median": float(np.median(spread)),
        "minimum": float(spread.min()),
        "maximum": float(spread.max()),
        "average_to_price": float((spread / entry).mean()),
        "average_to_barrier": float((spread / barrier).mean()),
        "total_cost_usd": float(trades["spread_cost_usd"].sum()),
        "gross_pnl_before_spread": float(trades["gross_pnl"].sum()),
        "net_pnl_after_spread": float(trades["pnl"].sum()),
        "exit_spread_assumption": "spread saat tutup = spread saat buka",
        "average_exit_candle_spread": average_exit_spread,
        "exit_candle_spread_rows": int(len(known_exit_spread)),
    }


def summarize_trades(trades: pd.DataFrame) -> dict:
    """Ringkasan hasil simulasi: win rate, profit factor, drawdown, equity."""
    if trades.empty:
        return {"trades": 0}

    pnl = trades["pnl"].to_numpy(dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    equity = np.cumsum(pnl)
    running_peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    drawdown = equity - running_peak

    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = None

    exit_counts = {}
    exit_shares = {}
    for reason, count in trades["exit_reason"].value_counts().items():
        exit_counts[str(reason)] = int(count)
        exit_shares[str(reason)] = float(count / len(trades))

    holding_minutes = trades["exit_step"].to_numpy(dtype=float) * M5_MINUTES
    overlapping = count_overlapping(trades)

    return {
        "trades": int(len(trades)),
        "buy_trades": int((trades["direction"] == "BUY").sum()),
        "sell_trades": int((trades["direction"] == "SELL").sum()),
        "win_rate": float(len(wins) / len(trades)),
        "net_pnl": float(pnl.sum()),
        "net_move_price_total": float(trades["net_move"].sum()),
        "average_pnl": float(pnl.mean()),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "max_drawdown": float(drawdown.min()),
        "total_spread_cost": float(trades["spread_cost"].sum()),
        "exit_reasons": exit_counts,
        "exit_reason_share": exit_shares,
        "average_holding_minutes": float(holding_minutes.mean()),
        "overlapping_trades": overlapping,
        "overlapping_share": float(overlapping / len(trades)),
    }
