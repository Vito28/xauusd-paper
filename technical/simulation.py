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
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


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
    """
    selected = predictions[
        predictions["signal"].isin(["BUY", "SELL"]) & (predictions["label_status"] == "AVAILABLE")
    ]
    columns = list(asdict(SimulatedTrade("BUY", 0, 0, "", 0, 0, 0, 0, 0, 0, 0)).keys())
    if selected.empty:
        return pd.DataFrame(columns=[*columns, "prediction_time"])

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
        }
    )
    trades["prediction_time"] = selected["prediction_time"].to_numpy()
    return trades


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
    for reason, count in trades["exit_reason"].value_counts().items():
        exit_counts[str(reason)] = int(count)

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
    }
