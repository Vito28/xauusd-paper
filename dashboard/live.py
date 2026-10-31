"""Live market monitoring: MT5 -> LiveService -> FastAPI WebSocket -> dashboard.

Alur data:
  1. LiveFeed membaca tick terakhir dan bar M5 terakhir dari terminal MT5
     (melalui market.collector.MT5Provider, read-only, tanpa order).
  2. LiveService memanggil feed setiap POLL_SECONDS di thread terpisah, mendeteksi
     candle yang baru selesai, lalu meminta LivePredictor membuat prediksi.
  3. Setiap perubahan dikirim ke browser melalui WebSocket /ws/live.

Jika paket MetaTrader5 tidak tersedia (misalnya Linux/Docker) atau terminal belum
login, status dikirim sebagai "disconnected" beserta alasannya. Tidak ada data
dummy: grafik hanya diisi data historis PostgreSQL dengan label yang jelas.

Prediksi live dicatat terpisah dari prediksi backtest di runs/live/predictions.csv.
"""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from dashboard.settings import load_dashboard_settings
from market.collector import MT5Provider
from market.contracts import Environment
from technical.config import CLASS_NAMES, PROJECT_ROOT
from technical.data import prepare_candles
from technical.decision import apply_temperature, decide
from technical.features import average_true_range, build_features
from technical.labels import make_labels
from technical.run import clean_for_json

POLL_SECONDS = 1.0
RECONNECT_SECONDS = 10.0
TIMEFRAME = "M5"
BAR_MINUTES = 5
# 1.500 bar (~5 hari) cukup untuk warm-up fitur terpanjang (rolling 288 bar + EMA).
FEATURE_HISTORY_BARS = 1500
CHART_BARS = 500
MAX_LIVE_PREDICTIONS = 500


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def epoch_to_utc(seconds) -> pd.Timestamp:
    # Sama dengan collector: epoch MT5 diperlakukan sebagai UTC agar konsisten
    # dengan data historis yang dipakai training.
    return pd.Timestamp(int(seconds), unit="s", tz="UTC")


# ---------------------------------------------------------------------------
# 1. Sumber data MT5
# ---------------------------------------------------------------------------


class LiveFeed:
    """Pembungkus MT5Provider untuk satu simbol. Semua method bersifat blocking."""

    def __init__(self, symbol: str, terminal_path: Path | None = None, provider=None):
        self.symbol = symbol
        if provider is None:
            provider = MT5Provider(terminal_path)
        self.provider = provider
        self.point = 0.01
        self.connected = False

    def connect(self) -> None:
        self.provider.initialize()
        self.provider.ensure_symbol_selected(self.symbol)
        metadata = self.provider.symbol_metadata(self.symbol)
        if metadata.point is not None:
            self.point = float(metadata.point)
        self.connected = True

    def disconnect(self) -> None:
        self.provider.shutdown()
        self.connected = False

    def read(self) -> dict:
        """Ambil tick terakhir dan bar M5 terakhir (bar terakhir masih terbentuk)."""
        tick = self.provider.latest_tick(self.symbol)
        raw_bars = self.provider.recent_bars(self.symbol, TIMEFRAME, FEATURE_HISTORY_BARS + 1)
        return {"tick": self.normalize_tick(tick), "bars": self.normalize_bars(raw_bars)}

    def normalize_tick(self, tick: dict) -> dict:
        if tick.get("time_msc"):
            time = pd.Timestamp(int(tick["time_msc"]), unit="ms", tz="UTC")
        else:
            time = epoch_to_utc(tick["time"])
        bid = float(tick["bid"])
        ask = float(tick["ask"])
        return {"time": time.isoformat(), "bid": bid, "ask": ask, "spread": ask - bid}

    def normalize_bars(self, raw_bars: list[dict]) -> pd.DataFrame:
        rows = []
        for bar in raw_bars:
            open_time = epoch_to_utc(bar["time"])
            rows.append(
                {
                    "open_time_utc": open_time,
                    "close_time_utc": open_time + pd.Timedelta(minutes=BAR_MINUTES),
                    "open": float(bar["open"]),
                    "high": float(bar["high"]),
                    "low": float(bar["low"]),
                    "close": float(bar["close"]),
                    "tick_volume": float(bar.get("tick_volume", np.nan)),
                    "native_spread_price": float(bar.get("spread", np.nan)) * self.point,
                }
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Prediksi live dari bundle hasil final-test
# ---------------------------------------------------------------------------


def find_latest_bundle(runs_root: Path) -> Path | None:
    if not runs_root.exists():
        return None
    bundles = sorted(runs_root.glob("*/live_bundle.joblib"), reverse=True)
    if len(bundles) == 0:
        return None
    return bundles[0]


class LivePredictor:
    """Memakai model yang dilatih dengan data <= development_end (tidak dilatih ulang)."""

    def __init__(self, bundle: dict, path: Path):
        self.bundle = bundle
        self.path = path
        self.model_version = bundle["model_version"]
        self.horizon_bars = int(bundle["horizon_bars"])

    @classmethod
    def load(cls, path: Path) -> "LivePredictor":
        return cls(joblib.load(path), path)

    def describe(self) -> dict:
        return {
            "available": True,
            "model_version": self.model_version,
            "trained_until": self.bundle.get("trained_until"),
            "horizon_minutes": self.horizon_bars * BAR_MINUTES,
            "bundle": str(self.path.relative_to(PROJECT_ROOT))
            if self.path.is_relative_to(PROJECT_ROOT)
            else str(self.path),
        }

    def predict_last(self, closed_bars: pd.DataFrame) -> dict:
        """Prediksi untuk candle terakhir yang SUDAH selesai."""
        candles, _ = prepare_candles(closed_bars)
        features, _ = build_features(candles, self.bundle["feature_groups"])
        features = features[self.bundle["features"]]
        last = features.iloc[[-1]]

        raw = self.model_predict(last)
        calibrated = apply_temperature(raw, self.bundle["temperature"])
        decisions, reasons = decide(calibrated, self.bundle["thresholds"])
        decision = int(decisions[0])

        atr = average_true_range(
            candles["high"], candles["low"], candles["close"], self.bundle["atr_period"]
        )
        last_candle = candles.iloc[-1]
        spread = float(last_candle["spread"])
        barrier = self.bundle["barrier_atr_multiple"] * float(atr.iloc[-1]) + spread
        prediction_time = last_candle["prediction_time"]
        target_time = prediction_time + pd.Timedelta(minutes=self.horizon_bars * BAR_MINUTES)

        probabilities = {}
        for index, name in enumerate(CLASS_NAMES):
            probabilities[name] = float(calibrated[0, index])

        return {
            "prediction_time": prediction_time.isoformat(),
            "candle_open_time": last_candle["open_time_utc"].isoformat(),
            "target_time": target_time.isoformat(),
            "signal": CLASS_NAMES[decision],
            "confidence": float(calibrated[0, decision]),
            "probabilities": probabilities,
            "reason_code": str(reasons[0]),
            "bid_reference": float(last_candle["close"]),
            "ask_reference": float(last_candle["close"]) + spread,
            "spread": spread,
            "barrier": barrier,
            "status": "PENDING",
            "actual_label": None,
            "future_close": None,
            "model_version": self.model_version,
        }

    def model_predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.bundle["model"].predict_proba(features)

    def evaluate(self, prediction: dict, closed_bars: pd.DataFrame) -> dict:
        """Isi hasil aktual setelah horizon selesai, memakai definisi label yang sama."""
        if prediction["status"] != "PENDING":
            return prediction
        candles, _ = prepare_candles(closed_bars)
        labels = make_labels(
            candles,
            "triple_barrier",
            self.horizon_bars,
            self.bundle["atr_period"],
            self.bundle["barrier_atr_multiple"],
        )
        match = labels[labels["prediction_time"] == pd.Timestamp(prediction["prediction_time"])]
        if match.empty:
            return prediction
        row = match.iloc[0]
        if row["label_status"] == "PENDING":
            return prediction
        updated = dict(prediction)
        if row["label_status"] != "AVAILABLE":
            updated["status"] = "UNAVAILABLE"
            updated["reason_code"] = f"LABEL_{row['label_status']}"
            return updated
        updated["status"] = "EVALUATED"
        updated["actual_label"] = CLASS_NAMES[int(row["label"])]
        updated["future_close"] = float(row["future_close"])
        return updated


# ---------------------------------------------------------------------------
# 3. Layanan live: polling + broadcast WebSocket
# ---------------------------------------------------------------------------


class LiveService:
    def __init__(self, feed: LiveFeed, runs_root: Path, enabled: bool = True):
        self.feed = feed
        self.runs_root = runs_root
        self.enabled = enabled
        self.clients: set[WebSocket] = set()
        self.status = {
            "connected": False,
            "source": "mt5",
            "symbol": feed.symbol,
            "timeframe": TIMEFRAME,
            "reason": "Belum mencoba koneksi",
            "last_update": None,
        }
        self.tick = None
        self.bars = pd.DataFrame()
        self.last_closed_time = None
        self.predictions: list[dict] = []
        self.predictor = None
        self.predictor_status = {"available": False, "reason": "Belum dimuat"}
        self.task = None
        self.log_path = runs_root / "live" / "predictions.csv"

    # -- siklus hidup -------------------------------------------------------

    async def start(self) -> None:
        self.load_predictor()
        if not self.enabled:
            self.status["reason"] = "Live feed dinonaktifkan (GOLD_LIVE_ENABLED=false)"
            return
        self.task = asyncio.create_task(self.loop())

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        if self.feed.connected:
            await asyncio.to_thread(self.feed.disconnect)

    def load_predictor(self) -> None:
        path = find_latest_bundle(self.runs_root)
        if path is None:
            self.predictor = None
            self.predictor_status = {
                "available": False,
                "reason": "Belum ada model. Jalankan: python -m technical.run final-test --confirm",
            }
            return
        try:
            self.predictor = LivePredictor.load(path)
            self.predictor_status = self.predictor.describe()
        except Exception as exc:  # noqa: BLE001 - bundle rusak tidak boleh mematikan server
            self.predictor = None
            self.predictor_status = {"available": False, "reason": f"Gagal memuat model: {exc}"}

    async def loop(self) -> None:
        while True:
            if not self.feed.connected:
                try:
                    await asyncio.to_thread(self.feed.connect)
                except Exception as exc:  # noqa: BLE001 - semua error MT5 = disconnected
                    self.mark_disconnected(exc)
                    await self.broadcast(self.update_message())
                    await asyncio.sleep(RECONNECT_SECONDS)
                    continue
            try:
                snapshot = await asyncio.to_thread(self.feed.read)
            except Exception as exc:  # noqa: BLE001
                self.mark_disconnected(exc)
                await asyncio.to_thread(self.feed.disconnect)
                await self.broadcast(self.update_message())
                await asyncio.sleep(RECONNECT_SECONDS)
                continue

            new_bar = self.apply_snapshot(snapshot)
            if new_bar:
                await asyncio.to_thread(self.on_new_closed_bar)
                await self.broadcast(self.snapshot_message())
            else:
                await self.broadcast(self.update_message())
            await asyncio.sleep(POLL_SECONDS)

    def mark_disconnected(self, error: Exception) -> None:
        self.status["connected"] = False
        self.status["reason"] = str(error) or type(error).__name__
        self.status["last_update"] = utc_now_iso()

    # -- pengolahan data ----------------------------------------------------

    def apply_snapshot(self, snapshot: dict) -> bool:
        """Simpan snapshot; return True jika ada candle baru yang selesai."""
        self.tick = snapshot["tick"]
        self.bars = snapshot["bars"]
        self.status["connected"] = True
        self.status["reason"] = None
        self.status["last_update"] = utc_now_iso()
        if len(self.bars) < 2:
            return False
        last_closed = self.bars["open_time_utc"].iloc[-2]
        if last_closed == self.last_closed_time:
            return False
        self.last_closed_time = last_closed
        return True

    def closed_bars(self) -> pd.DataFrame:
        return self.bars.iloc[:-1]

    def on_new_closed_bar(self) -> None:
        if self.predictor is None:
            return
        closed = self.closed_bars()
        try:
            for position, prediction in enumerate(self.predictions):
                self.predictions[position] = self.predictor.evaluate(prediction, closed)
            prediction = self.predictor.predict_last(closed)
        except Exception as exc:  # noqa: BLE001 - data live tidak lengkap
            self.predictor_status["last_error"] = str(exc)
            return
        already = False
        for existing in self.predictions:
            if existing["prediction_time"] == prediction["prediction_time"]:
                already = True
        if not already:
            self.predictions.append(prediction)
        self.predictions = self.predictions[-MAX_LIVE_PREDICTIONS:]
        self.write_log()

    def write_log(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for prediction in self.predictions:
            row = dict(prediction)
            probabilities = row.pop("probabilities")
            for name, value in probabilities.items():
                row[f"probability_{name.lower()}"] = value
            rows.append(row)
        pd.DataFrame(rows).to_csv(self.log_path, index=False)

    # -- pesan WebSocket ----------------------------------------------------

    def bar_records(self, frame: pd.DataFrame) -> list[dict]:
        records = []
        for _, bar in frame.iterrows():
            records.append(
                {
                    "time": int(bar["open_time_utc"].timestamp()),
                    "open": bar["open"],
                    "high": bar["high"],
                    "low": bar["low"],
                    "close": bar["close"],
                    "volume": bar["tick_volume"],
                    "spread": bar["native_spread_price"],
                }
            )
        return records

    def base_message(self, kind: str) -> dict:
        return {
            "type": kind,
            "status": self.status,
            "tick": self.tick,
            "predictor": self.predictor_status,
            "predictions": self.predictions[-30:],
        }

    def snapshot_message(self) -> dict:
        message = self.base_message("snapshot")
        message["bars"] = self.bar_records(self.bars.tail(CHART_BARS))
        return clean_for_json(message)

    def update_message(self) -> dict:
        message = self.base_message("update")
        message["bars"] = self.bar_records(self.bars.tail(2))
        return clean_for_json(message)

    async def broadcast(self, message: dict) -> None:
        stale = []
        for client in list(self.clients):
            try:
                await client.send_json(message)
            except Exception:  # noqa: BLE001 - klien yang putus dibuang
                stale.append(client)
        for client in stale:
            self.clients.discard(client)


def create_live_service() -> LiveService:
    settings = load_dashboard_settings()
    feed = LiveFeed(settings.live_symbol, settings.mt5_terminal_path)
    return LiveService(feed, settings.runs_dir, settings.live_enabled)


# ---------------------------------------------------------------------------
# 4. Endpoint
# ---------------------------------------------------------------------------

router = APIRouter(tags=["Live"])

HISTORY_QUERY = """
SELECT open_time_utc, open, high, low, close, tick_volume, native_spread_price
FROM market.processed_m5
WHERE timeframe = 'M5' AND (dataset_symbol = :symbol OR actual_symbol = :symbol)
ORDER BY open_time_utc DESC
LIMIT :limit
"""


@router.get("/api/live/status", summary="Status MT5 dan model live")
def live_status(request: Request) -> dict:
    service: LiveService = request.app.state.live
    return clean_for_json(
        {
            "status": service.status,
            "tick": service.tick,
            "predictor": service.predictor_status,
            "predictions": service.predictions[-30:],
        }
    )


def read_history(database, symbol: str, limit: int) -> list[dict]:
    with database.session() as session:
        rows = session.execute(text(HISTORY_QUERY), {"symbol": symbol, "limit": limit}).mappings()
        records = []
        for row in rows:
            records.append(
                {
                    "time": int(pd.Timestamp(row["open_time_utc"]).timestamp()),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["tick_volume"]) if row["tick_volume"] is not None else None,
                }
            )
    records.reverse()
    return records


@router.get("/api/live/history", summary="Candle M5 historis dari PostgreSQL")
def live_history(
    symbol: Annotated[str, Query(min_length=1, max_length=32)] = "XAUUSD",
    limit: Annotated[int, Query(ge=50, le=5000)] = CHART_BARS,
) -> dict:
    """Fallback grafik saat MT5 tidak terhubung. Data ini BUKAN harga live."""
    from market.dashboard import DashboardConfigurationError, get_database

    try:
        database = get_database(Environment.DEMO)
        bars = read_history(database, symbol, limit)
    except (DashboardConfigurationError, SQLAlchemyError) as exc:
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL tidak tersedia. Isi GOLD_DASHBOARD_DEMO_DATABASE_URL di .env.",
        ) from exc
    return {"source": "postgres", "symbol": symbol, "bars": bars}


@router.websocket("/ws/live")
async def live_socket(websocket: WebSocket) -> None:
    service: LiveService = websocket.app.state.live
    await websocket.accept()
    service.clients.add(websocket)
    try:
        await websocket.send_json(service.snapshot_message())
        while True:
            # Klien boleh mengirim "ping"; isi pesan tidak dipakai.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        service.clients.discard(websocket)
