"""Uji API dashboard backtest, komponen live, dan WebSocket (tanpa MT5 dan PostgreSQL)."""

import json
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dashboard.live import LiveFeed, LivePredictor, LiveService
from technical.backtest import build_prediction_frame, period_candles, save_period
from technical.config import TechnicalConfig
from technical.data import prepare_candles
from technical.run import build_dataset
from tests.test_technical_pipeline import make_candles

RUN_ID = "20260101T000000Z_final-test_abc"


@pytest.fixture()
def runs_dir(tmp_path, monkeypatch):
    """Satu run sintetis dengan file backtest, tanpa melatih model."""
    candles, _ = prepare_candles(make_candles(800, start="2026-01-02T00:00:00Z"))
    config = TechnicalConfig()
    _, _, labels, _ = build_dataset(candles, config, config.features.groups)
    rows = labels.index[300:700]
    probabilities = np.tile([0.15, 0.25, 0.6], (len(rows), 1))
    decisions = np.where(np.arange(len(rows)) % 3 == 0, 0, 2)
    decisions[::7] = 1
    reasons = np.full(len(rows), "TEST", dtype=object)
    frame = build_prediction_frame(
        candles, labels, rows, probabilities, decisions, reasons, "final_test", "model-test"
    )
    run_directory = tmp_path / RUN_ID
    run_directory.mkdir()
    with open(run_directory / "run.json", "w", encoding="utf-8") as file:
        json.dump({"config": config.to_dict()}, file)
    chart = period_candles(
        candles, frame["prediction_time"].min(), frame["prediction_time"].max(), 6
    )
    save_period(run_directory, "xgboost", "final_test", frame, chart)
    monkeypatch.setenv("GOLD_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("GOLD_LIVE_ENABLED", "0")
    return tmp_path


@pytest.fixture()
def client(runs_dir):
    from dashboard.app import app

    with TestClient(app) as test_client:
        yield test_client


def test_pages_are_served(client) -> None:
    for path in ["/", "/backtest", "/data", "/static/app.css", "/static/backtest.js"]:
        assert client.get(path).status_code == 200


def test_backtest_runs_summary_candles_predictions(client) -> None:
    runs = client.get("/api/backtest/runs").json()["runs"]
    assert runs == [{"run_id": RUN_ID, "models": {"xgboost": ["final_test"]}}]

    base = f"/api/backtest/{RUN_ID}/xgboost/final_test"
    summary = client.get(base + "/summary").json()
    assert summary["horizon_minutes"] == 30
    assert summary["signal_counts"]["NO_TRADE"] > 0
    assert summary["classification"]["confusion_matrix"]["values"]
    assert summary["simulation"]["trades"] > 0

    candles = client.get(base + "/candles").json()
    assert len(candles["bars"]) > 0
    assert set(candles["bars"][0]) >= {"time", "open", "high", "low", "close", "volume"}

    data = client.get(base + "/predictions", params={"signal": "SELL"}).json()
    signal_column = data["columns"].index("signal")
    assert data["count"] > 0
    assert all(row[signal_column] == "SELL" for row in data["rows"])
    assert data["model_version"] == "model-test"


def test_backtest_rejects_unknown_run(client) -> None:
    assert client.get("/api/backtest/../secret/xgboost/final_test/summary").status_code == 404
    assert client.get("/api/backtest/unknown/xgboost/final_test/summary").status_code == 404


def test_websocket_reports_disconnected_without_dummy_prices(client) -> None:
    with client.websocket_connect("/ws/live") as socket:
        message = socket.receive_json()
    assert message["type"] == "snapshot"
    assert message["status"]["connected"] is False
    assert message["tick"] is None
    assert message["bars"] == []
    assert message["predictor"]["available"] is False


# ---------------------------------------------------------------------------
# Komponen live dengan provider palsu
# ---------------------------------------------------------------------------


class FakeProvider:
    def __init__(self, bars):
        self.bars = bars
        self.initialized = False

    def initialize(self):
        self.initialized = True

    def shutdown(self):
        self.initialized = False

    def ensure_symbol_selected(self, symbol):
        return None

    def symbol_metadata(self, symbol):
        return SimpleNamespace(point=Decimal("0.01"))

    def latest_tick(self, symbol):
        return {"time": 1767225600, "time_msc": 1767225600123, "bid": 2000.10, "ask": 2000.35}

    def recent_bars(self, symbol, timeframe, count):
        return self.bars[-count:]


def mt5_bars(count):
    raw = make_candles(count, start="2026-01-05T00:00:00Z")
    bars = []
    for row in raw.itertuples(index=False):
        bars.append(
            {
                "time": int(row.open_time_utc.timestamp()),
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "tick_volume": int(row.tick_volume),
                "spread": 20,
            }
        )
    return bars


def test_live_feed_normalizes_tick_and_bars() -> None:
    feed = LiveFeed("XAUUSD", provider=FakeProvider(mt5_bars(50)))
    feed.connect()
    snapshot = feed.read()
    assert snapshot["tick"]["spread"] == pytest.approx(0.25)
    bars = snapshot["bars"]
    assert len(bars) == 50
    assert bars["native_spread_price"].iloc[0] == pytest.approx(0.2)  # 20 point x 0.01
    assert (bars["close_time_utc"] - bars["open_time_utc"]).eq(pd.Timedelta(minutes=5)).all()


class ConstantModel:
    def predict_proba(self, features):
        return np.tile([0.1, 0.2, 0.7], (len(features), 1))


def test_live_service_predicts_on_closed_bar_and_evaluates_later(tmp_path) -> None:
    config = TechnicalConfig()
    candles, _ = prepare_candles(make_candles(10))
    features, _ = build_dataset(candles, config, config.features.groups)[:2]
    bundle = {
        "model": ConstantModel(),
        "temperature": 1.0,
        "thresholds": {
            "BUY": {"enabled": True, "threshold": 0.5},
            "SELL": {"enabled": True, "threshold": 0.5},
        },
        "feature_groups": config.features.groups,
        "features": list(features.columns),
        "horizon_bars": 6,
        "atr_period": 14,
        "barrier_atr_multiple": 1.0,
        "model_version": "live-test",
    }
    all_bars = mt5_bars(400)
    provider = FakeProvider(all_bars[:300])
    feed = LiveFeed("XAUUSD", provider=provider)
    feed.connect()
    service = LiveService(feed, tmp_path)
    service.predictor = LivePredictor(bundle, tmp_path / "bundle.joblib")

    assert service.apply_snapshot(feed.read()) is True
    service.on_new_closed_bar()
    # Candle yang sama dibaca lagi -> tidak ada prediksi baru.
    assert service.apply_snapshot(feed.read()) is False
    first = service.predictions[0]
    assert first["signal"] == "BUY"
    assert first["status"] == "PENDING"
    # Prediksi dibuat dari candle yang SUDAH selesai (bukan candle yang masih terbentuk).
    forming_open = pd.Timestamp(all_bars[299]["time"], unit="s", tz="UTC")
    assert pd.Timestamp(first["candle_open_time"]) < forming_open

    provider.bars = all_bars[:320]
    assert service.apply_snapshot(feed.read()) is True
    service.on_new_closed_bar()
    assert service.predictions[0]["status"] == "EVALUATED"
    assert service.predictions[0]["actual_label"] in {"BUY", "SELL", "NO_TRADE"}
    assert (tmp_path / "live" / "predictions.csv").exists()
