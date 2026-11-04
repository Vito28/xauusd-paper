"""Aplikasi FastAPI gabungan untuk riset XAUUSD.

Halaman:
  /          Live market monitoring (MT5 -> WebSocket)
  /backtest  Analisis backtesting XGBoost (prediksi out-of-sample vs harga aktual)
  /data      Data browser PostgreSQL (bars, ticks, processed M5)
  /docs      Dokumentasi API (OpenAPI)

Jalankan:
  uv run uvicorn dashboard.app:app --host 127.0.0.1 --port 8080
"""

from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from dashboard.backtest_api import router as backtest_router
from dashboard.live import create_live_service
from dashboard.live import router as live_router
from market.dashboard import dispose_databases
from market.dashboard import router as market_router

STATIC_DIRECTORY = Path(__file__).with_name("static")


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.live = create_live_service()
    await application.state.live.start()
    yield
    await application.state.live.stop()
    dispose_databases()


app = FastAPI(
    title="XAUUSD Research Dashboard",
    summary="Live market monitoring, backtesting XGBoost, dan data browser.",
    description=(
        "Dashboard read-only. Tidak ada endpoint yang mengirim order ke broker; "
        "prediksi live hanya dicatat dan dievaluasi setelah horizon selesai."
    ),
    version="0.4.0",
    lifespan=lifespan,
)
app.include_router(market_router)
app.include_router(backtest_router)
app.include_router(live_router)
app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")


def page(name: str) -> HTMLResponse:
    return HTMLResponse((STATIC_DIRECTORY / name).read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def live_page() -> HTMLResponse:
    return page("live.html")


@app.get("/backtest", response_class=HTMLResponse, include_in_schema=False)
def backtest_page() -> HTMLResponse:
    return page("backtest.html")


def run() -> None:
    uvicorn.run("dashboard.app:app", host="127.0.0.1", port=8080)


if __name__ == "__main__":
    run()
