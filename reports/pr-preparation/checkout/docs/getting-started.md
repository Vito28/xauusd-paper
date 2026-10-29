# Panduan Cepat

Halaman ini adalah langkah minimum untuk menjalankan collector sebagai data
acquisition platform. Pastikan MetaTrader 5 Terminal sudah terpasang di Windows
dan login secara manual ke account yang sesuai.

## 1. Sinkronkan environment Python

```powershell
uv sync --dev
Copy-Item .env.example .env
```

Edit `.env`, terutama environment dan URL PostgreSQL. Demo dan live wajib
memakai database fisik berbeda.

```dotenv
GOLD_ENVIRONMENT=demo
GOLD_DATABASE_URL=postgresql+psycopg://gold_demo:password@localhost:5432/gold_market_demo
GOLD_MT5_TERMINAL_PATH=C:\Program Files\Broker MT5\terminal64.exe
```

Profile broker dipilih secara eksplisit pada setiap command melalui
`--profile <nama_profile>`.

## 2. Sesuaikan profile broker

Salin `configs/brokers/example_demo.toml`, kemudian sesuaikan broker, server,
dan exact symbol broker seperti `XAUUSD`, `XAUUSDm`, atau `GOLD`. Jangan letakkan
credential pada profile.

## 3. Buat struktur database

```powershell
uv run alembic upgrade head
```

## 4. Validasi terminal dan source

```powershell
uv run python -m market.collector mt5-info --profile example_demo
uv run python -m market.collector symbol-info --profile example_demo
uv run python -m market.collector source-register --profile example_demo
```

## 5. Jalankan historical bars

```powershell
uv run python -m market.collector ingest-bars `
  --profile example_demo `
  --start 2016-01-01T00:00:00Z `
  --end 2026-01-01T00:00:00Z `
  --timeframe M5
```

## 6. Periksa hasil

```powershell
uv run python -m market.collector status --profile example_demo
uv run python -m market.collector coverage --profile example_demo
uv run python -m market.collector validate --profile example_demo
```

## 7. Lihat data dalam tabel

Pastikan migration terbaru sudah diterapkan dan historical tick sudah dikoleksi,
kemudian jalankan dashboard lokal:

```powershell
uv run alembic upgrade head
uv run python -m market.dashboard
```

Buka `http://127.0.0.1:8080/`. Pilih tab Market bars, Raw ticks, atau Gabungan
M5. Kolom Bid/Ask/spread pada gabungan akan `NULL` apabila broker tidak
menyediakan historical tick untuk candle terkait.

## Menjalankan website dokumentasi

```powershell
uv run mkdocs serve
```

Kemudian buka `http://127.0.0.1:8000/` pada browser. Server akan memperbarui
halaman secara otomatis ketika file dokumentasi berubah.
