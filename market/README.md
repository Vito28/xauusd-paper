# Market — profile dan collector

Dua file utama memakai section `# %%` seperti notebook/Colab. Penanda section
memisahkan kelompok fungsi; pemanggilan proses tetap eksplisit melalui fungsi
atau CLI. Import modul tidak membaca runtime settings, membuka terminal MT5,
atau melakukan transaksi PostgreSQL.

```text
market/
  profile.py        profile broker, loader/saver TOML, dan GOLD_* runtime settings
  collector.py      seluruh workflow acquisition, validation, coverage, dan CLI
  contracts.py      record bersama untuk collector dan repository database
  dashboard.py      browser/API read-only; template berada di static/
  cli.py            penghubung untuk console script/perintah lama
  __init__.py       identitas package
```

`contracts.py` menjaga record data tetap independen dari collector dan database.
Dashboard tetap menjadi modul API dengan dokumentasi OpenAPI. `cli.py` hanya
mengekspor app dari `collector.py`; tidak menyimpan implementasi collection.

## Profile dan penulisan konfigurasi

Profile tersedia: `example_demo` dan `example_live` sebagai template,
`metaquotes_demo` untuk MetaQuotes-Demo, serta `finex_live` untuk
FinexBisnisSolusi-Real. Profile hanya berisi identitas broker/server/simbol
dan policy collection; credential berasal dari environment dan terminal MT5.
Pilih profile yang sesuai dengan environment dan account terminal aktif.

Section `profile.py`: import → schema profile broker → runtime settings →
penulisan konfigurasi. `load_broker_profile()` membaca TOML;
`save_broker_profile()` memvalidasi lalu menulis satu profile TOML.
Nama profile harus berupa stem file. File yang sudah ada tidak diganti kecuali
`overwrite=True` diberikan secara eksplisit. Runtime credential tetap melalui
environment/terminal; profile menyimpan identitas source dan policy non-secret.

Contoh membuat profile dari template yang tersedia:

```python
from market.profile import BrokerProfile, load_broker_profile, save_broker_profile

fields = load_broker_profile("example_demo").model_dump()
fields["source"].update(
    source_key="broker-baru-demo-xauusd",
    broker_name="Broker Baru",
    server_name="BrokerBaru-Demo",
    requested_symbol="XAUUSD",
)
profile = BrokerProfile.model_validate(fields)
path = save_broker_profile("broker_baru_demo", profile)
```

Contoh tersebut hanya menulis `configs/brokers/broker_baru_demo.toml`.
Pendaftaran source dan ingestion merupakan operasi CLI yang terpisah.
Seluruh profile TOML yang sudah ada tetap memakai format sebelumnya.

## Section collector

| Section | Proses | Komentar di source |
|---|---|---|
| 01 | Import dependensi | Import tidak mengaktifkan MT5/database |
| 02 | Resolusi exact symbol | `SYMBOL-*` |
| 03 | MT5 provider, preflight, metadata, fetch bars/ticks | `MT5-*` |
| 04 | Partition bulanan bars dan harian ticks dalam UTC | `PARTITION-*` |
| 05 | Normalisasi, fingerprint, validasi OHLC/quote dan rejection | `VALIDATE-*` |
| 06 | Derived aggregation M5 dan candidate gaps | `DERIVED-*` |
| 07 | Ingestion, resume, retry, transaksi, audit, checkpoint | `INGEST-*` |
| 08 | Coverage requested/actual, status run, invariant report | `REPORT-*` |
| 09 | Logging JSON dan redaksi credential | `LOG-*` |
| 10 | Register, inspect, backfill, sync, reports | `CLI-*` |

Alur utama: profile → MT5 preflight → partition → fetch → normalisasi/validasi →
PostgreSQL observations/rejections/checkpoint → coverage/status/validation report.
Data, rejection dan checkpoint disimpan atomik per partition; harga tidak
di-fill atau diinterpolasi. Laporan membaca PostgreSQL tanpa membuka MT5.

## Menjalankan

```powershell
uv run python -m market.collector --help
uv run python -m market.collector mt5-info --profile example_demo
uv run python -m market.collector ingest-bars --help
uv run python -m market.collector coverage --profile example_demo
```

Console script `uv run market ...` dan perintah lama `python -m market.cli ...`
tetap mengarah ke app yang sama. Dashboard tetap dijalankan melalui
`uv run python -m market.dashboard`.

## Verifikasi

```powershell
uv run pytest tests/market tests/database -m "not integration" -q
uv run ruff check market database tests/market tests/database
```

Unit tests memakai fake provider/session dan tidak memanggil terminal/database
nyata. Test integrasi PostgreSQL tetap membutuhkan database disposable yang
ditentukan melalui `GOLD_TEST_DATABASE_URL`.

## Panduan cepat collector

Halaman ini adalah langkah minimum untuk menjalankan collector sebagai data
acquisition platform. Pastikan MetaTrader 5 Terminal sudah terpasang di Windows
dan login secara manual ke account yang sesuai.

### 1. Sinkronkan environment Python

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

### 2. Sesuaikan profile broker

Salin `configs/brokers/example_demo.toml`, kemudian sesuaikan broker, server,
dan exact symbol broker seperti `XAUUSD`, `XAUUSDm`, atau `GOLD`. Jangan letakkan
credential pada profile.

### 3. Buat struktur database

```powershell
uv run alembic upgrade head
```

### 4. Validasi terminal dan source

```powershell
uv run python -m market.collector mt5-info --profile example_demo
uv run python -m market.collector symbol-info --profile example_demo
uv run python -m market.collector source-register --profile example_demo
```

### 5. Jalankan historical bars

```powershell
uv run python -m market.collector ingest-bars `
  --profile example_demo `
  --start 2016-01-01T00:00:00Z `
  --end 2026-01-01T00:00:00Z `
  --timeframe M5
```

### 6. Periksa hasil

```powershell
uv run python -m market.collector status --profile example_demo
uv run python -m market.collector coverage --profile example_demo
uv run python -m market.collector validate --profile example_demo
```

### 7. Lihat data dalam tabel

Pastikan migration terbaru sudah diterapkan dan historical tick sudah dikoleksi,
kemudian jalankan dashboard lokal:

```powershell
uv run alembic upgrade head
uv run python -m market.dashboard
```

Buka `http://127.0.0.1:8080/`. Pilih tab Market bars, Raw ticks, atau Gabungan
M5. Kolom Bid/Ask/spread pada gabungan akan `NULL` apabila broker tidak
menyediakan historical tick untuk candle terkait.

### 8. Docker (opsional, API/dashboard + PostgreSQL 18)

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Compose menjalankan `alembic upgrade head` lalu FastAPI di `http://127.0.0.1:8080/`.
Pengambilan data MT5 tetap dari Windows host karena package `MetaTrader5` hanya tersedia di Windows.
