# XAUUSD Market Data Collector

Platform Python 3.12+ untuk mengambil native OHLC bar dan historical tick
XAUUSD/GOLD dari API resmi MetaTrader 5, memvalidasinya, lalu menyimpannya ke
PostgreSQL 16+. Lapisan collector menangani **data acquisition** untuk riset.
Modul analytics menyediakan EDA, analisis time series, dan eksperimen forecast
melalui pembacaan data secara read-only. Dashboard FastAPI menampilkan dataset
tanpa menjalankan terminal MT5 atau mengubah data sumber.

## Arsitektur

```text
MetaTrader 5 (terminal sudah login)
        |
        v
MT5Provider -- preflight terminal/account/broker/server/symbol
        |
        +---- native bars (partition bulanan)
        +---- native ticks (partition harian)
        |
        v
normalisasi UTC + validasi tanpa fill/interpolasi
        |
        +---- valid   -> PostgreSQL market.*
        +---- invalid -> market.rejection
        |
        v
ingest.checkpoint + ingest.run -> coverage/audit -> pipeline riset berikutnya
```

PostgreSQL adalah satu-satunya source of truth market data. Tidak ada
persistence market data ke CSV, Parquet, SQLite, pickle, JSON file, atau folder
lokal. Penjelasan komponen dan boundary tersedia di
[arsitektur](docs/architecture.md), sedangkan alur transaksi rinci ada di
[alur data](docs/data-flow.md).

Workflow acquisition kini dipusatkan ke dua file dengan section seperti notebook:
`market/profile.py` untuk membaca/menulis profile konfigurasi dan runtime settings,
serta `market/collector.py` untuk MT5, validation, ingestion, coverage dan CLI.
Lihat [panduan market](market/README.md) untuk peta section dan contoh penulisan TOML.

Persistence memakai dua file utama: `database/schema.py` untuk tabel/relasi dan
`database/storage.py` untuk koneksi, transaksi, serta operasi repository.
Riwayat Alembic tetap berada di `database/migrations/`. Peta section dan import
terbaru dijelaskan di [panduan database](database/README.md).

## Persyaratan

- Windows 10/11 (package Python MetaTrader 5 dan terminal berjalan di Windows).
- Python 3.12 atau lebih baru; versi dipilih oleh `uv` melalui `.python-version`.
- [`uv`](https://docs.astral.sh/uv/) untuk dependency, lock, environment, dan
  command execution.
- PostgreSQL 16+; Compose menggunakan PostgreSQL 18.
- MetaTrader 5 Terminal yang sudah login ke account yang tepat.

Collector tidak melakukan login otomatis dan tidak menyimpan password/account
number MT5. Install terminal dari broker, login secara manual, tampilkan simbol
yang akan dipakai di Market Watch, lalu pastikan history sudah diunduh.

## Instalasi dengan `uv`

```powershell
uv sync --dev
Copy-Item .env.example .env
```

Jangan membuat `requirements.txt`; dependency authoritative ada di
`pyproject.toml` dan resolusi terkunci di `uv.lock`. Panduan khusus, alasan
penggunaan, serta daftar command `uv` tersedia di [docs/tooling/uv.md](docs/tooling/uv.md).

## Menjalankan FastAPI dan PostgreSQL 18 dengan Docker

Salin konfigurasi contoh lalu jalankan stack:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Compose menunggu PostgreSQL sehat, menjalankan `alembic upgrade head`, kemudian
menjalankan FastAPI di `http://127.0.0.1:8080/`. Swagger tersedia di
`http://127.0.0.1:8080/docs`. Data PostgreSQL disimpan dalam named volume
`postgres_data`, sehingga tetap tersedia setelah container dihentikan.

Container Linux hanya menjalankan API/dashboard dan migrasi. Pengambilan data
MetaTrader 5 tetap dijalankan dari Windows host karena package resmi
`MetaTrader5` hanya tersedia untuk Windows. Panduan command, konfigurasi, dan
operasional lengkap tersedia di [docs/tooling/docker.md](docs/tooling/docker.md).

## PostgreSQL demo dan live

Pisahkan kedua environment secara fisik, idealnya termasuk user dan secret:

```powershell
createdb gold_market_demo
createdb gold_market_live
```

Contoh `.env` demo:

```dotenv
GOLD_ENVIRONMENT=demo
GOLD_DATABASE_URL=postgresql+psycopg://gold_demo:password@localhost:5432/gold_market_demo
GOLD_MT5_TERMINAL_PATH=C:\Program Files\Broker MT5\terminal64.exe
GOLD_LOG_LEVEL=INFO
```

`GOLD_ENVIRONMENT` dan `GOLD_DATABASE_URL` wajib. Aplikasi menolak database yang
jelas bernama environment lawan, profile yang tidak cocok, serta account MT5
demo/live yang salah. `.env` diabaikan Git; jangan commit credential.

## Broker profile

Salin `configs/brokers/example_demo.toml`, lalu ubah identitas non-secret:

```toml
[source]
source_key = "broker-a-demo-xauusd"
broker_name = "Broker A"
server_name = "Broker-A-Demo"
environment = "demo"
dataset_symbol = "XAUUSD"
requested_symbol = "XAUUSDm"
```

`dataset_symbol` adalah identitas canonical riset. `requested_symbol` harus
cocok persis dengan simbol broker. Jika hanya ada kandidat seperti `GOLD` atau
`XAUUSD.a`, collector melaporkannya tetapi tidak memilih diam-diam.

## Migrasi dan registrasi

Jalankan migrasi pada database environment aktif:

```powershell
uv run alembic upgrade head
```

Validasi terminal/account tanpa menampilkan nomor account:

```powershell
uv run python -m market.collector mt5-info --profile example_demo
uv run python -m market.collector symbol-info --profile example_demo
```

Registrasikan provenance broker/server/environment/symbol:

```powershell
uv run python -m market.collector source-register --profile example_demo
```

`source_key` stabil. Kunci yang sama tidak dapat digunakan ulang untuk source
dengan broker, server, environment, atau actual symbol berbeda.

## Historical collection

Base timeframe riset adalah native M5. Bar diambil per bulan, bukan satu call
10 tahun:

```powershell
uv run python -m market.collector ingest-bars `
  --profile example_demo `
  --start 2016-01-01T00:00:00Z `
  --end 2026-01-01T00:00:00Z `
  --timeframe M5
```

Tick memiliki availability berbeda dan diambil per UTC day:

```powershell
uv run python -m market.collector ingest-ticks `
  --profile example_demo `
  --start 2025-01-01T00:00:00Z `
  --end 2026-01-01T00:00:00Z
```

Request 10 tahun tidak berarti broker menyediakan 10 tahun. Empty partition
dicatat sebagai coverage yang valid; collector tidak menciptakan data untuk
menutupi history yang tidak tersedia.

## Incremental sync, resume, dan idempotensi

```powershell
uv run python -m market.collector sync-bars --profile example_demo --timeframe M5
uv run python -m market.collector sync-ticks --profile example_demo
```

`ingest.checkpoint` adalah state resume authoritative. Partition `fetched` atau
`empty` dilewati pada request identik. Exact bar duplicate di-skip; payload bar
yang direvisi broker disimpan sebagai observation baru dan `market.bar_latest`
memilih revision terbaru. Tick memakai fingerprint native + occurrence ordinal
karena MT5 tidak memberi global tick primary key.

Bulk bar dan tick ditulis dalam batch terbatas (maksimal 1.000 row per
statement) agar jumlah parameter setiap query tetap di bawah batas protokol
PostgreSQL 65.535. Jika satu partisi gagal saat penyimpanan, transaksinya
di-rollback dan checkpoint berstatus `failed`; jalankan ulang rentang yang sama
setelah penyebab diperbaiki.

Setiap partition memakai retry terbatas (default jeda 1, 2, 4 detik) dan
transaction sendiri. Batch lama yang sudah committed tidak hilang bila batch
baru gagal.

## Status, coverage, dan validasi

```powershell
uv run python -m market.collector status --profile example_demo
uv run python -m market.collector coverage --profile example_demo
uv run python -m market.collector validate --profile example_demo
```

`market.data_coverage` membedakan requested range dan actual first/last record.
Gap adalah candidate untuk investigasi—weekend, holiday, session broker, atau
history unavailable dapat menjelaskannya—dan tidak diperbaiki otomatis.

## Partitioning PostgreSQL

Parent `market.bar_observation` dan `market.tick_observation` dipartisi RANGE
berdasarkan timestamp UTC. Fungsi `market.ensure_monthly_partitions` membuat
child partition bulanan tepat sebelum bulk insert. Jangan membuat ratusan
migration manual. Index utama mengikuti `(source_id, timeframe, open_time_utc)`
untuk bar dan `(source_id, tick_time_utc)` untuk tick.

## Semantik data penting

- Native MT5 OHLC berasal dari `copy_rates_range`; field `spread` adalah spread
  point pada bar, bukan historical Bid/Ask OHLC lengkap.
- `market.quote_ohlc_m5` dihitung hanya dari historical Bid/Ask tick yang valid.
  Jika tick tidak tersedia, quote OHLC juga tidak tersedia—bukan nol/synthetic.
- `market.processed_m5` menyediakan satu view siap-baca yang menggabungkan
  native M5 terbaru dengan `bid_*`, `ask_*`, statistik `spread_*`, dan provenance
  source. Kolom quote bernilai `NULL` jika tick interval tersebut tidak tersedia.
- `tick_volume` pada bar adalah jumlah perubahan/tick yang dilaporkan broker.
  `real_volume` pada bar dan `volume_real` pada tick adalah field berbeda dan
  sering bernilai nol bila broker tidak menyediakannya; nol bukan error otomatis.
- Semua timestamp canonical berupa timezone-aware UTC / `TIMESTAMPTZ`.
- Bar invalid dan tick invalid masuk `market.rejection`; tidak ada forward fill,
  backfill harga, outlier deletion, atau interpolation.

Lihat dataset M5 gabungan tanpa membuat file Parquet:

```sql
SELECT *
FROM market.processed_m5
ORDER BY open_time_utc DESC
LIMIT 100;
```

Atau jalankan tabel web lokal read-only:

```powershell
uv run alembic upgrade head
uv run python -m market.dashboard
```

Buka `http://127.0.0.1:8080/`. Dashboard dapat berpindah antara market bars,
raw ticks, dan gabungan M5. Tersedia filter waktu/symbol/source/timeframe,
sorting kolom ASC/DESC, pagination, refresh otomatis, kelompok kolom, dan
unduhan CSV untuk halaman aktif.

## Testing dan dokumentasi

```powershell
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv run python scripts/generate_code_reference.py
```

Integration test PostgreSQL hanya berjalan jika `GOLD_TEST_DATABASE_URL`
menunjuk database test disposable yang sudah dimigrasi. Jangan arahkan variable
itu ke demo/live research database. Referensi AST hasil generate berada di
[docs/reference/code-reference.md](docs/reference/code-reference.md).

## Website dokumentasi lokal

Dokumentasi Markdown dapat dibuka sebagai website lokal dengan MkDocs:

```powershell
uv run mkdocs serve
```

Lalu buka `http://127.0.0.1:8000/`. Server melakukan reload otomatis saat file
di `docs/` atau `mkdocs.yml` berubah. Untuk memeriksa website tanpa menjalankan
server, gunakan `uv run mkdocs build --strict`.

## Analytics Matplotlib

Analytics berisi dua file Python: `data.py` untuk query read-only dan
persiapan data; `analytics.py` untuk UNI, BI, MULTI, grafik time series,
model return dengan validasi waktu, evaluasi dan forecast. Kode memakai
pemisah `# %%` seperti notebook. Lihat [analytics README](docs/analytics.md)
untuk metodologi, batas interpretasi dan referensi paper.

```powershell
uv run python -B -m analytics.analytics univariate `
  --source-id <uuid> `
  --symbol XAUUSD `
  --timeframe M5 `
  --start 2026-01-01T00:00:00Z `
  --end 2026-02-01T00:00:00Z `
  --columns close tick_volume spread_price
```

Mode EDA yang tersedia adalah `univariate`, `bivariate`, `multivariate`, dan
`feature-eda`. Evaluasi dijalankan melalui `python -m analytics.analytics evaluate`.
Feature EDA dan model evaluation memerlukan tabel dari pipeline feature/model
mendatang; migration collector saat ini baru menyediakan raw market data. PNG
tersimpan di `reports/analytics/` dan tidak menggantikan PostgreSQL sebagai
source of truth. Subfolder outputnya adalah `univariate/`, `bivariate/`,
`multivariate/`, `feature_eda/`, dan `model_evaluation/`. Menjalankan command
yang sama akan mengganti PNG turunan dengan hasil terbaru.

Agar EDA terus membaca perubahan baru setelah ingestion, gunakan polling:

```powershell
uv run python -B -m analytics.analytics univariate `
  --source-id <uuid> `
  --symbol XAUUSD `
  --timeframe M5 `
  --start 2016-01-01T00:00:00Z `
  --end 2027-01-01T00:00:00Z `
  --watch-seconds 60 `
  --columns close tick_volume spread_price
```

Setiap 60 detik command meng-query PostgreSQL ulang dan menimpa PNG yang sama.
Tekan `Ctrl+C` untuk berhenti. Watcher menggunakan polling, bukan trigger
database; perubahan terlihat setelah transaksi ingestion sudah committed.

### Forecast XGBoost 2026 tanpa data leakage

analytics.analytics mengagregasi bar native menjadi close harian lalu melatih
XGBoost hanya dengan data sebelum 2026-01-01T00:00:00Z, sehingga harga terakhir
yang dapat dipakai model adalah 31 Desember 2025. Prediksi 2026 dibuat secara
recursive: setiap lag setelah Januari 2026 berasal dari forecast sebelumnya,
bukan harga aktual. Harga 2026 dimuat pada query terpisah hanya untuk garis
hijau dan metrik pembanding.

```powershell
uv run python -B -m analytics.analytics forecast `
  --source-id <uuid> `
  --symbol XAUUSD `
  --timeframe M5 `
  --history-start 2016-01-01T00:00:00Z `
  --actual-end 2026-11-01T00:00:00Z
```

Output berada di reports/analytics/xgboost_forecast/: dashboard.html interaktif
yang mendukung zoom, pan, hover, dan date-range slider; grafik historis
biru/forecast merah/aktual hijau, actual-vs-predicted, residual, perbandingan
bulanan, distribusi error, serta metrics.json berisi MAE, MSE, RMSE, dan MAPE.
--actual-end hanya menentukan
seberapa jauh ground truth tersedia untuk dibandingkan; close pada rentang itu
tidak pernah menjadi input XGBoost.

## Troubleshooting

- **MT5 initialization failed**: buka terminal Windows, login, cek path
  `terminal64.exe`, lalu jalankan `mt5-info`.
- **Environment/account mismatch**: jangan mengubah label profile untuk
  mengakali validasi; login ke account yang sesuai atau gunakan `.env`/database
  environment yang benar.
- **Server/broker mismatch**: samakan profile dengan nilai aktual yang tampil
  dari terminal. Source key lama tidak boleh dipakai untuk provenance baru.
- **Symbol unavailable/ambiguous**: gunakan nama exact dari Market Watch dan
  aktifkan simbol tersebut; jangan mengandalkan substring.
- **History pendek/empty**: di MT5 buka chart M5 dan muat history, cek batas
  “Max bars in chart”, lalu periksa `coverage`. Availability broker tetap batas
  akhir dan tidak boleh disebut lengkap tanpa bukti.
- **PostgreSQL unavailable**: cek service, database, role, network, dan DSN.
  DSN disembunyikan dari logs.
- **Migration gagal**: pastikan role boleh membuat schema, table, view, function,
  dan partition pada database yang dituju.

## Struktur proyek

```text
market/                 domain, MT5 boundary, validation, services, CLI
database/               SQLAlchemy mappings, repositories, Alembic migration
analytics/              pembacaan read-only, EDA, time series dan forecast
configs/brokers/        profile non-secret per broker/server/environment
tests/                  unit tests dan integration test opt-in
docs/                   arsitektur, alur, tooling, dan generated reference
scripts/                deterministic documentation generator
```

Dokumentasi adalah bagian Definition of Done: perubahan behavior harus mengubah
docstring, test, serta dokumen arsitektur terkait pada commit yang sama.

## Insight time series

Gunakan `uv run python -B -m analytics.analytics time-series --source-id <uuid> --symbol XAUUSD --timeframe M5 --start 2016-01-01T00:00:00Z --end 2026-10-01T00:00:00Z --include-models`.
Laporan mencakup tren/drawdown, return/volatilitas, distribusi tahunan, return
bulanan, ACF, aktivitas/spread, Spearman, counts dan perbandingan model
UNI/BI/MULTI versus baseline. Tahun/bulan terakhir mengikuti cakupan
observasi dan dapat parsial; harga meningkat jangka panjang belum berarti
setiap sesi naik. Output berada di `reports/analytics/time_series/`.
