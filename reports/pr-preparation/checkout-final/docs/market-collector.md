# Build Request — XAUUSD / GOLD Market Data Ingestion Platform

Buat project Python baru dari nol untuk mengumpulkan historical dan incremental market data **XAUUSD / GOLD** dari **MetaTrader 5** dan menyimpannya ke **PostgreSQL**.

Project ini merupakan tahap **Data Acquisition / Data Collection** untuk penelitian Machine Learning prediksi XAUUSD.

Jangan membuat model Machine Learning pada tahap ini.

Jangan menggunakan, menyalin, atau bergantung pada project lama.

---

# 1. Tujuan

Bangun sistem market-data ingestion dengan flow:

```text
MetaTrader 5
      ↓
MT5 Python API
      ↓
Source / Broker Validation
      ↓
Historical / Incremental Collector
      ↓
Data Validation
      ↓
PostgreSQL
      ↓
Data Understanding
      ↓
Preprocessing
      ↓
Feature Engineering
      ↓
Machine Learning
```

Sistem harus:

1. Mengambil historical market data XAUUSD/GOLD hingga sekitar **10 tahun**, jika tersedia dari broker.
2. Mengambil incremental market data setelah historical backfill selesai.
3. Mengambil native OHLC bars dari MetaTrader 5.
4. Mengambil historical Bid/Ask ticks jika tersedia.
5. Menyimpan `tick_volume`, `real_volume`, spread, dan field market-data lain yang diberikan MT5.
6. Menyimpan seluruh persistence market data di PostgreSQL.
7. Mendukung banyak broker.
8. Mendukung banyak server.
9. Mendukung account `demo` dan `live`.
10. Menjaga provenance sehingga setiap record diketahui berasal dari broker/server/symbol mana.
11. Mendukung retry.
12. Mendukung checkpoint dan resume.
13. Bersifat idempotent.
14. Menyediakan audit trail.
15. Menyediakan data validation.
16. Bersifat **READ-ONLY terhadap trading account**.

Project TIDAK BOLEH:

- membuka order;
- menutup order;
- modify order;
- melakukan trading;
- memanggil `MetaTrader5.order_send`;
- membuat BUY/SELL signal;
- membuat technical indicator;
- melakukan feature engineering;
- melakukan normalization ML;
- menghapus outlier;
- melakukan interpolation harga;
- melakukan training model.

Catatan:

> Ini bukan web scraping HTML. Gunakan API resmi Python `MetaTrader5`.

---

# 2. Prinsip Utama

Prioritas implementasi:

```text
Correctness
    ↓
Data Integrity
    ↓
Reproducibility
    ↓
Auditability
    ↓
Performance
    ↓
Convenience
```

Raw market data harus dipertahankan semirip mungkin dengan data yang diberikan broker.

Collector hanya bertanggung jawab pada:

```text
Extract
Validate
Normalize metadata
Normalize timestamps
Store
Audit
```

Bukan:

```text
Feature Engineering
Outlier Removal
Interpolation
Standardization
Normalization ML
Prediction
Trading
```

---

# 3. Teknologi

Gunakan:

- Python 3.12+
- `uv` untuk dependency management
- MetaTrader5 Python package
- PostgreSQL 16+
- SQLAlchemy 2.x
- psycopg
- Alembic
- Pydantic Settings
- pytest
- structured logging menggunakan `structlog` atau JSON logging

Environment utama adalah **Windows**, karena MetaTrader 5 Terminal dijalankan pada Windows.

Tambahkan dependency menggunakan:

```powershell
uv add <package>
```

Jangan membuat `requirements.txt` secara manual jika project menggunakan `uv`.

Gunakan:

```text
pyproject.toml
uv.lock
```

---

# 4. Struktur Project

Gunakan struktur sederhana berikut:

```text
gold-market-data/
│
├── market/
│   ├── __init__.py
│   ├── contracts.py
│   ├── profile.py        # schema, loader/saver config TOML, runtime settings
│   ├── collector.py      # MT5, validation, partition, ingestion, coverage, CLI
│   ├── dashboard.py      # API/browser read-only
│   ├── static/
│   └── cli.py
│
├── database/
│   ├── __init__.py
│   ├── schema.py          # ORM tables, relasi, constraints, metadata
│   ├── storage.py         # engine/session/transaksi dan seluruh repository
│   │
│   └── migrations/
│       ├── env.py
│       └── versions/
│
├── tests/
│   ├── market/
│   └── database/
│
├── configs/
│   └── brokers/
│       ├── example_demo.toml
│       └── example_live.toml
│
├── .env.example
├── .gitignore
├── alembic.ini
├── pyproject.toml
├── uv.lock
└── README.md
```

Jangan membuat folder:

```text
data/
raw/
parquet/
csv/
collector/
goldmind/
```

Market data tidak boleh menggunakan filesystem sebagai primary persistence.

---

# 5. PostgreSQL sebagai Single Source of Truth

Seluruh market data harus disimpan di PostgreSQL.

Jangan gunakan:

- Parquet;
- CSV;
- SQLite;
- JSON files;
- pickle;
- local filesystem;

sebagai source of truth.

Flow:

```text
MetaTrader 5
      ↓
PostgreSQL
      ↓
Research / ML Pipeline
```

---

# 6. Pemisahan Demo dan Live

Pisahkan environment demo dan live secara fisik.

Gunakan database berbeda.

```text
Research / Demo:

gold_market_demo
```

```text
Live:

gold_market_live
```

Environment harus eksplisit.

Gunakan:

```env
GOLD_ENVIRONMENT=demo
```

atau:

```env
GOLD_ENVIRONMENT=live
```

Jangan mempunyai default:

```text
live
```

Jika environment tidak ditentukan, aplikasi harus gagal startup.

Database:

```text
demo
→ gold_market_demo
```

```text
live
→ gold_market_live
```

Jangan mengizinkan collector demo menulis ke database live atau sebaliknya.

Untuk production, live database idealnya menggunakan:

- PostgreSQL user berbeda;
- secret berbeda;
- deployment berbeda;
- network permission berbeda.

---

# 7. Sensitive Configuration

Jangan hardcode:

- PostgreSQL password;
- PostgreSQL production DSN;
- MT5 login;
- MT5 password;
- account number;
- token;
- secret;
- private broker server;
- terminal path pribadi.

Gunakan environment variable.

Buat:

```env
GOLD_ENVIRONMENT=demo

GOLD_DATABASE_URL=postgresql+psycopg://<user>:<password>@<host>:5432/<database>

GOLD_MT5_TERMINAL_PATH=

GOLD_LOG_LEVEL=INFO
```

`.env` tidak boleh masuk Git.

Tambahkan ke `.gitignore`.

---

# 8. MT5 Authentication Strategy

Jangan menyimpan credential MT5 di application database.

Asumsikan MetaTrader 5 Terminal sudah login pada account yang ingin digunakan.

Aplikasi melakukan:

```python
mt5.initialize(...)
```

kemudian membaca:

```python
mt5.terminal_info()
mt5.account_info()
```

Collector tidak perlu melakukan automatic login menggunakan password.

Jika nanti login otomatis diperlukan, buat sebagai extension terpisah dan jangan menyimpan credential di repository.

---

# 9. Mandatory MT5 Preflight Validation

Sebelum mengambil data, lakukan validasi berikut.

## Terminal

Pastikan:

```text
MT5 initialized
terminal connected
terminal path sesuai konfigurasi jika diberikan
```

## Account

Ambil:

```python
mt5.account_info()
```

Validasi minimal:

```text
server
company
trade_mode
currency
```

Mapping account harus mengenali:

```text
DEMO
LIVE / REAL
CONTEST jika broker mendukung
```

Jika:

```text
GOLD_ENVIRONMENT=demo
```

tetapi account MT5 adalah live/real:

```text
FAIL IMMEDIATELY
```

Jika:

```text
GOLD_ENVIRONMENT=live
```

tetapi account MT5 adalah demo:

```text
FAIL IMMEDIATELY
```

Jangan hanya mempercayai nama profile.

Validasi profile terhadap **account aktual yang sedang login di MT5**.

Jangan log account number.

Account number boleh digunakan sementara untuk validation internal tetapi tidak boleh muncul pada log biasa.

---

# 10. Broker Profile

Simpan konfigurasi non-secret pada:

```text
configs/brokers/<profile>.toml
```

Contoh:

```toml
[source]
source_key = "example-demo-xauusd"
broker_name = "Example Broker"
server_name = "Example-Demo"
environment = "demo"

dataset_symbol = "XAUUSD"
requested_symbol = "XAUUSD"

[market]
base_timeframe = "M5"
additional_native_timeframes = ["M15", "H1"]
timezone = "UTC"

[collection]
bar_chunk = "1month"
tick_chunk = "1day"

max_attempts = 3
retry_delays_seconds = [1, 2, 4]
```

Validasi:

```text
profile.environment == GOLD_ENVIRONMENT
```

Validasi juga:

```text
profile.server_name
```

dengan server account yang sedang aktif jika informasi tersebut tersedia.

Jika mismatch, fail fast.

---

# 11. Symbol Normalization

Broker dapat menggunakan nama instrument berbeda.

Contoh:

```text
XAUUSD
XAUUSDm
XAUUSD.a
XAUUSD.pro
GOLD
GOLDm
```

Pisahkan:

```text
dataset_symbol
```

dan:

```text
actual_symbol
```

Contoh:

```text
dataset_symbol = XAUUSD
actual_symbol = XAUUSDm
```

`dataset_symbol` adalah nama canonical untuk penelitian.

`actual_symbol` adalah symbol asli broker.

Jangan melakukan silent auto-mapping ke instrument yang tidak pasti.

Jika symbol yang dikonfigurasi tidak ditemukan:

1. cari candidate symbols;
2. log candidate;
3. fail dengan error yang jelas.

Jangan otomatis memilih symbol hanya berdasarkan substring jika terdapat beberapa kandidat ambigu.

---

# 12. Symbol Metadata

Gunakan:

```python
mt5.symbol_info(symbol)
```

Simpan metadata instrument seperti:

```text
digits
point
trade_tick_size
trade_tick_value
trade_contract_size
volume_min
volume_max
volume_step
currency_base
currency_profit
```

Metadata instrument dapat berubah.

Jangan menganggap symbol specification hari ini berlaku untuk seluruh data 10 tahun.

Karena itu gunakan versioning.

---

# 13. PostgreSQL Schema

Gunakan PostgreSQL schemas:

```text
core
ingest
market
audit
```

---

# 14. core.data_source

Simpan identitas source.

Kolom minimum:

```text
id UUID PRIMARY KEY

source_key TEXT UNIQUE NOT NULL

broker_name TEXT NOT NULL
server_name TEXT NOT NULL

environment TEXT NOT NULL

dataset_symbol TEXT NOT NULL
actual_symbol TEXT NOT NULL

created_at_utc TIMESTAMPTZ NOT NULL

active BOOLEAN NOT NULL
```

Constraint:

```text
environment IN ('demo', 'live')
```

Contoh:

```text
metaquotes-demo-xauusd
icmarkets-demo-xauusd
icmarkets-live-xauusd
broker-a-live-xauusdm
```

`source_key` harus:

- unik;
- stabil;
- tidak berubah antar ingestion run.

Jangan mencampur:

```text
broker
server
environment
actual_symbol
```

yang berbeda ke source yang sama.

---

# 15. core.symbol_spec_version

Simpan versioned symbol metadata.

Kolom minimum:

```text
id UUID PRIMARY KEY

source_id UUID NOT NULL

observed_at_utc TIMESTAMPTZ NOT NULL

actual_symbol TEXT NOT NULL

digits INTEGER

point NUMERIC(20,10)

trade_tick_size NUMERIC(20,10)

trade_tick_value NUMERIC(30,10)

trade_contract_size NUMERIC(30,10)

volume_min NUMERIC(30,10)

volume_max NUMERIC(30,10)

volume_step NUMERIC(30,10)

currency_base TEXT

currency_profit TEXT

payload_hash TEXT NOT NULL
```

Optional:

```text
raw_payload JSONB
```

Karena symbol specification jumlah record-nya kecil, JSONB diperbolehkan pada tabel ini.

---

# 16. ingest.run

Setiap proses collection mempunyai satu ingestion run.

Kolom:

```text
id UUID PRIMARY KEY

source_id UUID NOT NULL

kind TEXT NOT NULL

requested_start_utc TIMESTAMPTZ

requested_end_utc TIMESTAMPTZ

started_at_utc TIMESTAMPTZ NOT NULL

finished_at_utc TIMESTAMPTZ

status TEXT NOT NULL

rows_received BIGINT NOT NULL DEFAULT 0

rows_inserted BIGINT NOT NULL DEFAULT 0

rows_rejected BIGINT NOT NULL DEFAULT 0

rows_skipped BIGINT NOT NULL DEFAULT 0

error_code TEXT

error_message TEXT

collector_version TEXT
```

`kind`:

```text
bars
ticks
symbol_spec
```

`status`:

```text
running
completed
partial
failed
```

---

# 17. ingest.checkpoint

Checkpoint digunakan menggantikan manifest file.

Kolom:

```text
id UUID PRIMARY KEY

source_id UUID NOT NULL

kind TEXT NOT NULL

timeframe TEXT

partition_start_utc TIMESTAMPTZ NOT NULL

partition_end_utc TIMESTAMPTZ NOT NULL

status TEXT NOT NULL

last_success_at_utc TIMESTAMPTZ

last_error TEXT

run_id UUID NOT NULL
```

Status:

```text
fetched
empty
partial
failed
```

Unique constraint:

```text
source_id
kind
timeframe
partition_start_utc
partition_end_utc
```

Checkpoint hanya berubah menjadi:

```text
fetched
```

setelah:

```text
fetch
→ validation
→ database insert
→ transaction commit
```

berhasil.

---

# 18. Historical Coverage

JANGAN menganggap request 10 tahun berarti broker pasti menyediakan data 10 tahun.

Contoh request:

```text
2016-01-01
→
2026-01-01
```

bukan berarti seluruh periode tersedia.

Collector harus mencatat:

```text
requested_start
requested_end

actual_first_record
actual_last_record

requested_partitions
fetched_partitions
empty_partitions
failed_partitions
```

Sediakan query atau view:

```text
market.data_coverage
```

agar coverage setiap:

```text
source
symbol
timeframe
```

dapat diperiksa.

Contoh:

```text
Requested:
2016-01-01 → 2026-01-01

Actual available:
2017-05-03 → 2026-01-01
```

Ini bukan error jika broker memang tidak menyediakan data sebelumnya.

Tetapi harus terlihat jelas dalam report.

---

# 19. MT5 Historical Limitation Check

MetaTrader 5 dapat membatasi jumlah bars berdasarkan history yang tersedia pada terminal dan konfigurasi terminal.

Sebelum historical backfill:

1. baca `terminal_info()`;
2. record relevant terminal configuration;
3. beri warning jika history/max bars dapat membatasi request;
4. dokumentasikan pada README bagaimana memastikan history tersedia.

Jangan mengklaim dataset lengkap jika source tidak menyediakan periode tersebut.

---

# 20. Base Timeframe

Gunakan:

```text
M5
```

sebagai **base research timeframe**.

M5 menjadi dataset utama untuk pipeline ML.

Timeframe berikut nantinya dapat dibentuk dari M5:

```text
M15
M30
H1
H4
```

Tetapi collector boleh mengambil native broker bars:

```text
M15
H1
```

jika dikonfigurasi untuk:

- auditing;
- cross-validation;
- membandingkan hasil resampling.

Default:

```text
base_timeframe = M5
```

Higher timeframe tidak perlu menjadi source of truth utama.

---

# 21. market.bar_observation

Simpan native MT5 bar observations.

Partition:

```text
RANGE(open_time_utc)
```

dengan partition bulanan.

Kolom:

```text
id UUID NOT NULL

source_id UUID NOT NULL

symbol_spec_version_id UUID

actual_symbol TEXT NOT NULL

timeframe TEXT NOT NULL

open_time_utc TIMESTAMPTZ NOT NULL

close_time_utc TIMESTAMPTZ NOT NULL

open NUMERIC(20,10) NOT NULL

high NUMERIC(20,10) NOT NULL

low NUMERIC(20,10) NOT NULL

close NUMERIC(20,10) NOT NULL

tick_volume BIGINT NOT NULL

real_volume NUMERIC(30,10)

spread_points BIGINT

spread_price NUMERIC(20,10)

ingested_at_utc TIMESTAMPTZ NOT NULL

payload_hash TEXT NOT NULL

quality_status TEXT NOT NULL
```

Karena tabel dipartisi berdasarkan:

```text
open_time_utc
```

jangan membuat:

```text
id UUID PRIMARY KEY
```

sendirian.

Gunakan primary key yang kompatibel dengan PostgreSQL partitioning, misalnya:

```text
PRIMARY KEY (open_time_utc, id)
```

---

# 22. Bar Idempotency dan Revision

Logical identity bar:

```text
source_id
actual_symbol
timeframe
open_time_utc
```

Tetapi broker dapat memberikan revisi data.

Jangan diam-diam overwrite data lama.

Hitung:

```text
payload_hash
```

berdasarkan field market bar asli.

Gunakan uniqueness yang memungkinkan revision tetapi mencegah exact duplicate.

Contoh:

```text
UNIQUE (
    open_time_utc,
    source_id,
    actual_symbol,
    timeframe,
    payload_hash
)
```

Dengan demikian:

```text
same timestamp + same payload
→ duplicate
→ skip
```

tetapi:

```text
same timestamp + different payload
→ revision
→ insert observation baru
```

Buat view:

```text
market.bar_latest
```

yang memilih observation terbaru berdasarkan:

```text
source_id
actual_symbol
timeframe
open_time_utc
```

untuk penggunaan downstream.

Raw observation tetap append-only.

---

# 23. Bar Validation

Validasi:

```text
open > 0
high > 0
low > 0
close > 0
```

dan:

```text
high >= open
high >= close
high >= low

low <= open
low <= close
```

Validasi:

```text
tick_volume >= 0
real_volume >= 0 jika tersedia
spread >= 0 jika tersedia
```

Jangan:

```text
forward fill
back fill
interpolation
```

Jangan membuat candle synthetic untuk mengisi gap.

---

# 24. Candle yang Belum Selesai

Candle yang sedang berjalan tidak boleh dianggap sebagai final historical bar.

Untuk incremental ingestion, hanya finalisasi candle jika:

```text
current_utc >= close_time_utc + safety_lag
```

Gunakan safety lag configurable.

Contoh:

```toml
finalization_delay_seconds = 5
```

Candle unfinished dapat:

- dilewati;
- atau disimpan pada staging;

tetapi tidak boleh masuk sebagai final observation.

---

# 25. Spread pada Native Bars

MT5 memberikan field:

```text
spread
```

pada bar.

Simpan sebagai:

```text
spread_points
```

Jika `point` tersedia dari symbol specification:

```text
spread_price = spread_points * point
```

Jangan menganggap native bar spread sama dengan complete historical Bid/Ask OHLC.

---

# 26. market.tick_observation

Bid/Ask harus berasal dari historical ticks.

Gunakan:

```python
mt5.copy_ticks_range(...)
```

dengan mode yang sesuai.

Simpan field native yang tersedia:

```text
time
time_msc
bid
ask
last
volume
flags
volume_real
```

Schema:

```text
id UUID NOT NULL

source_id UUID NOT NULL

actual_symbol TEXT NOT NULL

tick_time_utc TIMESTAMPTZ NOT NULL

timestamp_ms BIGINT NOT NULL

occurrence INTEGER NOT NULL

bid NUMERIC(20,10)

ask NUMERIC(20,10)

last NUMERIC(20,10)

volume NUMERIC(30,10)

volume_real NUMERIC(30,10)

flags BIGINT

spread_price NUMERIC(20,10)

is_valid_quote BOOLEAN NOT NULL

payload_hash TEXT NOT NULL

quality_status TEXT NOT NULL

ingested_at_utc TIMESTAMPTZ NOT NULL
```

Partition:

```text
RANGE(tick_time_utc)
```

per bulan.

Primary key:

```text
PRIMARY KEY (tick_time_utc, id)
```

bukan:

```text
id UUID PRIMARY KEY
```

sendirian.

---

# 27. Jangan Simpan JSONB pada Setiap Tick Valid

Jangan menyimpan:

```text
raw_payload JSONB
```

pada setiap tick valid secara default.

Alasannya:

10 tahun tick dapat menghasilkan data sangat besar dan field native MT5 sudah disimpan dalam structured columns.

Gunakan:

```text
payload_hash
```

untuk integrity.

JSONB raw payload digunakan terutama untuk:

```text
rejection
symbol metadata
error diagnostics
```

Jika nantinya benar-benar dibutuhkan, buat feature flag:

```text
STORE_VALID_TICK_RAW_PAYLOAD=false
```

Default harus:

```text
false
```

---

# 28. Tick Identity

Jangan menggunakan:

```text
timestamp
```

saja sebagai unique key.

Dalam satu millisecond dapat terdapat beberapa tick.

Gunakan:

```text
timestamp_ms
bid
ask
last
volume
volume_real
flags
```

untuk membentuk:

```text
payload_hash
```

Kemudian buat:

```text
occurrence
```

sebagai ordinal untuk payload identik dalam partition fetch yang sama.

Karena tick di-fetch dengan fixed daily partitions, occurrence harus dihitung secara deterministic.

Contoh logical unique identity:

```text
tick_time_utc
source_id
actual_symbol
payload_hash
occurrence
```

Dokumentasikan bahwa MetaTrader 5 tidak memberikan global tick primary key sehingga deduplication menggunakan deterministic fingerprint + occurrence.

---

# 29. Tick Validation

Jika quote memiliki Bid/Ask:

```text
bid > 0
ask > 0
ask >= bid
```

Hitung:

```text
spread_price = ask - bid
```

Jika tick hanya merupakan trade tick dan tidak membawa Bid/Ask valid:

jangan membuat Bid/Ask palsu.

Jangan:

```text
bid = close
ask = close
```

Jangan mengganti missing Bid/Ask dengan candle price.

Invalid record masuk ke rejection/audit.

---

# 30. Historical Tick Availability

JANGAN mensyaratkan broker menyediakan 10 tahun complete tick data.

Historical bars dan historical ticks memiliki coverage berbeda.

Collector harus mendukung:

```text
10-year bar backfill
```

dan:

```text
configurable tick backfill
```

secara terpisah.

Contoh:

```powershell
uv run python -m market.collector ingest-bars `
  --profile example_demo `
  --start 2016-01-01T00:00:00Z `
  --end 2026-01-01T00:00:00Z `
  --timeframe M5
```

Sedangkan ticks:

```powershell
uv run python -m market.collector ingest-ticks `
  --profile example_demo `
  --start 2025-01-01T00:00:00Z `
  --end 2026-01-01T00:00:00Z
```

Jika broker ternyata menyediakan lebih banyak historical ticks, collector harus dapat melanjutkan ke periode tersebut.

---

# 31. market.quote_ohlc_m5

Native MT5 OHLC dan Bid/Ask OHLC merupakan dataset berbeda.

Buat:

```text
market.quote_ohlc_m5
```

sebagai SQL view atau materialized view berdasarkan:

```text
market.tick_observation
```

Kolom:

```text
source_id
actual_symbol

open_time_utc

bid_open
bid_high
bid_low
bid_close

ask_open
ask_high
ask_low
ask_close

spread_open
spread_high
spread_low
spread_close

spread_mean
spread_median
spread_std

tick_count

first_tick_time_utc
last_tick_time_utc
```

Rules:

```text
Bid OHLC
→ aggregate Bid ticks

Ask OHLC
→ aggregate Ask ticks
```

Jangan mengarang Ask OHLC dari native candle.

Jika historical ticks tidak tersedia:

```text
quote OHLC = unavailable
```

bukan:

```text
0
```

dan bukan synthetic values.

---

# 32. market.rejection

Data invalid jangan dibuang tanpa jejak.

Simpan:

```text
id UUID PRIMARY KEY

run_id UUID NOT NULL

source_id UUID NOT NULL

record_kind TEXT NOT NULL

event_time_utc TIMESTAMPTZ

reason_code TEXT NOT NULL

raw_payload JSONB NOT NULL

rejected_at_utc TIMESTAMPTZ NOT NULL
```

Contoh reason:

```text
INVALID_OHLC
NEGATIVE_VOLUME
NEGATIVE_SPREAD
ASK_BELOW_BID
INVALID_TIMESTAMP
MISSING_SYMBOL
UNFINISHED_BAR
MALFORMED_PAYLOAD
```

---

# 33. Historical Collection Strategy

Jangan melakukan:

```text
2016 → 2026
```

dalam satu API call.

Bars menggunakan fixed partition:

```text
1 month
```

Contoh:

```text
2016-01
2016-02
2016-03
...
2025-12
2026-01
```

Ticks menggunakan:

```text
1 day
```

Contoh:

```text
2025-01-01
2025-01-02
2025-01-03
...
```

---

# 34. Database Transaction

Untuk setiap partition:

```text
Create / update ingestion run
        ↓
Fetch MT5
        ↓
Validate
        ↓
Bulk insert
        ↓
Insert rejection
        ↓
Commit
        ↓
Update checkpoint = fetched
```

Jika error:

```text
Rollback current transaction
        ↓
Checkpoint = failed / partial
        ↓
Save error
        ↓
Retry jika diperbolehkan
```

Batch sebelumnya yang sudah committed tidak boleh hilang.

---

# 35. Retry

Gunakan bounded retry.

Contoh:

```text
Attempt 1
wait 1 second

Attempt 2
wait 2 seconds

Attempt 3
wait 4 seconds
```

Jangan infinite retry.

Error permanent seperti:

```text
invalid symbol
environment mismatch
wrong broker
wrong server
```

harus fail fast dan tidak perlu retry.

---

# 36. Resume

Collector harus resumable.

Jika proses terhenti pada:

```text
2019-06
```

dan:

```text
2016-01 → 2019-05
```

sudah fetched:

run berikutnya harus melanjutkan partition yang belum selesai.

Jangan fetch ulang seluruh periode secara otomatis.

Gunakan:

```text
ingest.checkpoint
```

sebagai authoritative resume state.

---

# 37. Idempotency

Menjalankan command yang sama dua kali tidak boleh membuat exact duplicate.

Contoh:

```text
ingest-bars 2019-01 → 2020-01

run again

ingest-bars 2019-01 → 2020-01
```

hasil database harus tetap konsisten.

Tetapi apabila broker mengembalikan revisi data:

```text
same logical candle
different payload_hash
```

simpan sebagai revision.

---

# 38. Gap Detection

Buat utility untuk mendeteksi candidate gap.

Tetapi jangan langsung menganggap semua gap sebagai missing data.

Pertimbangkan:

- weekend;
- broker session;
- holiday;
- market close;
- instrument session;
- historical data unavailable.

Output gap detection:

```text
candidate_gap
```

bukan:

```text
automatically repaired
```

Jangan interpolation.

Jangan synthetic candle.

---

# 39. Aggregation M5 → Higher Timeframe

Buat service yang dapat membangun:

```text
M15
M30
H1
H4
```

dari M5.

OHLC:

```text
OPEN  = first(open)
HIGH  = max(high)
LOW   = min(low)
CLOSE = last(close)
```

Volume:

```text
tick_volume = sum(tick_volume)
real_volume = sum(real_volume)
```

Jangan menjumlahkan spread.

Spread dapat menggunakan:

```text
spread_open
spread_high
spread_low
spread_close
spread_mean
spread_median
```

Aggregation merupakan derived dataset.

Native M5 tetap dipertahankan.

---

# 40. Timezone

Gunakan UTC sebagai canonical timezone.

Database:

```text
TIMESTAMPTZ
```

Semua datetime object untuk MT5 harus dibuat timezone-aware.

Jangan gunakan naive datetime.

Simpan:

```text
*_utc
```

untuk timestamp canonical.

Jangan melakukan conversion berdasarkan timezone Windows secara diam-diam.

---

# 41. Incremental Mode

Setelah historical backfill selesai, support:

```text
incremental collection
```

Contoh:

```powershell
uv run python -m market.collector sync-bars `
  --profile example_demo `
  --timeframe M5
```

Incremental collector:

```text
Read latest completed candle
        ↓
Request newer MT5 data
        ↓
Ignore unfinished candle
        ↓
Validate
        ↓
Insert
        ↓
Checkpoint
```

Ticks:

```powershell
uv run python -m market.collector sync-ticks `
  --profile example_demo
```

---

# 42. CLI

Sediakan:

```powershell
uv run alembic upgrade head
```

Register source:

```powershell
uv run python -m market.collector source-register `
  --profile example_demo
```

Inspect MT5:

```powershell
uv run python -m market.collector mt5-info `
  --profile example_demo
```

Inspect symbol:

```powershell
uv run python -m market.collector symbol-info `
  --profile example_demo
```

Historical bars:

```powershell
uv run python -m market.collector ingest-bars `
  --profile example_demo `
  --start 2016-01-01T00:00:00Z `
  --end 2026-01-01T00:00:00Z `
  --timeframe M5
```

Ticks:

```powershell
uv run python -m market.collector ingest-ticks `
  --profile example_demo `
  --start 2025-01-01T00:00:00Z `
  --end 2026-01-01T00:00:00Z
```

Incremental:

```powershell
uv run python -m market.collector sync-bars `
  --profile example_demo `
  --timeframe M5
```

Status:

```powershell
uv run python -m market.collector status `
  --profile example_demo
```

Coverage:

```powershell
uv run python -m market.collector coverage `
  --profile example_demo
```

Validate:

```powershell
uv run python -m market.collector validate `
  --profile example_demo
```

---

# 43. Explicit MT5 Access

Jangan fetch MT5 otomatis ketika:

```text
module import
pytest startup
database migration
application import
```

MT5 hanya boleh diakses oleh explicit command.

Contoh:

```text
ingest-bars
ingest-ticks
sync-bars
sync-ticks
mt5-info
symbol-info
```

---

# 44. Performance

Gunakan:

```text
bulk insert
batch transaction
PostgreSQL COPY jika sesuai
ON CONFLICT jika sesuai
```

Repository membatasi bulk insert bar dan tick menjadi maksimal 1.000 row per
statement. Batas ini menjaga jumlah bind parameter tetap di bawah limit
protokol PostgreSQL 65.535 ketika satu partisi bulanan berisi ribuan candle
atau satu partisi tick berisi banyak record. Setiap partisi tetap diproses
secara atomik: kegagalan satu batch me-roll back seluruh partisi dan checkpoint
berstatus `failed`, sehingga partisi tersebut aman untuk di-resume.

Hindari:

```text
INSERT satu per satu
SELECT berulang untuk setiap row
```

Index berdasarkan pola query utama.

Bar:

```text
(source_id, timeframe, open_time_utc)
```

Tick:

```text
(source_id, tick_time_utc)
```

Jangan membuat terlalu banyak indexes pada tick table karena dapat memperlambat ingestion.

---

# 45. PostgreSQL Partition Management

Karena data dapat mencakup 10 tahun, buat mekanisme membuat monthly partitions.

Contoh:

```text
bar_observation_2016_01
bar_observation_2016_02
...
```

Ticks:

```text
tick_observation_2025_01
tick_observation_2025_02
...
```

Migration harus membuat parent table.

Buat helper/service untuk memastikan required monthly partition tersedia sebelum insert.

Jangan manual membuat ratusan migration satu per bulan.

---

# 46. Logging

Gunakan structured JSON logging.

Field minimum:

```text
run_id
source_id

broker_name
server_name
environment

actual_symbol
timeframe

operation
partition

status

rows_received
rows_inserted
rows_skipped
rows_rejected

reason_code

timestamp_utc
```

Jangan log:

```text
PostgreSQL password
full DSN
MT5 password
MT5 login
account number
secret
token
```

Contoh log:

```json
{
  "operation": "ingest_bars",
  "source": "example-demo-xauusd",
  "timeframe": "M5",
  "partition": "2025-01",
  "rows_received": 6432,
  "rows_inserted": 6430,
  "rows_skipped": 2,
  "rows_rejected": 0,
  "status": "completed"
}
```

---

# 47. Error Handling

Handle minimal:

```text
MT5 initialize failure
MT5 disconnected
wrong account environment
wrong broker
wrong server

symbol unavailable
symbol disabled
ambiguous symbol

historical data unavailable
empty result

PostgreSQL unavailable
database timeout
database conflict

invalid timestamp
invalid OHLC
invalid quote

timezone issue

partial fetch
network interruption
```

Gunakan:

```python
mt5.last_error()
```

jika MT5 mengembalikan failure.

Pastikan:

```python
mt5.shutdown()
```

dipanggil dengan benar.

---

# 48. Testing

Gunakan fake/mock MT5 provider untuk unit tests.

Unit test tidak boleh membutuhkan real MetaTrader terminal.

Wajib test:

1. OHLC validation.
2. Bid/Ask validation.
3. Spread calculation.
4. Symbol normalization.
5. Environment mismatch.
6. Demo profile + live account harus ditolak.
7. Live profile + demo account harus ditolak.
8. Broker mismatch.
9. Server mismatch.
10. Resume checkpoint.
11. Historical partition generation.
12. Ingestion idempotency.
13. Bar revision handling.
14. Duplicate bars.
15. Multiple ticks dalam millisecond yang sama.
16. Exact repeated ticks.
17. Gap detection.
18. Unfinished candle rejection.
19. M5 → M15 aggregation.
20. M5 → M30 aggregation.
21. M5 → H1 aggregation.
22. Quote OHLC tidak dibuat tanpa historical ticks.
23. Tidak ada synthetic Bid/Ask.
24. Tidak ada forward fill.
25. Tidak ada interpolation.
26. Tidak ada filesystem market-data persistence.
27. Tidak ada `order_send`.
28. Sensitive fields tidak muncul di logs.

Integration tests boleh menggunakan PostgreSQL test database terpisah.

---

# 49. Read-Only Trading Safety Test

Tambahkan test yang memastikan source project tidak menggunakan:

```python
MetaTrader5.order_send
```

dan tidak mempunyai trading execution module.

Tidak boleh ada method:

```text
buy
sell
open_position
close_position
modify_position
place_order
```

dalam ingestion platform.

Project ini adalah:

```text
MARKET DATA COLLECTOR
```

bukan:

```text
TRADING BOT
```

---

# 50. README

README minimal harus menjelaskan:

1. Tujuan project.
2. Architecture.
3. Requirement Python.
4. Requirement Windows.
5. Install MetaTrader 5 Terminal.
6. Install dependency menggunakan `uv`.
7. Setup PostgreSQL.
8. Membuat `gold_market_demo`.
9. Membuat `gold_market_live`.
10. `.env`.
11. Broker profile.
12. Alembic migration.
13. Source registration.
14. MT5 connection validation.
15. Symbol validation.
16. Historical bar collection.
17. Tick collection.
18. Incremental sync.
19. Checkpoint/resume.
20. Coverage checking.
21. Data validation.
22. PostgreSQL partitioning.
23. Troubleshooting.
24. Historical data availability limitation.
25. MT5 terminal history / max bars considerations.

Gunakan contoh command PowerShell.

---

# 51. Important Data Semantics

Pastikan dokumentasi menjelaskan perbedaan:

```text
Native MT5 OHLC
```

vs:

```text
Bid/Ask OHLC derived from tick data
```

dan:

```text
tick_volume
```

vs:

```text
real_volume
```

dan:

```text
tick volume
```

vs:

```text
tick volume_real
```

Jangan menganggap semua broker menyediakan real volume.

Nilai:

```text
0
```

dari broker tidak otomatis berarti application error.

---

# 52. Target Research Dataset

Target utama historical bars:

```text
Instrument:
XAUUSD / GOLD

Canonical symbol:
XAUUSD

Base timeframe:
M5

Target historical range:
approximately 10 years
```

Contoh:

```text
2016-01-01
→
latest available
```

Tetapi actual dataset harus mengikuti:

```text
broker availability
```

bukan memaksakan synthetic completeness.

Higher timeframe:

```text
M15
M30
H1
H4
```

nantinya dapat dihasilkan dari M5.

---

# 53. Future ML Pipeline

Project ini nantinya digunakan oleh pipeline:

```text
MetaTrader 5
      ↓
PostgreSQL Raw Market Data
      ↓
Data Understanding
      ↓
Data Quality Analysis
      ↓
Preprocessing
      ↓
Feature Engineering
      ↓
M5 / M15 / M30 / H1 Features
      ↓
Market Regime Analysis
      ↓
Training Dataset
      ↓
XGBoost / ML Models
```

Jangan implementasikan bagian ML tersebut sekarang.

Architecture cukup dibuat agar ingestion database dapat digunakan oleh pipeline tersebut nantinya.

---

# 54. Acceptance Criteria

Project dianggap selesai apabila:

- MT5 dapat terhubung secara read-only.
- Account environment tervalidasi.
- Broker dan server tervalidasi.
- Symbol dapat dikonfigurasi per broker.
- XAUUSD variant dapat dibedakan.
- Historical M5 bars dapat diambil.
- Native M15/H1 dapat diambil jika dikonfigurasi.
- Historical ticks dapat diambil sesuai availability broker.
- Bid dan Ask tick tersimpan.
- `tick_volume` tersimpan.
- `real_volume` tersimpan jika diberikan source.
- spread tersimpan.
- Semua timestamps canonical menggunakan UTC.
- PostgreSQL menjadi single source of truth.
- Demo dan live menggunakan database terpisah.
- Multi-broker menggunakan `source_id` berbeda.
- Dataset dapat resume setelah failure.
- Duplicate ingestion tidak menggandakan exact record.
- Bar revision dapat diaudit.
- Invalid data dapat dilihat melalui rejection.
- Historical coverage dapat diperiksa.
- Missing historical source data tidak diganti synthetic data.
- Tidak ada Parquet.
- Tidak ada CSV persistence.
- Tidak ada SQLite.
- Tidak ada raw-data filesystem persistence.
- Tidak ada hardcoded secret.
- Tidak ada order trading.
- Tidak ada `order_send`.
- Unit tests berjalan.
- Database integration tests tersedia.
- README lengkap.

JANGAN jadikan requirement berikut sebagai acceptance criteria absolut:

```text
"10 tahun tick harus lengkap"
```

karena availability historical tick bergantung pada broker.

Gunakan:

```text
"collector mengambil seluruh tick yang tersedia untuk requested range dan mencatat actual coverage"
```

Begitu juga untuk bar history.

---

# 55. Implementation Order

Kerjakan secara bertahap.

## Phase 1 — Foundation

1. Inspect repository.
2. Setup `uv`.
3. Setup project structure.
4. Setup settings.
5. Setup PostgreSQL connection.
6. Setup Alembic.

## Phase 2 — Database

7. Buat database schemas.
8. Buat `core.data_source`.
9. Buat `core.symbol_spec_version`.
10. Buat `ingest.run`.
11. Buat `ingest.checkpoint`.
12. Buat `market.bar_observation`.
13. Buat `market.tick_observation`.
14. Buat `market.rejection`.
15. Buat monthly partition helper.

## Phase 3 — MT5

16. Implement MT5 provider.
17. Implement terminal validation.
18. Implement account validation.
19. Implement demo/live validation.
20. Implement broker/server validation.
21. Implement symbol resolver.
22. Implement symbol metadata snapshot.

## Phase 4 — Historical Bars

23. Implement monthly bar partition fetch.
24. Implement validation.
25. Implement bulk insert.
26. Implement payload hash.
27. Implement revisions.
28. Implement checkpoint.
29. Implement retries.
30. Implement coverage.

## Phase 5 — Historical Ticks

31. Implement daily tick fetch.
32. Implement deterministic tick fingerprint.
33. Implement occurrence.
34. Implement Bid/Ask validation.
35. Implement bulk insertion.
36. Implement checkpoint.
37. Implement coverage.

## Phase 6 — Derived Data

38. Implement `bar_latest`.
39. Implement Bid/Ask M5 OHLC view.
40. Implement M5 aggregation utilities.

## Phase 7 — CLI

41. Implement source-register.
42. Implement mt5-info.
43. Implement symbol-info.
44. Implement ingest-bars.
45. Implement ingest-ticks.
46. Implement sync-bars.
47. Implement sync-ticks.
48. Implement status.
49. Implement coverage.
50. Implement validate.

## Phase 8 — Quality

51. Unit tests.
52. Integration tests.
53. Logging review.
54. Security review.
55. README.

---

# 56. Coding Rules

Gunakan:

- type hints;
- dataclass/Pydantic where appropriate;
- dependency injection untuk MT5 provider;
- repository abstraction secukupnya;
- explicit transaction boundaries;
- timezone-aware datetime;
- UTC;
- structured logging;
- deterministic behavior;
- small focused functions.

Hindari overengineering.

Jangan membuat interface/abstract class yang tidak memberikan manfaat nyata.

Jangan membuat microservices.

Project ini cukup menjadi satu Python application modular.

---

# 57. Important Instructions for Codex

Sebelum menulis code:

1. Inspect repository.
2. Jelaskan architecture singkat.
3. Identifikasi file yang akan dibuat.
4. Identifikasi database migration yang diperlukan.
5. Baru lakukan implementation.

Jangan membuat seluruh application dalam satu file.

Jangan membuat placeholder implementation jika dapat dibuat benar.

Setelah implementation:

1. Jalankan formatter/linter jika tersedia.
2. Jalankan unit tests.
3. Jalankan database tests jika PostgreSQL test environment tersedia.
4. Periksa migration.
5. Periksa imports.
6. Periksa bahwa tidak ada `order_send`.
7. Periksa bahwa tidak ada market-data persistence ke filesystem.
8. Tampilkan daftar file yang dibuat/diubah.
9. Tampilkan command yang harus saya jalankan berikutnya.

Jika real MT5 terminal tidak tersedia pada environment Codex:

- jangan fake bahwa connection berhasil;
- implementasikan provider dan tests menggunakan fake provider;
- jelaskan langkah real validation yang harus dijalankan di Windows saya.

Jangan mengarang hasil MT5 yang tidak benar-benar diuji.

---

# Final Requirement

Saya ingin hasil akhirnya memiliki konsep:

```text
                       ┌──────────────────┐
                       │ MetaTrader 5     │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ MT5 Provider     │
                       └────────┬─────────┘
                                │
                  ┌─────────────┴─────────────┐
                  │                           │
                  ▼                           ▼
          Historical Bars              Historical Ticks
                  │                           │
                  ▼                           ▼
             Validation                  Validation
                  │                           │
                  └─────────────┬─────────────┘
                                ▼
                       ┌──────────────────┐
                       │ PostgreSQL       │
                       │                  │
                       │ core             │
                       │ ingest           │
                       │ market           │
                       │ audit            │
                       └────────┬─────────┘
                                │
                                ▼
                         Research / ML
```

Data source hierarchy:

```text
Environment
│
├── DEMO DATABASE
│   │
│   ├── Broker A
│   │   └── XAUUSD
│   │       ├── M5 Bars
│   │       └── Ticks
│   │
│   └── Broker B
│       └── XAUUSDm
│           ├── M5 Bars
│           └── Ticks
│
└── LIVE DATABASE
    │
    ├── Broker A
    │   └── XAUUSD
    │       ├── M5 Bars
    │       └── Ticks
    │
    └── Broker B
        └── GOLD
            ├── M5 Bars
            └── Ticks
```

Canonical research identity:

```text
dataset_symbol = XAUUSD
```

Broker-specific identity:

```text
actual_symbol = XAUUSD
actual_symbol = XAUUSDm
actual_symbol = GOLD
```

Semua data harus tetap dapat ditelusuri kembali ke source asli.

Tujuan tahap ini adalah membangun **clean, reproducible, auditable, multi-broker XAUUSD market-data foundation** untuk penelitian Machine Learning selanjutnya.
