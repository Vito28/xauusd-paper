# Arsitektur Sistem

## Model matematis mandiri

`analisis_teknikall/xgboost_matematis.py` merupakan satu file standard-library
untuk training Newton tree dari nol (classifier/regressor terpisah) dan inference
manual pohon JSON technical. CSV fitur dan bobot JSON adalah input data, tanpa
import ke pipeline technical saat runtime. Kesamaan yang diuji adalah inference
dari fitur/bobot sama, bukan training histogram end-to-end. Baca
[panduan matematika](xgboost-matematis.md) untuk rumus, CLI, dan batas cakupan.

## Boundary

Lapisan collector hanya melakukan extract, validation, metadata/timestamp
normalization, store, dan audit. Collector tidak melakukan trading, indikator,
feature engineering, normalisasi ML, penghapusan outlier, interpolation, atau
training. Pipeline riset terpisah di `technical/` membaca view clean secara
read-only dan tidak mengubah boundary collector tersebut.

```text
MetaTrader 5
  -> market.profile (broker TOML + validated GOLD_* settings)
  -> market.collector / MT5 provider (external boundary, read-only)
  -> market.collector / validation (pure normalization + invariant checks)
  -> market.collector / ingestion (partition/retry/transaction orchestration)
  -> database.storage (session/transaction + bulk persistence)
  -> PostgreSQL schemas core/ingest/market/audit
  -> market.dashboard (local read-only table and JSON API)
  -> technical.xgboost_model (M5 features + purged expanding walk-forward evaluation)
```

Untuk deployment lokal berbasis container, Docker Compose menyediakan dua
service: `postgres` menjalankan PostgreSQL 18 dan `api` menjalankan FastAPI
dengan Python 3.12 serta environment yang dikelola `uv`. Service API menunggu
health check PostgreSQL, menjalankan migrasi Alembic, lalu membuka dashboard.
MetaTrader 5 tetap berada di boundary Windows host karena library resminya tidak
tersedia untuk container Linux.

MT5 hanya diakses oleh command `source-register`, `mt5-info`, `symbol-info`,
`ingest-bars`, `ingest-ticks`, `sync-bars`, dan `sync-ticks`. Import module,
pytest discovery, serta Alembic tidak menginisialisasi terminal.

## Tanggung jawab folder

| Folder | Tanggung jawab | Bukan tanggung jawab |
|---|---|---|
| `market/` | Kontrak domain, profile/settings, MT5 adapter, validasi, partition, aggregation, reporting, CLI | SQL schema, ML, trading |
| analytics/ | Query data, EDA, evaluasi model, dan laporan forecast XGBoost dengan PNG Matplotlib | Mengubah market data, memakai target masa depan saat fitting/prediksi, tuning pada test, trading |
| `technical/` | Fitur M5, Quality/ReliefF/RFECV, direct XGBoost classifier/regressor, calibration, policy, evaluasi dan dashboard HTML offline | Menulis data sumber, fitting label TEST, recursive harga dalam model utama, eksekusi trading |
| `database/` | Session, mappings, repository bulk operations, migration PostgreSQL | Provider MT5, transform harga |
| `configs/brokers/` | Identitas source dan policy non-secret | Password/login/token |
| `tests/market/` | Unit test tanpa terminal/database nyata | Menulis production data |
| `tests/database/` | Smoke test database disposable | Menggunakan database demo/live |
| `docs/` | Penjelasan why/how dan generated reference | Menjadi source behavior pengganti code |
| `scripts/` | Tool dokumentasi deterministik | Ingestion market data |

## Workflow market

Dua file utama acquisition adalah `market/profile.py` dan `market/collector.py`,
keduanya memakai section `# %%` dan komentar proses seperti notebook.
`profile.py` berisi schema/loader/saver broker TOML dan runtime settings.
`save_broker_profile()` memvalidasi serta menyimpan konfigurasi non-secret ke
`configs/brokers/`, dengan penggantian file melalui `overwrite=True` eksplisit.

`collector.py` menyatukan symbol resolution, MT5 provider/preflight, UTC partition,
bar/tick validation, derived aggregation/gap, ingestion/retry/checkpoint,
coverage/status/validation reports, logging dan Typer CLI.
`contracts.py` tetap menjadi kontrak record bersama repository; `dashboard.py`
tetap menjadi API/browser read-only. `cli.py` hanya menghubungkan launcher lama
ke app collector. Repository menunda import fingerprint collector sampai
`save_symbol_spec()` dipanggil untuk menghindari circular import.

Import tidak mengaktifkan settings, MT5, atau koneksi database. Profile TOML,
policy retry, transaksi atomik dan format data tidak berubah oleh penggabungan.
Lihat `market/README.md` dalam source repository untuk peta section dan contoh config.

## Persistence database

Dua file utama persistence adalah `database/schema.py` dan `database/storage.py`,
dengan section `# %%` dan komentar proses. `schema.py` memuat seluruh ORM mappings:
source, symbol spec, ingestion run, checkpoint, bars, ticks dan rejection.
`storage.py` memuat lazy engine/session lifecycle, transaksi dan empat repository
source/market data/ingestion/rejection. Collector, dashboard dan analytics
menggunakan API storage yang sama; schema metadata tersedia untuk Alembic.

`database/migrations/` tetap menyimpan runtime/template serta riwayat revisi
Alembic. Struktur tabel, constraints dan migration revision IDs dipertahankan;
refactor ini tidak menjalankan perubahan pada database aktif. Lihat
`database/README.md` dalam source repository untuk peta section dan import baru.

## Pipeline technical

Kedua file workflow berada langsung di `technical/` dengan section `# %%` seperti notebook.
`data_preparation.py` menangani loader read-only, preparation, quality report,
labeling, walk-forward folds dan purged CV. `xgboost_model.py` menggabungkan
causal features/manifest, Quality/ReliefF/RFECV, XGBoost, calibration, policy,
prediction/evaluation, orchestration/artifacts dan CLI. Seleksi fit hanya TRAIN;
fitting model hanya menerima empat development blocks. Model frozen memprediksi
TEST sebelum actual dipasang untuk evaluasi. Import modul tidak menjalankan
training; gunakan CLI `technical.xgboost_model` atau `run_pipeline()`.

`config.py` menyimpan konfigurasi bersama. File `modeling.py` dan subfolder
`data/` dihapus. `load_model_bundle()` dalam `xgboost_model.py` menyelesaikan
path class kalibrasi artifact lama di memory tanpa mengubah file artifact.
Classifier memprediksi SELL/NO_TRADE/BUY; regressor memprediksi future return
direct dengan horizon default tiga candle M5 (15 menit).
`dashboard.py` menanamkan Plotly,
prediksi, audit serta template `dashboard.html` dalam satu file HTML offline.

Prediksi diterbitkan pada close time candle M5. Label H bar harus mencapai
H×5 menit; gap-crossing/pending outcomes tidak masuk metrik. Boundary purge
menggunakan target timestamp. Default final TRAIN sampai akhir 2024, TUNE
Jan–Jun 2025, CALIBRATION Jul–Sep, POLICY Oct–Dec, TEST Jan–Sep 2026.
Semua batas final dapat dikonfigurasi; fitting tidak membaca label TEST.
Regressor memilih fitur return secara terpisah dan memakai early stopping,
robust-loss comparison serta quantile interval. Classifier dioptimasi macro F1;
BUY/SELL memerlukan validasi precision per arah, kesesuaian regressor dan biaya.
Lihat [metodologi selective XGBoost](technical-selective.md).
Versi direction_v3 mengisolasi Quality/ReliefF pada prefix TRAIN internal sebelum
validation pertama. Outcome arah, net biaya dan status actual dipisahkan.
`load_frozen_models()` memverifikasi hash/fitur; `predict_latest()` memakai
histori aktual lengkap dan menolak candle basi. `replay_single_position()`
menilai delayed entry, bid/ask dan satu posisi. Total BUY/SELL dilaporkan
dalam unit harga, terpisah dari laba akun. Output berada di
`technical/artifacts/direction_v3/`; hasil v2 dipertahankan.
Direct future-return regression menggantikan recursive daily
forecast dalam workflow utama. Legacy tersedia sebagai ZIP lokal untuk pemulihan.

Lihat `technical/README.md` dalam source repository untuk CLI, artifacts, metodologi
dan batas trading proxy. Generated reference mengambil modul `technical/`.
| `Dockerfile`, `compose.yaml` | Runtime FastAPI Linux dan PostgreSQL 18 untuk development lokal | Menjalankan terminal MetaTrader 5 |

## Data model dan provenance

`core.data_source` mengikat satu `source_key` ke broker, server, environment,
canonical dataset symbol, dan actual symbol. Source berbeda tidak dicampur.
`core.symbol_spec_version` menyimpan snapshot metadata berdasarkan payload hash,
karena specification sekarang tidak boleh diasumsikan berlaku untuk 10 tahun.

`ingest.run` adalah audit satu invocation. `ingest.checkpoint` adalah resume state
per fixed partition. `market.bar_observation` dan `market.tick_observation`
append-only serta dipartisi bulanan. `market.rejection` mempertahankan payload
invalid untuk diagnosis.

## Views

- `market.bar_latest`: revision terbaru per logical candle; raw revision lama
  tetap ada.
- `market.data_coverage`: requested partitions versus actual record coverage.
- `market.quote_ohlc_m5`: Bid/Ask OHLC dan statistik spread dari tick valid saja.
- `market.processed_m5`: view siap-riset yang menggabungkan native M5 terbaru,
  provenance source, dan quote M5 dengan `LEFT JOIN`. Native candle tetap tampil
  ketika historical tick tidak tersedia; pada kondisi itu kolom quote bernilai
  `NULL`.

Native bar dan quote OHLC adalah dataset berbeda. Higher timeframe dari M5 juga
merupakan derived dataset; native M5 tetap dipertahankan.

`market.dashboard` membaca `market.bar_latest`, `market.tick_observation`, dan
`market.processed_m5` melalui query terparameterisasi. Dataset serta kolom
sorting dipilih dari whitelist server. Dashboard tidak menginisialisasi MT5 dan
tidak menyediakan operasi tulis database. Satu instance dashboard dapat
membaca database demo atau live melalui pilihan environment yang dibatasi ke
`demo`/`live`; URL koneksi masing-masing tetap berada di server pada
`GOLD_DASHBOARD_DEMO_DATABASE_URL` dan
`GOLD_DASHBOARD_LIVE_DATABASE_URL`, bukan di browser.

## Security dan isolation

Demo/live wajib memakai database berbeda. Runtime memvalidasi environment pada
settings, profile, dan account MT5 aktual. Logs tidak memuat password, DSN,
login, account number, token, atau secret. Credential MT5 tidak disimpan dalam
database aplikasi maupun repository.

## Analytics dalam dua file

`analytics/data.py` menangani query PostgreSQL read-only dan transformasi
data pada salinan: UTC/provenance, candle tertutup, return native dengan gap
mask, serta agregat sesi UTC harian tanpa imputation. `analytics/analytics.py`
menangani section UNI/BI/MULTI, grafik time series, model Ridge return dengan
expanding folds dan holdout 2026, evaluasi split model beku, forecast XGBoost
yang dipertahankan, serta satu CLI. Scaler fit training per fold; target yang
melewati boundary dipurge. Plot dan metrik ditulis sebagai artifact turunan,
sedangkan database tetap source of truth. Panduan dan paper tersedia di
[analytics README](analytics.md).
