# XAUUSD Research — Technical Agent BUY / SELL / NO_TRADE

Proyek penelitian klasifikasi arah harga emas (XAUUSD) intraday dengan **XGBoost**.
Data candle M5 diambil dari MetaTrader 5, disimpan di PostgreSQL, lalu dipakai untuk
memprediksi apakah dalam **H candle ke depan** (default 6 × M5 = 30 menit) harga
bergerak cukup jauh untuk **BUY**, **SELL**, atau sebaiknya **NO_TRADE**.

Hasil model dapat diperiksa di dua dashboard: **live market** (MT5 → WebSocket) dan
**backtesting** (sinyal out-of-sample dibandingkan dengan harga aktual).
Tidak ada order yang dikirim ke broker; semua transaksi hanya disimulasikan.

> **Status (30 Oktober 2026).** Pipeline, dashboard, dan tes sudah diverifikasi pada
> data sintetis. Eksperimen pada data XAUUSD riil belum dijalankan, sehingga repository
> ini belum memuat angka performa model.

## Dokumentasi

| Dokumen | Isi |
|---|---|
| `README.md` (file ini) | gambaran proyek, struktur, instalasi, cara menjalankan, dashboard |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | dataset, fitur, label, model, walk-forward, metrik, backtesting |
| [`SOURCE_OF_TRUTH.md`](SOURCE_OF_TRUTH.md) | bukti, penurunan rumus XGBoost, log keputusan, referensi |
| [`market/README.md`](market/README.md) | collector MT5 → PostgreSQL (Windows) |
| [`database/README.md`](database/README.md) | schema dan migrasi Alembic |
| [`analytics/README.md`](analytics/README.md) | grafik EDA univariat, bivariat, multivariat |
| [`reports/arsip_pipeline_lama/`](reports/arsip_pipeline_lama/README.md) | arsip laporan pipeline generasi sebelumnya |

## Teknologi

Python 3.12 · uv · MetaTrader5 (Windows) · PostgreSQL 18 + SQLAlchemy + Alembic ·
pandas / NumPy / scikit-learn · XGBoost (resmi) + XGBoost from scratch (NumPy) ·
FastAPI + WebSocket · TradingView Lightweight Charts · pytest + ruff · Docker Compose.

## Alur data

```text
MetaTrader 5 ──(market/collector, Windows)──► PostgreSQL market.processed_m5
      │                                                │
      │ live (dashboard/live.py)                       ▼
      │                          technical/data.py      preprocessing & validasi
      │                          technical/features.py  41 fitur kausal
      │                          technical/labels.py    label triple-barrier
      │                          technical/models.py    XGBoost (+ baseline, scratch)
      │                          technical/splits.py    walk-forward + purging/embargo
      │                          technical/evaluation.py metrik klasifikasi
      │                          technical/backtest.py  prediksi vs harga aktual
      │                          technical/simulation.py simulasi P&L offline
      ▼                                                ▼
FastAPI WebSocket /ws/live ──► dashboard live     runs/<id>/ ──► dashboard backtest
```

## Struktur folder

```text
configs/             profile broker (brokers/*.toml) dan konfigurasi eksperimen (technical.toml)
market/              collector MT5, profile broker, router data browser (/data)
database/            schema, repository, migrasi Alembic
analytics/           EDA time series read-only
technical/           pipeline penelitian (data, fitur, label, split, model, evaluasi,
                     backtest, simulasi, XGBoost from scratch, CLI run.py)
dashboard/           aplikasi FastAPI: live (/), backtest (/backtest), static UI
docs/METHODOLOGY.md  metodologi penelitian
reports/             grafik EDA dan arsip laporan lama
tests/               unit test (tanpa MT5/PostgreSQL nyata)
runs/                output eksperimen dan model (di-ignore git, dapat direproduksi)
```

## Instalasi

```bash
uv sync --dev
cp .env.example .env    # isi URL PostgreSQL; GOLD_MT5_TERMINAL_PATH untuk live (Windows)
uv run alembic upgrade head
```

Pengambilan data historis MT5 (hanya Windows dengan terminal MT5 yang sudah login)
dijelaskan di [`market/README.md`](market/README.md).

## Training dan backtesting

```bash
# 1. Distribusi label 2016–2025 untuk memilih horizon dan k (tanpa model)
uv run python -m technical.run label-scan

# 2. Walk-forward 2021–2025 (expanding window). Prediksi setiap fold disimpan untuk dashboard.
uv run python -m technical.run walk-forward --models always_no_trade,persistence,logistic,xgboost

# 3. (opsional) XGBoost resmi vs from scratch, ablation kelompok fitur
uv run python -m technical.run compare-scratch --year 2025
uv run python -m technical.run ablation

# 4. Setelah konfigurasi dikunci: uji out-of-sample 2026 (sekali). Juga membuat
#    runs/<id>/live_bundle.joblib yang dipakai prediksi live.
uv run python -m technical.run final-test --model xgboost --confirm
```

Tambahkan `--input-csv ekspor.csv` untuk membaca ekspor `market.processed_m5` tanpa
PostgreSQL. Setiap run menulis `runs/<waktu>_<perintah>_<hash>/` berisi `run.json`
(config, hash, commit git, versi paket), metrik, dan `backtest/`.
Data 2026 tidak dipakai untuk training, tuning, kalibrasi, atau threshold
(lihat [metodologi bagian 11–12](docs/METHODOLOGY.md#11-strategi-training-validation-dan-testing)).

Folder `runs/` tidak disimpan di git, jadi setelah clone dashboard backtest kosong
sampai langkah 2 dijalankan. Prediksi live baru tersedia setelah langkah 4.
Seluruh angka performa untuk laporan diambil dari run pada data riil, bukan dari tes.

## Logika sinyal: kapan masuk dan kapan keluar

Model **menilai** setiap candle M5 yang sudah selesai, tetapi **tidak selalu masuk**.

**Masuk.**

1. XGBoost menghitung probabilitas BUY, SELL, dan NO_TRADE, lalu probabilitas dikalibrasi
   (temperature scaling pada blok kalibrasi).
2. BUY atau SELL hanya keluar jika probabilitas arah itu **≥ threshold** arah tersebut.
   Threshold dipilih pada blok Okt–Des tahun sebelumnya, bukan pada data uji
   (`technical/decision.py`).
3. Jika tidak lolos threshold, keputusannya **NO_TRADE**: tidak ada transaksi.

**Keluar.** Posisi ditutup oleh kejadian **pertama** dari tiga kondisi berikut, sama
dengan definisi label triple-barrier sehingga yang dipelajari model sejalan dengan cara
posisi dinilai.

| Kejadian | Kondisi (contoh BUY) | Hasil bersih per oz |
|---|---|---|
| **TP** | bid naik setinggi barrier (k·ATR14 + spread) dari close | +k·ATR |
| **SL** | bid turun sejauh barrier yang sama | −(k·ATR + 2·spread) |
| **Waktu habis** | TP dan SL tidak tersentuh selama H candle (default 6 = 30 menit) | close candle ke-H dikurangi harga masuk |

- BUY masuk di ask (= close + spread), SELL masuk di bid (= close). OHLC MT5 adalah bid.
- Jika TP dan SL tersentuh di candle yang sama, hasilnya dihitung **SL** (asumsi pesimis).
- P&L = gerak bersih × lot × 100 oz; margin = nilai kontrak ÷ leverage.

**Tiga ukuran yang dipisahkan** agar tidak tertukar:

| Ukuran | Pertanyaan | Nilai |
|---|---|---|
| Hasil prediksi | apakah sinyal sama dengan label triple-barrier? | CORRECT / INCORRECT / NEUTRAL / PENDING / UNAVAILABLE |
| Arah | apakah close t+H searah sinyal, tanpa melihat barrier? | HIT / MISS / FLAT |
| Simulasi | berapa P&L setelah spread? | TP / SL / TIME, equity, drawdown |

NEUTRAL berarti model memberi sinyal tetapi harga tidak mencapai barrier mana pun:
kenaikan kecil di bawah barrier **tidak** dihitung "BUY benar". UNAVAILABLE berarti
candle masa depan terputus (pasar tutup/gap) sehingga label tidak bisa dihitung.

**Keterbatasan simulasi saat ini.** Simulasi **mengizinkan posisi tumpang tindih**: jika
BUY muncul pukul 10:00 dan BUY lagi pukul 10:05, posisi kedua dibuka walaupun posisi
pertama belum keluar, sehingga bisa ada sampai H posisi sekaligus dan P&L, drawdown,
serta margin terlihat lebih besar dari praktik nyata. Rencana perbaikan: aturan
**satu posisi pada satu waktu** sebagai default (sinyal saat posisi masih terbuka
ditandai `SKIPPED_POSITION_OPEN`), opsi *cooldown* N candle, dan pilihan
"satu posisi / tumpang tindih" di dashboard. Aturan ini hanya mengubah lapisan simulasi
dan tidak perlu training ulang. Metrik klasifikasi tetap dihitung untuk semua sinyal.

## Keputusan desain

- **M5, bukan M1.** M1 lebih bising, spread relatif terhadap gerak harga lebih besar, dan
  datanya belum dikumpulkan (SoT D-01).
- **Horizon 30 menit, bukan 15.** Pada 15 menit gerak harga terlalu kecil dibanding
  spread. Horizon tetap dapat diubah lewat config; dashboard membaca horizon dari run
  (SoT D-18).
- **Walk-forward expanding window 2021–2025 + uji 2026.** Setiap prediksi backtest
  berasal dari model fold-nya sendiri dengan purging dan embargo; 2026 hanya dipakai
  sekali di `final-test` (SoT D-16).
- **Live tanpa data dummy.** Jika MT5 tidak tersedia, status **Tidak terhubung**
  ditampilkan apa adanya (SoT D-19). Dashboard tidak pernah mengirim order.

## Dashboard

```bash
uv run uvicorn dashboard.app:app --host 127.0.0.1 --port 8080
# atau: uv run python -m dashboard.app
```

| URL | Isi |
|---|---|
| `http://127.0.0.1:8080/` | **Live**: Bid/Ask/spread MT5, candlestick M5 + volume, status koneksi, prediksi XGBoost per candle yang selesai (PENDING → EVALUATED) |
| `http://127.0.0.1:8080/backtest` | **Backtest**: candlestick dengan marker ▲BUY ▼SELL, entry/TP/SL, tabel prediksi vs harga t+H, filter, confusion matrix, hit rate arah, simulasi P&L |
| `http://127.0.0.1:8080/data` | Data browser PostgreSQL (bars, ticks, processed M5) |
| `http://127.0.0.1:8080/docs` | Dokumentasi API |

Catatan live:

- Paket `MetaTrader5` hanya berjalan di **Windows** dengan terminal MT5 yang login.
  Di Linux/Docker halaman live menampilkan status **Tidak terhubung** beserta alasannya,
  dan grafik memakai data historis PostgreSQL (diberi label jelas, tanpa data dummy).
- Prediksi live memakai model dari `final-test` terbaru (tidak dilatih ulang) dan
  dicatat terpisah di `runs/live/predictions.csv`.

Docker (PostgreSQL + dashboard, tanpa MT5): `docker compose up --build`.

## Tes dan lint

```bash
uv run pytest -m "not integration"
uv run ruff check .
```

Tes integrasi PostgreSQL membutuhkan `GOLD_TEST_DATABASE_URL` ke database disposable.
