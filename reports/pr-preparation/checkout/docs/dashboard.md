# Dashboard Market Data

Dashboard adalah browser lokal read-only untuk tiga dataset PostgreSQL:

| Mode UI | Sumber database | Isi utama |
|---|---|---|
| Gabungan M5 | `market.processed_m5` | Native M5, Bid/Ask OHLC, spread, provenance |
| Market bars | `market.bar_latest` | Revision terbaru OHLC, volume, native spread |
| Raw ticks | `market.tick_observation` | Bid, Ask, last, volume, flags, validitas quote |

Dashboard tidak menginisialisasi MetaTrader 5 dan tidak menyediakan operasi
insert, update, atau delete. Semua nilai filter dikirim sebagai SQL parameter.
Dataset dan kolom sorting dipilih dari whitelist pada server.

## Prasyarat

1. Isi `.env` dengan `GOLD_ENVIRONMENT` dan `GOLD_DATABASE_URL` untuk collector,
   serta kedua URL dashboard berikut:

   ```dotenv
   GOLD_DASHBOARD_DEMO_DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/gold_market_demo
   GOLD_DASHBOARD_LIVE_DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/gold_market_live
   ```

   Keduanya harus menunjuk database fisik berbeda yang telah dimigrasikan.
   Command collector selalu menerima profile broker melalui `--profile`; tidak
   ada `GOLD_BROKER_PROFILE` karena profile dibaca dari `configs/brokers/`.
2. Terapkan seluruh migration:

```powershell
uv run alembic upgrade head
```

3. Jalankan ingestion bar dan/atau tick. Migration hanya membuat struktur;
   migration tidak mengambil data dari MetaTrader 5.

## Menjalankan dashboard

```powershell
uv run python -m market.dashboard
```

Buka `http://127.0.0.1:8080/`. Dokumentasi Swagger tersedia di
`http://127.0.0.1:8080/docs`.

## Fitur antarmuka

- berpindah antara gabungan M5, market bars, dan raw ticks;
- memilih database `Demo` atau `Live` melalui dropdown Environment;
- filter exact symbol, source key, timeframe, dan rentang UTC;
- filter quote tersedia atau tidak tersedia untuk dataset yang relevan;
- urutan ASC/DESC dari dropdown atau klik judul kolom;
- sorting hanya pada kolom yang tersedia di dataset aktif;
- pagination 25–200 baris;
- kelompok kolom yang dapat ditampilkan atau disembunyikan;
- refresh manual atau otomatis setiap 30 detik, 1 menit, atau 5 menit;
- unduh kolom dan halaman aktif sebagai CSV;
- preferensi dataset, symbol, source key, dan jumlah baris disimpan lokal di browser.

Nilai `NULL` ditampilkan sebagai tanda `—`. Pada gabungan M5, kolom Bid/Ask
bernilai `NULL` jika historical tick tidak tersedia untuk interval candle itu.
Setelah memilih Environment, tekan **Tampilkan** untuk menjalankan query ke
database yang dipilih. Dropdown tidak mengubah akun MT5, profile broker, atau
URL database pada browser.

## API read-only

Endpoint utama yang dipakai UI:

```text
GET /api/market-data
```

Parameter penting:

```text
dataset=processed_m5|bars|ticks
environment=demo|live
symbol=XAUUSD
source_key=metaquotes-demo-xauusd
timeframe=M5
start_utc=2016-01-01T00:00:00Z
end_utc=2027-01-01T00:00:00Z
quote_status=all|available|missing
sort_by=open_time_utc
sort_direction=asc|desc
limit=50
offset=0
```

Endpoint lama tetap tersedia untuk kompatibilitas:

```text
GET /api/processed-m5
```

Contoh membaca tick terbaru:

```text
GET /api/market-data?dataset=ticks&sort_by=tick_time_utc&sort_direction=desc&limit=50
```
