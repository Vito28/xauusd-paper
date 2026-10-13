# Alur Data dan Transaksi

## Preflight

```text
explicit CLI command
  -> load GOLD_* settings (environment dan DSN wajib)
  -> load broker TOML
  -> initialize MT5
  -> validate connected terminal/path
  -> read account_info tanpa menyimpan login
  -> validate demo/live, company, server
  -> exact symbol resolution
  -> symbol_select + symbol_info snapshot
  -> start collection
```

Mismatch environment, broker, server, atau symbol adalah permanent error dan
gagal cepat sebelum fetch.

## Historical bars

```text
requested UTC range
  -> split per calendar month
  -> skip checkpoint fetched/empty
  -> copy_rates_range
  -> preserve native fields + calculate deterministic payload hash
  -> validate OHLC/volume/spread/finalization lag
  -> one PostgreSQL transaction:
       ensure monthly partition
       bulk insert valid bars (exact duplicate ignored, revision retained)
       bulk insert rejections
       checkpoint fetched/empty
  -> commit
```

`close_time_utc` berasal dari native open time + timeframe duration. Candle hanya
final bila current UTC melewati close time + configured safety delay.

## Historical ticks

```text
requested UTC range
  -> split per UTC day
  -> copy_ticks_range(COPY_TICKS_ALL)
  -> preserve time/time_msc/bid/ask/last/volume/flags/volume_real
  -> fingerprint native fields + deterministic occurrence ordinal
  -> validate quote if Bid/Ask is present
  -> bulk insert + rejection + checkpoint transaction
```

Trade-only tick tidak diberi Bid/Ask palsu. Valid tick tidak menyimpan JSONB per
row; structured native fields dan SHA-256 payload hash menjaga integrity.

## Failure dan retry

Fetch/transaction mendapat retry maksimal sesuai profile, default tiga attempt
dengan delay 1 dan 2 detik sebelum attempt berikutnya (daftar policy menyediakan
1, 2, 4 untuk perluasan). Setiap attempt gagal me-roll back transaction aktif.
Setelah batas tercapai, partition ditandai `failed`, error disimpan, dan run
menjadi `partial` bila partition lain berhasil atau `failed` bila tidak ada data
yang berhasil diproses. Run berikutnya hanya mengulang state selain
`fetched`/`empty`.

## Idempotensi

Logical bar identity adalah source/symbol/timeframe/open time. Exact payload hash
yang sama diabaikan, sedangkan hash berbeda pada logical identity yang sama
adalah revision baru. Logical tick identity memakai source/symbol/timestamp,
payload fingerprint, dan occurrence. Fixed daily fetch membuat occurrence stabil.

## Gap dan aggregation

`detect_candidate_gaps` hanya menghasilkan interval kandidat; tidak melakukan
repair. `aggregate_m5_bars` hanya membentuk M15/M30/H1/H4 jika seluruh slot M5
dalam bucket tersedia. OHLC memakai first/max/min/last dan volume dijumlahkan;
spread tidak dijumlahkan.

## Unified M5 research view

`market.processed_m5` menggabungkan revision native M5 terbaru dari
`market.bar_latest` dengan Bid/Ask/spread OHLC dari `market.quote_ohlc_m5`.
Penggabungan memakai `source_id`, `actual_symbol`, dan `open_time_utc` agar data
antarbroker tidak tercampur. `LEFT JOIN` mempertahankan native candle ketika
historical tick tidak tersedia; kolom `bid_*`, `ask_*`, dan `spread_*` akan
bernilai `NULL` dan tidak pernah diisi nilai sintetis.
