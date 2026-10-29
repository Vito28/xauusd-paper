# Dokumentasi XAUUSD Market Data Collector

Dokumentasi ini menjelaskan sistem acquisition serta pipeline riset M5 terpisah
di `technical/`. Boundary collector tetap hanya mengumpulkan/menyimpan data;
feature engineering dan model tidak mengubah data sumber.

Workflow M5 [technical direction_v3](technical-selective.md) menilai BUY/SELL
dari harga sinyal ke close aktual setelah 15 menit; NO_TRADE selalu netral.
Dashboard memisahkan ketepatan arah, total hasil harga setelah spread/slippage,
dan replay satu posisi. Hasil harga belum merupakan laba akun tanpa ukuran posisi.

Untuk menjalankan dokumentasi ini sebagai website lokal, gunakan:

```powershell
uv run mkdocs serve
```

Lalu buka `http://127.0.0.1:8000/`.

- [Panduan cepat](getting-started.md)
- [Arsitektur dan tanggung jawab folder](architecture.md)
- [Alur ingestion dan transaksi](data-flow.md)
- [Panduan khusus uv](tooling/uv.md)
- [Strategi testing](testing.md)
- [Referensi kode hasil AST](reference/code-reference.md)
- [Spesifikasi awal](market-collector.md)

Source code dan docstring adalah sumber kebenaran perilaku. Referensi kode
dihasilkan ulang dari AST dan menandai dokumentasi yang hilang secara eksplisit.

## Technical M5

`technical.xgboost_model` menjalankan preparation/validation, causal feature
engineering, Quality Filter → ReliefF → RFECV, XGBoost, calibration dan
selective precision policy. Fold TEST 2020–2025 memakai expanding window; default final TEST
Januari–September 2026 membekukan development sebelum Januari 2026. Base M5 dengan horizon
default 3 bar (15 menit), tanpa recursive daily forecast pada workflow utama.

```powershell
uv run python -B -m technical.xgboost_model --source-key metaquotes-demo-xauusd
```

Dashboard self-contained berada di
`technical/artifacts/selective_v2/final_2026/dashboard_2026.html`.

[Metodologi selective XGBoost](technical-selective.md) menjelaskan tuning terpisah,
target precision, abstention, interval, penggunaan candle aktual dan referensi.
Mendukung candle/line, scroll/zoom, titik sinyal, probabilitas, ledger berfilter,
walk-forward performance dan audit fitur. Actual yang belum tersedia/gap-crossing
tidak masuk evaluasi. Detail struktur dan batas informasi dijelaskan dalam
[arsitektur](architecture.md) dan README di folder source `technical/`.

## Analytics PNG

Analytics berisi dua file Python: `data.py` untuk query read-only dan
persiapan data; `analytics.py` untuk UNI, BI, MULTI, grafik time series,
model return dengan validasi waktu, evaluasi dan forecast. Kode memakai
pemisah `# %%` seperti notebook. Lihat [panduan analytics](analytics.md)
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

PNG dibuat di `reports/analytics/` dalam subfolder sesuai mode analisis. Folder
ini hanya artifact visual dan bukan source data; semua input tetap dibaca dari
PostgreSQL. Jalankan ulang command untuk mengganti PNG dengan hasil terbaru,
atau tambahkan polling agar otomatis refresh:

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

Watcher melakukan query ulang setiap interval dan berhenti dengan `Ctrl+C`.

## Forecast XGBoost 2026

Modul forecast memisahkan history tepat pada 2026-01-01T00:00:00Z. XGBoost
hanya melihat close harian sampai 31 Desember 2025, lalu memprediksi 2026 secara
recursive. Harga aktual 2026 dibaca terpisah sebagai ground truth visual dan
evaluasi, bukan sebagai fitur, target training, atau state inference.

```powershell
uv run python -B -m analytics.analytics forecast `
  --source-id <uuid> `
  --symbol XAUUSD `
  --timeframe M5 `
  --history-start 2016-01-01T00:00:00Z `
  --actual-end 2026-11-01T00:00:00Z
```

PNG, dashboard.html interaktif, dan metrics.json berisi MAE, MSE, RMSE, serta
MAPE disimpan di reports/analytics/xgboost_forecast/. --actual-end adalah batas
eksklusif data pembanding, bukan batas training.

## Insight time series

Gunakan `uv run python -B -m analytics.analytics time-series --source-id <uuid> --symbol XAUUSD --timeframe M5 --start 2016-01-01T00:00:00Z --end 2026-10-01T00:00:00Z --include-models`.
Laporan mencakup tren/drawdown, return/volatilitas, distribusi tahunan, return
bulanan, ACF, aktivitas/spread, Spearman, counts dan perbandingan model
UNI/BI/MULTI versus baseline. Tahun/bulan terakhir mengikuti cakupan
observasi dan dapat parsial; harga meningkat jangka panjang belum berarti
setiap sesi naik. Output berada di `reports/analytics/time_series/`.
