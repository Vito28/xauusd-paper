# Hasil analisis XAUUSD M5

Run menggunakan source `metaquotes-demo-xauusd` dan query read-only
`[2016-01-01 UTC, 2026-10-01 UTC)`. Observasi aktual:
**4 Januari 2016 01:05 UTC – 30 September 2026 06:40 UTC**.
Terdapat **754.660 candle**; seluruh bar yang dimuat memiliki status valid,
harga valid, dan sudah tertutup saat dianalisis. Status valid belum membuktikan
bahwa seluruh candle yang seharusnya ada sudah tersedia.

## Apa yang terlihat pada data

- Perubahan close sesi pertama ke close sesi terakhir: **+288,61%**.
  Ini perubahan pada cakupan observasi, bukan CAGR atau return investasi.
- Terdapat **1.284 sesi dengan log return negatif**. Level akhir yang lebih
  tinggi tidak menghapus penurunan di tengah perjalanan.
- Maksimum drawdown dari puncak close sesi UTC: **26,52%**.
  Drawdown intraday atau suatu strategi bisa berbeda.
- Tahun 2018 dan 2021 mengalami perubahan negatif dari close sesi pertama
  ke terakhir yang terobservasi pada tahun tersebut, masing-masing sekitar
  **−2,85%** dan **−5,92%**. Ini bukan return tahun kalender berbasis anchor
  penutupan tahun sebelumnya.
- Tahun 2026 hanya memiliki **193 sesi UTC yang terobservasi**, sampai tanggal
  terakhir yang cakupannya parsial. Jangan membandingkannya sebagai tahun penuh.
- Ada **2.836 transisi gap** antarbar. Gap meliputi penutupan pasar dan perlu
  kalender broker untuk membedakan jeda normal dari candle yang hilang.
  **751.823 return native** dapat dihitung dari pasangan berjarak tepat M5.

Mulai membaca `uni_trend_drawdown.png`, lalu `uni_monthly_returns.png` dan
`uni_return_volatility.png`. Grafik harga menunjukkan arah besar; heatmap
menunjukkan bulan naik/turun; volatilitas menunjukkan besar fluktuasinya.
Gunakan boxplot tahunan dan ACF untuk pemeriksaan perubahan distribusi serta
dependensi. Grafik BI/MULTI menunjukkan hubungan deskriptif, belum kausalitas.

## Pembandingan model return

Model memakai informasi yang tersedia setelah akhir sesi saat ini untuk
memprediksi log return sesi terobservasi berikutnya. Lima fold expanding
window memakai periode sebelum 2026, scaler fit hanya training, dan label
overlap dipurge. Holdout 2026 tidak dipakai untuk memilih fitur/parameter.
Tanggal observasi terakhir dikeluarkan secara konservatif dari model agar
target pada sesi yang mungkin parsial tidak masuk metrik. Ada 192 target
holdout pada sampel bersama yang lengkap.

| Model | MAE holdout | RMSE holdout |
|---|---:|---:|
| Baseline nol return | 0,014219 | **0,019002** |
| Baseline return terakhir | 0,021185 | 0,026720 |
| UNI: return saat ini | **0,014186** | 0,019014 |
| BI: return + range relatif | 0,014212 | 0,019012 |
| MULTI: BI + spread relatif + aktivitas tick | 0,014276 | 0,019017 |

Angka memakai unit log return, bukan USD. UNI memiliki MAE sedikit lebih kecil,
tetapi **ketiga model belum mengalahkan baseline nol return pada RMSE**.
Selisihnya sangat kecil dan belum diuji signifikansinya. Menambah variabel
belum memberi bukti peningkatan prediksi pada rancangan sederhana ini.
Grafik `models_return_comparison.png` memisahkan hasil tiap fold dan holdout.

## Langkah pengolahan berikutnya

Periksa cakupan dengan kalender broker dan riwayat requested partitions sebelum
mengambil keputusan tentang kelengkapan sesi. Evaluasi perubahan distribusi
perperiode, lalu pilih target/horizon yang jelas dan baseline sesuai target.
Jika menambah indikator, scaling dan seleksi harus tetap berada di dalam
training setiap fold. Data ekonomi/sentimen membutuhkan waktu publikasi dan
as-of join agar informasi yang belum tersedia tidak masuk fitur.

Metode dan pemetaan lima paper terdapat pada `docs/analytics.md`.
Semua angka rinci dan batas waktu tiap fold terdapat dalam `insights.json`;
plot serta metrik ini belum mengukur profit atau biaya transaksi.
