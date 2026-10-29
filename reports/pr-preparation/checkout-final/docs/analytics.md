# Analytics: data dan analisis dalam dua file

`data.py` memuat query PostgreSQL read-only, pemeriksaan provenance/UTC,
persiapan candle native, return yang memperhatikan gap, dan agregasi harian.
`analytics.py` memuat grafik, model UNI/BI/MULTI, evaluasi model beku,
eksperimen forecast XGBoost lama, serta satu CLI. Keduanya memakai pemisah
`# %%` seperti sel notebook. Import tidak menjalankan query atau training.

## Analisis sebelum memilih model

Harga yang lebih tinggi pada akhir 2016–2026 tidak berarti selalu naik.
Pisahkan pertanyaan tren harga, distribusi perubahan, risiko penurunan, dan
kemampuan memprediksi perubahan berikutnya. Level harga membantu membaca
tren; log return `ln(close_t / close_t-1)` membuat perubahan antarperiode
lebih mudah dibandingkan pada skala harga berbeda. Transformasi ini belum
membuktikan stationarity: lihat perubahan distribusi/volatilitas dan ACF;
uji ADF/KPSS dapat menjadi tahap penelitian berikutnya jika diperlukan.
Penjelasan metodologinya tersedia di
[Forecasting: Principles and Practice](https://otexts.com/fpp3/stationarity.html).

Database yang diperiksa berisi satu source `metaquotes-demo-xauusd`, XAUUSD M5,
dengan observasi hingga 30 September 2026. Cakupan akhir ini perlu diperiksa
lagi pada setiap run; tahun 2026 dan tanggal terakhir belum bisa dianggap
periode penuh. Hasil numerik run tersedia dalam `insights.json`, termasuk
jumlah sesi turun, perubahan harga, maksimum drawdown, audit kualitas, jumlah
observasi per tahun, serta metrik model jika diminta.

## Pengelolaan data

1. Baca `market.bar_latest` untuk satu source, actual symbol, dan timeframe.
   Query membatasi `[start, end)` dengan parameter terikat. Jangan gabungkan
   broker/demo/live tanpa rancangan harmonisasi yang eksplisit.
2. Wajibkan timezone; normalisasikan ke UTC dan urutkan. Tolak identitas
   campuran serta timestamp duplikat. Timestamp naive tidak diasumsikan UTC.
3. Gunakan candle yang sudah tertutup. Harga nonpositive/nonfinite atau
   OHLC tidak konsisten menjadi `NaN` pada salinan analisis dan dihitung
   dalam audit. Volume/spread negatif tidak dipakai. Data sumber tetap utuh.
4. Return native M5 hanya dihitung ketika pasangan candle berjarak tepat
   lima menit. Perubahan melintasi weekend, jeda sesi, atau gap tidak
   dianggap return satu candle. Gap belum tentu data hilang; coverage yang
   benar membutuhkan kalender trading broker dan requested partitions.
5. Agregasi harian memakai tanggal open candle UTC dan close candle terakhir
   yang terobservasi. Tidak ada interpolasi, forward-fill atau bar buatan.
   Ini ringkasan sesi UTC, bukan klaim daily candle resmi broker. Return
   harian membandingkan dua sesi terobservasi, termasuk jeda hingga empat
   hari kalender; jeda lebih panjang membatalkan return. Hari dengan harga
   terakhir invalid tetap memiliki close `NaN`.
6. Nilai awal rolling window tetap missing. Volatilitas rolling memakai
   20 return sesi terobservasi, tanpa annualisasi asumsi 252 hari. Realized
   volatility hanya memakai return native yang tersedia dan disertai counts.
   Tick volume menunjukkan aktivitas broker; `real_volume` nol tidak
   diinterpretasikan sebagai total volume pasar emas.
7. Heatmap return bulanan memakai close bulan sebelumnya, sehingga bulan
   pertama tanpa anchor tetap kosong. Bulan tanpa data tidak diisi. Nilai
   akhir bulan/tahun parsial harus dibaca bersama batas akhir observasi.
8. Simpan hanya artifact turunan PNG/JSON/HTML. Seluruh transformasi memakai
   salinan DataFrame dan tidak memperbarui database.

## Grafik dan insight

| Section | Grafik | Pertanyaan yang dijawab |
|---|---|---|
| UNI | Harga, indeks awal 100 dengan skala log, drawdown | Seberapa besar kenaikan relatif dan penurunan dari puncak? |
| UNI | Return dan volatilitas rolling | Kapan fluktuasi menguat atau melemah? |
| UNI | Heatmap return bulanan | Bulan/tahun mana turun atau naik; mana belum tersedia? |
| UNI | Boxplot return per tahun + jumlah sampel | Apakah distribusi dan outlier berubah antarperiode? |
| UNI | ACF return dan absolute return | Apakah ada dependensi lag atau volatility clustering? |
| BI | Hexbin aktivitas tick/spread relatif vs absolute return | Apakah gerakan besar berasosiasi dengan aktivitas/biaya spread? |
| MULTI | Spearman fitur relatif + jumlah pasangan | Variabel mana berasosiasi atau redundan? |
| DATA | Heatmap jumlah candle per bulan | Periode mana perlu pemeriksaan cakupan? |
| MODEL | RMSE tiap fold dan holdout | Apakah model memberi perbaikan dibanding baseline sederhana? |

Korelasi bersifat deskriptif, bukan bukti kausal atau kualitas prediksi.
Korelasi OHLC mentah bisa didominasi tren dan skala; default analisis BI/MULTI
memakai return, range relatif, spread relatif, dan aktivitas tick.
Grafik tidak melakukan clipping outlier. ACF memakai lag observasi sesi,
bukan hari kalender, tanpa klaim signifikansi atau stationarity.

## Model UNI, BI dan MULTI

Model memprediksi **log return sesi terobservasi berikutnya** setelah akhir
sesi saat ini, menggunakan Ridge `alpha=1` yang tetap:

- UNI: return saat ini.
- BI: return dan rata-rata `(high-low)/close` sesi saat ini.
- MULTI: fitur BI + rata-rata `spread_price/close` + `log1p(total tick_volume)`.
- Baseline: prediksi nol return dan prediksi sama dengan return terakhir.

Semua model memakai complete-case sample yang sama agar perbandingannya adil.
Lima fold expanding window dibentuk pada periode sebelum 2026. Label yang
mencapai waktu keputusan test dibuang dari training. StandardScaler di-fit
hanya pada training setiap fold. Tidak ada shuffle atau tuning menggunakan
test. Model akhir di-fit dengan label yang waktunya **sebelum** 1 Januari 2026.
Holdout menerima fitur yang sudah tersedia pada akhir sesi saat itu;
prediksi bersifat satu sesi ke depan, bukan recursive satu tahun.
Tanggal terakhir yang terobservasi dikeluarkan secara konservatif dari model
karena ingestion bisa belum mencakup satu sesi UTC penuh. Plot tetap
menampilkannya. Periksa counts/calendar sebelum memperlakukan sesi lain
sebagai lengkap; row count sendiri belum membuktikan kelengkapan.

MAE/RMSE dihitung pada return, tanpa MAPE yang bermasalah pada return sekitar
nol. Model lebih kompleks belum tentu lebih baik: bandingkan tiap fold dan
holdout dengan baseline. Metrik ini belum mencakup spread transaksi, slippage,
atau profit strategi. Model pada modul analytics menggunakan agregat harian;
metriknya mengikuti target dan horizon harian tersebut.

Eksperimen recursive XGBoost harga lama tersedia pada mode `forecast` dengan
pemisahan pre-2026 dan actual 2026 yang sama. Regresi pohon pada level harga
dan horizon recursive panjang perlu diperiksa dengan baseline/residual;
kenaikan historis sendiri tidak menjamin kemampuan extrapolation.

## Pemakaian

```powershell
uv run python -B -m analytics.analytics time-series `
  --source-id 654c1aa6-b15d-4135-99b9-145e16b2b899 `
  --symbol XAUUSD --timeframe M5 `
  --start 2016-01-01T00:00:00Z `
  --end 2026-10-01T00:00:00Z `
  --include-models
```

Output default berada di
`reports/analytics/time_series/<source-id>/<timeframe>/`.
`--output-dir` memilih direktori lain; `--watch-seconds 60` memperbarui
artifact dengan query baru. Model tambahan hanya dijalankan jika memakai
`--include-models`; tanpa flag tersebut tersedia delapan PNG + insights JSON.

Mode lain: `univariate`, `bivariate`, `multivariate`, `feature-eda`,
`evaluate`, `forecast`. Gunakan `--help` pada masing-masing mode.
Sepuluh plot univariat lama tetap tersedia. `feature-eda` dan `evaluate`
masih membutuhkan tabel feature/model yang belum dibuat migration collector;
model return baru langsung menggunakan native bars sehingga tidak bergantung
pada tabel tersebut. Pemanggilan forecast lama yang langsung memakai flags
masih diterima CLI. Import lama dari modul EDA/evaluation perlu diarahkan ke
`analytics.analytics`; kedua file lama telah digabung.

## Referensi paper dan keputusan penerapan

| Paper | Yang relevan | Batas penerapan di proyek |
|---|---|---|
| [JCDE 2023](https://academic.oup.com/jcde/article/10/3/1110/7148134) | Indikator teknikal dan seleksi fitur untuk sinyal | Bukan alasan memakai semua indikator tanpa validasi; seleksi technical tetap hanya fit training. |
| [Sadorsky 2021](https://www.mdpi.com/1911-8074/14/5/198) | Klasifikasi arah harga dengan tree-based models | ETF emas/perak dan horizon paper berbeda dari native XAUUSD M5; akurasi paper tidak dipindahkan ke proyek. |
| [Amini & Kalantari 2024](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0298426) | Forecast daily close CNN-Bi-LSTM dengan evaluasi error | Dataset historis dan target berbeda; studi ini menjadi referensi pembandingan, belum implementasi deep learning. |
| [Taneva-Angelova dkk. 2025](https://www.mdpi.com/2227-7072/13/2/102) | Penggabungan finansial, ekonomi dan sentimen untuk forecast multivariat | Fitur eksternal belum tersedia di database; perlu sumber, release timestamp dan as-of join sebelum digunakan. |
| [Arif dkk., JMASIF](https://ejournal.undip.ac.id/index.php/jmasif/article/view/78091/0) | Multivariate intraday LSTM dengan rolling time-series validation | Ambil prinsip validasi waktu; scaler tetap fit training saja dan H1 paper tidak disamakan dengan M5 proyek. |

Tahap ini menerapkan pengelolaan data, grafik, baseline dan model return
yang transparan terlebih dahulu. Pemilihan model lanjutan harus didasarkan
pada hasil validasi di data proyek, bukan menyalin skor atau model terbaik
dari paper dengan dataset/target berbeda.
