# Bab 4 — Eksperimen XGBoost dan Pembandingan Baseline

Dokumen ini disusun untuk bahan laporan berdasarkan kode dan hasil yang
tersimpan di [artifacts/direction_v3](artifacts/direction_v3/). Pemeriksaan
dilakukan pada 14 Oktober 2026 tanpa training ulang. Pembahasan berfokus pada
`final_2026`, dengan fold 2024 dan 2025 sebagai konteks evaluasi historis.

**Batas penggunaan istilah baseline:** artifact yang tersedia sudah melalui
seleksi fitur, tuning parameter, kalibrasi, dan pemilihan policy. Hasil ini
merupakan hasil pipeline XGBoost yang dikonfigurasi, bukan hasil XGBoost default
sebelum optimasi. Tidak tersedia hasil terpisah sebelum tuning yang dapat
dipakai untuk mengukur peningkatan akibat optimasi. Judul dan penomoran
4.1–4.6 dipertahankan sesuai kerangka laporan, tetapi perbedaan tersebut perlu
dinyatakan dalam naskah penelitian.

## 4.1 Baseline Model Selection

Pemilihan model pada implementasi ini berfokus pada XGBoost untuk dua tugas:
klasifikasi sinyal SELL, NO_TRADE, dan BUY; serta regresi return harga pada
horizon 15 menit. XGBoost dipilih sebagai model ensemble berbasis tree boosting
yang dapat mempelajari interaksi nonlinear pada fitur tabular. Dasar algoritma
dan pemrosesan data sparse dijelaskan oleh
[Chen dan Guestrin, XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754).
Karakteristik algoritma ini merupakan alasan rancangan eksperimen, bukan bukti
bahwa XGBoost lebih akurat daripada algoritma lain pada dataset proyek.

| Algoritma/pembanding | Peran metodologis | Status pada hasil yang diperiksa |
|---|---|---|
| Dummy Classifier | Pembanding klasifikasi sederhana tanpa pembelajaran hubungan fitur, misalnya prediksi kelas mayoritas TRAIN | Belum dilatih/dievaluasi sebagai pembanding |
| Logistic Regression | Pembanding classifier dengan batas keputusan linear pada representasi fitur | Belum dievaluasi sebagai classifier trading pembanding |
| Random Forest | Pembanding ensemble pohon yang memakai banyak tree pada subsampel data | Belum dilatih/dievaluasi sebagai pembanding |
| XGBClassifier | Model klasifikasi utama untuk tiga kelas | Hasil tersedia |
| XGBRegressor | Model regresi return utama | Hasil tersedia |
| Zero-return/persistence | Prediksi return nol, ekuivalen dengan harga masa depan sama dengan close saat keputusan | Hasil baseline regresi tersedia |

Peran kandidat klasifikasi mengacu pada dokumentasi resmi
[DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html),
[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
dan [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
Tabel tersebut tidak menyatakan ketiga algoritma sudah menjadi bagian eksperimen.

Walaupun LogisticRegression terdapat di kode, penggunaannya adalah untuk
kalibrasi probabilitas one-vs-rest, bukan classifier trading alternatif.
Prediksi NO_TRADE setelah policy juga tidak boleh disebut hasil training
DummyClassifier. Pembanding nol return pada regresi dihitung secara analitik
dan tidak membutuhkan fitting classifier.

**Narasi laporan:** Penelitian ini mengimplementasikan XGBoost sebagai model
klasifikasi sinyal dan regresi return. Kinerja regresi dibandingkan dengan
baseline nol return atau persistence price. Dummy Classifier, Logistic
Regression, dan Random Forest merupakan kandidat pembanding yang belum
dievaluasi pada artifact ini, sehingga superioritas XGBoost terhadap ketiganya
belum dapat disimpulkan.

## 4.2 Model Configuration

### Data, fitur, dan target

Data berasal dari view canonical `market.processed_m5`, untuk sumber
`metaquotes-demo-xauusd` dan instrumen XAUUSD. Sebanyak 754.660 candle M5
lolos preparation. Open timestamp pertama adalah 4 Januari 2016 01:05 UTC,
sedangkan timestamp keputusan terakhir adalah 30 September 2026 06:45 UTC.
Angka tersebut merupakan cakupan observasi dan tidak membuktikan kelengkapan
seluruh kalender perdagangan broker.

| Komponen eksperimen | Konfigurasi yang tersimpan |
|---|---|
| Timeframe input | M5 |
| Horizon target | 3 candle, wajib tepat 15 menit waktu nyata |
| Kandidat fitur | 223, terdiri atas 196 fitur tanpa lag tambahan dan 27 fitur lag |
| Fitur final classifier 2026 | 15 |
| Fitur final regressor 2026 | 40 |
| Urutan kelas | SELL=0, NO_TRADE=1, BUY=2 |
| Target regresi | Simple return `close_target / close_decision − 1` |
| Satuan target saat fitting regresi | Return dikalikan 10.000 atau bps |
| Minimum edge | 1 bps |
| Slippage asumsi round-trip | 0 bps |
| Seed | 42 |
| CV internal | 3 fold expanding window dengan purging |
| Early stopping | 20 boosting rounds tanpa perbaikan metrik evaluasi |

Target klasifikasi menggunakan ambang:

`threshold = spread_price / close + minimum_edge_bps / 10000 + slippage_bps / 10000`.

BUY diberikan jika future return lebih besar dari threshold, SELL jika
lebih kecil dari negatif threshold, dan NO_TRADE untuk rentang di antaranya.
Target yang melintasi gap, belum tersedia, atau tidak memiliki spread yang
diketahui tidak diubah menjadi NO_TRADE. Fitur memakai informasi yang tersedia
pada candle close; future close dan label tidak dimasukkan ke matriks input.

### Parameter konstruktor dan parameter hasil tuning

Nilai konstruktor berikut berasal dari `classifier()` dan `regressor()`.
Nilai tersebut tidak berarti hasil evaluasi model default sudah disimpan.

| Parameter | Default classifier | Classifier terpilih | Default regressor | Regressor terpilih |
|---|---|---|---|---|
| Objective | `multi:softprob` | `multi:softprob` | `reg:squarederror` | `reg:squarederror` |
| `n_estimators` | 160 | 160 | 160 | 21 |
| `max_depth` | 4 | 4 | 3 | 4 |
| `learning_rate` | 0,05 | 0,03 | 0,04 | 0,03 |
| `min_child_weight` | 8 | 24 | 16 | 24 |
| `reg_lambda` | 5,0 | 12,0 | 8,0 | 12,0 |
| `reg_alpha` | 0,1 | 0,1 | 0,3 | 0,3 |
| `subsample` | 0,8 | 0,8 | 0,8 | 0,8 |
| `colsample_bytree` | 0,8 | 0,8 | 0,8 | 0,8 |
| `tree_method` | `hist` | `hist` | `hist` | `hist` |
| `max_bin` | 128 | 128 | 128 | 128 |

`n_estimators` menunjukkan boosting rounds. Pada classifier multiclass,
jumlahnya tidak boleh langsung disebut jumlah keseluruhan pohon individual.
Classifier terpilih memakai balanced sample weights. Shrinkage keluaran
regresi terpilih adalah 1,0. Model quantile terpisah memakai 160 boosting
rounds, quantile 0,1 dan 0,9, lalu lebar interval dikalibrasi pada CALIBRATION.
Nominal coverage interval 80% berbeda dari target precision sinyal 90%.

**Narasi laporan:** Konfigurasi eksperimen menggunakan candle M5 dengan
horizon prediksi 15 menit. Dari 223 kandidat fitur, dipilih 15 fitur untuk
classifier dan 40 untuk regressor pada fold final. Parameter hasil tuning
ditetapkan berdasarkan blok development, sedangkan data TEST tidak digunakan
untuk memilih parameter.

## 4.3 Training Strategy

Training mengikuti pembagian waktu kronologis, tanpa shuffle.

| Blok | Periode UTC final 2026 | Jumlah baris | Tujuan |
|---|---|---:|---|
| TRAIN | 1 Jan 2016–sebelum 1 Jan 2025 | 625.909 | Seleksi fitur dan fitting awal |
| TUNE | 1 Jan 2025–sebelum 1 Jul 2025 | 34.465 | Parameter, early stopping, shrinkage |
| CALIBRATION | 1 Jul 2025–sebelum 1 Okt 2025 | 17.657 | Kalibrasi probabilitas dan interval |
| POLICY | 1 Okt 2025–sebelum 1 Jan 2026 | 17.507 | Aturan abstain dan konfirmasi sinyal |
| TEST | 1 Jan 2026–sebelum 1 Okt 2026 | 51.236 | Evaluasi historis model beku |

Tahapan fitting adalah:

1. Pilih fitur classifier melalui Quality Filter, ReliefF, dan RFECV pada TRAIN.
   Screening awal dibekukan dari prefix TRAIN sebelum validation internal pertama.
   Pilih fitur regresi secara terpisah melalui Quality Filter dan gain screening.
2. Latih kandidat classifier pada TRAIN dengan tiga kombinasi parameter, masing-masing
   diuji dengan dan tanpa balanced sample weights. Macro F1 pada TUNE memilih kandidat,
   sedangkan early stopping classifier memantau multiclass log loss pada TUNE.
3. Latih kandidat regresi dengan squared error dan pseudo-Huber. Early stopping
   memantau RMSE pada TUNE; return RMSE juga digunakan memilih shrinkage keluaran.
4. Fit kembali classifier dan regressor menggunakan TRAIN+TUNE dengan subset fitur,
   parameter, dan jumlah boosting rounds yang telah dipilih. Kedua model memakai
   observasi development yang sama, meskipun subset fiturnya berbeda.
5. Fit kalibrasi probabilitas dan perluasan interval pada CALIBRATION, kemudian
   pilih policy pada POLICY. Kedua blok ini tidak digunakan untuk refit tree.
6. Bekukan seluruh komponen sebelum evaluasi TEST.

Model tidak dilatih ulang pada setiap candle TEST. Informasi input diperbarui
dari candle aktual yang telah selesai. Prediksi hari/candle berikutnya tidak
menggunakan candle hasil prediksi sebagai pengganti observasi aktual.

**Narasi laporan:** Strategi training memisahkan fitting, tuning, kalibrasi,
dan pemilihan aturan sinyal ke dalam blok waktu berbeda. Model akhir di-fit
kembali pada TRAIN+TUNE setelah konfigurasi dipilih, kemudian dibekukan untuk
evaluasi pada periode TEST.

## 4.4 Validation Strategy

Istilah validation pada laporan ini perlu dibedakan menurut fungsinya:

| Tahap | Periode/ruang data | Penggunaan |
|---|---|---|
| Validation internal RFECV | Hanya dalam TRAIN | Pemilihan subset fitur classifier |
| TUNE | Jan–Jun 2025 | Pemilihan parameter dan early stopping |
| CALIBRATION | Jul–Sep 2025 | Fitting kalibrasi, bukan final test |
| POLICY | Okt–Des 2025 | Pemilihan aturan dan konfirmasi, bukan final test |
| Held-out TEST | Jan–Sep 2026 | Penilaian model yang sudah beku |

TRAIN untuk validation internal memakai expanding window. Label training
yang targetnya mencapai awal validation internal dibuang. Pada batas blok
development, hanya label yang diketahui dan targetnya berakhir sebelum batas
blok yang digunakan. Purging ini mempertimbangkan waktu target sebenarnya.

Pada POLICY, data dibagi secara kronologis menjadi selection dan confirmation
dengan horizon tidak tumpang tindih. Setiap arah memerlukan sedikitnya 50
sinyal pada tiap tahap dan precision arah minimal 90% sesuai konfigurasi,
beserta persyaratan probabilitas, prediksi return, dan kesesuaian interval.
Jika tidak ada aturan yang memenuhi persyaratan, arah tersebut dinonaktifkan.
Ini kriteria pemilihan policy, bukan jaminan precision TEST 90%.

Artifact berisi evaluasi tahunan 2024, 2025, dan final 2026. Kode menyediakan
fold lain, tetapi keberadaan definisi fold tidak membuktikan semuanya telah
dijalankan. Angka dalam bagian 4.5 adalah hasil held-out TEST 2026, bukan hasil
prediksi TUNE 2025. Ringkasan skor kandidat TUNE tersedia pada audit tuning,
tetapi tabel prediction results yang dibahas di sini bersumber dari CSV TEST.

Sebanyak 50.653 dari 51.236 keputusan TEST memiliki actual tersedia. Sebanyak
580 target melintasi gap dan tiga masih pending; keduanya tidak dimasukkan
ke metrik pada actual yang diketahui. Periode 2026 ini sudah terobservasi dan
ditinjau secara retrospektif; validasi forward baru tetap diperlukan untuk
menilai penggunaan setelah periode artifact.

**Narasi laporan:** Validasi dilakukan secara berurutan berdasarkan waktu.
Development validation digunakan untuk pemilihan komponen pipeline, sedangkan
hasil akhir dilaporkan pada held-out TEST yang tidak masuk fitting. Target
yang tidak tersedia atau melintasi gap dikeluarkan dari evaluasi actual.

## 4.5 Prediction Results

### Keluaran classifier sebelum selective policy

Nama kolom artifact adalah `raw_predicted_signal`. Keluaran ini sebenarnya
merupakan argmax probabilitas yang sudah dikalibrasi, sebelum selective policy;
bukan argmax probabilitas model sebelum kalibrasi.

| Sinyal | Seluruh keputusan TEST | Keputusan dengan actual tersedia |
|---|---:|---:|
| BUY | 47.904 | 47.409 |
| SELL | 3.317 | 3.232 |
| NO_TRADE | 15 | 12 |
| Total | 51.236 | 50.653 |

Distribusi actual cost-aware pada 50.653 observasi adalah BUY 23.392,
SELL 23.307, dan NO_TRADE 3.954. Dengan demikian, kelas BUY dan SELL pada
actual relatif berimbang, tetapi keluaran classifier sangat condong ke BUY.

| Kelas | Precision | Recall | F1-score | Actual support |
|---|---:|---:|---:|---:|
| BUY | 46,47% | 94,18% | 0,6223 | 23.392 |
| SELL | 48,64% | 6,74% | 0,1185 | 23.307 |
| NO_TRADE | 41,67% | 0,13% | 0,0025 | 3.954 |

Accuracy sebelum selective policy adalah **46,61%** dan macro F1 **0,2478**.
Recall BUY tinggi terjadi bersama precision yang rendah dan recall SELL
yang rendah. Oleh karena itu, hasil belum menunjukkan klasifikasi seimbang
atau precision BUY/SELL mendekati target 90%. Precision pada tabel mengukur
kesesuaian dengan label cost-aware tiga kelas, bukan precision arah trade
yang diterima selective policy.

### Keluaran sesudah selective policy

| Sinyal final | Jumlah keputusan |
|---|---:|
| BUY | 0 |
| SELL | 0 |
| NO_TRADE | 51.236 |

Policy menonaktifkan BUY dan SELL karena tidak ada aturan yang memenuhi
persyaratan selection. Audit gate menunjukkan tidak ada kandidat yang
memenuhi kesesuaian interval terhadap biaya; kandidat point return yang
melewati biaya juga sangat sedikit. Hasil ini tidak boleh ditulis sebagai
precision terukur yang gagal atau berhasil mencapai 90% pada sekumpulan
trade yang memenuhi jumlah minimum, karena sekumpulan trade tersebut tidak ada.

Trade coverage adalah **0%**, dan precision BUY/SELL final **tidak terdefinisi**
karena tidak ada sinyal trade. NO_TRADE dianggap netral pada evaluasi arah,
bukan transaksi berhasil atau akurasi 100%. Kinerja trading positif belum
dapat dibuktikan dari hasil abstain seluruhnya.

### Hasil regresi

| Metrik TEST 2026 | Hasil |
|---|---:|
| Observasi actual tersedia | 50.653 |
| MAE harga | 5,6251 satuan harga XAUUSD |
| RMSE harga | 9,1810 satuan harga XAUUSD |
| MAE return | 12,2541 bps |
| RMSE return | 19,6623 bps |
| Accuracy arah regresi | 50,30% |
| Coverage interval empiris | 66,92% |
| Coverage interval nominal | 80% |

Satuan harga bukan profit uang akun. Interval nominal 80% juga tidak
menghasilkan coverage 80% pada TEST. Coverage empiris yang lebih rendah
menunjukkan bahwa hasil kalibrasi pada development tidak otomatis berlaku
pada periode berikutnya. MAPE harga yang kecil tidak dipakai sebagai bukti
ketepatan BUY/SELL; pembandingan return dan arah tetap diperlukan.

**Narasi laporan:** Pada TEST 2026, keluaran classifier sebelum selective
policy mencapai accuracy 46,61% dan macro F1 0,2478, dengan kecenderungan
prediksi BUY. Sesudah policy, seluruh keputusan menjadi NO_TRADE karena
tidak tersedia aturan trade yang lolos persyaratan development. Model regresi
memiliki RMSE return 19,6623 bps dan accuracy arah 50,30%, sehingga hasil
belum menunjukkan ketepatan sinyal trading yang ditargetkan.

## 4.6 Baseline Model Comparison

### Perbandingan klasifikasi

| Model | Accuracy TEST 2026 | Macro F1 | Status |
|---|---:|---:|---|
| Dummy Classifier | — | — | Belum dievaluasi |
| Logistic Regression sebagai classifier trading | — | — | Belum dievaluasi |
| Random Forest | — | — | Belum dievaluasi |
| XGBoost setelah kalibrasi, sebelum selective policy | 46,61% | 0,2478 | Tersedia |

Tanda — berarti hasil tidak tersedia, bukan skor nol. Tidak ada dasar
untuk menentukan pemenang antaralgoritma klasifikasi dari tabel ini.
Jika perbandingan tersebut dijalankan kemudian, algoritma perlu memakai
target, horizon, batas waktu, dan observasi evaluasi yang sama. Seleksi fitur
berbasis XGBoost juga perlu dinyatakan sebagai bagian rancangan, agar
perbandingan algoritma tidak disajikan seolah-olah seleksi fiturnya netral.

### Perbandingan regresi dengan baseline sederhana

Baseline memprediksi return nol. Dalam satuan harga, baseline tersebut
mempertahankan close pada waktu keputusan sebagai prediksi harga target.
Kedua kolom RMSE berikut memakai observasi evaluasi yang sama dalam tiap fold.

| Periode TEST | RMSE return XGBoost (bps) | RMSE baseline nol return (bps) | Hasil deskriptif |
|---|---:|---:|---|
| 2024 | 9,1455 | 9,1452 | XGBoost sedikit lebih tinggi |
| 2025 | 11,7652 | 11,7665 | XGBoost sedikit lebih rendah |
| 2026 | 19,6623 | 19,6599 | XGBoost sedikit lebih tinggi |

Pada TEST 2026, RMSE harga XGBoost sebesar 9,1810 dibandingkan persistence
9,1798 satuan harga. Pada TUNE 2025, RMSE return XGBoost terpilih sebesar
11,4216 bps dibandingkan baseline 11,4226 bps. Keunggulan yang sangat kecil
pada TUNE tidak bertahan pada TEST 2026. Selisih kecil tersebut belum
diperiksa dengan uji statistik dan tidak cukup untuk menyatakan keunggulan
praktis atau keuntungan trading.

**Narasi laporan:** Pembandingan yang tersedia menunjukkan RMSE return
XGBoost hampir sama dengan baseline nol return. XGBoost sedikit lebih baik
pada fold 2025, tetapi sedikit lebih buruk pada fold 2024 dan TEST 2026.
Hasil tersebut belum memberikan bukti peningkatan prediksi yang konsisten.
Perbandingan antaralgoritma klasifikasi masih belum dapat dilaporkan karena
hasil Dummy Classifier, Logistic Regression, dan Random Forest belum tersedia.

## Sumber hasil dan penelusuran

- [Konfigurasi eksperimen](artifacts/direction_v3/config.json).
- [Data quality](artifacts/direction_v3/data_quality.json).
- [Parameter terpilih dan seluruh trial tuning](artifacts/direction_v3/final_2026/hyperparameters.json).
- [Jumlah fitur dan screening classifier](artifacts/direction_v3/final_2026/quality_filter_report.json).
- [Fitur classifier](artifacts/direction_v3/final_2026/selected_features.json).
- [Seleksi fitur regresi](artifacts/direction_v3/final_2026/regression_feature_selection.json).
- [Prediksi TEST per candle](artifacts/direction_v3/final_2026/predictions.csv).
- [Metrik final 2026](artifacts/direction_v3/final_2026/metrics.json).
- [Kalibrasi](artifacts/direction_v3/final_2026/calibration.json).
- [Policy dan gate audit](artifacts/direction_v3/final_2026/policy.json).
- [Metrik fold 2024](artifacts/direction_v3/fold_2024/metrics.json).
- [Metrik fold 2025](artifacts/direction_v3/fold_2025/metrics.json).
- [Penjelasan persiapan data](artifacts/direction_v3/laporan_metodologi_data_xgboost.md).

README ini melaporkan artifact yang tersedia. Pembuatan laporan tidak
menjalankan ulang training, mengubah ambang policy, atau menambahkan hasil
algoritma yang belum diuji.
