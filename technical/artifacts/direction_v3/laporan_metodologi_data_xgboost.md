# Persiapan data dan metodologi pipeline XGBoost XAUUSD M5

Dokumen ini menjelaskan implementasi yang diperiksa pada 14 Oktober 2026.
Penjelasan mengacu pada `technical/data_preparation.py`,
`technical/xgboost_model.py`, dan artifact `technical/artifacts/direction_v3`.
Angka merupakan hasil run yang tersimpan, bukan hasil training baru pada tanggal
pemeriksaan. Pipeline dan dataset tidak diubah untuk menyusun laporan ini.

## Pemetaan tahap ke kode

| Tahap | File dan fungsi |
|---|---|
| Pembacaan data | `data_preparation.py`: `load_data()` |
| Cleaning dan normalisasi | `data_preparation.py`: `prepare_data()` |
| Target labeling | `data_preparation.py`: `make_labels()` |
| Pembagian waktu | `data_preparation.py`: `Fold.split()`, `make_folds()`, `purged_cv()` |
| Feature engineering | `xgboost_model.py`: `build_features()` |
| Seleksi klasifikasi | `xgboost_model.py`: `quality_filter()`, `relieff_rank()`, `select_features()` |
| Seleksi regresi | `xgboost_model.py`: `select_regression_features()` |
| Tuning, fitting, kalibrasi, policy | `xgboost_model.py`: `fit_development()` |

Feature engineering dan selection tidak berada di `data_preparation.py`.
Keduanya dijalankan dari file pemodelan dan menerima candle hasil preparation.

## 1. Data cleaning

### Sumber dan batas pemeriksaan

Data dibaca dari view canonical `market.processed_m5` melalui transaksi
PostgreSQL read-only. Query membatasi satu sumber, satu simbol, timeframe M5,
rentang waktu, dan candle yang sudah tertutup. Karena input sudah berasal dari
view canonical, angka kualitas di bawah berlaku untuk data yang dimuat pipeline,
bukan seluruh observasi mentah sebelum pemrosesan database.

### Missing value pada kolom

Kolom harga, volume, dan spread dikonversi ke numerik. Nilai nonnumerik pada
konversi menjadi `NaN`; nilai infinity positif/negatif juga diganti `NaN`.
Baris dengan salah satu OHLC missing ditolak melalui pemeriksaan OHLC.
Volume negatif atau missing tetap `NaN`, bukan nol buatan.

Untuk spread, kode mengutamakan `native_spread_price` yang nonnegatif. Jika
nilai tersebut tidak tersedia, digunakan `spread_mean` dari candle yang sama
bila nonnegatif. Asal nilai dicatat melalui `spread_source`: `native`,
`same_candle_ticks`, atau `missing`. Bila kedua sumber tidak tersedia, spread
tetap `NaN`. Ini fallback observasi pada candle yang sama, bukan imputasi dari
nilai rata-rata dataset atau dari masa depan.

Tidak ada forward-fill, backward-fill, interpolasi harga, atau pembuatan
candle sintetis. Laporan preparation mencatat `spread_missing` dan
`tick_volume_missing` setelah normalisasi, tetapi belum menyediakan tabel
missing awal untuk setiap kolom OHLC maupun distribusi seluruh sumber spread.
Karena itu, jangan mengklaim adanya audit missing perkolom lengkap sebelum
cleaning hanya berdasarkan artifact ini.

### Duplikasi

Duplikasi dihitung berdasarkan `open_time_utc`, dalam dataset yang sudah
divalidasi sebagai satu sumber/simbol. Setelah candle invalid dan belum tutup
disaring, data diurutkan dengan stable sort dan duplikat valid terakhir
dipertahankan (`keep="last"`). Query awal mengurutkan waktu dan
`bar_observation_id`, sehingga pemilihan mengikuti urutan input canonical.
Pemeriksaan ini berbeda dari pemeriksaan duplikasi kolom fitur pada tahap
feature selection.

### Candle tidak valid dan konsistensi waktu

Candle diterima jika:

1. Semua nilai OHLC tersedia dan lebih besar dari nol.
2. High tidak lebih rendah dari open, close, atau low.
3. Low tidak lebih tinggi dari open, close, atau high.
4. Timestamp open dan close tersedia, dengan durasi tepat lima menit.
5. Waktu open berada pada grid M5: menit kelipatan lima, tanpa detik,
   mikrodetik, atau nanodetik tambahan.
6. Waktu close tidak melewati batas `as_of` atau waktu pemeriksaan.

Timestamp tanpa timezone ditolak. Identitas source/symbol yang tercampur,
identitas missing jika kolomnya tersedia, dan timeframe selain M5 juga ditolak.
Jika close timestamp tidak tersedia sebagai kolom, kode menurunkannya dari
open timestamp ditambah lima menit; ini penentuan batas waktu, bukan pengisian harga.

### Gap dan anomali

Setelah cleaning, selisih open timestamp melebihi lima menit dicatat sebagai
gap. Banyak slot grid di antara dua candle dihitung dengan:

`missing_slots = selisih_detik / 300 − 1`.

Slot tersebut dapat mencakup weekend, penutupan sesi, libur, dan kehilangan
observasi. Kode belum membedakan penyebabnya memakai kalender perdagangan
broker. Karena itu, nama `missing_rows` pada artifact tidak boleh langsung
ditafsirkan sebagai jumlah candle yang hilang ketika pasar seharusnya aktif.

Anomali yang diperiksa saat ini adalah pelanggaran aturan harga, OHLC, waktu,
volume, dan spread. Tidak ada detektor outlier statistik khusus berupa IQR,
z-score harga, Hampel filter, atau Isolation Forest. Pergerakan ekstrem yang
memenuhi aturan validitas tetap dipertahankan. Fitur z-score volume/spread
yang dibuat kemudian bukan mekanisme penghapusan outlier pada cleaning.

### Hasil artifact yang tersimpan

| Pemeriksaan | Hasil |
|---|---:|
| Baris canonical dimuat | 754.660 |
| Duplikasi open timestamp | 0 |
| Baris OHLC invalid | 0 |
| Baris timestamp invalid | 0 |
| Baris belum tertutup | 0 |
| Baris setelah preparation | 754.660 |
| Spread missing setelah preparation | 0 |
| Tick volume missing setelah preparation | 0 |
| Transisi gap | 2.836 |
| Slot grid dalam gap, termasuk penutupan pasar | 374.944 |

Open timestamp pertama adalah 4 Januari 2016 01:05 UTC; timestamp keputusan
terakhir adalah 30 September 2026 06:45 UTC. Jumlah kategori pelanggaran adalah
hitungan masing-masing aturan dan pada dataset lain dapat saling tumpang tindih.
Hasil nol pelanggaran tidak membuktikan bahwa harga telah diverifikasi dengan
sumber pasar independen atau bahwa cakupan data lengkap 100%.

**Narasi laporan:** Data cleaning dilakukan melalui pemeriksaan identitas data,
normalisasi numerik, penanganan missing value, resolusi duplikasi timestamp,
validasi hubungan OHLC, dan pemeriksaan waktu candle M5. Pada data canonical
yang dimuat, seluruh 754.660 baris lolos preparation. Ditemukan 2.836 transisi
gap yang tidak diisi dengan harga sintetis. Gap dicatat untuk audit tanpa
menganggap semua jeda sebagai kehilangan data pada jam perdagangan aktif.

## 2. Data preprocessing

Preprocessing menyiapkan candle yang telah dibersihkan untuk pembentukan
fitur dan target. Timestamp dinormalisasi ke UTC, data diurutkan secara
kronologis, dan waktu keputusan (`prediction_timestamp`) ditetapkan sebagai
close time candle. Artinya, OHLC candle saat ini baru digunakan setelah
candle selesai.

Nilai input banyak diubah menjadi ukuran relatif, seperti return, jarak
close terhadap moving average, ATR/close, dan spread/close. Matriks fitur
akhir dikonversi ke `float32`. Infinity hasil perhitungan fitur diganti `NaN`.
Missing akibat warm-up indikator atau data tidak tersedia dibiarkan untuk
penanganan missing native XGBoost; tidak diterapkan imputer global.

Tidak digunakan StandardScaler atau MinMaxScaler untuk input classifier
dan regressor utama. Pada perhitungan ReliefF saja, missing diisi median
sampel screening TRAIN dan fitur dinormalisasi memakai minimum/range sampel
tersebut untuk perhitungan jarak Manhattan. Transformasi lokal ini bukan
pengisian missing seluruh dataset atau preprocessing input model final.

Target regresi dikalikan 10.000 saat fitting agar berada dalam unit basis
point (bps), lalu keluaran dikembalikan ke unit return. Satu bps setara
dengan 0,01%. Prediksi harga diperoleh dari `close × (1 + predicted_return)`.

**Narasi laporan:** Preprocessing dilakukan dengan normalisasi waktu UTC,
pengurutan kronologis, penggunaan candle tertutup, dan transformasi harga
menjadi karakteristik relatif. Matriks fitur menggunakan tipe float32,
sedangkan missing pada fitur dipertahankan untuk mekanisme native XGBoost.
Transformasi median dan range hanya digunakan secara lokal pada seleksi ReliefF.

## 3. Target labeling

Konfigurasi yang tersimpan menggunakan horizon tiga candle M5 atau 15 menit.
Misalkan `C_t` adalah close candle keputusan dan `C_(t+3)` adalah close
target. Target regresi adalah simple return:

`R_t = C_(t+3) / C_t − 1`.

Ini bukan log return. Tiga langkah observasi juga diwajibkan tepat setara
15 menit waktu nyata. Target yang melintasi gap dinyatakan tidak tersedia.

Ambang klasifikasi adalah:

`threshold_t = spread_t / C_t + minimum_edge_bps / 10000 + slippage_bps / 10000`.

Konfigurasi artifact menggunakan minimum edge 1 bps dan asumsi slippage
round-trip 0 bps. Nilai nol tersebut adalah konfigurasi, bukan bukti eksekusi
nyata tanpa slippage.

| Kelas | Kode | Aturan |
|---|---:|---|
| SELL | 0 | `R_t < −threshold_t` |
| NO_TRADE | 1 | `−threshold_t <= R_t <= threshold_t` |
| BUY | 2 | `R_t > threshold_t` |

Label missing tidak diganti NO_TRADE. Status `PENDING` menandai target yang
melewati akhir observasi, `GAP_INVALID` menandai horizon yang tidak kontinu,
dan `COST_UNAVAILABLE` menandai spread yang tidak tersedia. Baris development
memerlukan target class yang diketahui; classifier dan regressor menggunakan
observasi development yang sama.

Label pelatihan di atas berbeda dari evaluasi keberhasilan arah sinyal.
Pada evaluasi arah, BUY dinilai sesuai bila actual return positif dan SELL
bila negatif; NO_TRADE bersifat netral. Profit setelah biaya diperiksa secara
terpisah. Ambang label menggunakan biaya keputusan sebagai pendekatan,
bukan janji bahwa setiap label trade menghasilkan profit pada fill nyata.

**Narasi laporan:** Target regresi berupa return harga 15 menit ke depan.
Target klasifikasi dibentuk menjadi SELL, NO_TRADE, dan BUY berdasarkan return
masa depan terhadap ambang spread dan minimum edge. Label hanya dianggap
tersedia jika horizon waktu kontinu dan biaya spread diketahui. Observasi
dengan target yang belum tersedia tidak diperlakukan sebagai kelas netral.

## 4. Feature engineering

`build_features()` menghasilkan 223 kandidat fitur: 196 fitur tanpa lag
tambahan dan 27 fitur lag. Jumlah ini diperiksa dari manifest artifact,
bukan 233. Pengelompokan berikut mencakup lag di dalam kelompok asalnya.

| Kelompok | Jumlah | Contoh informasi |
|---|---:|---|
| Return | 21 | Simple/log return dan rolling sum |
| Bentuk candle | 14 | Body, wick, range, posisi close |
| SMA | 27 | Jarak close, slope, perubahan MA |
| EMA | 27 | Jarak close, slope, perubahan MA |
| MACD | 15 | MACD, signal, histogram, delta |
| RSI | 15 | Momentum, jarak dari 50, delta |
| ATR | 15 | ATR relatif, perubahan, rasio median |
| Stochastic | 6 | %K dan %D |
| Momentum | 5 | ROC |
| Volatilitas | 15 | Std return dan estimator high-low |
| Bollinger | 9 | Bandwidth, posisi, jarak mid |
| Donchian | 16 | Posisi channel dan breakout |
| Aktivitas tick | 14 | OBV, perubahan, z-score, rasio MA |
| Spread | 16 | Spread relatif, median, z-score, spread/ATR |
| Waktu | 4 | Encoding sin/cos jam dan weekday |
| Sesi | 4 | Asia, London, New York, overlap |
| Total | 223 | |

Moving average memakai periode 5/8/12/24/48/96/144/288 bar. RSI dan ATR
memakai periode 7/14/21/28 bar. Lag 1/3/6 bar ditambahkan pada sembilan
fitur asal: return_1, rsi_14, macd_hist_12_26_9, atr_pct_14, rolling_std_24,
spread_pct, volume_zscore_24, close_vs_ema_24, dan close_vs_sma_24.

Seluruh input memakai candle saat ini yang sudah selesai dan riwayat
sebelumnya. Future close dan target tidak dimasukkan ke matriks X.
Sesi menggunakan timezone lokal yang mengikuti DST. Manifest menyimpan
nama, kelompok, rumus, parameter, lookback, lag, dan sifat rekursif indikator.
EMA/RSI/ATR mempunyai ketergantungan rekursif pada riwayat sebelumnya;
lookback pada manifest bukan batas keras seluruh riwayatnya.

Perhitungan fitur rolling/shift saat ini mengikuti urutan bar terobservasi,
termasuk melintasi penutupan sesi. Kode tidak mereset seluruh indikator pada
setiap gap. Jadi, n bar atau lag n pada fitur tidak selalu berarti n × 5 menit
waktu nyata. Ini berbeda dari target labeling yang mewajibkan horizon kontinu.

**Narasi laporan:** Feature engineering menghasilkan 223 kandidat yang
merepresentasikan return, bentuk candle, tren, momentum, volatilitas, aktivitas
tick, biaya spread, dan waktu perdagangan. Fitur dibentuk secara kausal dari
candle yang telah selesai dan riwayat sebelumnya. Sebanyak 27 fitur lag
ditambahkan untuk memberikan konteks temporal, dengan metadata setiap fitur
disimpan pada manifest agar perhitungannya dapat ditelusuri.

## 5. Feature selection

### Classifier

Seleksi klasifikasi memakai alur Quality Filter → ReliefF → RFECV.
Quality Filter membuang kolom nonnumerik, infinity, seluruhnya null,
missing lebih dari 40%, konstan, nilai dominan minimal 99,9% pada nilai
nonmissing, dan duplikat kolom yang persis sama.

Quality Filter dan ReliefF di-fit pada prefix TRAIN pertama yang telah
dipurge sebelum validation internal pertama. Untuk final 2026, prefix ini
memiliki 156.475 baris dengan target terakhir 6 April 2018 22:20 UTC.
Daftar kandidat dibekukan untuk seluruh RFECV agar label validation internal
tidak dipakai dalam screening. Ini screening tetap, bukan screening ulang
yang sepenuhnya nested pada setiap fold internal.

ReliefF pada artifact menggunakan sampel referensi maksimal 10.000 baris,
512 anchor, dan maksimal 10 tetangga tiap kelas. Peringkat mempertimbangkan
perbedaan terhadap tetangga sekelas dan kelas lain dengan bobot prior kelas.
Sebanyak 40 kandidat teratas masuk RFECV.

RFECV menggunakan classifier XGBoost, tiga fold expanding window yang
dipurge, dan macro F1 untuk tiga kelas tetap. Eliminasi dilakukan dengan
step lima fitur, dengan batas minimum sepuluh fitur. Hasil final 2026:

`223 kandidat → 223 lolos quality → 40 ReliefF → 15 fitur RFECV`.

Fitur final classifier adalah hour_cos, hour_sin, london_session, spread_pct,
spread_price, spread_to_atr, volume_zscore_48, spread_pct_lag_3,
volume_zscore_96, stoch_k_21, obv_change, volume_zscore_24_lag_1,
rolling_std_288, high_low_volatility_288, dan high_low_volatility_96.

### Regressor

Seleksi regresi dilakukan terpisah menggunakan target return dalam bps.
Quality Filter di-fit pada seluruh TRAIN yang dipertahankan. Regressor
screening, dengan maksimal 80 estimator pada konfigurasi ini, memberi
peringkat gain; 40 fitur teratas dipilih. Artifact final menunjukkan
223 fitur lolos quality dan 40 fitur regresi dipertahankan.

Regressor tidak menggunakan hasil ReliefF/RFECV classifier sebagai daftar
fiturnya. Gain adalah ukuran screening model, bukan bukti pengaruh kausal.
Jumlah dan nama fitur dapat berbeda antar-fold: classifier yang tersimpan
memiliki 20 fitur pada fold 2024, 10 pada fold 2025, dan 15 pada final 2026.

**Narasi laporan:** Seleksi fitur dilakukan hanya pada blok TRAIN.
Classifier menggunakan Quality Filter, ReliefF, dan RFECV berbasis macro F1,
sedangkan regressor menggunakan Quality Filter dan gain screening terhadap
target return. Pada fold final 2026, classifier memakai 15 fitur dan
regressor memakai 40 fitur. Kedua subset dipilih secara terpisah sesuai
target masing-masing model.

## 6. Data splitting dan pencegahan kebocoran waktu

Dataset tidak dibagi secara acak atau memakai proporsi 80:20. Pembagian
menggunakan lima blok waktu yang berurutan dengan interval setengah terbuka
`start <= prediction_timestamp < end`.

| Blok final 2026 | Periode UTC | Baris pada artifact | Fungsi |
|---|---|---:|---|
| TRAIN | 1 Jan 2016–sebelum 1 Jan 2025 | 625.909 | Seleksi fitur dan fitting awal |
| TUNE | 1 Jan 2025–sebelum 1 Jul 2025 | 34.465 | Pemilihan parameter, early stopping, shrinkage regresi |
| CALIBRATION | 1 Jul 2025–sebelum 1 Okt 2025 | 17.657 | Kalibrasi probabilitas dan lebar interval return |
| POLICY | 1 Okt 2025–sebelum 1 Jan 2026 | 17.507 | Seleksi dan konfirmasi aturan sinyal |
| TEST | 1 Jan 2026–sebelum 1 Okt 2026 | 51.236 | Evaluasi model beku |

Periode tabel adalah batas konfigurasi, bukan klaim adanya observasi pada
setiap tanggal tersebut. Baris development hanya dipakai bila target diketahui
dan target timestamp berada sebelum akhir bloknya. Baris yang targetnya
mencapai atau melewati batas berikutnya dipurge agar label tidak melintasi
blok waktu. CV internal juga membuang target TRAIN yang mencapai awal
validation internal.

TEST mempertahankan keputusan yang actual-nya belum tersedia atau tidak valid
untuk audit; tidak seluruh 51.236 baris otomatis dapat dinilai. Pada artifact
final, 50.653 actual tersedia, 580 melintasi gap, dan tiga target masih pending.
Jumlah seluruh blok lebih kecil daripada jumlah candle prepared karena
filter label dan purge batas diterapkan pada development.

Setelah parameter dan jumlah tree dipilih pada TUNE, model final di-fit
kembali pada gabungan TRAIN+TUNE dengan subset fitur yang sudah ditetapkan.
CALIBRATION dan POLICY tidak digunakan untuk refit tree. TEST tidak diberikan
ke fungsi fitting. Selama TEST, fitur diperbarui dari candle aktual yang
sudah selesai, sementara parameter model tetap beku.

Kode menyediakan fold tahunan 2020–2025 dan final 2026. Artifact direction_v3
yang diperiksa berisi fold 2024, fold 2025, dan final 2026; laporan tidak
mengklaim tujuh fold telah dijalankan pada artifact ini. Seluruh periode 2026
dalam artifact merupakan evaluasi historis atas data yang sudah terobservasi.

Target precision 90% adalah kriteria pemilihan policy, bukan akurasi yang
dijamin tercapai oleh preprocessing atau model. Seleksi fitur classifier
mengoptimalkan macro F1; prioritas precision trade diterapkan pada tahap policy
yang terpisah.

**Narasi laporan:** Pembagian data dilakukan secara kronologis melalui blok
TRAIN, TUNE, CALIBRATION, POLICY, dan TEST. Purging diterapkan agar target
pelatihan tidak melintasi batas blok berikutnya. Parameter dipilih pada TUNE,
kemudian tree di-fit kembali menggunakan TRAIN+TUNE. Kalibrasi dan pemilihan
aturan sinyal menggunakan blok terpisah, sedangkan TEST digunakan untuk
evaluasi model beku dengan input candle aktual yang telah selesai.

## Artifact pendukung

- `data_quality.json`: hasil preparation dan daftar gap.
- `config.json`: konfigurasi horizon, indikator, seleksi, dan batas waktu.
- `feature_manifest.json`: 223 fitur dan metadata rumusnya.
- `final_2026/quality_filter_report.json`: screening dan hasil seleksi classifier.
- `final_2026/selected_features.json`: 15 fitur classifier.
- `final_2026/regression_feature_selection.json`: ranking dan 40 fitur regressor.
- `final_2026/metrics.json`: rentang blok, jumlah baris, dan evaluasi.

Dokumen ini membedakan aturan yang diterapkan, angka yang tersimpan, dan
keterbatasan pemeriksaan. Tidak diterapkan detektor anomali baru, imputasi
baru, perubahan label, atau training ulang untuk menyusun laporan.
