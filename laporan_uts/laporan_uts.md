<!--
Sumber LAPORAN UTS MILESTONE 1.docx. Jalankan dari root repo yang memuat technical/simulation.py terbaru
(branch feat/dashboard-buka-tutup-spread atau main setelah di-merge), dengan D = ../xauusd-paper-documents:
  1. uv run python $D/laporan_uts/ringkas_runs.py --out $D/laporan_uts runs/<id_label_scan> runs/<id_walk_forward_1> runs/<id_walk_forward_2> runs/<id_final_test>
     (menulis hasil_label_scan.md, hasil_walk_forward.md, hasil_final_test.md)
  2. sesuaikan $D/laporan_uts/analisis_walk_forward.md bila angka berubah
  3. uv run python $D/laporan_uts/md2docx.py build "$D/BUSINESS UNDERSTANDING.docx" $D/laporan_uts/laporan_uts.md "$D/LAPORAN UTS MILESTONE 1.docx" .
-->
%title LAPORAN UTS MILESTONE 1
%subtitle Technical Agent XAU/USD M5: Klasifikasi Selektif BUY / SELL / NO_TRADE dengan XGBoost

Proyek Kelompok Deep Learning · versi 8 November 2026

> **Status dokumen.** Dokumen ini mengikuti 12 kriteria Tahap UTS Milestone 1, ditambah poin 13 tentang alasan nilai konstanta di dalam kode. Semua angka berasal dari artefak proyek yang disebut sumbernya. Tabel hasil eksperimen (bab 3.2, 10, dan 11) dibuat otomatis dari folder runs/ dengan skrip `laporan_uts/ringkas_runs.py`, tanpa angka yang diketik manual. Eksperimen yang sudah dijalankan: label-scan, walk-forward 2021–2025 untuk tujuh model, dan final-test 2026 untuk XGBoost. Yang belum dijalankan: xgboost_selected (feature selection), compare-scratch, dan ablation.

%toc

%pagebreak

# 1. Problem Formulation

## 1.1 Latar Belakang Masalah

Harga emas terhadap dolar Amerika Serikat (XAU/USD) dapat bergerak besar dalam waktu singkat. Pada data penelitian ini, harga penutupan harian pernah turun 26,52% dari puncak sebelumnya selama 2016–2026, walaupun tren jangka panjangnya naik. Trader individu juga rentan terhadap biaya transaksi dan bias perilaku, sehingga keputusan perdagangan yang terukur dan dapat diaudit lebih bernilai daripada sekadar tebakan arah harga. Proyek ini membangun Technical Agent, yaitu komponen analisis teknikal dari sistem multi-agen XAU/USD, yang memberi sinyal BUY, SELL, atau NO_TRADE beserta probabilitas dan alasan keputusannya.

## 1.2 Rumusan Masalah

Rumusan masalah pada tahap UTS adalah:

1) Bagaimana data candle M5 XAU/USD dapat diubah menjadi sinyal BUY, SELL, dan NO_TRADE yang lebih presisi daripada tebakan acak setelah biaya spread diperhitungkan?
2) Apakah XGBoost lebih baik daripada baseline sederhana (selalu NO_TRADE, persistence, dan regresi logistik) bila diuji secara kronologis pada beberapa tahun berbeda?
3) Apakah ensemble beberapa model dan feature selection di dalam fold memperbaiki kinerja dibandingkan XGBoost tunggal dengan 41 fitur?
4) Apakah implementasi XGBoost from scratch menghasilkan perhitungan yang sama dengan library XGBoost?

## 1.3 Formulasi sebagai Masalah Machine Learning

Masalah dirumuskan sebagai klasifikasi tiga kelas yang selektif, bukan regresi harga. Penelitian emas yang meregresikan level harga cenderung menghasilkan R² tinggi yang tidak bermakna untuk keputusan arah (SoT A-01), dan pada pipeline generasi sebelumnya regresi return 2026 memperoleh RMSE 19,6623 bps, sedikit lebih buruk daripada baseline zero-return 19,6599 bps (SoT EXP-L03).

Tabel 1.1 Formulasi Masalah
<!-- widths: 28,72 -->
| Komponen | Definisi |
|---|---|
| Unit observasi | Satu candle M5 XAU/USD yang sudah tutup (keputusan pada waktu close candle) |
| Input X | 41 fitur teknikal kausal dalam 7 kelompok, hanya memakai candle t dan sebelumnya |
| Target y | Label triple-barrier: BUY, SELL, atau NO_TRADE dalam horizon 6 candle (30 menit) |
| Model | Classifier multikelas yang mengeluarkan probabilitas [p_SELL, p_NO_TRADE, p_BUY] |
| Keputusan | BUY/SELL hanya jika probabilitas terkalibrasi melewati threshold arah; selain itu NO_TRADE |
| Output agen | `TechnicalSignal`: sinyal, confidence, probabilitas, horizon, reason code, ringkasan rezim pasar |

Sistem bersifat selektif: NO_TRADE boleh dominan, dan BUY/SELL hanya dikeluarkan bila model cukup yakin. Pendekatan ini mengikuti kerangka reject option dan selective classification (SoT A-02).

## 1.4 Aturan Keputusan dan Transaksi

Agar hasil model dapat dinilai dari sudut pandang trader, setiap sinyal BUY atau SELL diperlakukan sebagai satu posisi simulasi dengan aturan yang sama dengan definisi label (`technical/simulation.py`, bab Business Understanding).

Tabel 1.2 Aturan Buka dan Tutup Posisi
<!-- widths: 24,76 -->
| Aspek | Aturan |
|---|---|
| Frekuensi penilaian | Setiap candle M5 yang sudah tutup, yaitu setiap 5 menit |
| Buka posisi | Pada close candle keputusan. BUY di ask = close (bid) + spread; SELL di bid = close |
| Target | Barrier b = k · ATR14 + spread (k = 1,0). TP dan SL = close ± b pada harga bid |
| Tutup posisi | Barrier searah tersentuh dulu → TP; barrier berlawanan dulu → SL; keduanya di candle yang sama → dianggap SL; tidak ada yang tersentuh sampai candle ke-6 → tutup di close candle ke-6 (waktu habis) |
| Spread saat tutup | Dianggap sama dengan spread saat buka, karena spread di titik sentuh tidak tercatat |
| Hasil bersih | TP = +k·ATR; SL = −(k·ATR + 2 × spread) |

Contohnya, dengan close 4.000,00, spread 0,30, dan ATR14 3,00, barrier bernilai 3,30. Posisi BUY dibuka di 4.000,30, mencapai TP bila bid menyentuh 4.003,30 (+3,00), dan SL bila bid menyentuh 3.996,70 (−3,60). Setelah 30 menit, label aktual dihitung dari candle asli dengan aturan yang sama, lalu dibandingkan dengan sinyal: CORRECT bila barrier searah tersentuh lebih dulu, INCORRECT bila barrier lawan lebih dulu, dan NEUTRAL bila tidak ada barrier tersentuh. Karena aturannya sama, CORRECT selalu berarti TP dan INCORRECT selalu berarti SL.

## 1.5 Kriteria Keberhasilan

Tabel 1.3 Kriteria Keberhasilan Tahap UTS
<!-- widths: 30,70 -->
| Kriteria | Ukuran |
|---|---|
| Sinyal lebih baik dari acak | Precision BUY dan SELL ≥ 1,2 × prevalensi kelas tersebut (lift ≥ 1,2) |
| Sinyal cukup banyak untuk dinilai | Minimal 30 sinyal per arah pada blok pemilihan threshold |
| Konsisten antarperiode | Rata-rata dan simpangan baku metrik pada lima fold walk-forward 2021–2025 |
| Kualitas probabilitas | Log loss, Brier score, dan expected calibration error |
| Bernilai setelah biaya | Simulasi transaksi setelah spread: win rate, profit factor, maximum drawdown |
| Implementasi benar | Unit test lolos dan XGBoost from scratch identik dengan library pada data uji |

## 1.6 Ruang Lingkup dan Batasan

- Hanya data teknikal (OHLC, tick volume, spread, waktu); data fundamental ditangani Fundamental Agent di luar tahap ini (SoT A-07).
- Spread berasal dari akun demo MetaQuotes yang belum tentu sama dengan akun live (SoT A-09).
- Tahun 2026 adalah uji out-of-sample retrospektif, bukan blind test, karena hasil 2026 pernah dilihat pada pipeline sebelumnya (SoT A-05).
- Hasil klasifikasi bukan bukti profitabilitas; simulasi transaksi dilaporkan terpisah (SoT A-06).

# 2. Dataset dan Sumber Data

## 2.1 Sumber Data

Data adalah candle M5 XAUUSD dari MetaTrader 5 pada server MetaQuotes-Demo, diambil oleh collector `market/collector.py` dengan fungsi `copy_rates_range`, lalu disimpan di PostgreSQL pada view `market.processed_m5`. Modul `technical/data.py` hanya membaca data secara read-only atau dari ekspor CSV; tidak ada pengambilan data dari broker di dalam pipeline model.

Tabel 2.1 Sumber dan Cakupan Data
<!-- widths: 35,65 -->
| Atribut | Nilai |
|---|---|
| Platform | MetaTrader 5, MetaQuotes Ltd., server MetaQuotes-Demo (akun demo) |
| Simbol dan timeframe | XAUUSD, M5 (5 menit) |
| Periode bar | 4 Januari 2016 01:05 UTC – 30 September 2026 06:40 UTC (waktu buka bar) |
| Jumlah bar valid | 754.660 |
| Penyimpanan | PostgreSQL, view `market.processed_m5`, semua timestamp UTC |
| Akses pipeline | `load_from_postgres` (read-only) atau `load_from_csv` |

Sumber: BUSINESS UNDERSTANDING.docx Tabel 2.1, dari `reports/analytics/time_series/654c1aa6-b15d-4135-99b9-145e16b2b899/M5/insights.json`.

## 2.2 Atribut yang Dipakai

Tabel 2.2 Atribut Data dan Kegunaannya
<!-- widths: 24,16,60 -->
| Atribut | Tipe | Kegunaan dalam pipeline |
|---|---|---|
| open_time_utc, close_time_utc | Datetime | Mengurutkan data, waktu keputusan (prediction_time = close), deteksi gap |
| open, high, low, close | Numerik | Fitur harga, ATR, label triple-barrier, simulasi TP/SL |
| tick_volume | Numerik | Fitur aktivitas pasar (jumlah perubahan harga, bukan volume kontrak) |
| native_spread_price, spread_mean | Numerik | Spread dalam satuan harga untuk fitur, lebar barrier, dan biaya simulasi |

## 2.3 Pembagian Periode

Data 1 Januari 2016 – 31 Desember 2025 adalah data pengembangan: seluruh training, kalibrasi, pemilihan threshold, dan pemilihan model hanya memakai periode ini. Data 2026 hanya dibaca oleh perintah `final-test` yang wajib diberi `--confirm`. Pada run walk-forward, data pengembangan berisi 703.425 candle valid (4 Januari 2016 – 31 Desember 2025, 2.642 transisi gap), sedangkan final-test membaca 754.660 candle sampai 30 September 2026. Batas periode diatur pada konfigurasi berikut.

Kode 2.1 Konfigurasi data (configs/technical.toml, baris 14–29)
%code configs/technical.toml 14 29

# 3. Data Understanding

## 3.1 Karakteristik Dataset

Tabel 3.1 Karakteristik Dataset XAU/USD M5
<!-- widths: 50,50 -->
| Karakteristik | Nilai |
|---|---|
| Jumlah bar valid | 754.660 |
| Bar duplikat / OHLC tidak valid / bar belum tutup | 0 / 0 / 0 |
| Transisi gap waktu (akhir pekan, libur, data hilang) | 2.836 |
| Return M5 yang dapat dihitung tanpa melintasi gap | 751.823 |
| Jumlah sesi harian UTC | 2.773 (2026: 193 sesi, data parsial) |

Sumber: `reports/analytics/time_series/654c1aa6-b15d-4135-99b9-145e16b2b899/M5/insights.json`.

## 3.2 Memahami Target

Target tidak tersedia langsung di data, melainkan dibentuk oleh `make_labels` di `technical/labels.py`. Untuk setiap candle t dihitung barrier b = k · ATR14 + spread; label BUY bila high menyentuh close + b lebih dulu dalam 6 candle, SELL bila low menyentuh close − b lebih dulu, dan NO_TRADE bila tidak ada yang tersentuh atau keduanya tersentuh pada candle yang sama. Label diberi status GAP, PENDING, atau INPUT_MISSING (dan tidak dipakai) bila horizon melintasi gap, data masa depan belum ada, atau ATR/spread belum tersedia.

Kode 3.1 Pembentukan barrier dan langkah sentuhan pertama (technical/labels.py, baris 83–100)
%code technical/labels.py 83 100

Spread ditambahkan ke barrier agar gerakan yang mencapai TP masih menghasilkan +k·ATR setelah biaya spread. Desain ini menggantikan label pipeline sebelumnya (return 15 menit dengan ambang spread + 1 bps) yang hanya menghasilkan 7,81% NO_TRADE pada 2026, sehingga model dipaksa menebak arah di hampir setiap candle (SoT EXP-L01). Perintah label-scan menghitung distribusi label 2016–2025 untuk tiga horizon dan empat pengali barrier sebelum model dilatih (Tabel 3.2).

%include hasil_label_scan.md

Pada konfigurasi yang dipakai (triple-barrier, 6 candle, k = 1,0), NO_TRADE mencakup 24,7% label, sedangkan BUY dan SELL masing-masing 37,5% dan 37,8%. Barrier yang lebih sempit (k = 0,5) mengulangi masalah pipeline sebelumnya karena NO_TRADE hanya 8,1%, sedangkan k = 1,5 membuat NO_TRADE menjadi 48,6% sehingga BUY dan SELL jauh lebih jarang. Konfigurasi 6 candle dan k = 1,0 dipertahankan sebagai titik tengah. Kombinasi lain belum diuji dengan walk-forward dan masuk rencana tuning (bab 12.5). Distribusi label per fold ditampilkan pada Tabel 10.9.

## 3.3 Keterbatasan Data

- Tick volume adalah jumlah perubahan harga pada feed broker, bukan volume transaksi pasar emas global.
- Spread akun demo dapat berbeda dari akun live, padahal spread masuk langsung ke definisi label.
- Data 2026 bersifat parsial (sampai 30 September 2026) dan sudah pernah dilihat pada pipeline sebelumnya.

# 4. EDA

Seluruh grafik pada bab ini dihasilkan modul `analytics/` dari data pada bab 2 dan sama dengan Gambar 2.1–2.7 pada BUSINESS UNDERSTANDING.docx. Angka yang dibaca dari grafik ditandai sebagai pembacaan visual.

## 4.1 Tren Harga dan Drawdown

%image rId6

Gambar 4.1 Tren Harga Penutupan, Indeks Harga, dan Drawdown XAU/USD 2016–2026

Harga naik 288,61% antara sesi pertama dan terakhir (indeks 100 menjadi 388,61), dengan kenaikan +27,47% pada 2024 dan +62,49% pada 2025, sedangkan 2026 (Januari–September) berubah −3,62%. Namun 1.284 dari 2.772 return sesi bernilai negatif dan maximum drawdown mencapai 26,52%. Kenaikan jangka panjang tidak stabil, sehingga model perlu kelas NO_TRADE untuk menahan diri.

## 4.2 Return dan Volatilitas

%image rId7

Gambar 4.2 Return dan Volatilitas Rolling XAU/USD 2016–2026

Volatilitas relatif rendah pada 2017–2019 dan meningkat pada 2020, 2025, dan terutama 2026, dengan puncak sekitar 3,6% (pembacaan visual). Karena volatilitas berubah terhadap waktu, barrier label dan sebagian besar fitur dinormalisasi dengan ATR.

## 4.3 Return Bulanan dan Distribusi Tahunan

%image rId8

Gambar 4.3 Heatmap Return Bulanan XAU/USD 2016–2026

%image rId9

Gambar 4.4 Boxplot Distribusi Return XAU/USD 2016–2026

Pola return bulanan tidak konsisten antartahun, dan distribusi return 2020, 2025, serta 2026 jauh lebih lebar daripada 2017–2018. Karena karakteristik pasar berubah, evaluasi memakai walk-forward per tahun, bukan satu pembagian train/test.

## 4.4 Autokorelasi

%image rId10

Gambar 4.5 ACF Log Return dan Absolute Return XAU/USD 2016–2026

Autokorelasi log return mendekati nol, sedangkan autokorelasi absolute return positif pada lag awal (volatility clustering). Arah harga sulit diprediksi dari return masa lalu saja, tetapi besarnya pergerakan lebih dapat diprediksi; temuan ini mendukung fitur volatilitas seperti ATR, volatilitas realisasi, dan lebar Bollinger Bands.

## 4.5 Analisis Bivariat dan Multivariat

%image rId11

Gambar 4.6 Hubungan Tick Volume dan Spread Relatif terhadap Absolute Return M5

%image rId12

Gambar 4.7 Heatmap Korelasi Spearman Antarvariabel Harian

Tick volume yang tinggi disertai sebaran perubahan harga yang lebih lebar, sedangkan spread relatif tidak memperlihatkan hubungan konsisten dengan besarnya pergerakan. Korelasi antarvariabel lebih terlihat pada besaran pergerakan dan aktivitas pasar daripada pada arah harga.

## 4.6 Dari Temuan EDA ke Desain Pipeline

Tabel 4.1 Hubungan Temuan EDA dengan Keputusan Desain
<!-- widths: 34,40,26 -->
| Temuan EDA | Keputusan desain | Kode |
|---|---|---|
| Tren naik dengan drawdown 26,52% | Kelas NO_TRADE dan keputusan selektif | `technical/decision.py` |
| Volatilitas berubah antartahun | Barrier dan fitur dinormalisasi ATR | `technical/labels.py`, `technical/features.py` |
| Pola return berubah antarperiode | Walk-forward lima fold per tahun | `technical/splits.py` |
| Volatility clustering | Kelompok fitur volatility (ATR, realized vol, Bollinger) | `technical/features.py` |
| Tick volume terkait besarnya gerak | Kelompok fitur volume (z-score, rasio) | `technical/features.py` |
| Spread tidak konsisten dengan gerak, tetapi menentukan biaya | Spread masuk ke barrier dan simulasi; fitur spread diuji lewat seleksi | `technical/labels.py`, `technical/simulation.py` |

# 5. Pre-processing

## 5.1 Validasi dan Pembersihan Data

Pre-processing dilakukan oleh `prepare_candles` di `technical/data.py`. Harga tidak pernah diisi atau diinterpolasi; candle yang tidak valid dibuang dan gap hanya ditandai.

Kode 5.1 Validasi OHLC, durasi candle, duplikat, dan penanda gap (technical/data.py, baris 106–132)
%code technical/data.py 106 132

Tabel 5.1 Aturan Pre-processing
<!-- widths: 30,70 -->
| Langkah | Aturan dan alasan |
|---|---|
| Validasi OHLC | Harga harus ada dan positif; high ≥ max(open, close), low ≤ min(open, close) |
| Validasi waktu | Durasi tepat 5 menit dan open time pada kelipatan 5 menit; bar belum tutup dibuang |
| Duplikat | Timestamp kembar diselesaikan dengan observasi terakhir |
| Gap | `gap_before` = True bila candle sebelumnya tidak tepat 5 menit sebelumnya; tidak ada interpolasi |
| Spread | `native_spread_price` lebih dulu, lalu rata-rata spread tick pada candle yang sama |
| Waktu keputusan | `prediction_time` = close candle, agar fitur hanya memakai candle yang sudah selesai |

## 5.2 Nilai Hilang dan Skala

Nilai warm-up indikator dibiarkan NaN karena XGBoost menangani nilai hilang secara native dengan arah default pada setiap split. Model berbasis pohon tidak memerlukan scaling. Imputasi median dan StandardScaler hanya dipakai regresi logistik, dan keduanya di-fit pada blok TRAIN setiap fold di dalam Pipeline scikit-learn. Seluruh fitur disimpan sebagai float32.

## 5.3 Pencegahan Kebocoran Data

- Fitur hanya memakai `rolling`, `ewm`, dan `shift` positif; diuji otomatis oleh `test_features_do_not_change_when_future_candles_change`.
- Label tidak pernah masuk matriks fitur; label boleh memakai harga masa depan karena merupakan target.
- Purging dan embargo pada batas blok (bab 9.2) dan `assert_no_overlap` memastikan label suatu blok selesai sebelum blok berikutnya dimulai.
- Preprocessing, kalibrasi, threshold, bobot ensemble, dan feature selection di-fit ulang di setiap fold tanpa melihat blok EVALUATION.
- Perintah pengembangan tidak memuat data setelah 31 Desember 2025.

# 6. Feature Engineering Awal

## 6.1 Prinsip

Fungsi `build_features` di `technical/features.py` menghasilkan 41 fitur dalam tujuh kelompok. Ada tiga prinsip. Pertama, fitur bersifat kausal. Kedua, fitur harga dibagi ATR14 atau harga agar nilai tahun 2016 (emas sekitar 1.100 USD) dan 2026 sebanding. Ketiga, nilai yang belum bisa dihitung tetap NaN.

Tabel 6.1 Kelompok Fitur
<!-- widths: 16,9,75 -->
| Kelompok | Jumlah | Fitur |
|---|---|---|
| price_action | 11 | log return 1/3/6/12 candle, return 6 dan 24 candle ÷ ATR, badan dan rentang candle ÷ ATR, rasio sumbu atas dan bawah, posisi close dalam rentang |
| trend | 7 | jarak close ke EMA 12/48/288 ÷ ATR, selisih EMA 12–48 ÷ ATR, kemiringan EMA 48, ADX 14, selisih +DI dan −DI |
| momentum | 5 | RSI 14, perubahan RSI 3 candle, histogram MACD (12, 26, 9) ÷ ATR, stochastic %K 14, posisi Donchian 48 |
| volatility | 6 | ATR ÷ close, ATR ÷ rata-rata ATR 288 candle, volatilitas realisasi 12 dan 48 candle, lebar dan posisi Bollinger Bands 20 |
| volume | 2 | z-score log tick volume 48 candle, rasio rata-rata tick volume 12 ÷ 288 candle |
| spread | 2 | spread ÷ ATR, z-score spread 288 candle |
| time | 8 | jam (sin dan cos), hari, sesi Asia, London, dan New York, overlap London–New York, jumlah candle sejak gap |

## 6.2 Contoh Implementasi

Kode 6.1 Kelompok fitur trend (technical/features.py, baris 134–150)
%code technical/features.py 134 150

Panjang jendela dipilih dalam satuan candle M5 agar bermakna sebagai waktu: 12 candle = 1 jam, 48 candle = 4 jam, dan 288 candle = 24 jam. Periode indikator klasik (RSI, ATR, dan ADX 14; MACD 12, 26, 9; Bollinger 20, 2) adalah konvensi penemunya, bukan nilai optimal yang sudah dibuktikan pada data ini (SoT tabel C). Relevansinya diuji pada tahap feature selection dan ablation.

# 7. Baseline

## 7.1 Mengapa Baseline Diperlukan

Tanpa pembanding sederhana, angka seperti precision 45% tidak dapat ditafsirkan. Baseline menjawab apakah model yang rumit benar-benar menambah nilai. Semua baseline diuji pada fold, blok, dan aturan penilaian yang sama dengan model utama (`technical/models.py`).

Tabel 7.1 Model Baseline
<!-- widths: 20,46,34 -->
| Baseline | Cara kerja | Pertanyaan yang dijawab |
|---|---|---|
| always_no_trade | Selalu NO_TRADE (coverage 0) | Batas bawah: apakah model lebih berguna daripada tidak pernah trading? |
| persistence | BUY bila close naik lebih dari barrier selama 6 candle terakhir, SELL bila turun lebih dari barrier | Apakah model lebih baik daripada aturan momentum naif? |
| logistic | Regresi logistik multinomial, imputasi median, StandardScaler, C = 1,0, max_iter = 500 | Apakah model non-linear (pohon) lebih baik daripada model linear? |

Kode 7.1 Baseline persistence memakai barrier yang sama dengan label (technical/models.py, baris 56–78)
%code technical/models.py 56 78

Persistence memakai barrier yang sama dengan label sehingga perbandingannya adil, dan seluruh inputnya hanya memakai informasi sampai candle t. Regresi logistik memakai C = 1,0 (default scikit-learn, tanpa tuning) dan max_iter = 500 (lebih besar dari default 100 agar solver konvergen pada data ratusan ribu baris).

## 7.2 Baseline dari Pipeline Generasi Sebelumnya

Pipeline generasi sebelumnya (label 15 menit, 15 fitur hasil seleksi ReliefF dan RFECV, Platt scaling, dan aturan sinyal precision ≥ 90%) menjadi titik awal perancangan ulang.

Tabel 7.2 Hasil Pipeline Generasi Sebelumnya pada TEST Januari–September 2026
<!-- widths: 50,50 -->
| Metrik | Hasil |
|---|---|
| Accuracy | 46,61% |
| Macro F1-score | 0,2478 |
| Precision BUY / SELL / NO_TRADE | 46,47% / 48,64% / 41,67% |
| Recall BUY / SELL / NO_TRADE | 94,18% / 6,74% / 0,13% |
| Macro F1 pada blok TUNE (konfigurasi terbaik) | 0,3525 |
| Keputusan BUY/SELL setelah aturan sinyal | 0 dari 51.236 keputusan |

Sumber: `git show a0e31c0:technical/artifacts/direction_v3/final_2026/metrics.json` dan `hyperparameters.json` (dikutip pada BUSINESS UNDERSTANDING.docx Tabel 4.2 dan 5.2).

Hasil baseline pipeline baru ditampilkan bersama kandidat model pada Tabel 10.3: always_no_trade memperoleh macro F1 0,1025 tanpa sinyal, persistence 0,3043 dengan lift hanya 0,98–1,08 (Tabel 10.8), dan regresi logistik 0,1244 dengan coverage 3,05%.

# 8. Beberapa Kandidat Model Awal

## 8.1 Daftar Kandidat

Tabel 8.1 Kandidat Model
<!-- widths: 20,46,34 -->
| Model | Cara kerja | Peran |
|---|---|---|
| xgboost | XGBClassifier multi:softprob, early stopping pada CALIBRATION | Model utama |
| xgboost_scratch | Implementasi NumPy di `technical/xgboost_scratch.py` | Verifikasi rumus; dibandingkan lewat compare-scratch |
| random_forest | Bagging 300 pohon dalam | Anggota ensemble, menurunkan variance |
| xgboost_balanced | XGBoost dengan bobot kelas balanced | Pembanding penanganan imbalance |
| ensemble | Soft voting berbobot xgboost + random_forest + logistic | Menguji manfaat kombinasi model |
| xgboost_selected | XGBoost pada fitur hasil seleksi di dalam fold | Menguji apakah subset fitur menyamai 41 fitur |

Semua kandidat memiliki antarmuka yang sama, yaitu `fit(train, calibration)` dan `predict_proba(X)`, lalu melalui kalibrasi temperature scaling dan pemilihan threshold yang sama di `run_fold` (`technical/run.py`).

## 8.2 XGBoost (Library)

Kode 8.1 Pembuatan XGBClassifier dari konfigurasi (technical/models.py, baris 99–124)
%code technical/models.py 99 124

Objective `multi:softprob` dipilih karena decision layer membutuhkan probabilitas setiap kelas, bukan hanya label. `tree_method="hist"` mempercepat pencarian split pada ratusan ribu candle, dan `eval_metric="mlogloss"` membuat early stopping memantau kualitas probabilitas, bukan accuracy yang bisa tinggi hanya karena NO_TRADE dominan. Jumlah pohon sebenarnya ditentukan early stopping pada blok CALIBRATION (`best_iteration` dicatat di metrics.json). Alasan setiap nilai hyperparameter dibahas pada bab 13.

## 8.3 XGBoost from Scratch

Implementasi from scratch hanya memakai NumPy dan mengikuti Chen dan Guestrin (2016). Tujuannya membuktikan pemahaman matematis dan membandingkan perhitungan dengan library secara terkendali, bukan menggantikan library. Setiap ronde boosting membangun satu pohon regresi per kelas.

$$ p_ik = exp(z_ik) / Σ_j exp(z_ij) $$ (8.1)

$$ g_ik = p_ik − y_ik $$ (8.2)

$$ h_ik = max(2 · p_ik · (1 − p_ik), 10⁻¹⁶) $$ (8.3)

$$ w* = −T_α(G) / (H + λ) $$ (8.4)

$$ Gain = ½ · [S(G_L, H_L) + S(G_R, H_R) − S(G, H)] − γ,   S(G, H) = T_α(G)² / (H + λ) $$ (8.5)

$$ z_ik ← z_ik + η · f_k(x_i) $$ (8.6)

Persamaan (8.1) adalah softmax yang distabilkan dengan mengurangi margin maksimum, (8.2) dan (8.3) adalah gradien dan Hessian diagonal cross-entropy, (8.4) adalah bobot daun optimal dengan soft-thresholding L1 T_α, (8.5) adalah gain split, dan (8.6) adalah pembaruan margin dengan learning rate η.

Kode 8.2 Gradien dan Hessian softmax (technical/xgboost_scratch.py, baris 57–72)
%code technical/xgboost_scratch.py 57 72

Kode 8.3 Bobot daun, skor struktur, dan gain split (technical/xgboost_scratch.py, baris 80–108)
%code technical/xgboost_scratch.py 80 108

Kode 8.4 Satu ronde boosting multikelas (technical/xgboost_scratch.py, baris 454–477)
%code technical/xgboost_scratch.py 454 477

Faktor 2 pada Hessian dipakai karena XGBoost resmi melakukan hal yang sama; tanpa faktor ini kedua implementasi tidak sebanding. Pembaruan margin dilakukan setelah ketiga pohon dalam satu ronde selesai, karena gradien setiap kelas dihitung dari probabilitas yang sama.

Tabel 8.2 Bukti Kesesuaian XGBoost from Scratch dengan Library
<!-- widths: 14,46,40 -->
| ID | Pengujian | Hasil |
|---|---|---|
| EXP-S01 | Data sintetis 600 baris, kandidat split identik, 20 ronde | Selisih probabilitas maksimum < 1,1 × 10⁻⁷ |
| EXP-S02 | Base score multi:softprob XGBoost 3.4.1 | Sama dengan log prior yang di-center: [−0,3619; 0,2218; 0,1400] |
| EXP-S03 | 20.000 baris kontinu dengan 5% NaN, data uji terpisah | Argmax sama 97,86%; accuracy 0,6400 (library) vs 0,6428 (scratch) |
| EXP-S04 | 700.000 × 41 fitur, waktu per ronde | Scratch ≈ 3,4 detik; library ≈ 0,32 detik |
| EXP-S05 | Konvensi gamma | gamma library = 2 × gamma scratch menghasilkan model identik |
| Contoh numerik | Data contoh 3 observasi, 2 ronde | Selisih maksimum rumus vs library 1,94 × 10⁻⁸ |

Sumber: SOURCE_OF_TRUTH.md bagian F.1, `tests/test_xgboost_scratch.py`, dan BUSINESS UNDERSTANDING.docx Tabel 6.7.

Perbedaan yang tersisa pada data riil berasal dari kandidat split: kode manual memakai kuantil biasa, sedangkan XGBoost hist memakai weighted quantile sketch berbobot Hessian. Subsampling, paralelisasi, dan optimasi sistem juga tidak diimplementasikan. Perbandingan pada data XAU/USD dilakukan dengan perintah `compare-scratch`, yang menyamakan hyperparameter dan mematikan subsampling; perintah ini belum dijalankan pada tahap UTS.

## 8.4 Random Forest dan XGBoost Balanced

Random forest (`RandomForestClassifier`, 300 pohon, kedalaman 12, daun minimal 100 candle, 33% fitur per split, dan 50% baris per pohon) adalah pohon dalam yang dirata-rata. Pohon dalam memiliki bias rendah tetapi variance tinggi; rata-rata ratusan pohon menurunkan variance. Model ini terutama berperan sebagai anggota ensemble yang kesalahannya berbeda dari boosting.

`xgboost_balanced` sama dengan XGBoost utama, tetapi memberi bobot sampel n / (K · n_kelas) sehingga kesalahan pada kelas jarang lebih mahal. Model ini menguji apakah class weight lebih baik daripada threshold moving, yaitu metode utama proyek. SMOTE tidak dipakai karena label triple-barrier pada candle berdekatan saling tumpang tindih, sehingga titik sintetis dapat membawa informasi masa depan.

## 8.5 Ensemble Learning

### Mengapa Memakai Ensemble

Error model terdiri atas bias, variance, dan noise. Boosting seperti XGBoost menurunkan bias, sedangkan bagging seperti random forest menurunkan variance. Data XAU/USD M5 sangat bising dan polanya berubah antartahun (bab 4), sehingga satu model mudah terlalu yakin pada pola satu periode. Ensemble hanya efektif bila tiga syarat terpenuhi: setiap anggota lebih baik daripada acak, kesalahan antaranggota beragam, dan keluarannya digabung dengan cara yang tepat (Dietterich, 2000).

Tabel 8.3 Anggota Ensemble
<!-- widths: 18,30,52 -->
| Anggota | Jenis | Alasan dipilih |
|---|---|---|
| xgboost | Boosting, pohon dangkal berurutan | Menurunkan bias; model utama |
| random_forest | Bagging, pohon dalam paralel | Menurunkan variance; pola kesalahan berbeda dari boosting |
| logistic | Linear | Batas keputusan linear, pola kesalahan berbeda dari kedua model pohon |

### Cara Kombinasi: Soft Voting Berbobot

Kelas `SoftVotingEnsemble` di `technical/ensemble.py` bekerja dalam empat langkah. (1) Setiap anggota dilatih pada blok TRAIN. (2) Probabilitas setiap anggota pada blok CALIBRATION dikalibrasi dengan temperature scaling, karena soft voting hanya adil bila probabilitas antaranggota sebanding. (3) Probabilitas digabung dengan bobot:

$$ p_ens = Σ_m w_m · p_m,   w_m ≥ 0,   Σ_m w_m = 1 $$ (8.7)

(4) Bobot dipilih dari semua kombinasi kelipatan 0,05 yang jumlahnya 1, yaitu yang meminimalkan log loss campuran pada blok CALIBRATION:

$$ w* = argmin_w  LogLoss(y_cal, Σ_m w_m · p_m) $$ (8.8)

Kode 8.5 Grid bobot simpleks dan pencarian bobot (technical/ensemble.py, baris 28–63)
%code technical/ensemble.py 28 63

Untuk tiga anggota, langkah 0,05 menghasilkan 231 kombinasi bobot. Grid ini juga memuat bobot satu anggota saja, misalnya (1, 0, 0), sehingga campuran tidak pernah lebih buruk daripada anggota terbaik pada blok CALIBRATION; bila anggota lain tidak membantu, bobotnya otomatis nol dan anggota itu dilewati saat prediksi.

Tabel 8.4 Alternatif Metode Kombinasi
<!-- widths: 22,78 -->
| Alternatif | Alasan tidak dipakai |
|---|---|
| Hard voting | Hanya menghitung suara kelas dan membuang tingkat keyakinan, padahal keputusan selektif bergantung pada probabilitas |
| Stacking dengan meta-learner | Meta-learner yang belajar dari prediksi in-sample rawan bocor; versi out-of-fold membutuhkan blok data tambahan |
| Rata-rata tanpa bobot | Anggota yang lemah ikut menurunkan kualitas; bobot terpilih dapat menolak anggota tersebut |
| SMOTE atau oversampling | Label tumpang tindih pada candle berdekatan, sehingga data sintetis dapat membocorkan informasi masa depan |

### Output yang Dihasilkan

Untuk setiap fold, `metrics.json` mencatat di bawah kunci `ensemble`: bobot, temperatur, dan log loss setiap anggota; log loss campuran; anggota terbaik; serta keragaman setiap pasangan anggota, yaitu proporsi candle dengan prediksi berbeda (disagreement) dan korelasi indikator salah (error correlation). Korelasi kesalahan yang rendah menandakan syarat keragaman terpenuhi. Struktur output tersebut adalah:

```
"ensemble": {
  "method": "weighted_soft_voting",
  "fitted_on": "calibration",
  "members": {"xgboost": {"temperature": …, "log_loss": …, "weight": …}, …},
  "ensemble_log_loss": …,
  "best_single_member": "…",
  "best_single_log_loss": …,
  "diversity": {"xgboost|random_forest": {"disagreement": …, "error_correlation": …}, …}
}
```

### Hasil

Hasil ensemble pada data riil ditampilkan pada Tabel 10.3 (ringkasan), 10.7 (per tahun), 10.11 (bobot), dan 10.12 (keragaman). Bobot terpilih didominasi XGBoost (0,35–0,80) dan random forest (0,10–0,65), sedangkan regresi logistik hampir tidak dipakai (0–0,10). Log loss campuran pada blok CALIBRATION hanya 0,0003–0,0008 lebih kecil daripada anggota terbaik. Pada blok EVALUATION, ensemble memperoleh precision BUY 47,84% dan SELL 47,73%, hampir sama dengan XGBoost (47,82% dan 47,24%), tetapi directional F1-nya lebih rendah (0,1253 dibanding 0,1371) karena coverage-nya lebih kecil. Ensemble hanya dipakai sebagai model final bila lebih baik daripada XGBoost tunggal secara konsisten pada fold validasi, karena ensemble melatih tiga model dan lebih lambat. Syarat ini belum terpenuhi, sehingga model utama tetap XGBoost (bab 11.2, Temuan 4).

## 8.6 Feature Selection di Dalam Fold

### Mengapa Feature Selection Diperlukan

Fitur yang tidak relevan atau saling mirip menambah noise dan variance, membuat model mudah overfitting pada satu periode, dan menyulitkan penjelasan model. Seleksi harus diulang di setiap fold dengan hanya memakai blok TRAIN dan CALIBRATION fold tersebut; seleksi sekali di awal pada seluruh data akan membocorkan informasi dari periode evaluasi (SoT D-21).

Tabel 8.5 Tahapan Feature Selection (xgboost_selected)
<!-- widths: 24,14,44,18 -->
| Tahap | Kategori | Aturan | Data |
|---|---|---|---|
| 1. Nilai hilang dan konstan | Filter | Buang fitur dengan nilai hilang > 40% atau simpangan baku nol | TRAIN |
| 2. Audit kebocoran | Filter | Catat fitur dengan Spearman terhadap arah label > 0,95 | TRAIN |
| 3. Redundansi | Filter | Dari pasangan dengan Spearman > 0,95, pertahankan yang mutual information-nya lebih tinggi | Sampel 30.000 baris TRAIN |
| 4. Gain XGBoost | Embedded | Dicatat sebagai pembanding, bukan dasar keputusan | TRAIN |
| 5. Permutation importance | Model-based | Pertahankan fitur yang menaikkan log loss bila diacak (minimal 5 fitur) | CALIBRATION |
| 6. Ablation kelompok | Wrapper | Walk-forward diulang tanpa satu kelompok fitur (perintah ablation) | Seluruh fold |

Permutation importance fitur j adalah rata-rata kenaikan log loss pada blok CALIBRATION ketika kolom j diacak R = 3 kali:

$$ PI_j = (1/R) · Σ_r [ LogLoss(X_cal dengan kolom j diacak) − LogLoss(X_cal) ] $$ (8.9)

Kode 8.6 Permutation importance (technical/feature_selection.py, baris 114–131)
%code technical/feature_selection.py 114 131

Kode 8.7 Aturan memilih fitur (technical/feature_selection.py, baris 168–173)
%code technical/feature_selection.py 168 173

Keputusan memakai permutation importance pada data di luar TRAIN, bukan gain XGBoost, karena gain cenderung tinggi pada fitur dengan banyak nilai unik dan terbagi di antara fitur yang berkorelasi. Korelasi peringkat gain dengan permutation importance tetap dicatat untuk melihat seberapa sejalan keduanya.

### Output yang Dihasilkan

Untuk setiap fold, metrics.json mencatat status setiap fitur (terpilih, atau alasan dibuang), mutual information, gain, permutation importance, dan dugaan kebocoran. Stabilitas seleksi, yaitu berapa fold sebuah fitur terpilih, ditulis ke `feature_selection_xgboost_selected.csv`, dan terminal mencetak jumlah fitur terpilih per fold. Model `xgboost_selected` belum dijalankan pada data riil (tidak ada pada folder runs/ yang tersedia), sehingga hasil seleksi belum dapat dilaporkan. Perintahnya adalah `walk-forward --models xgboost,xgboost_selected`. Subset fitur hanya dipakai bila kinerja `xgboost_selected` tidak lebih buruk daripada `xgboost` dengan 41 fitur.

# 9. Strategi Evaluasi

## 9.1 Walk-Forward Expanding Window

Data deret waktu tidak boleh dibagi acak. Setiap tahun 2021–2025 menjadi satu fold evaluasi, data training selalu dimulai pada awal 2016, dan setiap fold terdiri atas empat blok berurutan.

Tabel 9.1 Blok Waktu Setiap Fold
<!-- widths: 18,22,18,18,24 -->
| Fold | TRAIN | CALIBRATION | THRESHOLD | EVALUATION |
|---|---|---|---|---|
| validation_2021 | 2016 – 30 Jun 2020 | Jul – Sep 2020 | Okt – Des 2020 | 2021 |
| validation_2022 | 2016 – 30 Jun 2021 | Jul – Sep 2021 | Okt – Des 2021 | 2022 |
| validation_2023 | 2016 – 30 Jun 2022 | Jul – Sep 2022 | Okt – Des 2022 | 2023 |
| validation_2024 | 2016 – 30 Jun 2023 | Jul – Sep 2023 | Okt – Des 2023 | 2024 |
| validation_2025 | 2016 – 30 Jun 2024 | Jul – Sep 2024 | Okt – Des 2024 | 2025 |
| final_test | 2016 – 30 Jun 2025 | Jul – Sep 2025 | Okt – Des 2025 | 2026 (sekali, --confirm) |

TRAIN dipakai untuk melatih model, CALIBRATION untuk early stopping, temperature scaling, bobot ensemble, dan permutation importance, THRESHOLD untuk memilih threshold BUY/SELL, dan EVALUATION hanya untuk mengukur kinerja.

## 9.2 Purging dan Embargo

Label candle t melihat harga sampai t + 30 menit, sehingga label di akhir suatu blok bisa "melihat" blok berikutnya. Baris hanya dipakai bila `label_end_time` ditambah embargo 6 candle masih di dalam bloknya.

Kode 9.1 Pembagian blok dengan purging dan embargo (technical/splits.py, baris 95–113)
%code technical/splits.py 95 113

## 9.3 Kalibrasi dan Pemilihan Threshold

Probabilitas mentah dikalibrasi dengan temperature scaling satu parameter, p_T = softmax(log p / T), dengan T dipilih dari 121 nilai log-spaced 0,25–4 yang meminimalkan log loss pada CALIBRATION. Threshold BUY dan SELL dipilih terpisah pada blok THRESHOLD dari grid 0,34–0,90 dengan langkah 0,02. Sebuah threshold lolos bila menghasilkan minimal 30 sinyal dan precision minimal 1,2 × prevalensi kelas, lalu dipilih yang F1-nya tertinggi. Bila tidak ada yang lolos, arah tersebut dinonaktifkan.

Kode 9.2 Syarat threshold yang lolos (technical/decision.py, baris 104–131)
%code technical/decision.py 104 131

## 9.4 Metrik Evaluasi

Tabel 9.2 Metrik dan Alasannya
<!-- widths: 30,70 -->
| Metrik | Alasan |
|---|---|
| Precision BUY/SELL dan interval Wilson 95% | Ukuran utama: seberapa sering sinyal yang dikeluarkan benar |
| Lift = precision / prevalensi | Membandingkan sinyal dengan tebakan acak pada periode yang sama |
| Recall dan F1 per kelas, macro F1, directional F1 | Keseimbangan precision dan cakupan; directional F1 = rata-rata F1 BUY dan SELL |
| Coverage sinyal | Proporsi candle yang menghasilkan BUY/SELL; precision tanpa coverage dapat menyesatkan |
| Kesalahan arah berlawanan | BUY saat label SELL atau sebaliknya, kesalahan yang paling mahal |
| Log loss, Brier, ECE | Kualitas probabilitas setelah kalibrasi |
| Accuracy | Dilaporkan, tetapi bukan ukuran utama karena NO_TRADE dapat mendominasi |

Hasil antarfold dirangkum dengan rata-rata dan simpangan baku (`summarize_folds`). Konfigurasi dipilih hanya dari hasil walk-forward 2021–2025.

## 9.5 Tiga Ukuran Backtest

Backtest memakai prediksi out-of-sample setiap fold, sehingga setiap prediksi dibuat oleh model yang memang tersedia pada waktu itu. Tiga ukuran dipisahkan dengan sengaja: (1) akurasi klasifikasi terhadap label triple-barrier, (2) ketepatan arah, yaitu tanda close t+6 − close t tanpa ambang, dan (3) simulasi transaksi setelah spread dengan aturan pada Tabel 1.2. Ketiganya ditampilkan di dashboard `/backtest`, termasuk waktu buka dan tutup setiap transaksi, alasan tutup, serta persentase spread terhadap harga dan terhadap jarak TP.

## 9.6 Perintah Eksperimen

Tabel 9.3 Perintah dan Hasilnya
<!-- widths: 42,58 -->
| Perintah | Hasil di runs/<waktu>_<perintah>_<hash>/ |
|---|---|
| `technical.run label-scan` | label_scan.csv: distribusi label untuk horizon 3/6/12 dan k 0,5–2,0 |
| `technical.run walk-forward --models …` | metrics.json, run.json, backtest/<model>/validation_<tahun>/predictions.csv |
| `technical.run ablation` | ablation.json: walk-forward tanpa satu kelompok fitur |
| `technical.run compare-scratch --year 2025` | comparison.json: library vs scratch pada satu fold |
| `technical.run final-test --model <pemenang> --confirm` | metrics.json, test_period.json, latest_signal.json, live_bundle.joblib |

# 10. Hasil Eksperimen Awal

## 10.1 Bukti yang Sudah Tersedia

Tabel 10.1 Bukti Bahwa Pipeline Berjalan
<!-- widths: 28,44,28 -->
| Bukti | Hasil | Cara mengulang |
|---|---|---|
| Unit test seluruh proyek | 128 tes lolos, 1 dilewati (7 November 2026) | `uv run pytest` |
| Kausalitas fitur | Fitur tidak berubah bila candle masa depan diubah | `test_features_do_not_change_when_future_candles_change` |
| Kesesuaian XGBoost from scratch | Identik dengan library pada kandidat split yang sama (Tabel 8.2) | `tests/test_xgboost_scratch.py` |
| Simulasi transaksi | TP = +k·ATR, SL = −(k·ATR + 2 × spread), waktu buka/tutup per transaksi | `tests/test_backtest_simulation.py` |
| Dashboard backtest | Ringkasan, riwayat transaksi, dan laporan spread terbaca dari folder runs | `tests/test_dashboard_app.py` |
| Kualitas data dan EDA | 754.660 bar valid; Gambar 4.1–4.7 | `reports/analytics/` |
| Reproduksibilitas | Ringkasan metrik XGBoost identik pada tiga run walk-forward (3 dan 7 November 2026) | `runs/*_walk-forward_*/metrics.json` |

## 10.2 Uji Kewajaran pada Data Sintetis

Sebelum data riil dibaca, pipeline walk-forward lengkap dijalankan pada 130.000 candle random walk sintetis (generator `make_candles` pada `tests/test_technical_pipeline.py`, seed 7, satu fold evaluasi 2025). Data random walk tidak memiliki pola yang dapat diprediksi, sehingga sistem yang benar seharusnya menahan diri. Hasilnya, threshold BUY dan SELL pada xgboost, logistic, random_forest, ensemble, dan xgboost_selected dinonaktifkan karena tidak ada threshold dengan precision ≥ 1,2 × prevalensi, sehingga coverage-nya 0%. Persistence tetap mengeluarkan sinyal pada 45,86% candle dengan precision BUY 42,63% terhadap prevalensi BUY 43,01% (lift ≈ 0,99), yaitu setara tebakan acak. Uji ini menunjukkan decision layer tidak memaksakan sinyal ketika model tidak memiliki keunggulan. Angka pada bagian ini **bukan** hasil XAU/USD.

## 10.3 Hasil Walk-Forward pada Data XAU/USD

Walk-forward dijalankan pada 703.425 candle 2016–2025 dalam dua run dengan konfigurasi yang sama (Tabel 10.2). Run pertama memuat baseline, regresi logistik, dan XGBoost; run kedua memuat XGBoost, xgboost_balanced, random_forest, dan ensemble. Prediksi per candle setiap fold disimpan di folder backtest/ dan dipakai untuk simulasi transaksi (Tabel 10.13–10.15) serta dashboard backtest.

%include hasil_walk_forward.md

## 10.4 Hasil Final Test 2026

Final-test dijalankan satu kali pada 3 November 2026 untuk XGBoost, setelah walk-forward baseline dan XGBoost selesai. Model dilatih dengan data sampai Juni 2025, dikalibrasi pada Juli–September 2025, memilih threshold pada Oktober–Desember 2025, lalu diuji pada 2026. Statusnya out-of-sample retrospektif (bab 1.6).

%include hasil_final_test.md

# 11. Analisis Awal

## 11.1 Analisis Pipeline Generasi Sebelumnya

Pipeline sebelumnya menunjukkan tiga masalah (Tabel 7.2). Classifier cenderung memprediksi BUY (recall 94,18%) sementara recall SELL hanya 6,74%, sehingga model praktis tidak mengenali penurunan. Macro F1 turun dari 0,3525 pada blok TUNE menjadi 0,2478 pada TEST 2026, yang menandakan konfigurasi terlalu cocok dengan satu blok tuning dan karakteristik pasar berubah antarperiode. Aturan sinyal dengan syarat precision 90% terlalu ketat sehingga tidak menghasilkan satu pun sinyal dari 51.236 keputusan.

Tabel 11.1 Tanggapan Pipeline Baru terhadap Masalah Sebelumnya
<!-- widths: 36,64 -->
| Masalah | Perubahan pada pipeline baru |
|---|---|
| Label hampir tanpa NO_TRADE (7,81%) | Triple-barrier berbasis ATR dan spread; NO_TRADE muncul alami |
| Konfigurasi dipilih pada satu blok TUNE | Walk-forward lima fold dengan purging dan embargo |
| 223 fitur dengan seleksi satu kali | 41 fitur kausal; seleksi diulang di setiap fold |
| Aturan precision ≥ 90% (0 sinyal) | Threshold per arah: precision ≥ 1,2 × prevalensi, minimal 30 sinyal |
| Tanpa pembanding sederhana | Baseline always_no_trade, persistence, logistic |

## 11.2 Analisis Hasil Walk-Forward

%include analisis_walk_forward.md


## 11.3 Risiko dan Keterbatasan

- Blok CALIBRATION dipakai untuk early stopping, temperatur, bobot ensemble, dan permutation importance, sehingga kalibrasi sedikit optimistis (SoT D-13); threshold dan evaluasi tetap memakai blok bersih.
- Interval Wilson mengasumsikan percobaan independen, padahal label berdekatan tumpang tindih, sehingga intervalnya cenderung terlalu sempit.
- Hyperparameter belum di-tuning (bab 13); hasil awal bisa berubah setelah pencarian hyperparameter antarfold.
- Simulasi berada pada tingkat candle dan mengizinkan posisi tumpang tindih, sehingga P&L mengukur kualitas sinyal, bukan saldo akun nyata.

# 12. Rencana Pengembangan Setelah UTS

Prioritas pengembangan ditentukan oleh temuan pada bab 11.2 (Tabel 12.1); subbab berikutnya merinci setiap area pengembangan.

Tabel 12.1 Prioritas Pengembangan Berdasarkan Hasil Eksperimen Awal
<!-- widths: 40,60 -->
| Temuan (bab 11.2) | Tindakan setelah UTS |
|---|---|
| Precision belum mencapai titik impas sekitar 53% setelah spread (Temuan 8) | Uji k = 1,5 dan horizon 12 candle agar spread menjadi porsi barrier yang lebih kecil, threshold berbasis nilai ekonomi, dan model dua tahap (meta-labeling) |
| Batas 300 pohon tersentuh di semua fold (Temuan 9) | Naikkan n_estimators dengan early stopping, atau ubah learning rate, lalu bandingkan rata-rata walk-forward |
| Threshold tidak selalu cocok untuk tahun berikutnya; SELL 2023 dan BUY 2026 nonaktif (Temuan 6 dan 10) | Analisis stabilitas threshold antarfold dan uji blok THRESHOLD yang lebih panjang |
| NO_TRADE pada TRAIN 26–33%, pada EVALUATION 16–20% (Temuan 7) | Bandingkan expanding window dengan rolling window atau bobot sampel yang lebih besar untuk data terbaru |
| Ensemble belum lebih baik (Temuan 4) | Ganti anggota yang lemah (regresi logistik) dengan model yang lebih beragam, misalnya LSTM/CNN-LSTM, sebelum mengulang ensemble |
| xgboost_selected, compare-scratch, dan ablation belum dijalankan | Jalankan ketiganya pada konfigurasi yang sama dan lengkapi bab 10 |

## 12.1 Pengembangan Kandidat Model

- Menjalankan skenario E2 dan E3 pada Skripsi Bab III (Tabel 3.6): LSTM dan CNN-LSTM pada jendela 48, 96, dan 144 candle, dengan fold, kalibrasi, dan threshold yang sama agar dapat dibandingkan langsung dengan XGBoost.
- Model dua tahap atau meta-labeling: tahap pertama menilai apakah candle layak ditradingkan, tahap kedua menentukan arah (SoT D-08, future work).
- Stacking out-of-fold bila ensemble soft voting terbukti membantu, dengan blok data tambahan agar meta-learner tidak belajar dari prediksi in-sample.
- LightGBM atau CatBoost hanya ditambahkan bila ada alasan yang terukur, karena menambah dependency (SoT D-04).

## 12.2 Feature Selection Lanjutan

- Memakai tabel stabilitas `feature_selection_xgboost_selected.csv` untuk menyusun subset fitur yang terpilih di hampir semua fold.
- Menjalankan `ablation` kelompok fitur (EXP-R04) dan mengisi kolom ablation pada SoT bagian D.
- Menguji fitur multi-timeframe (M15 dan H1 sudah dikumpulkan) dan fitur rezim pasar.
- Menambahkan penjelasan per prediksi (misalnya SHAP) untuk analisis kesalahan, bukan untuk memilih fitur.

## 12.3 Analisis Error

- Memecah precision dan kesalahan arah berlawanan per sesi (Asia, London, New York), per jam, per hari, per rezim volatilitas (`atr_14_vs_daily_mean`), dan per kekuatan trend (ADX ≥ 25).
- Menganalisis sinyal NEUTRAL: seberapa jauh harga bergerak sebelum horizon habis, untuk menilai apakah horizon atau k terlalu sempit.
- Membandingkan confusion matrix antarfold untuk melihat apakah kegagalan terpusat pada tahun tertentu, misalnya 2025 dengan kenaikan +62,49%.

## 12.4 Analisis Threshold

- Kurva precision–coverage per arah pada blok THRESHOLD dan EVALUATION untuk melihat trade-off antara jumlah sinyal dan ketepatannya.
- Stabilitas threshold antarfold: threshold yang terlalu berbeda antar tahun menandakan probabilitas belum stabil.
- Sensitivitas terhadap `min_precision_lift` (1,1; 1,2; 1,5) dan `min_signals` (30; 50; 100), dipilih dari rata-rata walk-forward.
- Threshold berbasis nilai ekonomi, yaitu memaksimalkan P&L harapan setelah spread, sebagai pembanding threshold berbasis F1.

## 12.5 Hyperparameter Tuning

Pencarian dilakukan antarfold walk-forward, bukan pada satu blok, karena pipeline sebelumnya turun dari macro F1 0,3525 pada TUNE menjadi 0,2478 pada TEST. Ruang pencarian mengikuti Skripsi Bab III Tabel 3.3.

Tabel 12.2 Ruang Pencarian Hyperparameter Setelah UTS
<!-- widths: 24,30,46 -->
| Hyperparameter | Nilai awal | Nilai yang diuji |
|---|---|---|
| max_depth | 4 | 3–6 |
| learning_rate | 0,05 | 0,03–0,1 |
| min_child_weight | 20 | 10–50 |
| reg_lambda | 5 | 1–20 |
| n_estimators / early stopping | 300 / 30 | batas dinaikkan (misalnya sampai 1.000) dengan early stopping 30 ronde, karena batas 300 tersentuh di semua fold |
| Label: horizon dan k | 6 candle, k = 1,0 | 3/6/12 candle; k = 0,5–2,0 (label-scan, EXP-R01) |
| Bobot kelas | tanpa | seimbang atau tanpa |

Kriteria pemilihan adalah rata-rata directional F1 dan lift pada lima fold, dengan simpangan baku sebagai pembeda bila rata-ratanya mirip. Data 2026 tidak dipakai untuk memilih apa pun.

## 12.6 Perbandingan Model

- Tabel rata-rata ± simpangan baku setiap model pada lima fold, ditambah jumlah fold tempat model mengalahkan XGBoost.
- Uji berpasangan per fold dan interval bootstrap blok (block bootstrap) untuk selisih precision, karena observasi berdekatan tidak independen.
- Perbandingan yang adil: semua model memakai fold, kalibrasi, threshold, dan aturan simulasi yang sama.

## 12.7 Simulasi, Integrasi, dan Pengujian Forward

- Aturan satu posisi pada satu waktu dan spread keluar dari candle tempat barrier tersentuh, sebagai pembanding simulasi saat ini.
- Integrasi dengan Agen Manajemen Risiko (skenario E4) dan Agen Fundamental berbasis LLM (E5).
- Mengunci konfigurasi lalu menjalankan uji forward pada periode baru, misalnya mulai November 2026, karena 2026 bukan blind test (SoT D-12).

Tabel 12.3 Urutan Pekerjaan Setelah UTS
<!-- widths: 8,52,40 -->
| No. | Pekerjaan | Keluaran |
|---|---|---|
| 1 | Walk-forward xgboost_selected, compare-scratch, dan ablation | Tabel feature selection, comparison.json, ablation.json |
| 2 | Tuning n_estimators, learning rate, dan hyperparameter lain antarfold | Konfigurasi XGBoost baru di configs/technical.toml |
| 3 | Walk-forward kombinasi horizon/k dari label-scan (mis. 6/1,5 dan 12/1,5) | Keputusan D-02/D-03 berdasarkan kinerja setelah spread |
| 4 | Analisis error dan analisis threshold | Tabel per sesi/rezim, kurva precision–coverage, stabilitas threshold |
| 5 | Aturan simulasi satu posisi pada satu waktu | Simulasi yang lebih dekat dengan akun nyata |
| 6 | Kunci konfigurasi, lalu uji forward mulai November 2026 | Uji yang benar-benar buta (SoT D-12) |
| 7 | LSTM/CNN-LSTM, agen risiko, agen fundamental | Skenario E2–E5 |

# 13. Justifikasi Nilai Konstanta dalam Kode

## 13.1 Penanda Sumber Alasan

Setiap nilai di `configs/technical.toml` dan `technical/config.py` diberi komentar dengan penanda sumber: [SoT X] berarti alasannya tertulis di SOURCE_OF_TRUTH.md; [design] berarti pilihan rancangan yang belum di-tuning dan akan divalidasi dengan walk-forward; [teknis] berarti konsekuensi matematis atau komputasi yang pasti. Nilai hyperparameter XGBoost termasuk [design]: alasannya berupa penalaran teknis, bukan hasil tuning.

Kode 13.1 Nilai default XGBoost beserta alasannya (technical/config.py, baris 89–121)
%code technical/config.py 89 121

## 13.2 Hyperparameter XGBoost

Tabel 13.1 Alasan Nilai Hyperparameter XGBoost (default library sebagai pembanding)
<!-- widths: 23,16,9,52 -->
| Parameter | Nilai | Default | Alasan |
|---|---|---|---|
| n_estimators | 300 | 100 | Batas atas; jumlah pohon dipilih early stopping. Pada data riil 295–300 pohon terpakai di semua fold (Tabel 10.6), jadi batas ini perlu dinaikkan |
| learning_rate | 0,05 | 0,3 | Langkah kecil (shrinkage) agar model tidak cepat menghafal noise M5; 0,01 membutuhkan ribuan pohon |
| early_stopping_rounds | 30 | – | Sekitar 10% dari 300; cukup sabar terhadap fluktuasi log loss, tidak membuang waktu |
| max_depth | 4 | 6 | Maksimal 16 daun, interaksi sampai 4 fitur; pohon dangkal sebagai weak learner |
| min_child_weight | 20 | 1 | Jumlah Hessian minimum per daun. Hessian softprob 2p(1−p) ≤ 0,5, jadi minimal sekitar 40 candle per daun (p = 0,5), 63 (p = 0,2), dan 111 (p = 0,1) |
| reg_lambda | 5 | 1 | Bobot daun −G/(H+λ): daun kecil (H = 20) tinggal 80% langkah Newton, daun besar (H = 1000) tetap 99,5% |
| reg_alpha | 0 | 0 | Fitur padat, bukan sparse; L2 dan min_child_weight sudah cukup |
| gamma | 0 | 0 | Pruning lewat max_depth dan min_child_weight; library dan scratch identik walau konvensi gamma berbeda faktor 2 (EXP-S05) |
| subsample, colsample_bytree | 0,8 | 1 | Setiap pohon melihat 80% baris dan sekitar 32–33 dari 41 fitur, sehingga pohon lebih beragam |
| max_bin | 64 | 256 | Sekitar 1,6% data per bin; lebih cepat dan tidak membuat potongan super halus yang cocok dengan noise |
| objective, tree_method, eval_metric | multi:softprob, hist, mlogloss | – | Probabilitas untuk decision layer, histogram cepat, early stopping berbasis kualitas probabilitas |
| random_state | 42 | – | Angka bebas untuk reproduksibilitas; tidak bermakna statistik |

## 13.3 Konstanta XGBoost from Scratch

Tabel 13.2 Konstanta pada technical/xgboost_scratch.py
<!-- widths: 28,72 -->
| Konstanta | Alasan |
|---|---|
| HESSIAN_FLOOR = 10⁻¹⁶ | Mencegah pembagian nol saat p mendekati 0 atau 1; sama dengan XGBoost resmi |
| Faktor 2 pada Hessian | Meniru XGBoost resmi agar kedua implementasi sebanding; langkah per pohon menjadi kira-kira setengah |
| LOG_LOSS_EPSILON = 10⁻¹⁵ | Probabilitas di-clip agar log(0) tidak menjadi −∞; loss per sampel maksimal sekitar 34,5 |
| max_bin antara 2 dan 255 | Indeks bin disimpan sebagai uint8 dan satu bin dicadangkan untuk NaN |
| base_score = "prior" | Model mulai dari distribusi kelas TRAIN; terbukti sama dengan XGBoost 3.x (EXP-S02) |
| n_estimators = 150 | Waktu komputasi: sekitar 3,4 detik per ronde, sehingga 150 ronde ≈ 9 menit per fold (EXP-S04) |
| Berhenti bila H < 2 × min_child_weight | Konsekuensi pasti: node tidak mungkin dibagi menjadi dua anak yang masing-masing memenuhi batas |

## 13.4 Label, Split, Ensemble, Feature Selection, dan Decision Layer

Tabel 13.3 Konstanta Pipeline Lainnya
<!-- widths: 26,14,46,14 -->
| Konstanta | Nilai | Alasan | Sumber |
|---|---|---|---|
| horizon_bars | 6 (30 menit) | 15 menit terlalu kecil dibanding spread; nilai final dari label-scan | SoT D-02, D-18 |
| atr_period | 14 | Konvensi Wilder, bukan nilai optimal terbukti | SoT tabel C |
| barrier_atr_multiple | 1,0 | Usulan awal; k besar membuat NO_TRADE lebih banyak | SoT D-03 |
| embargo_bars | 6 | Disamakan dengan horizon label | SoT D-11 |
| validation_years | 2021–2025 | Fold pertama tetap memiliki sekitar 4,5 tahun TRAIN | design |
| weight_step ensemble | 0,05 | 231 kombinasi untuk 3 anggota; 0,1 terlalu kasar (66), 0,01 lambat dan rawan overfit (5.151) | design |
| max_missing / redundancy / leakage | 0,4 / 0,95 / 0,95 | Hanya fitur yang hampir kosong atau hampir identik yang dibuang | SoT D-21 |
| sample_rows, permutation_repeats, min_features | 30.000 / 3 / 5 | Kecepatan mutual information, rata-rata pengacakan, dan jaring pengaman jumlah fitur | design |
| Grid temperatur | 121 nilai log-spaced 0,25–4 | Simetris di skala log dengan titik tengah tepat T = 1 | design |
| Grid threshold | 0,34–0,90, langkah 0,02 | 0,34 sedikit di atas 1/3 (tebakan seragam tiga kelas); 29 kandidat | design |
| min_precision_lift | 1,2 | Model acak memiliki precision = prevalensi; minimal 20% lebih baik dari acak | SoT D-09 |
| min_signals | 30 | Precision dari segelintir sinyal tidak stabil | SoT D-09 |
| Random forest min_samples_leaf | 100 | Data M5 bising, daun minimal 100 candle | design |
| Lot simulasi | 0,01 lot × 100 oz | Ukuran kontrak umum XAUUSD; hanya mengubah skala P&L | design |

## 13.5 Ringkasan

Nilai yang konsekuensinya pasti (misalnya batas 255 bin dan HESSIAN_FLOOR) dan nilai yang dibuktikan eksperimen (base score, konvensi gamma, waktu per ronde) dapat dipertanggungjawabkan langsung. Hyperparameter XGBoost, grid threshold, dan pengali barrier masih berupa design decision yang akan diuji pada tahap setelah UTS (bab 12.5). Karena itu, hasil pada bab 10 dilaporkan sebagai hasil konfigurasi awal, bukan konfigurasi optimal. Walk-forward sudah memberi bukti pertama untuk dua nilai: batas 300 pohon terlalu kecil, dan syarat lift 1,2 cukup ketat sehingga beberapa arah dinonaktifkan (SELL 2023 dan BUY 2026).

# Daftar Pustaka

T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., 2016, pp. 785–794, doi: 10.1145/2939672.2939785.

J. H. Friedman, "Greedy function approximation: A gradient boosting machine," Ann. Statist., vol. 29, no. 5, pp. 1189–1232, 2001, doi: 10.1214/aos/1013203451.

M. López de Prado, Advances in Financial Machine Learning. Hoboken, NJ, USA: Wiley, 2018 (§3.4 triple-barrier, §7.4 purging dan embargo).

T. G. Dietterich, "Ensemble methods in machine learning," in Multiple Classifier Systems (MCS 2000), LNCS, vol. 1857. Springer, 2000, pp. 1–15, doi: 10.1007/3-540-45014-9_1.

C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in Proc. 34th ICML, PMLR, vol. 70, 2017, pp. 1321–1330.

A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning," in Proc. 22nd ICML, 2005, pp. 625–632, doi: 10.1145/1102351.1102430.

C. K. Chow, "On optimum recognition error and reject tradeoff," IEEE Trans. Inf. Theory, vol. 16, no. 1, pp. 41–46, 1970, doi: 10.1109/TIT.1970.1054406.

Y. Geifman and R. El-Yaniv, "Selective classification for deep neural networks," in Adv. Neural Inf. Process. Syst. 30, 2017.

P. Sadorsky, "Predicting gold and silver price direction using tree-based classifiers," J. Risk Financial Manag., vol. 14, no. 5, Art. 198, 2021, doi: 10.3390/jrfm14050198.

H. He and E. A. Garcia, "Learning from imbalanced data," IEEE Trans. Knowl. Data Eng., vol. 21, no. 9, pp. 1263–1284, 2009, doi: 10.1109/TKDE.2008.239.

E. B. Wilson, "Probable inference, the law of succession, and statistical inference," J. Amer. Statist. Assoc., vol. 22, no. 158, pp. 209–212, 1927, doi: 10.1080/01621459.1927.10502953.

J. W. Wilder Jr., New Concepts in Technical Trading Systems. Greensboro, NC, USA: Trend Research, 1978.
