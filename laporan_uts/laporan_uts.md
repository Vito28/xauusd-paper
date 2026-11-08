<!--
Sumber LAPORAN UTS MILESTONE 1.docx. Jalankan dari root repo yang memuat technical/simulation.py terbaru
(branch feat/dashboard-buka-tutup-spread atau main setelah di-merge), dengan D = ../xauusd-paper-documents:
  0a. (bab 3.4, 4.6, 6.3) uv run python $D/laporan_uts/eda_m5.py --walk-forward runs/<id_walk_forward_2> --final-test runs/<id_final_test> --out $D/laporan_uts
  0b. (bab 10.5) uv run python $D/laporan_uts/uji_scratch_bin.py --year 2025 --out $D/laporan_uts/uji_scratch_bin_2025.json --candles-dir runs/<id_walk_forward_2>/backtest/candles --history-start 2021-01-01T00:00:00Z
  1. uv run python $D/laporan_uts/ringkas_runs.py --out $D/laporan_uts --uji-scratch $D/laporan_uts/uji_scratch_bin_2025.json runs/<id_label_scan> runs/<id_walk_forward_1> runs/<id_walk_forward_2> runs/<id_walk_forward_3> runs/<id_final_test> [runs/<id_compare_scratch>] [runs/<id_ablation>]
     (menulis hasil_label_scan.md, hasil_walk_forward.md, hasil_final_test.md, hasil_uji_scratch.md, hasil_blok_fold.md, dan bila ada hasil_compare_scratch.md, hasil_ablation.md)
  2. sesuaikan $D/laporan_uts/analisis_walk_forward.md bila angka berubah
  3. uv run python $D/laporan_uts/md2docx.py build "$D/BUSINESS UNDERSTANDING.docx" $D/laporan_uts/laporan_uts.md "$D/LAPORAN UTS MILESTONE 1.docx" .
  4. render ke PDF (soffice --headless --convert-to pdf), lalu
     uv run python $D/laporan_uts/md2docx.py toc-pages <laporan.pdf> $D/laporan_uts/laporan_uts.md . $D/laporan_uts/toc_pages.json
     dan ulangi langkah 3 dengan argumen tambahan $D/laporan_uts/toc_pages.json agar daftar isi bernomor halaman.
-->
%cover
judul: Technical Agent XAU/USD M5: Klasifikasi Selektif BUY / SELL / NO_TRADE dengan XGBoost
jenis: LAPORAN UTS MILESTONE 1
keterangan: Mata Kuliah Deep Learning
kelompok: KELOMPOK GOLDMIND
anggota: DELVIN DE VITO | 231111185
anggota: DICKSON YANGGA | 231110389
anggota: ELVINA ESTELITA | 231112380
logo: ../Draft Skripsi Bab I-III - Sistem Multi-Agen XAUUSD.docx#word/media/image1.png
instansi: PROGRAM STUDI S-1 TEKNIK INFORMATIKA
instansi: FAKULTAS INFORMATIKA
instansi: UNIVERSITAS MIKROSKIL
instansi: MEDAN
instansi: 2026
%endcover
%toc

%pagebreak

# Ringkasan

Proyek ini membangun Technical Agent untuk XAU/USD pada timeframe M5. Setiap 5 menit, agen memutuskan BUY, SELL, atau NO_TRADE untuk horizon 30 menit dan hanya mengeluarkan BUY atau SELL bila probabilitas terkalibrasinya cukup tinggi. Data berasal dari terminal MetaTrader 5 broker MetaQuotes-Demo: 754.660 candle M5 dari Januari 2016 sampai September 2026. Target dibentuk dengan label triple-barrier berbasis ATR dan spread. Model utama adalah XGBoost dengan 41 fitur teknikal kausal, yang dibandingkan dengan tiga baseline, random forest, XGBoost berbobot kelas, ensemble soft voting, XGBoost dengan feature selection di dalam fold, dan XGBoost yang ditulis ulang dari nol (from scratch).

Evaluasi memakai walk-forward lima fold (2021–2025) dengan purging dan embargo, ditambah uji out-of-sample retrospektif pada 2026. Hasil awal menunjukkan sinyal XGBoost lebih presisi daripada tebakan acak (lift 1,09–1,20, precision sekitar 47–48% terhadap prevalensi 40–42%), tetapi keunggulan ini belum cukup untuk menutup biaya spread (profit factor 0,91). Ensemble dan feature selection belum memperbaiki XGBoost tunggal. Implementasi from scratch terbukti identik dengan library bila batas bin-nya disamakan. EDA juga menemukan dua masalah kualitas data yang perlu diperbaiki setelah UTS: timestamp yang tersimpan sebagai UTC sebenarnya adalah jam server broker, dan 5,35% candle memiliki spread tercatat nol.

Pemetaan kriteria Milestone 1 ke bab laporan:

<!-- widths: 26,52,22 -->
| Kriteria | Yang dijawab | Bab |
|---|---|---|
| Problem formulation | Masalah, alasan penting, target, jenis pendekatan ML | 1.1–1.5 |
| Dataset dan sumber data | Sumber dan tautan, jumlah observasi, unit observasi, jumlah fitur, deskripsi atribut, target | 2 |
| Data understanding | Karakteristik data, pembentukan target, masalah kualitas yang ditemukan | 3 |
| EDA dan temuan awal | Visualisasi, ringkasan statistik, interpretasi, dampak ke desain | 4 |
| Pre-processing | Langkah dan alasannya, data bermasalah, nilai hilang, pencegahan kebocoran | 5 |
| Feature engineering awal | Transformasi fitur, alasan, dampak yang diharapkan dan terukur | 6 |
| Baseline | Baseline dan alasan pemilihannya | 7 |
| Kandidat model awal | XGBoost, from scratch, random forest, ensemble, feature selection | 8 |
| Strategi evaluasi | Pembagian data, metode validasi, metrik, kesesuaian dengan karakteristik data | 9 |
| Hasil eksperimen awal | Tabel hasil seluruh model, uji 2026, frekuensi sinyal, from scratch | 10 |
| Analisis awal | Perbedaan hasil, temuan, keterbatasan, hal yang perlu diperbaiki | 11 |
| Dokumentasi dan rencana | Repositori GitHub, cara mengulang eksperimen, rencana setelah UTS | 12 |
| Alasan pemilihan nilai | Justifikasi setiap konstanta di kode | 13 |

# 1. Problem Formulation

## 1.1 Masalah yang Ingin Diselesaikan

Proyek ini adalah bagian dari sistem multi-agen untuk perdagangan XAU/USD (emas terhadap dolar Amerika Serikat). Technical Agent bertugas membaca data harga dan memberi rekomendasi arah jangka pendek. Masalah yang ingin diselesaikan adalah: **pada setiap candle M5 yang baru tutup, menentukan apakah layak membuka posisi BUY, SELL, atau tidak sama sekali (NO_TRADE) untuk 30 menit berikutnya, sedemikian rupa sehingga sinyal yang dikeluarkan lebih sering benar daripada tebakan acak setelah biaya spread.**

Kata kuncinya adalah *selektif*. Sebagian besar candle M5 tidak memberikan peluang yang jelas, sehingga sistem yang baik harus tahu kapan menahan diri. Pipeline generasi sebelumnya gagal tepat di titik ini. Classifier-nya hampir selalu menebak BUY (recall BUY 94,18%, recall SELL hanya 6,74%), dan setelah aturan sinyal dengan syarat precision 90% diterapkan, tidak ada satu pun sinyal dari 51.236 keputusan pada 2026 (Tabel 7.2). Pipeline baru harus menghasilkan sinyal yang cukup sering untuk berguna, cukup presisi untuk lebih baik daripada acak, dan dapat diaudit alasannya.

## 1.2 Mengapa Masalah Ini Penting

- **Pergerakan harga besar dan berubah-ubah.** Selama 2016–2026 harga emas naik 288,61%, tetapi drawdown maksimumnya 26,52%. Pada 2025 harga naik 62,49%, sedangkan pada 2026 return bulanan berayun antara −11,6% dan +13,3% (bab 4). Keputusan yang salah pada kondisi seperti ini mahal.
- **Biaya spread menghabiskan keunggulan kecil.** Pada data ini, spread rata-rata 6,1% dari jarak take-profit (Tabel 10.16). Akibatnya, posisi harus mencapai take-profit pada sekitar 53% transaksi hanya untuk impas (Persamaan 11.1). Model yang sedikit lebih baik daripada acak tetap rugi bila sinyalnya tidak selektif.
- **Disiplin keputusan.** Trader individu rentan terhadap bias perilaku seperti overtrading. Kelas NO_TRADE dengan threshold yang terukur berfungsi sebagai filter yang disiplin dan dapat dijelaskan.
- **Kebutuhan sistem multi-agen.** Keluaran Technical Agent dipakai agen manajemen risiko, sehingga agen harus memberi probabilitas yang terkalibrasi dan reason code, bukan hanya label.
- **Kebutuhan metodologis.** Banyak penelitian emas meregresikan level harga dan melaporkan R² tinggi yang tidak bermakna untuk keputusan arah (SoT A-01). Proyek ini menguji klaim secara kronologis pada beberapa tahun, dengan baseline dan biaya transaksi.

## 1.3 Rumusan Masalah

Rumusan masalah pada tahap UTS adalah:

1) Bagaimana data candle M5 XAU/USD dapat diubah menjadi sinyal BUY, SELL, dan NO_TRADE yang lebih presisi daripada tebakan acak setelah biaya spread diperhitungkan?
2) Apakah XGBoost lebih baik daripada baseline sederhana (selalu NO_TRADE, persistence, dan regresi logistik) bila diuji secara kronologis pada beberapa tahun berbeda?
3) Apakah ensemble beberapa model dan feature selection di dalam fold memperbaiki kinerja dibandingkan XGBoost tunggal dengan 41 fitur?
4) Apakah implementasi XGBoost from scratch menghasilkan perhitungan yang sama dengan library XGBoost?

## 1.4 Target yang Ingin Dihasilkan

Target terdiri atas dua lapis: variabel target yang dipelajari model, dan keluaran yang dihasilkan agen.

Tabel 1.1 Formulasi Masalah dan Keluaran Agen
<!-- widths: 28,72 -->
| Komponen | Definisi |
|---|---|
| Unit observasi | Satu candle M5 XAU/USD yang sudah tutup (keputusan pada waktu close candle) |
| Input X | 41 fitur teknikal kausal dalam 7 kelompok, hanya memakai candle t dan sebelumnya |
| Variabel target y | Label triple-barrier: BUY, SELL, atau NO_TRADE dalam horizon 6 candle (30 menit), bab 3.2 |
| Model | Classifier multikelas yang mengeluarkan probabilitas [p_SELL, p_NO_TRADE, p_BUY] |
| Keputusan | BUY/SELL hanya jika probabilitas terkalibrasi melewati threshold arahnya; selain itu NO_TRADE |
| Keluaran agen | `TechnicalSignal`: sinyal, confidence, probabilitas, horizon, reason code, ringkasan rezim pasar |

Target terukur pada tahap UTS ditetapkan sebelum eksperimen dijalankan:

Tabel 1.2 Target Terukur (Kriteria Keberhasilan) Tahap UTS
<!-- widths: 30,70 -->
| Kriteria | Ukuran |
|---|---|
| Sinyal lebih baik dari acak | Precision BUY dan SELL ≥ 1,2 × prevalensi kelas tersebut (lift ≥ 1,2) |
| Sinyal cukup banyak untuk dinilai | Minimal 30 sinyal per arah pada blok pemilihan threshold |
| Konsisten antarperiode | Rata-rata dan simpangan baku metrik pada lima fold walk-forward 2021–2025 |
| Kualitas probabilitas | Log loss, Brier score, dan expected calibration error |
| Bernilai setelah biaya | Simulasi transaksi setelah spread: win rate dan profit factor |
| Implementasi benar | Unit test lolos dan XGBoost from scratch identik dengan library pada kandidat split yang sama |

## 1.5 Jenis Pendekatan Machine Learning

Pendekatan yang dipilih adalah **supervised learning untuk klasifikasi multikelas yang selektif** pada data tabular hasil rekayasa fitur deret waktu. Model utamanya adalah gradient boosted decision trees (XGBoost). Probabilitasnya dikalibrasi, lalu diubah menjadi keputusan dengan threshold per arah (reject option). Seluruh model divalidasi secara kronologis.

Tabel 1.3 Pilihan Pendekatan dan Alasannya
<!-- widths: 26,30,44 -->
| Aspek | Pilihan | Alasan |
|---|---|---|
| Jenis tugas | Klasifikasi 3 kelas, bukan regresi harga | Regresi level harga memberi R² tinggi yang menyesatkan (SoT A-01); pada pipeline sebelumnya regresi return 2026 (RMSE 19,6623 bps) sedikit lebih buruk daripada baseline zero-return (19,6599 bps; SoT EXP-L03) |
| Selektivitas | Reject option: NO_TRADE bila model kurang yakin | Arah M5 hampir tidak berautokorelasi (lag 1 = −0,002, Tabel 4.3), sehingga model hanya layak bertindak pada sebagian kecil candle (Chow, 1970; Geifman dan El-Yaniv, 2017) |
| Model utama | XGBoost (gradient boosting pohon keputusan) | Cocok untuk fitur tabular heterogen, menangani nilai hilang secara native, dan menangkap interaksi non-linear (Chen dan Guestrin, 2016; Sadorsky, 2021) |
| Keluaran | Probabilitas terkalibrasi (temperature scaling) | Threshold dan agen risiko membutuhkan probabilitas yang dapat dipercaya, bukan hanya skor (Guo et al., 2017) |
| Validasi | Walk-forward kronologis dengan purging dan embargo | Data deret waktu tidak boleh diacak, dan label 30 menit tumpang tindih antarcandle (López de Prado, 2018) |
| Deep learning | LSTM/CNN-LSTM ditunda ke setelah UTS | Pada tahap UTS fondasi data, label, dan evaluasi diuji dahulu dengan model tabular yang cepat; model sekuensial memakai fold dan aturan yang sama sebagai pembanding (bab 12.2) |

## 1.6 Aturan Keputusan dan Transaksi

Agar hasil model dapat dinilai dari sudut pandang trader, setiap sinyal BUY atau SELL diperlakukan sebagai satu posisi simulasi dengan aturan yang sama dengan definisi label (`technical/simulation.py`).

Tabel 1.4 Aturan Buka dan Tutup Posisi
<!-- widths: 24,76 -->
| Aspek | Aturan |
|---|---|
| Frekuensi penilaian | Setiap candle M5 yang sudah tutup, yaitu setiap 5 menit. Sinyal hanya keluar bila probabilitas arah tersebut ≥ threshold-nya; selain itu NO_TRADE. Frekuensi nyata per tahun ada pada Tabel 10.20 |
| Arah posisi | BUY membuka posisi beli (long) dan SELL membuka posisi jual (short). SELL bukan perintah untuk menutup BUY; setiap posisi ditutup sendiri oleh aturan di bawah |
| Buka posisi | Pada close candle keputusan. BUY di ask = close (bid) + spread; SELL di bid = close |
| Target | Barrier b = k · ATR14 + spread (k = 1,0). TP dan SL = close ± b pada harga bid |
| Tutup posisi | Barrier searah tersentuh dulu → TP; barrier berlawanan dulu → SL; keduanya di candle yang sama → dianggap SL; tidak ada yang tersentuh sampai candle ke-6 → tutup di close candle ke-6 (waktu habis) |
| Spread saat tutup | Dianggap sama dengan spread saat buka, karena spread di titik sentuh tidak tercatat |
| Hasil bersih | TP = +k·ATR; SL = −(k·ATR + 2 × spread) |

Contohnya, dengan close 4.000,00, spread 0,30, dan ATR14 3,00, barrier bernilai 3,30. Posisi BUY dibuka di 4.000,30, mencapai TP bila bid menyentuh 4.003,30 (+3,00), dan SL bila bid menyentuh 3.996,70 (−3,60). Setelah 30 menit, label aktual dihitung dari candle asli dengan aturan yang sama, lalu dibandingkan dengan sinyal: CORRECT bila barrier searah tersentuh lebih dulu, INCORRECT bila barrier lawan lebih dulu, dan NEUTRAL bila tidak ada barrier tersentuh. Karena aturannya sama, CORRECT selalu berarti TP dan INCORRECT selalu berarti SL.

## 1.7 Ruang Lingkup dan Batasan

- Hanya data teknikal (OHLC, tick volume, spread, waktu). Data fundamental ditangani Fundamental Agent di luar tahap ini (SoT A-07).
- Spread berasal dari akun demo MetaQuotes yang belum tentu sama dengan akun live (SoT A-09).
- Tahun 2026 adalah uji out-of-sample retrospektif, bukan blind test, karena hasil 2026 pernah dilihat pada pipeline sebelumnya (SoT A-05).
- Hasil klasifikasi bukan bukti profitabilitas; simulasi transaksi dilaporkan terpisah (SoT A-06).

# 2. Dataset dan Sumber Data

## 2.1 Sumber Data

Data adalah candle M5 XAUUSD yang diambil langsung dari terminal MetaTrader 5 (MT5) yang terhubung ke server broker MetaQuotes-Demo. Pengambilan dilakukan oleh collector proyek (`market/collector.py`) melalui paket Python resmi MetaTrader5, fungsi `copy_rates_range`. Hasilnya disimpan di PostgreSQL pada view `market.processed_m5`. Modul `technical/data.py` hanya membaca data tersebut secara read-only atau dari ekspor CSV, sehingga pipeline model tidak pernah mengambil data langsung dari broker.

Tabel 2.1 Sumber, Tautan, dan Cakupan Data
<!-- widths: 30,70 -->
| Atribut | Nilai |
|---|---|
| Platform dan broker | MetaTrader 5 (MetaQuotes Ltd.), server MetaQuotes-Demo, akun demo. https://www.metatrader5.com |
| API pengambilan | Paket Python MetaTrader5, `copy_rates_range`. https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesrange_py |
| Kode pengambilan | `market/collector.py` pada repositori https://github.com/Vito28/xauusd-paper |
| Simbol dan timeframe | XAUUSD, M5 (5 menit) |
| Periode bar | 4 Januari 2016 01:05 – 30 September 2026 06:40 (waktu buka bar, jam server; bab 3.4) |
| Penyimpanan | PostgreSQL, view `market.processed_m5` |
| Ketersediaan | Data broker tidak dipublikasikan sebagai file. Data dapat diambil ulang dengan collector dan akun demo MT5; candle setiap periode evaluasi juga tersimpan di `runs/*/backtest/candles/` pada repositori |

Sumber: `reports/analytics/time_series/654c1aa6-b15d-4135-99b9-145e16b2b899/M5/insights.json`.

## 2.2 Jumlah Observasi, Unit Observasi, Fitur, dan Target

Tabel 2.2 Ringkasan Dataset
<!-- widths: 38,62 -->
| Ukuran | Nilai |
|---|---|
| Unit observasi | Satu candle M5 XAUUSD yang sudah tutup |
| Jumlah observasi total | 754.660 candle valid (2016 – September 2026) |
| Data pengembangan 2016–2025 | 703.425 candle |
| Observasi berlabel lengkap 2016–2025 (horizon 6 candle) | 687.693 candle (Tabel 3.2) |
| Data uji 2026 | 51.236 candle diprediksi, 50.071 di antaranya berlabel lengkap |
| Atribut mentah | 13 kolom (Tabel 2.3), 9 di antaranya dipakai pipeline |
| Fitur model | 41 fitur turunan dalam 7 kelompok (bab 6, Tabel 6.3) |
| Target | Label triple-barrier tiga kelas: SELL (0), NO_TRADE (1), BUY (2) |

## 2.3 Deskripsi Atribut Data

Tabel 2.3 Atribut pada View market.processed_m5
<!-- widths: 26,14,60 -->
| Atribut | Tipe | Deskripsi dan kegunaan |
|---|---|---|
| source_key | Teks | Identitas sumber (`metaquotes-demo-xauusd`); dipakai untuk memfilter data |
| dataset_symbol, actual_symbol | Teks | Simbol standar dan simbol di broker (XAUUSD); dipakai untuk memfilter data |
| timeframe | Teks | Timeframe bar (M5); dipakai untuk memfilter data |
| open_time_utc, close_time_utc | Datetime | Waktu buka dan tutup candle; mengurutkan data, menentukan waktu keputusan (close), dan mendeteksi gap. Isinya jam server broker, bukan UTC (bab 3.4) |
| open, high, low, close | Numerik (USD) | Harga bid pembukaan, tertinggi, terendah, dan penutupan; dasar fitur harga, ATR, label triple-barrier, dan simulasi TP/SL |
| tick_volume | Bilangan bulat | Jumlah perubahan harga (tick) dalam candle pada feed broker, bukan volume kontrak; fitur aktivitas pasar |
| native_spread_price | Numerik (USD) | Spread bar dari MT5 dalam satuan harga; dipakai sebagai spread utama |
| spread_mean | Numerik (USD) | Rata-rata spread tick dalam candle; pengganti bila spread bar tidak tersedia |

## 2.4 Pembagian Periode

Data 1 Januari 2016 – 31 Desember 2025 adalah data pengembangan: seluruh training, kalibrasi, pemilihan threshold, dan pemilihan model hanya memakai periode ini. Data 2026 hanya dibaca oleh perintah `final-test` yang wajib diberi `--confirm`. Pada run walk-forward, data pengembangan berisi 703.425 candle valid dengan 2.642 transisi gap, sedangkan final-test membaca 754.660 candle sampai 30 September 2026. Pembagian di dalam data pengembangan dijelaskan pada bab 9.

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
| Jumlah sesi harian | 2.773 (2026: 193 sesi, data parsial) |
| Spread atau tick volume kosong pada data pengembangan | 0 / 0 |

Sumber: `insights.json` (bab 2.1) dan `data_report` pada `run.json` run walk-forward.

## 3.2 Memahami Target

Target tidak tersedia langsung di data, melainkan dibentuk oleh `make_labels` di `technical/labels.py`. Untuk setiap candle t dihitung barrier b = k · ATR14 + spread. Label BUY diberikan bila high menyentuh close + b lebih dulu dalam 6 candle, SELL bila low menyentuh close − b lebih dulu, dan NO_TRADE bila tidak ada yang tersentuh atau keduanya tersentuh pada candle yang sama. Label diberi status GAP, PENDING, atau INPUT_MISSING (dan tidak dipakai) bila horizon melintasi gap, data masa depan belum ada, atau ATR/spread belum tersedia.

Kode 3.1 Pembentukan barrier dan langkah sentuhan pertama (technical/labels.py, baris 83–100)
%code technical/labels.py 83 100

Spread ditambahkan ke barrier agar gerakan yang mencapai TP masih menghasilkan +k·ATR setelah biaya spread. Desain ini menggantikan label pipeline sebelumnya (return 15 menit dengan ambang spread + 1 bps) yang hanya menghasilkan 7,81% NO_TRADE pada 2026, sehingga model dipaksa menebak arah di hampir setiap candle (SoT EXP-L01). Perintah label-scan menghitung distribusi label 2016–2025 untuk tiga horizon dan empat pengali barrier sebelum model dilatih (Tabel 3.2).

%include hasil_label_scan.md

Pada konfigurasi yang dipakai (triple-barrier, 6 candle, k = 1,0), NO_TRADE mencakup 24,7% label, sedangkan BUY dan SELL masing-masing 37,5% dan 37,8%. Barrier yang lebih sempit (k = 0,5) mengulangi masalah pipeline sebelumnya karena NO_TRADE hanya 8,1%, sedangkan k = 1,5 membuat NO_TRADE menjadi 48,6% sehingga BUY dan SELL jauh lebih jarang. Konfigurasi 6 candle dan k = 1,0 dipertahankan sebagai titik tengah. Kombinasi lain belum diuji dengan walk-forward dan masuk rencana tuning (bab 12.6). Kelas BUY dan SELL hampir seimbang, sedangkan NO_TRADE menjadi kelas minoritas; distribusi per tahun ditampilkan pada Gambar 4.10.

## 3.3 Masalah Kualitas Data yang Ditemukan

Validasi otomatis tidak menemukan bar duplikat, OHLC yang tidak konsisten, atau nilai kosong pada kolom wajib (Tabel 3.1). Pemeriksaan lebih dalam pada saat EDA menemukan beberapa masalah lain yang memengaruhi label, fitur, dan cara membaca hasil.

Tabel 3.3 Masalah Kualitas Data, Dampak, dan Penanganannya
<!-- widths: 20,30,25,25 -->
| Masalah | Bukti | Dampak | Penanganan |
|---|---|---|---|
| Timestamp bukan UTC | Kolom `*_utc` berisi jam server MetaQuotes (EET/EEST): minggu selalu dibuka Senin 01.xx (Tabel 3.4) | Fitur jam dan sesi bergeser 2–3 jam; jam yang dilaporkan bukan UTC | Analisis jam di laporan dikoreksi ke UTC; perbaikan collector dan fitur masuk rencana (Tabel 12.1) |
| Spread tercatat nol | 5,35% candle 2021–2026; lebih dari 20% candle pada Mei–Juli 2021, Mei–Juli 2022, dan Juni–Juli 2026 (Tabel 4.3, Gambar 4.9) | Barrier dan biaya simulasi terlalu kecil pada bulan tersebut | Dicatat dan dibiarkan pada tahap UTS; rencana: ganti dengan spread tick atau median spread jam yang sama |
| Gap waktu | 2.836 transisi (akhir pekan, libur, data hilang) | Return dan label yang melintasi gap tidak bermakna | Ditandai `gap_before`, label berstatus GAP, fitur `bars_since_gap`; tanpa interpolasi |
| Lonjakan spread | Spread maksimum 5,95 USD; spread ÷ ATR mencapai 1.142% (Tabel 4.1) | Barrier sangat lebar pada candle tersebut | Dipertahankan karena merupakan biaya nyata; fitur `spread_z_288` menandainya |
| Ekor distribusi gemuk | Kurtosis log return 48,8; 0,74% return melebihi 4 simpangan baku (Tabel 4.1, 4.3) | Nilai ekstrem dapat mendominasi model yang sensitif skala | Bukan kesalahan data; fitur dinormalisasi ATR dan model pohon robust terhadap outlier |
| Nilai warm-up indikator | Indikator dengan jendela sampai 288 candle belum terdefinisi di awal data | Nilai kosong pada fitur | Dibiarkan NaN; XGBoost menangani nilai hilang secara native (bab 5.2) |
| Label tidak lengkap | Status GAP, PENDING, INPUT_MISSING; pada 2026 terdapat GAP 1.159 dan PENDING 6 | Label tidak dapat dinilai | Dikeluarkan dari training dan evaluasi |
| Tick volume dan spread demo | Tick volume adalah jumlah tick feed broker; spread berasal dari akun demo | Bukan volume pasar global; biaya live dapat berbeda | Dicatat sebagai keterbatasan (bab 1.7) |

## 3.4 Temuan Zona Waktu Timestamp

Kolom waktu di database diberi nama `open_time_utc`, tetapi isinya ternyata jam server broker. Collector membaca field `time` dari MT5 sebagai epoch UTC (`datetime.fromtimestamp(..., tz=UTC)`), padahal MT5 mengirim waktu dalam zona server. Buktinya terlihat dari jadwal pasar emas yang diketahui: pasar dibuka Minggu malam UTC dan ditutup Jumat malam UTC, tetapi data mencatat pembukaan pada Senin 01.xx dan penutupan Jumat 23.xx.

%include hasil_zona_waktu.md

Setelah dikonversi dengan zona EET/EEST (aturan DST Uni Eropa), pembukaan minggu jatuh tepat pada Minggu 22:00 atau 23:00 UTC. Pembukaan sesi pada pukul 00.xx hanya muncul pada Maret dan Oktober–November, yaitu minggu-minggu ketika jadwal DST Amerika Serikat dan Eropa berbeda, sehingga aturan DST server terbukti mengikuti Eropa. Dampaknya bersifat konsisten: semua candle bergeser dengan aturan yang sama, sehingga urutan data, label, dan pembagian fold tetap benar, dan model tetap mempelajari pola jam secara internal. Namun fitur sesi (Asia, London, New York) yang dihitung dengan anggapan input UTC menjadi bergeser 2–3 jam, dan semua jam yang ditampilkan dashboard bukan UTC. Pada laporan ini, analisis per jam (Gambar 4.8 dan Tabel 10.20) sudah dikoreksi ke UTC. Perbaikan di sumbernya (collector dan fitur waktu) menjadi prioritas pertama setelah UTS (Tabel 12.1).

# 4. EDA dan Temuan Awal

EDA dilakukan dua tingkat. Bab 4.1–4.5 merangkum candle menjadi sesi harian untuk melihat tren, volatilitas, dan distribusi jangka panjang 2016–2026. Grafiknya dihasilkan modul `analytics/` dan sama dengan Gambar 2.1–2.7 pada BUSINESS UNDERSTANDING.docx. Karena temuan harian tidak otomatis berlaku untuk horizon 30 menit, bab 4.6 mengulang pemeriksaan pada tingkat candle M5 yang dipakai model. Angka yang hanya dibaca dari grafik ditandai sebagai pembacaan visual.

## 4.1 Tren Harga dan Drawdown

%image rId6

Gambar 4.1 Tren Harga Penutupan, Indeks Harga, dan Drawdown XAU/USD 2016–2026

Harga naik 288,61% antara sesi pertama dan terakhir (indeks 100 menjadi 388,61), dengan kenaikan +27,47% pada 2024 dan +62,49% pada 2025, sedangkan 2026 (Januari–September) berubah −3,62%. Namun 1.284 dari 2.772 return sesi bernilai negatif dan drawdown maksimum mencapai 26,52%. Kenaikan jangka panjang tidak berlangsung stabil, sehingga model perlu kelas NO_TRADE untuk menahan diri.

## 4.2 Return dan Volatilitas

%image rId7

Gambar 4.2 Return dan Volatilitas Rolling XAU/USD 2016–2026

Volatilitas relatif rendah pada 2017–2019, lalu meningkat pada 2020, 2025, dan terutama 2026, dengan puncak sekitar 3,6% (pembacaan visual). Karena volatilitas berubah terhadap waktu, barrier label dan sebagian besar fitur dinormalisasi dengan ATR.

## 4.3 Return Bulanan dan Distribusi Tahunan

%image rId8

Gambar 4.3 Heatmap Return Bulanan XAU/USD 2016–2026

%image rId9

Gambar 4.4 Boxplot Distribusi Return XAU/USD 2016–2026

Pola return bulanan tidak konsisten antartahun, dan distribusi return 2020, 2025, serta 2026 jauh lebih lebar daripada 2017–2018. Karena karakteristik pasar berubah, evaluasi memakai walk-forward per tahun, bukan satu pembagian train/test.

## 4.4 Autokorelasi

%image rId10

Gambar 4.5 ACF Log Return dan Absolute Return XAU/USD 2016–2026

Autokorelasi log return harian mendekati nol, sedangkan autokorelasi absolute return positif pada lag awal (volatility clustering). Arah harga sulit diprediksi dari return masa lalu saja, tetapi besarnya pergerakan lebih dapat diprediksi. Bab 4.6 menunjukkan pola yang sama, bahkan lebih kuat, pada tingkat M5.

## 4.5 Analisis Bivariat dan Multivariat

%image rId11

Gambar 4.6 Hubungan Tick Volume dan Spread Relatif terhadap Absolute Return M5

%image rId12

Gambar 4.7 Heatmap Korelasi Spearman Antarvariabel Harian

Tick volume yang tinggi disertai sebaran perubahan harga yang lebih lebar, sedangkan spread relatif tidak memperlihatkan hubungan konsisten dengan besarnya pergerakan. Korelasi antarvariabel lebih terlihat pada besaran pergerakan dan aktivitas pasar daripada pada arah harga.

## 4.6 EDA pada Tingkat Candle M5

Bagian ini memakai 404.536 candle M5 periode 2021 – September 2026, yaitu seluruh periode evaluasi walk-forward dan uji 2026. Data diambil dari candle yang disimpan setiap run di folder runs/, karena data 2016–2020 hanya tersedia di database. Ringkasan statistik ditampilkan pada Tabel 4.1–4.3 dan visualisasinya pada Gambar 4.8–4.10.

%include hasil_eda_m5.md

**Distribusi return.** Rata-rata log return M5 hampir nol (0,01 bps) dengan simpangan baku 6,63 bps, tetapi distribusinya berekor sangat gemuk: excess kurtosis 48,8, dan 0,74% return melebihi 4 simpangan baku, sekitar 70 kali lebih sering daripada distribusi normal (Tabel 4.1, 4.3). Median absolute return (2,65 bps) jauh di bawah rata-ratanya (4,10 bps), artinya sebagian besar candle bergerak kecil dan sedikit candle bergerak sangat besar. Karena itu barrier memakai ATR, bukan ambang tetap, dan model pohon dipilih karena tidak sensitif terhadap nilai ekstrem.

**Arah vs besarnya gerakan.** Autokorelasi log return M5 praktis nol (−0,002 pada lag 1 dan lag 12), sedangkan autokorelasi absolute return tinggi: 0,353 pada lag 1, 0,258 pada lag 12 (1 jam), dan masih 0,184 pada lag 288 (24 jam). Data dengan jelas menunjukkan bahwa *besarnya* gerakan dapat diprediksi dari masa lalu, sedangkan *arahnya* tidak dapat diprediksi secara linear. Temuan ini mendasari dua keputusan: fitur volatilitas dan aktivitas diharapkan membantu membedakan NO_TRADE dari BUY/SELL, dan sistem harus selektif karena keunggulan arah, bila ada, kecil.

**Aktivitas dan biaya menurut jam.** Setelah timestamp dikoreksi ke UTC (bab 3.4), aktivitas pasar membentuk pola harian yang jelas (Gambar 4.8). Rata-rata absolute return dan tick volume memuncak pada 13–14 UTC (20–21 WIB), yaitu saat pembukaan sesi New York dan rilis data ekonomi AS, dengan kenaikan lebih kecil saat pembukaan London (07–08 UTC). Aktivitas terendah terjadi pada 20–23 UTC, menjelang dan sesudah penutupan harian. Pada jam sepi ini median spread mencapai 8,6–11,5% dari ATR, sedangkan pada 14 UTC hanya 2,3%, sehingga biaya relatif sebuah transaksi 4–5 kali lebih mahal. Tick volume berkorelasi positif dengan absolute return (Spearman 0,411), sedangkan spread berkorelasi negatif lemah (−0,117): spread lebih mencerminkan likuiditas jam tersebut daripada besarnya gerakan. Temuan ini mendukung fitur jam, sesi, volume, dan spread.

%figure gambar/eda_m5_profil_jam.png 12.5

Gambar 4.8 Profil Rata-Rata per Jam UTC: Absolute Return, Tick Volume, dan Spread ÷ ATR14 (2021–2026)

**Perubahan karakteristik antartahun.** Median ATR14 naik dari 1,06 USD (2021) menjadi 5,34 USD (2026), sebagian karena harga naik dari sekitar 1.794 menjadi 4.523 USD dan sebagian karena volatilitas relatif juga naik (rata-rata absolute return 3,50 menjadi 7,07 bps; Tabel 4.2). Median spread berubah tidak searah dengan ATR, sehingga porsi spread terhadap ATR berayun antara 0,8% dan 9,1% per tahun dan antara 0% dan 20% per bulan (Gambar 4.9). Porsi biaya yang tidak konstan ini menjadi alasan spread dimasukkan ke dalam barrier label dan ke dalam fitur.

%figure gambar/eda_m5_atr_spread_bulanan.png 12.5

Gambar 4.9 Median Bulanan ATR14, Spread, dan Spread ÷ ATR14 (2021–2026)

**Distribusi target per tahun.** Proporsi label BUY dan SELL hampir seimbang di setiap tahun (39,6–43,8%), sedangkan NO_TRADE hanya 15,1–20,1% (Gambar 4.10). Angka ini lebih kecil daripada rata-rata 2016–2025 sebesar 24,7% (Tabel 3.2): pada tahun-tahun evaluasi, harga lebih sering menyentuh barrier dalam 30 menit dibandingkan data training awal. Pergeseran distribusi ini dibahas lagi pada Temuan 7 (bab 11.2).

%figure gambar/eda_m5_label_tahun.png 15

Gambar 4.10 Distribusi Label Aktual Triple-Barrier per Tahun (2026 hanya Januari–September)

## 4.7 Ringkasan Temuan EDA dan Dampaknya pada Desain

Tabel 4.4 Hubungan Temuan EDA dengan Keputusan Desain
<!-- widths: 34,40,26 -->
| Temuan EDA | Keputusan desain | Kode |
|---|---|---|
| Tren naik dengan drawdown 26,52% | Kelas NO_TRADE dan keputusan selektif | `technical/decision.py` |
| Arah M5 tidak berautokorelasi (−0,002) | Klasifikasi selektif dengan threshold per arah, bukan prediksi di setiap candle | `technical/decision.py` |
| Volatility clustering (ACF absolute return 0,353) | Kelompok fitur volatility (ATR, realized vol, Bollinger) | `technical/features.py` |
| Ekor gemuk dan ATR naik 5 kali lipat 2021–2026 | Barrier dan fitur dinormalisasi ATR | `technical/labels.py`, `technical/features.py` |
| Pola return berubah antartahun | Walk-forward lima fold per tahun | `technical/splits.py` |
| Aktivitas dan biaya mengikuti jam (puncak 13–14 UTC) | Kelompok fitur time dan sesi; koreksi zona waktu direncanakan | `technical/features.py` |
| Tick volume terkait besarnya gerak (Spearman 0,411) | Kelompok fitur volume (z-score, rasio) | `technical/features.py` |
| Porsi spread terhadap ATR berubah 0–20% | Spread masuk ke barrier, simulasi, dan fitur spread | `technical/labels.py`, `technical/simulation.py` |
| NO_TRADE tahun evaluasi (15–20%) lebih kecil daripada rata-rata training (24,7%) | Threshold dipilih ulang setiap fold; rolling window diuji setelah UTS | `technical/decision.py` |

# 5. Pre-processing

## 5.1 Validasi dan Pembersihan Data

Pre-processing dilakukan oleh `prepare_candles` di `technical/data.py`. Prinsipnya adalah tidak mengarang data: harga tidak pernah diisi atau diinterpolasi, candle yang tidak valid dibuang, dan gap hanya ditandai.

Kode 5.1 Validasi OHLC, durasi candle, duplikat, dan penanda gap (technical/data.py, baris 106–132)
%code technical/data.py 106 132

Tabel 5.1 Langkah Pre-processing, Alasan, dan Hasilnya pada Data 2016–2025
<!-- widths: 18,34,32,16 -->
| Langkah | Aturan | Alasan | Hasil |
|---|---|---|---|
| Validasi OHLC | Harga ada dan positif; high ≥ max(open, close); low ≤ min(open, close) | Bar yang tidak konsisten akan merusak ATR, label, dan simulasi | 0 bar dibuang |
| Validasi waktu | Durasi tepat 5 menit dan open time pada kelipatan 5 menit; bar belum tutup dibuang | Fitur hanya boleh memakai candle yang sudah selesai | 0 bar dibuang |
| Duplikat | Timestamp kembar diselesaikan dengan observasi terakhir | Satu waktu hanya boleh punya satu candle | 0 duplikat |
| Gap | `gap_before` = True bila candle sebelumnya tidak tepat 5 menit sebelumnya | Interpolasi akan menciptakan harga yang tidak pernah ada | 2.642 gap ditandai |
| Pemilihan spread | `native_spread_price` lebih dulu, lalu rata-rata spread tick pada candle yang sama | Spread bar adalah biaya yang dilihat terminal; spread tick sebagai cadangan | 0 spread kosong |
| Waktu keputusan | `prediction_time` = close candle | Keputusan baru dapat dibuat setelah candle selesai | Semua baris |
| Status label | Label melintasi gap → GAP; horizon belum ada → PENDING; ATR/spread kosong → INPUT_MISSING | Label tanpa horizon lengkap tidak dapat dinilai | Dikeluarkan dari training dan evaluasi |

Sumber kolom hasil: `data_report` pada `run.json` setiap run walk-forward (Tabel 10.2).

## 5.2 Nilai Hilang, Data Bermasalah, dan Skala

- **Nilai hilang pada fitur.** Nilai warm-up indikator (misalnya EMA 288 candle pada awal data) dan pembagian dengan nol dibiarkan NaN. XGBoost menangani nilai hilang secara native dengan mempelajari arah default pada setiap split, sehingga imputasi tidak diperlukan dan tidak menambah informasi palsu.
- **Regresi logistik.** Model linear tidak dapat menerima NaN, sehingga memakai imputasi median dan StandardScaler. Keduanya di-fit hanya pada blok TRAIN setiap fold di dalam Pipeline scikit-learn.
- **Skala.** Model berbasis pohon tidak memerlukan scaling karena split hanya bergantung pada urutan nilai. Skala antartahun ditangani di tingkat fitur dengan normalisasi ATR (bab 6.3). Seluruh fitur disimpan sebagai float32.
- **Outlier.** Return dan spread ekstrem tidak dipotong karena merupakan kejadian pasar yang nyata (Tabel 3.3). Pengaruhnya dibatasi oleh normalisasi ATR dan oleh model pohon.
- **Masalah yang belum ditangani.** Spread nol dan zona waktu server (bab 3.3–3.4) belum dikoreksi di pipeline. Keduanya dicatat sebagai keterbatasan hasil UTS dan menjadi prioritas perbaikan (Tabel 12.1).

## 5.3 Pencegahan Kebocoran Data

Kebocoran data (data leakage) terjadi bila model memakai informasi yang belum tersedia pada saat keputusan. Pada data deret waktu dengan label berhorizon, kebocoran dapat terjadi lewat fitur, label, pembagian data, dan proses fitting. Setiap jalur ditutup sebagai berikut.

Tabel 5.2 Jalur Kebocoran dan Pencegahannya
<!-- widths: 26,46,28 -->
| Jalur kebocoran | Pencegahan | Bukti |
|---|---|---|
| Fitur memakai candle masa depan | Hanya `rolling`, `ewm`, dan `shift` positif; keputusan pada close candle | Tes `test_features_do_not_change_when_future_candles_change` |
| Label masuk ke fitur | Label disimpan terpisah dan tidak pernah masuk matriks fitur | `technical/run.py` (`build_dataset`) |
| Label di akhir blok melihat blok berikutnya | Purging dan embargo 6 candle pada batas blok | `assert_no_overlap` (bab 9.3) |
| Pembagian acak | Pembagian kronologis per tahun; tidak ada shuffle | `technical/splits.py` |
| Fitting memakai data evaluasi | Imputasi, scaling, kalibrasi, threshold, bobot ensemble, dan feature selection di-fit ulang setiap fold tanpa blok EVALUATION | `technical/run.py` (`run_fold`) |
| Data uji ikut dipakai memilih model | Perintah pengembangan tidak memuat data setelah 31 Desember 2025; 2026 hanya lewat `final-test --confirm` | `technical/run.py` |
| Fitur yang hampir sama dengan label | Audit Spearman fitur terhadap arah label > 0,95 di setiap fold | Tabel 10.13 (0 fitur terdeteksi) |

# 6. Feature Engineering Awal

## 6.1 Prinsip

Fungsi `build_features` di `technical/features.py` menghasilkan 41 fitur dalam tujuh kelompok dari kolom OHLC, tick volume, spread, dan waktu. Ada tiga prinsip. Pertama, fitur bersifat kausal: hanya memakai candle t dan sebelumnya. Kedua, fitur harga dibagi ATR14 atau harga agar nilai tahun 2016 (emas sekitar 1.100 USD) dan 2026 (sekitar 4.500 USD) sebanding. Ketiga, nilai yang belum bisa dihitung tetap NaN.

## 6.2 Kelompok Fitur, Alasan, dan Dampak yang Diharapkan

Tabel 6.1 Kelompok Fitur dan Dampak yang Diharapkan
<!-- widths: 14,8,40,38 -->
| Kelompok | Jumlah | Fitur | Alasan dan dampak yang diharapkan |
|---|---|---|---|
| price_action | 11 | Log return 1/3/6/12 candle, return 6 dan 24 candle ÷ ATR, badan dan rentang candle ÷ ATR, rasio sumbu atas dan bawah, posisi close dalam rentang | Menangkap momentum jangka sangat pendek dan bentuk candle (penolakan harga); diharapkan membantu arah, walau ACF return hampir nol |
| trend | 7 | Jarak close ke EMA 12/48/288 ÷ ATR, selisih EMA 12–48 ÷ ATR, kemiringan EMA 48, ADX 14, selisih +DI dan −DI | Posisi harga terhadap tren 1 jam, 4 jam, dan 24 jam; diharapkan membedakan kondisi trending dan sideways |
| momentum | 5 | RSI 14, perubahan RSI 3 candle, histogram MACD (12, 26, 9) ÷ ATR, stochastic %K 14, posisi Donchian 48 | Kondisi jenuh beli/jual dan percepatan harga; diharapkan memberi sinyal pembalikan jangka pendek |
| volatility | 6 | ATR ÷ close, ATR ÷ rata-rata ATR 288 candle, volatilitas realisasi 12 dan 48 candle, lebar dan posisi Bollinger Bands 20 | Volatility clustering terbukti kuat (ACF 0,353); diharapkan memprediksi apakah barrier tersentuh dalam 30 menit |
| volume | 2 | Z-score log tick volume 48 candle, rasio rata-rata tick volume 12 ÷ 288 candle | Tick volume berkorelasi dengan besarnya gerak (Spearman 0,411); diharapkan mendeteksi lonjakan aktivitas |
| spread | 2 | Spread ÷ ATR, z-score spread 288 candle | Porsi biaya terhadap barrier berubah 0–20%; diharapkan menurunkan keyakinan saat biaya relatif tinggi |
| time | 8 | Jam (sin dan cos), hari, sesi Asia, London, dan New York, overlap London–New York, jumlah candle sejak gap | Aktivitas dan biaya mengikuti jam (Gambar 4.8); diharapkan memisahkan jam ramai dan jam sepi |

Panjang jendela dipilih dalam satuan candle M5 agar bermakna sebagai waktu: 12 candle = 1 jam, 48 candle = 4 jam, dan 288 candle = 24 jam. Periode indikator klasik (RSI, ATR, dan ADX 14; MACD 12, 26, 9; Bollinger 20, 2) adalah konvensi penemunya, bukan nilai optimal yang sudah dibuktikan pada data ini (SoT tabel C). Relevansinya diuji dengan feature selection di dalam fold (bab 8.6) dan ablation.

## 6.3 Contoh Transformasi: Normalisasi ATR

Contoh utama feature engineering pada proyek ini adalah normalisasi dengan ATR14. Perubahan harga mentah dalam USD tidak sebanding antartahun: gerakan 3 USD adalah gerakan besar pada 2021 (ATR median 1,06 USD) tetapi gerakan biasa pada 2026 (ATR median 5,34 USD). Fitur `return_6_atr` membagi perubahan close 6 candle dengan ATR14 pada candle yang sama:

$$ x_t = (C_t − C_s) / ATR14_t,   s = t − 6 $$ (6.1)

dengan x = fitur `return_6_atr`, C = harga close, dan t = candle keputusan.

Kode 6.1 Fitur price action dengan normalisasi ATR (technical/features.py, baris 110–131)
%code technical/features.py 110 131

Dampak yang diharapkan adalah distribusi fitur yang stabil antartahun, sehingga pola yang dipelajari dari data 2016–2020 tetap berlaku pada 2025–2026. Dampak ini terukur pada data:

%include hasil_fe_atr.md

Simpangan baku perubahan harga mentah naik 5,6 kali dari 2021 ke 2026, sedangkan versi yang dibagi ATR hampir konstan (1,52–1,57, rasio 1,03). Tanpa normalisasi, sebuah split pohon seperti "perubahan > 3 USD" yang dipelajari pada data lama akan berarti hal yang sangat berbeda pada 2026. Prinsip yang sama dipakai pada fitur trend, momentum, dan spread (Tabel 6.1).

## 6.4 Bukti Awal Relevansi Fitur

Feature selection di dalam fold (bab 8.6, Tabel 10.13–10.14) mempertahankan 28–33 dari 41 fitur di setiap fold. Fitur dengan permutation importance tertinggi adalah `hour_cos`, `tick_volume_z_48`, `atr_14_vs_daily_mean`, dan `spread_atr`, yaitu fitur jam, aktivitas, rezim volatilitas, dan biaya. Urutan ini sejalan dengan temuan EDA: informasi yang paling berguna adalah kapan dan seberapa besar pasar bergerak, bukan arah return sebelumnya. Sebaliknya, `candle_body_atr` tidak pernah terpilih dan `log_return_3`, `return_6_atr`, `ema_48_slope_atr`, serta `bollinger_position_20` hanya terpilih pada satu fold.

## 6.5 Daftar Lengkap Fitur

%include tabel_fitur.md

# 7. Baseline

## 7.1 Mengapa Baseline Diperlukan

Tanpa pembanding sederhana, angka seperti precision 45% tidak dapat ditafsirkan. Baseline menjawab apakah model yang rumit benar-benar menambah nilai. Ketiga baseline dipilih untuk menjawab tiga pertanyaan berbeda, dan semuanya diuji pada fold, blok, dan aturan penilaian yang sama dengan model utama (`technical/models.py`).

Tabel 7.1 Model Baseline dan Alasan Pemilihannya
<!-- widths: 18,42,40 -->
| Baseline | Cara kerja | Alasan dipilih |
|---|---|---|
| always_no_trade | Selalu NO_TRADE (coverage 0) | Batas bawah: apakah model lebih berguna daripada tidak pernah trading? Juga menunjukkan bahwa accuracy saja menyesatkan |
| persistence | BUY bila close naik lebih dari barrier selama 6 candle terakhir, SELL bila turun lebih dari barrier | Aturan momentum naif yang sering dipakai trader; menjawab apakah model lebih baik daripada "ikuti arah terakhir" |
| logistic | Regresi logistik multinomial, imputasi median, StandardScaler, C = 1,0, max_iter = 500 | Model linear dengan fitur yang sama; menjawab apakah model non-linear (pohon) benar-benar diperlukan |

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

## 7.3 Hasil Baseline

Hasil baseline pipeline baru ditampilkan bersama kandidat model pada Tabel 10.3. always_no_trade memperoleh macro F1 0,1025 tanpa satu pun sinyal. Persistence memperoleh macro F1 tertinggi (0,3043) karena mengeluarkan sinyal pada 46,33% candle, tetapi lift-nya hanya 0,98–1,08 (Tabel 10.8), setara tebakan acak. Regresi logistik memperoleh macro F1 0,1244 dengan coverage 3,05%. Dengan kata lain, baseline sederhana tidak memiliki keunggulan precision, sehingga keunggulan XGBoost (bab 11.2) memang berasal dari model.

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

### Cara Membandingkan dengan Library pada Data XAU/USD

Perbandingan pada data riil mengikuti perintah `compare-scratch`. Hyperparameter library disamakan dengan scratch: 150 pohon, kedalaman 4, learning rate 0,05, min_child_weight 20, λ = 5, dan max_bin 64. Subsampling baris dan kolom dimatikan, dan gamma library diatur 2 × gamma scratch. Keduanya kemudian dilatih, dikalibrasi, dan diberi threshold pada fold 2025 yang sama. Dengan pengaturan ini, library pada perbandingan tersebut bukan model xgboost walk-forward, yang memakai 300 pohon dan subsample 0,8.

Bagian yang tetap berbeda hanya kandidat split. Kode scratch memakai kuantil biasa (`np.quantile`) pada data TRAINING dengan presisi float64. XGBoost hist membuat batas bin dengan quantile sketch sekali sebelum pohon pertama dibangun dan menyimpan fitur sebagai float32. Paper Chen dan Guestrin (2016) memberi bobot Hessian pada sketch, tetapi pada metode hist XGBoost 3.4.1 batas bin tidak berubah selama training, sehingga dapat diambil dan dipakai ulang. Subsampling, paralelisasi, dan optimasi sistem tidak diimplementasikan. Karena itu uji memakai dua varian scratch:

1) **Scratch dengan kuantil sendiri** mengukur seberapa mirip hasil akhirnya. Target awal kelompok adalah kelas argmax sama pada sekitar 95% candle, sejalan dengan EXP-S03 (97,86%).
2) **Scratch dengan batas bin library** mengambil batas bin dari XGBoost (`QuantileDMatrix.get_quantile_cut()`) dan membulatkan fitur ke float32, tanpa mengubah rumus. Bila hasilnya identik dengan library, terbukti bahwa gradien, Hessian, bobot daun, gain, base score, dan early stopping dihitung sama. Semua selisih pada varian 1 kemudian dapat dipastikan berasal dari batas bin.

Hasilnya dibahas pada bab 10.5.

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

### Ablation Kelompok Fitur

Ablation adalah feature selection tipe wrapper pada tingkat kelompok fitur (tahap 6 pada Tabel 8.5). Perintah `ablation` menjalankan walk-forward XGBoost lengkap delapan kali: satu kali dengan ketujuh kelompok (price_action, trend, momentum, volatility, volume, spread, time) dan tujuh kali dengan satu kelompok dihapus. Kontribusi kelompok g diukur dari selisih directional F1 rata-rata lima fold:

$$ Δ_g = DirF1(tanpa g) − DirF1(semua kelompok) $$ (8.10)

Δ_g negatif berarti kinerja turun ketika kelompok g dihapus, sehingga kelompok itu membawa informasi yang tidak tergantikan oleh kelompok lain. Δ_g mendekati nol atau positif berarti kelompok itu tidak membantu pada konfigurasi ini dan menjadi kandidat untuk dibuang atau diganti. Selisih yang lebih kecil daripada simpangan baku antarfold tidak dianggap berarti.

Ablation dan xgboost_selected menjawab pertanyaan yang berbeda. xgboost_selected memilih fitur satu per satu secara otomatis di dalam setiap fold, sedangkan ablation menjawab pertanyaan rancangan: kelompok indikator mana yang layak dipertahankan di pipeline. Ablation hanya memakai lima fold validasi 2021–2025 dan tidak menyentuh data 2026. Hasilnya menjadi dasar bab 12.3, misalnya membuang kelompok yang tidak berguna atau menggantinya dengan fitur multi-timeframe.

### Output yang Dihasilkan

Untuk setiap fold, metrics.json mencatat status setiap fitur (terpilih, atau alasan dibuang), mutual information, gain, permutation importance, dan dugaan kebocoran. Stabilitas seleksi, yaitu berapa fold sebuah fitur terpilih, ditulis ke `feature_selection_xgboost_selected.csv`, dan terminal mencetak jumlah fitur terpilih per fold. Aturan keputusannya ditetapkan sebelum eksperimen: subset fitur hanya dipakai bila kinerja `xgboost_selected` tidak lebih buruk daripada `xgboost` dengan 41 fitur. Hasilnya pada data XAU/USD dibahas pada bab 10.6. Perintah `ablation` menulis `ablation.json` berisi ringkasan walk-forward setiap varian; perintah ini belum dijalankan pada tahap UTS.

# 9. Strategi Evaluasi

Prosedur evaluasi diturunkan dari karakteristik data dan tujuan proyek, bukan dari kebiasaan pembagian train/test acak. Tabel 9.1 merangkum alasannya.

Tabel 9.1 Karakteristik Data, Konsekuensinya, dan Prosedur Evaluasi yang Dipilih
<!-- widths: 30,34,36 -->
| Karakteristik data atau tujuan | Konsekuensi | Prosedur |
|---|---|---|
| Deret waktu; pola berubah antartahun (bab 4.3, 4.6) | Pembagian acak membocorkan masa depan dan satu periode uji tidak mewakili | Walk-forward kronologis lima fold, satu tahun per fold |
| Label melihat 30 menit ke depan | Label di akhir blok tumpang tindih dengan blok berikutnya | Purging dan embargo 6 candle (bab 9.3) |
| Model harus selektif dan probabilitas dipakai agen risiko | Probabilitas mentah belum tentu terkalibrasi; threshold harus dipilih tanpa melihat data uji | Blok CALIBRATION untuk temperature scaling, blok THRESHOLD terpisah untuk threshold (bab 9.4) |
| NO_TRADE dapat dominan dan prevalensi berubah antartahun | Accuracy dan macro F1 menyesatkan | Precision, lift terhadap prevalensi, dan coverage sebagai metrik utama (bab 9.5) |
| Tujuan bisnis adalah keputusan setelah biaya | Klasifikasi benar belum tentu menguntungkan | Simulasi transaksi setelah spread dipisahkan dari metrik klasifikasi (bab 9.6) |
| Hasil 2026 pernah dilihat pipeline sebelumnya | 2026 tidak dapat menjadi blind test | 2026 hanya dipakai sekali setelah desain dikunci, dan statusnya ditulis sebagai out-of-sample retrospektif |

## 9.1 Pembagian Dataset

Setiap tahun 2021–2025 menjadi satu fold evaluasi. Data training selalu dimulai pada awal 2016 (expanding window), dan setiap fold terdiri atas empat blok berurutan.

Tabel 9.2 Blok Waktu Setiap Fold
<!-- widths: 18,22,18,18,24 -->
| Fold | TRAIN | CALIBRATION | THRESHOLD | EVALUATION |
|---|---|---|---|---|
| validation_2021 | 2016 – 30 Jun 2020 | Jul – Sep 2020 | Okt – Des 2020 | 2021 |
| validation_2022 | 2016 – 30 Jun 2021 | Jul – Sep 2021 | Okt – Des 2021 | 2022 |
| validation_2023 | 2016 – 30 Jun 2022 | Jul – Sep 2022 | Okt – Des 2022 | 2023 |
| validation_2024 | 2016 – 30 Jun 2023 | Jul – Sep 2023 | Okt – Des 2023 | 2024 |
| validation_2025 | 2016 – 30 Jun 2024 | Jul – Sep 2024 | Okt – Des 2024 | 2025 |
| final_test | 2016 – 30 Jun 2025 | Jul – Sep 2025 | Okt – Des 2025 | 2026 (sekali, --confirm) |

%include hasil_blok_fold.md

TRAIN dipakai untuk melatih model. CALIBRATION dipakai untuk early stopping, temperature scaling, bobot ensemble, dan permutation importance. THRESHOLD dipakai untuk memilih threshold BUY/SELL, dan EVALUATION hanya untuk mengukur kinerja. CALIBRATION dan THRESHOLD masing-masing tiga bulan: cukup untuk sekitar 17.000 candle, tetapi cukup dekat dengan periode evaluasi agar mencerminkan kondisi pasar terbaru.

## 9.2 Metode Validasi: Walk-Forward Expanding Window

Walk-forward mensimulasikan pemakaian nyata: model dilatih dengan semua data yang tersedia sampai suatu waktu, lalu dinilai pada tahun berikutnya yang belum pernah dilihat. Expanding window dipilih agar model memakai seluruh riwayat 2016–2020, termasuk periode volatilitas rendah dan tinggi. Lima fold memberi lima ukuran kinerja yang independen secara waktu, sehingga hasil dapat dilaporkan sebagai rata-rata ± simpangan baku dan tahun yang lemah (misalnya 2023) terlihat jelas. Rolling window, yang memberi bobot lebih pada data terbaru, diuji setelah UTS (Tabel 12.1).

## 9.3 Purging dan Embargo

Label candle t melihat harga sampai t + 30 menit, sehingga label di akhir suatu blok bisa "melihat" blok berikutnya. Baris hanya dipakai bila `label_end_time` ditambah embargo 6 candle masih di dalam bloknya.

Kode 9.1 Pembagian blok dengan purging dan embargo (technical/splits.py, baris 95–113)
%code technical/splits.py 95 113

## 9.4 Kalibrasi dan Pemilihan Threshold

Probabilitas mentah dikalibrasi dengan temperature scaling satu parameter, p_T = softmax(log p / T), dengan T dipilih dari 121 nilai log-spaced 0,25–4 yang meminimalkan log loss pada CALIBRATION. Threshold BUY dan SELL dipilih terpisah pada blok THRESHOLD dari grid 0,34–0,90 dengan langkah 0,02. Sebuah threshold lolos bila menghasilkan minimal 30 sinyal dan precision minimal 1,2 × prevalensi kelas, lalu dipilih yang F1-nya tertinggi. Bila tidak ada yang lolos, arah tersebut dinonaktifkan.

Kode 9.2 Syarat threshold yang lolos (technical/decision.py, baris 104–131)
%code technical/decision.py 104 131

## 9.5 Metrik Evaluasi

Tabel 9.4 Metrik dan Alasannya
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

Metrik utama adalah precision, lift, dan coverage karena tujuan sistem adalah sinyal yang jarang tetapi benar. Macro F1 dan accuracy tetap dilaporkan sebagai pembanding, tetapi dapat menyesatkan: persistence memperoleh macro F1 tertinggi hanya karena mengeluarkan sinyal di hampir separuh candle (Temuan 2). Hasil antarfold dirangkum dengan rata-rata dan simpangan baku (`summarize_folds`). Konfigurasi dipilih hanya dari hasil walk-forward 2021–2025.

## 9.6 Tiga Ukuran Backtest

Backtest memakai prediksi out-of-sample setiap fold, sehingga setiap prediksi dibuat oleh model yang memang tersedia pada waktu itu. Tiga ukuran dipisahkan dengan sengaja: (1) akurasi klasifikasi terhadap label triple-barrier, (2) ketepatan arah, yaitu tanda close t+6 − close t tanpa ambang, dan (3) simulasi transaksi setelah spread dengan aturan pada Tabel 1.4. Ketiganya ditampilkan di dashboard `/backtest`, termasuk waktu buka dan tutup setiap transaksi, alasan tutup, serta persentase spread terhadap harga dan terhadap jarak TP.

## 9.7 Perintah Eksperimen

Tabel 9.5 Perintah dan Hasilnya
<!-- widths: 42,58 -->
| Perintah | Hasil di runs/<waktu>_<perintah>_<hash>/ |
|---|---|
| `technical.run label-scan` | label_scan.csv: distribusi label untuk horizon 3/6/12 dan k 0,5–2,0 |
| `technical.run walk-forward --models …` | metrics.json, run.json, backtest/<model>/validation_<tahun>/predictions.csv |
| `technical.run ablation` | ablation.json: walk-forward tanpa satu kelompok fitur |
| `technical.run compare-scratch --year 2025` | comparison.json: library vs scratch pada satu fold |
| `laporan_uts/uji_scratch_bin.py --year 2025` | uji_scratch_bin_2025.json: langkah compare-scratch ditambah varian scratch dengan batas bin library |
| `laporan_uts/eda_m5.py` | hasil_eda_m5.md, hasil_zona_waktu.md, hasil_fe_atr.md, dan Gambar 4.8–4.10 |
| `technical.run final-test --model <pemenang> --confirm` | metrics.json, test_period.json, latest_signal.json, live_bundle.joblib |

# 10. Hasil Eksperimen Awal

## 10.1 Bukti yang Sudah Tersedia

Tabel 10.1 Bukti Bahwa Pipeline Berjalan
<!-- widths: 28,44,28 -->
| Bukti | Hasil | Cara mengulang |
|---|---|---|
| Unit test seluruh proyek | 128 tes lolos, 1 dilewati (7 November 2026) | `uv run pytest` |
| Kausalitas fitur | Fitur tidak berubah bila candle masa depan diubah | `test_features_do_not_change_when_future_candles_change` |
| Kesesuaian XGBoost from scratch | Identik dengan library pada kandidat split yang sama, baik pada data uji (Tabel 8.2) maupun pada data XAU/USD 2025 (Tabel 10.21–10.22) | `tests/test_xgboost_scratch.py`, `laporan_uts/uji_scratch_bin.py` |
| Simulasi transaksi | TP = +k·ATR, SL = −(k·ATR + 2 × spread), waktu buka/tutup per transaksi | `tests/test_backtest_simulation.py` |
| Dashboard backtest | Ringkasan, riwayat transaksi, dan laporan spread terbaca dari folder runs | `tests/test_dashboard_app.py` |
| Kualitas data dan EDA | 754.660 bar valid; Gambar 4.1–4.10, Tabel 3.3–3.4 dan 4.1–4.3 | `reports/analytics/`, `laporan_uts/eda_m5.py` |
| Reproduksibilitas | Ringkasan metrik XGBoost identik pada tiga run walk-forward (3 dan 7 November 2026) | `runs/*_walk-forward_*/metrics.json` |
| Repositori | Kode, run, dan skrip laporan tersimpan di GitHub | https://github.com/Vito28/xauusd-paper (bab 12.1) |

## 10.2 Uji Kewajaran pada Data Sintetis

Sebelum data riil dibaca, pipeline walk-forward lengkap dijalankan pada 130.000 candle random walk sintetis (generator `make_candles` pada `tests/test_technical_pipeline.py`, seed 7, satu fold evaluasi 2025). Data random walk tidak memiliki pola yang dapat diprediksi, sehingga sistem yang benar seharusnya menahan diri. Hasilnya, threshold BUY dan SELL pada xgboost, logistic, random_forest, ensemble, dan xgboost_selected dinonaktifkan karena tidak ada threshold dengan precision ≥ 1,2 × prevalensi, sehingga coverage-nya 0%. Persistence tetap mengeluarkan sinyal pada 45,86% candle dengan precision BUY 42,63% terhadap prevalensi BUY 43,01% (lift ≈ 0,99), yaitu setara tebakan acak. Uji ini menunjukkan decision layer tidak memaksakan sinyal ketika model tidak memiliki keunggulan. Angka pada bagian ini **bukan** hasil XAU/USD.

## 10.3 Hasil Walk-Forward pada Data XAU/USD

Walk-forward dijalankan pada 703.425 candle 2016–2025 dalam tiga run (Tabel 10.2). Run pertama memuat baseline, regresi logistik, dan XGBoost; run kedua memuat XGBoost, xgboost_balanced, random_forest, dan ensemble; run ketiga (8 November 2026) memuat xgboost_selected. Config hash run ketiga berbeda hanya karena bagian `[feature_selection]` sudah tercantum di konfigurasinya; seluruh parameter data, label, split, XGBoost, kalibrasi, dan threshold identik, sehingga hasilnya dapat dibandingkan langsung. Prediksi per candle setiap fold disimpan di folder backtest/ dan dipakai untuk simulasi transaksi (Tabel 10.15–10.17) serta dashboard backtest.

%include hasil_walk_forward.md

## 10.4 Hasil Final Test 2026

Final-test dijalankan satu kali pada 3 November 2026 untuk XGBoost, setelah walk-forward baseline dan XGBoost selesai. Model dilatih dengan data sampai Juni 2025, dikalibrasi pada Juli–September 2025, memilih threshold pada Oktober–Desember 2025, lalu diuji pada 2026. Statusnya out-of-sample retrospektif (bab 1.7). Tabel 10.20 membandingkan seberapa sering model mengeluarkan BUY dan SELL pada 2026 dengan tahun-tahun walk-forward.

%include hasil_final_test.md

## 10.5 Perbandingan XGBoost Library dan From Scratch

Uji dijalankan pada 8 November 2026 dengan skrip `laporan_uts/uji_scratch_bin.py`. Skrip ini mengikuti langkah `compare-scratch` (bab 8.3) dan menambah varian scratch dengan batas bin library. Database PostgreSQL tidak tersedia di mesin tempat uji dijalankan, sehingga data diambil dari candle 2021–2025 yang tersimpan di folder runs/ (`backtest/candles`). Akibatnya, TRAIN dimulai pada 2021, bukan 2016. Perintah `compare-scratch --year 2025` dengan data penuh 2016–2024 tetap perlu dijalankan sebagai konfirmasi; hasilnya ditabelkan otomatis setelah run-nya tersedia.

%include hasil_uji_scratch.md

Dengan kuantil sendiri, kelas argmax sama pada 92,79% candle, sedikit di bawah target 95%. Namun probabilitasnya hampir sama: selisih rata-rata 0,0037, korelasi per kelas ≥ 0,995, log loss 0,9958 dibanding 0,9957, dan accuracy argmax 41,62% dibanding 41,83%. Perbedaan argmax terpusat pada candle yang ragu. Pada 21,6% candle EVALUATION, selisih dua probabilitas teratas di bawah 0,01, sehingga perbedaan sekecil apa pun dapat membalik kelas teratas. Pada candle dengan selisih ≥ 0,01, kesepakatan naik menjadi 98,25%, dan pada selisih ≥ 0,02 menjadi 99,44%.

Dengan batas bin library, kelas argmax dan keputusan akhir sama pada 100% candle. Threshold-nya juga sama (BUY nonaktif, SELL 0,52), begitu pula 146 sinyal SELL dengan precision 41,10%, dan selisih probabilitas rata-rata hanya 2,4 × 10⁻⁶. Artinya, implementasi from scratch menghitung hal yang sama dengan library, dan seluruh selisih pada varian kuantil sendiri berasal dari batas bin, bukan dari rumus.

Selisih probabilitas yang kecil berdampak besar pada threshold. Scratch dengan kuantil sendiri mengaktifkan BUY (0,52) dan menurunkan threshold SELL ke 0,50, sehingga mengeluarkan 202 BUY dan 897 SELL, dibanding 146 SELL pada library. Keputusan per candle tetap sama pada 98,6% candle karena sebagian besar candle tetap NO_TRADE. Dari sisi waktu, 150 ronde scratch membutuhkan 127 detik, sedangkan library hanya 2,6 detik (sekitar 49 kali lebih cepat).

## 10.6 Feature Selection di Dalam Fold

Walk-forward `xgboost_selected` dijalankan pada 8 November 2026 (run `20261108T025726Z_walk-forward_2a8bb1db55`). Hasil seleksi per fold ada pada Tabel 10.13 dan 10.14, ringkasan kinerjanya pada Tabel 10.3, simulasinya pada Tabel 10.15–10.16, dan perbandingannya dengan xgboost pada Tabel 10.23.

%include hasil_seleksi.md

Seleksi mempertahankan 28–33 dari 41 fitur. Tahap filter membuang 4 fitur di setiap fold (37 lolos), tidak ada fitur yang terdeteksi bocor, dan korelasi peringkat gain dengan permutation importance 0,74–0,88, sehingga kedua ukuran kepentingan cukup sejalan (Tabel 10.13). Lima belas fitur terpilih di kelima fold, dipimpin `hour_cos`, `tick_volume_z_48`, `atr_14_vs_daily_mean`, dan `spread_atr` (Tabel 10.14).

Semua rata-rata xgboost_selected sedikit lebih rendah daripada xgboost, tetapi selisihnya jauh lebih kecil daripada simpangan baku antarfold. Simpangan baku directional F1 justru lebih kecil (0,0569 dibanding 0,0709). Sesuai aturan yang ditetapkan sebelumnya, subset fitur belum dipakai karena kinerjanya tidak lebih baik, dan model utama tetap memakai 41 fitur. Hasil ini tetap bernilai: sekitar seperempat fitur dapat dibuang tanpa penurunan yang berarti, dan daftar fitur yang tidak stabil menjadi bahan ablation setelah UTS. Ablation kelompok fitur (bab 8.6) belum dijalankan.

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
- Timestamp tersimpan adalah jam server broker (bab 3.4), sehingga fitur sesi bergeser 2–3 jam dan jam di dashboard bukan UTC.
- Spread tercatat nol pada 5,35% candle (bab 3.3), sehingga biaya simulasi pada bulan-bulan tersebut terlalu kecil.
- EDA M5 dan uji from scratch memakai candle 2021–2026 yang tersimpan di folder runs/, bukan seluruh data 2016–2026 di database.

# 12. Dokumentasi dan Rencana Pengembangan Setelah UTS

Bab ini menjelaskan di mana seluruh kode, data hasil, dan laporan dapat diperiksa (bab 12.1), lalu rencana pengembangan setelah UTS. Prioritas pengembangan ditentukan oleh temuan pada bab 11.2 (Tabel 12.1); subbab berikutnya merinci setiap area pengembangan.

Tabel 12.1 Prioritas Pengembangan Berdasarkan Hasil Eksperimen Awal
<!-- widths: 40,60 -->
| Temuan (bab 11.2) | Tindakan setelah UTS |
|---|---|
| Timestamp bukan UTC dan spread nol (Temuan 14) | Konversi jam server EET/EEST ke UTC di collector dan `technical/data.py`; ganti spread nol dengan spread tick atau median spread jam yang sama; ulangi walk-forward |
| Precision belum mencapai titik impas sekitar 53% setelah spread (Temuan 8) | Uji k = 1,5 dan horizon 12 candle agar spread menjadi porsi barrier yang lebih kecil, threshold berbasis nilai ekonomi, dan model dua tahap (meta-labeling) |
| Batas 300 pohon tersentuh di semua fold (Temuan 9) | Naikkan n_estimators dengan early stopping, atau ubah learning rate, lalu bandingkan rata-rata walk-forward |
| Threshold tidak selalu cocok untuk tahun berikutnya; SELL 2023 dan BUY 2026 nonaktif; selisih probabilitas 0,004 sudah mengubah threshold (Temuan 6, 10, dan 11) | Analisis stabilitas threshold antarfold dan uji blok THRESHOLD yang lebih panjang |
| NO_TRADE pada TRAIN 26–33%, pada EVALUATION 16–20% (Temuan 7) | Bandingkan expanding window dengan rolling window atau bobot sampel yang lebih besar untuk data terbaru |
| Ensemble belum lebih baik (Temuan 4) | Ganti anggota yang lemah (regresi logistik) dengan model yang lebih beragam, misalnya LSTM/CNN-LSTM, sebelum mengulang ensemble |
| Feature selection tidak memperbaiki kinerja (Temuan 13); ablation belum dijalankan; compare-scratch baru memakai data 2021–2025 | Jalankan ablation kelompok fitur dan compare-scratch --year 2025 dengan data penuh; buang fitur yang tidak stabil bila ablation mendukung |

## 12.1 Dokumentasi Proyek

Seluruh kode, konfigurasi, hasil run, dan sumber laporan ini disimpan di repositori GitHub **https://github.com/Vito28/xauusd-paper**.

Tabel 12.2 Isi Repositori
<!-- widths: 30,70 -->
| Lokasi | Isi |
|---|---|
| Branch `main` | Kode pipeline, konfigurasi, unit test, dan folder `runs/` berisi seluruh run yang dipakai laporan ini |
| Branch `documents` | Laporan ini, BUSINESS UNDERSTANDING.docx, dan skrip pembuat laporan di folder `laporan_uts/` |
| Branch `feat/dashboard-buka-tutup-spread` | Dashboard backtest dengan waktu buka/tutup posisi dan laporan spread |
| `technical/` | Data, fitur, label, split, model, kalibrasi, threshold, evaluasi, simulasi, XGBoost from scratch, dan CLI `run.py` |
| `configs/technical.toml` | Semua nilai konstanta eksperimen beserta alasannya (bab 13) |
| `market/`, `database/` | Collector MT5 dan skema PostgreSQL |
| `analytics/`, `reports/` | EDA harian (Gambar 4.1–4.7) |
| `dashboard/` | Aplikasi FastAPI: dashboard live (`/`) dan backtest (`/backtest`) |
| `runs/<waktu>_<perintah>_<hash>/` | run.json (config, hash, commit, versi paket), metrics.json, dan prediksi per candle |
| `SOURCE_OF_TRUTH.md`, `docs/METHODOLOGY.md` | Keputusan desain (kode SoT) dan metodologi |

Setiap angka di bab 3, 4, 6, 10, dan 11 dibuat ulang otomatis dari folder runs/ oleh skrip di `laporan_uts/` (`ringkas_runs.py`, `eda_m5.py`, `uji_scratch_bin.py`), sehingga tidak ada angka hasil yang diketik manual. Perintah untuk mengulang eksperimen:

```
uv sync --dev
uv run python -m technical.run label-scan
uv run python -m technical.run walk-forward --models always_no_trade,persistence,logistic,xgboost
uv run python -m technical.run walk-forward --models xgboost,xgboost_balanced,random_forest,ensemble
uv run python -m technical.run walk-forward --models xgboost,xgboost_selected
uv run python -m technical.run final-test --model xgboost --confirm
uv run pytest
uv run uvicorn dashboard.app:app --host 127.0.0.1 --port 8080
```

Tabel 12.3 Status Eksperimen pada Tahap UTS
<!-- widths: 40,22,38 -->
| Eksperimen | Status | Bukti |
|---|---|---|
| Label-scan | Selesai | Tabel 3.2 |
| Walk-forward baseline, logistic, XGBoost | Selesai | Tabel 10.3–10.10 |
| Walk-forward xgboost_balanced, random forest, ensemble | Selesai | Tabel 10.3, 10.7, 10.11–10.12 |
| Walk-forward xgboost_selected (feature selection) | Selesai | Tabel 10.13–10.14, 10.23 |
| Final-test 2026 (XGBoost) | Selesai | Tabel 10.18–10.20 |
| XGBoost library vs from scratch | Selesai (data 2021–2025) | Tabel 10.21–10.22 |
| compare-scratch dengan data penuh 2016–2024 | Belum | Rencana (Tabel 12.1) |
| Ablation kelompok fitur | Belum | Rencana (Tabel 12.1) |

## 12.2 Pengembangan Kandidat Model

- Menjalankan skenario E2 dan E3 pada Skripsi Bab III (Tabel 3.6): LSTM dan CNN-LSTM pada jendela 48, 96, dan 144 candle, dengan fold, kalibrasi, dan threshold yang sama agar dapat dibandingkan langsung dengan XGBoost.
- Model dua tahap atau meta-labeling: tahap pertama menilai apakah candle layak ditradingkan, tahap kedua menentukan arah (SoT D-08, future work).
- Stacking out-of-fold bila ensemble soft voting terbukti membantu, dengan blok data tambahan agar meta-learner tidak belajar dari prediksi in-sample.
- LightGBM atau CatBoost hanya ditambahkan bila ada alasan yang terukur, karena menambah dependency (SoT D-04).

## 12.3 Feature Selection Lanjutan

- Memakai tabel stabilitas `feature_selection_xgboost_selected.csv` (Tabel 10.14) untuk menyusun subset fitur yang terpilih di hampir semua fold, lalu mengujinya bersama hasil ablation.
- Menjalankan `ablation` kelompok fitur (EXP-R04) dan mengisi kolom ablation pada SoT bagian D.
- Menguji fitur multi-timeframe (M15 dan H1 sudah dikumpulkan) dan fitur rezim pasar.
- Menambahkan penjelasan per prediksi (misalnya SHAP) untuk analisis kesalahan, bukan untuk memilih fitur.

## 12.4 Analisis Error

- Memecah precision dan kesalahan arah berlawanan per sesi (Asia, London, New York), per jam, per hari, per rezim volatilitas (`atr_14_vs_daily_mean`), dan per kekuatan trend (ADX ≥ 25).
- Menganalisis sinyal NEUTRAL: seberapa jauh harga bergerak sebelum horizon habis, untuk menilai apakah horizon atau k terlalu sempit.
- Membandingkan confusion matrix antarfold untuk melihat apakah kegagalan terpusat pada tahun tertentu, misalnya 2025 dengan kenaikan +62,49%.

## 12.5 Analisis Threshold

- Kurva precision–coverage per arah pada blok THRESHOLD dan EVALUATION untuk melihat trade-off antara jumlah sinyal dan ketepatannya.
- Stabilitas threshold antarfold: threshold yang terlalu berbeda antar tahun menandakan probabilitas belum stabil.
- Sensitivitas terhadap `min_precision_lift` (1,1; 1,2; 1,5) dan `min_signals` (30; 50; 100), dipilih dari rata-rata walk-forward.
- Threshold berbasis nilai ekonomi, yaitu memaksimalkan P&L harapan setelah spread, sebagai pembanding threshold berbasis F1.
- Kepekaan threshold terhadap selisih probabilitas kecil (Temuan 11): grid yang lebih halus, threshold yang dihaluskan antarfold, atau syarat precision dengan batas bawah interval Wilson, bukan nilai titik.

## 12.6 Hyperparameter Tuning

Pencarian dilakukan antarfold walk-forward, bukan pada satu blok, karena pipeline sebelumnya turun dari macro F1 0,3525 pada TUNE menjadi 0,2478 pada TEST. Ruang pencarian mengikuti Skripsi Bab III Tabel 3.3.

Tabel 12.4 Ruang Pencarian Hyperparameter Setelah UTS
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

## 12.7 Perbandingan Model

- Tabel rata-rata ± simpangan baku setiap model pada lima fold, ditambah jumlah fold tempat model mengalahkan XGBoost.
- Uji berpasangan per fold dan interval bootstrap blok (block bootstrap) untuk selisih precision, karena observasi berdekatan tidak independen.
- Perbandingan yang adil: semua model memakai fold, kalibrasi, threshold, dan aturan simulasi yang sama.

## 12.8 Simulasi, Integrasi, dan Pengujian Forward

- Aturan satu posisi pada satu waktu dan spread keluar dari candle tempat barrier tersentuh, sebagai pembanding simulasi saat ini.
- Integrasi dengan Agen Manajemen Risiko (skenario E4) dan Agen Fundamental berbasis LLM (E5).
- Mengunci konfigurasi lalu menjalankan uji forward pada periode baru, misalnya mulai November 2026, karena 2026 bukan blind test (SoT D-12).

Tabel 12.5 Urutan Pekerjaan Setelah UTS
<!-- widths: 8,52,40 -->
| No. | Pekerjaan | Keluaran |
|---|---|---|
| 1 | Perbaikan zona waktu dan spread nol, lalu ulangi walk-forward; jalankan ablation dan compare-scratch dengan data penuh 2016–2024 | Data dan fitur waktu yang benar, ablation.json, comparison.json |
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
| n_estimators = 150 | Waktu komputasi: sekitar 3,4 detik per ronde pada 700.000 baris, sehingga 150 ronde ≈ 9 menit per fold (EXP-S04); pada 241.310 baris TRAIN 2021–2024, 150 ronde memakan 127 detik (Tabel 10.21) |
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

Nilai yang konsekuensinya pasti (misalnya batas 255 bin dan HESSIAN_FLOOR) dan nilai yang dibuktikan eksperimen (base score, konvensi gamma, waktu per ronde) dapat dipertanggungjawabkan langsung. Hyperparameter XGBoost, grid threshold, dan pengali barrier masih berupa design decision yang akan diuji pada tahap setelah UTS (bab 12.6). Karena itu, hasil pada bab 10 dilaporkan sebagai hasil konfigurasi awal, bukan konfigurasi optimal. Walk-forward sudah memberi bukti pertama untuk dua nilai: batas 300 pohon terlalu kecil, dan syarat lift 1,2 cukup ketat sehingga beberapa arah dinonaktifkan (SELL 2023 dan BUY 2026).

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

MetaQuotes Ltd., "MetaTrader 5 trading platform." [Online]. Tersedia: https://www.metatrader5.com (diakses 8 November 2026).

MetaQuotes Ltd., "Python integration: copy_rates_range," MQL5 Reference. [Online]. Tersedia: https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesrange_py (diakses 8 November 2026).
