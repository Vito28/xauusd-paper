# Metodologi Penelitian — Technical Agent XAUUSD

Dokumen ini menjelaskan dataset, preprocessing, fitur, label, model, validasi,
metrik, dan backtesting. Cara instalasi dan menjalankan program ada di
[`README.md`](../README.md). Bukti, penurunan rumus, dan log keputusan ada di
[`SOURCE_OF_TRUTH.md`](../SOURCE_OF_TRUTH.md) (disingkat **SoT**).

> **Status (30 Oktober 2026).** Pipeline, backtesting, dan dashboard sudah diverifikasi
> pada data sintetis. Eksperimen pada data XAUUSD riil dengan pipeline ini **belum
> dijalankan**, sehingga dokumen ini tidak memuat angka performa model.

## 1. Latar belakang dan permasalahan

Emas (XAUUSD) adalah aset yang banyak diperdagangkan dan berperan sebagai
lindung nilai serta *safe haven* [29], [30]. Pada timeframe intraday, pergerakan
harga emas bercampur dengan noise mikrostruktur, spread, dan pola volatilitas
yang bergantung pada jam perdagangan [18], [21]. Akibatnya:

- sebagian besar gerakan 5–30 menit terlalu kecil untuk ditransaksikan setelah biaya;
- model yang dipaksa memprediksi arah di **setiap** candle cenderung mendekati tebakan acak;
- metrik *accuracy* menyesatkan bila satu kelas mendominasi [28].

Penelitian emas berbasis ML yang ditemukan kebanyakan memakai data harian,
split acak, atau regresi level harga [4], [34], [35]. Studi klasifikasi arah
intraday yang **selektif** (boleh abstain), **memperhitungkan spread**, dan
divalidasi secara **walk-forward dengan purging** masih jarang; rujukan
terdekat adalah studi selective classification pada futures emas [37].
Analisis research gap lengkap ada di SoT bagian C.

## 2. Tujuan Technical Agent

1. Mengeluarkan BUY/SELL **hanya** ketika model cukup yakin; NO_TRADE boleh menjadi keputusan paling sering.
2. Mengutamakan **precision** sinyal BUY dan SELL, dengan **F1-score** sebagai penyeimbang precision–recall.
3. Menjaga metodologi yang valid: tanpa data leakage, tanpa memakai data 2026 untuk tuning.
4. Menyediakan output terstruktur yang dapat diaudit untuk integrasi multi-agent.
5. Membandingkan XGBoost resmi dengan implementasi from scratch berbasis rumus matematis.

## 3. Mengapa Machine Learning

Hubungan antara indikator teknikal dan pergerakan harga bersifat nonlinier,
berubah antarperiode, dan saling berinteraksi (mis. momentum hanya berarti pada
volatilitas tertentu). Model berbasis pohon seperti XGBoost dapat menangkap
interaksi tersebut tanpa spesifikasi manual dan tetap kuat pada data tabular
[1], [5], [7], [8]. Model ini juga menghasilkan probabilitas per kelas yang dapat
dikalibrasi dan dipakai untuk keputusan abstain [11], [13], [15].
Keputusan tidak memakai LSTM/GRU pada tahap ini dijelaskan di SoT D-05.

## 4. Dataset historis XAUUSD

| Atribut | Nilai |
|---|---|
| Sumber | MetaTrader 5, server MetaQuotes-Demo (akun demo) |
| Penyimpanan | PostgreSQL, view `market.processed_m5` |
| Timeframe | M5 native (M15 dan H1 juga dikumpulkan; M1 belum) |
| Rentang | Januari 2016 – September 2026 (sekitar 754 ribu candle M5 valid menurut audit pipeline lama) |
| Kolom | OHLC, tick volume, spread native, statistik spread dari tick |
| Timezone | UTC |

Data pengembangan dibatasi sampai **31 Desember 2025**. Data tahun 2026 hanya
untuk pengujian. Provenance, jumlah gap, dan angka mana yang belum diverifikasi
ulang ada di SoT bagian B. Spread akun demo belum tentu sama dengan akun live (SoT A-09).

## 5. Pemilihan timeframe

**M5 dipakai sebagai timeframe utama** (SoT D-01):

- **M1** lebih bising dan spread relatif terhadap gerak per bar lebih besar.
  Literatur realized volatility menunjukkan bahwa ketika noise mikrostruktur ada,
  sampling paling sering tidak otomatis lebih baik, dan return 5 menit lazim dipakai [18], [19], [20].
  Data M1 juga belum dikumpulkan. M1 direncanakan sebagai eksperimen ablation terpisah.
- **M15** mengurangi jumlah sampel empat kali lipat tanpa alasan kuat.
- Konteks jangka lebih panjang tetap tersedia lewat fitur M5 berjendela panjang
  (mis. EMA 288 bar = 24 jam), sehingga tidak ada look-ahead dari resampling.

Bukti literatur di atas bersifat tidak langsung (studi volatilitas, bukan
klasifikasi arah); karena itu pemilihan M5 berstatus *design decision*.

## 6. Pemilihan variabel dan indikator

Pipeline lama memiliki 223 kandidat fitur yang banyak redundan. Pipeline baru
memakai **41 fitur dalam 7 kelompok**:

| Kelompok | Contoh fitur | Alasan |
|---|---|---|
| price_action | log return 1/3/6/12 bar, body/range candle dibagi ATR | momentum jangka sangat pendek |
| trend | jarak close ke EMA 12/48/288 (÷ATR), ADX 14, +DI−−DI | arah dan kekuatan trend |
| momentum | RSI 14, histogram MACD (12,26,9) ÷ATR, stochastic 14, posisi Donchian 48 | jenuh beli/jual, perubahan momentum |
| volatility | ATR/close, ATR vs rata-rata 24 jam, realized vol, Bollinger (20,2) | rezim volatilitas |
| volume | z-score log tick volume, rasio tick volume | aktivitas pasar |
| spread | spread/ATR, z-score spread | biaya dan likuiditas |
| time | jam (sin/cos), hari, sesi Asia/London/New York, overlap, candle sejak gap | pola intraday [21] |

Periode indikator (RSI/ATR/ADX 14, MACD 12/26/9, Bollinger 20/2) adalah
**konvensi** dari sumber aslinya [25]–[27], bukan nilai yang terbukti optimal.
Definisi, rumus, dan risiko leakage setiap fitur ada di SoT bagian D.
Kontribusi setiap kelompok akan diukur lewat ablation (belum dijalankan).

**Feature selection** (`technical/feature_selection.py`, model `xgboost_selected`) diulang di
setiap fold dan hanya memakai blok TRAIN dan CALIBRATION fold tersebut:

| Tahap | Kategori | Aturan | Data |
|---|---|---|---|
| Nilai hilang dan konstan | filter | buang jika nilai hilang > 40% atau simpangan baku 0 | TRAIN |
| Audit kebocoran | filter | catat fitur dengan \|Spearman(fitur, arah label)\| > 0,95 | TRAIN |
| Redundansi | filter | dari pasangan \|Spearman\| > 0,95, pertahankan mutual information lebih tinggi | sampel TRAIN |
| Gain XGBoost | embedded | dicatat sebagai pembanding (bukan dasar keputusan) | TRAIN |
| Permutation importance | model-based | pertahankan fitur yang menaikkan log loss saat diacak | CALIBRATION |
| Ablation kelompok | wrapper | walk-forward tanpa satu kelompok (`technical.run ablation`) | per fold |

Model akhir dilatih ulang pada fitur terpilih lalu dibandingkan dengan `xgboost` (41 fitur)
pada blok EVALUATION yang sama. Stabilitas seleksi antarfold ditulis ke
`runs/<id>/feature_selection_xgboost_selected.csv`.

## 7. Preprocessing dan feature engineering

Preprocessing (`technical/data.py`):

1. Tolak OHLC tidak valid dan timestamp di luar grid 5 menit.
2. Hapus duplikat timestamp (pertahankan observasi terakhir).
3. Pilih spread: native bar, atau rata-rata spread tick pada candle yang sama.
4. Tandai gap (`gap_before`). Harga **tidak pernah** diisi atau diinterpolasi.

Feature engineering (`technical/features.py`):

- Fitur di baris *t* hanya memakai candle *t* dan sebelumnya (`rolling`, `ewm`, `shift(+n)`).
  Hal ini diuji otomatis: mengubah candle masa depan tidak mengubah fitur masa lalu.
- Sebagian besar fitur dinormalisasi dengan ATR atau harga agar harga emas tahun 2016 dan 2026 sebanding.
- Nilai warm-up dibiarkan NaN. Imputasi untuk Logistic Regression di-fit hanya pada data TRAIN.

## 8. Labeling tiga kelas

Label menggunakan **triple-barrier** [3] (SoT D-03). Untuk candle keputusan *t*:

$$\text{barrier}_t = k \cdot \text{ATR14}_t + \text{spread}_t$$

| Kondisi dalam H candle berikutnya | Label |
|---|---|
| `high` menyentuh `close_t + barrier` lebih dulu | **BUY** |
| `low` menyentuh `close_t − barrier` lebih dulu | **SELL** |
| Tidak ada yang tersentuh, atau keduanya tersentuh di candle yang sama | **NO_TRADE** |
| Horizon melewati gap, data belum lengkap, atau ATR/spread tidak tersedia | tidak dilabeli |

- **Entry reference**: close candle *t*, yaitu waktu keputusan setelah candle tutup.
- **Default**: H = 6 bar (30 menit) dan k = 1,0. Keduanya akan ditetapkan ulang dari
  distribusi label 2016–2025 (`label-scan`) **sebelum** model dilatih (SoT D-02).
- **Spread** ditambahkan ke barrier sehingga label BUY/SELL berarti gerak bersih
  minimal k·ATR setelah biaya spread.
- Metode **fixed horizon** (bandingkan `close_{t+H}` dengan barrier) tersedia sebagai pembanding.
- Harga masa depan hanya dipakai untuk membuat label (ground truth), tidak pernah sebagai fitur.

Label lama (spread + 1 bps, horizon 15 menit) menghasilkan NO_TRADE hanya sekitar 8%
pada 2026 (SoT EXP-L01). Itu bertentangan dengan filosofi selektif, sehingga labelnya diganti.

## 9. Arsitektur model: XGBoost dan XGBoost from scratch

| Model | Peran | File |
|---|---|---|
| `always_no_trade` | baseline: selalu abstain | `technical/models.py` |
| `persistence` | baseline naif: gerak H bar terakhir > barrier → searah | `technical/models.py` |
| `logistic` | baseline linear (multinomial logistic regression) | `technical/models.py` |
| `xgboost` | **model referensi utama** (`XGBClassifier`, `multi:softprob`, `hist`) | `technical/models.py` |
| `xgboost_balanced` | XGBoost dengan bobot kelas *balanced*; pembanding penanganan imbalance lewat threshold | `technical/models.py` |
| `xgboost_scratch` | gradient boosting multikelas dari nol, hanya NumPy | `technical/xgboost_scratch.py` |
| `random_forest` | bagging pohon dalam (anggota ensemble, menurunkan variance) | `technical/models.py` |
| `ensemble` | soft voting berbobot `xgboost` + `random_forest` + `logistic` | `technical/ensemble.py` |
| `xgboost_selected` | XGBoost dengan feature selection di dalam fold (bagian 7) | `technical/feature_selection.py` |

**Ensemble.** Anggota dipilih agar kesalahannya berbeda: boosting (menurunkan bias),
bagging (menurunkan variance), dan model linear. Setiap anggota dilatih pada TRAIN,
dikalibrasi dengan temperature scaling pada CALIBRATION, lalu bobotnya dipilih dari grid
simpleks (kelipatan 0,05, jumlah 1) yang meminimalkan log loss campuran pada CALIBRATION.
Bobot tidak dipelajari dari prediksi in-sample sehingga tidak bocor seperti stacking yang
salah, dan grid memuat bobot satu anggota saja sehingga campuran tidak lebih buruk daripada
anggota terbaik pada blok itu. Threshold tetap dipilih pada blok THRESHOLD. Laporan per fold
(`metrics.json`, kunci `ensemble`) memuat bobot, temperatur, log loss setiap anggota, dan
keragaman antaranggota (proporsi beda prediksi dan korelasi kesalahan).

Implementasi from scratch mencakup base score, softmax, gradien, Hessian, histogram
gradien, evaluasi split (termasuk arah default untuk nilai hilang), gain,
regularisasi L1/L2, pruning dengan γ, shrinkage, early stopping, `predict_proba`,
dan feature gain. Bagian ini **tidak setara penuh** dengan XGBoost resmi:
tidak ada weighted quantile sketch, subsampling, paralelisasi, atau optimasi
sistem (SoT E-10, E-11).

Verifikasi (SoT F.1):

- Pada fitur diskret yang kandidat split-nya sama, probabilitas scratch dan library
  **identik** (selisih < 10⁻⁷).
- Pada fitur kontinu, hasilnya berbeda sedikit karena titik potong kuantil
  (data sintetis: argmax sama di 97,9% baris).
- Konvensi `gamma` berbeda faktor 2 dan sudah dikonversi otomatis.

LightGBM, CatBoost, ensemble, dan LSTM tidak diimplementasikan (SoT D-04, D-05, D-08).

## 10. Rumus inti dan hubungannya dengan kode

| Rumus | Fungsi di `xgboost_scratch.py` |
|---|---|
| $\mathcal{L}=\sum_i l(y_i,\hat{y}_i)+\sum_k\Omega(f_k)$, $\Omega(f)=\gamma T+\tfrac12\lambda\sum_j w_j^2$ | parameter `gamma`, `reg_lambda`, `reg_alpha` |
| $\tilde{\mathcal{L}}^{(t)}=\sum_i[g_i f_t(x_i)+\tfrac12 h_i f_t(x_i)^2]+\Omega(f_t)$ | `GradientTree.build` (memakai g dan h) |
| $p_{ik}=\operatorname{softmax}(z_i)_k$ | `softmax` |
| $g_{ik}=p_{ik}-y_{ik}$, $h_{ik}=\max(2p_{ik}(1-p_{ik}),10^{-16})$ | `softmax_gradient_hessian` |
| $w_j^*=-\dfrac{G_j}{H_j+\lambda}$ (dengan L1: $-T_\alpha(G_j)/(H_j+\lambda)$) | `leaf_weight`, `soft_threshold` |
| $Gain=\tfrac12\left[\dfrac{G_L^2}{H_L+\lambda}+\dfrac{G_R^2}{H_R+\lambda}-\dfrac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma$ | `split_gain`, `GradientTree.find_best_split` |
| $\hat{y}^{(t)}=\hat{y}^{(t-1)}+\eta f_t(x)$ | loop boosting di `ScratchXGBoostClassifier.fit` |
| $F_{0k}=\log\pi_k-\tfrac1K\sum_j\log\pi_j$ | `compute_initial_margin` |

Penurunan lengkap setiap rumus, beserta tes yang memverifikasinya, ada di SoT bagian E.

## 11. Strategi training, validation, dan testing

Semua split bersifat **kronologis**, tanpa random split (`technical/splits.py`, SoT D-10).

**Walk-forward expanding window** untuk tahun validasi Y ∈ {2021, …, 2025}:

| Blok | Rentang | Digunakan untuk |
|---|---|---|
| TRAIN | 2016-01-01 → (Y−1)-07-01 | melatih model |
| CALIBRATION | (Y−1)-07-01 → (Y−1)-10-01 | early stopping dan temperature scaling |
| THRESHOLD | (Y−1)-10-01 → Y-01-01 | memilih threshold BUY/SELL |
| EVALUATION | Y-01-01 → (Y+1)-01-01 | mengukur performa saja |

- **Purging**: baris dipakai hanya jika `label_end_time + embargo` (6 bar) berada sebelum akhir bloknya [3].
- Pemilihan model, label, fitur, dan hyperparameter dilakukan berdasarkan **rata-rata hasil fold validasi**.
- Perintah pengembangan (`label-scan`, `walk-forward`, `ablation`, `compare-scratch`)
  tidak memuat data setelah 31 Desember 2025 sama sekali.
- `walk-forward` menyimpan prediksi setiap fold untuk **seluruh** candle tahun evaluasi
  (`runs/<id>/backtest/<model>/validation_<Y>/predictions.csv`). Setiap prediksi berasal
  dari model, kalibrasi, dan threshold fold itu sendiri, jadi model tidak pernah melihat
  data tahun yang diprediksinya.

## 12. Pengujian out-of-sample 2026

Fold final: TRAIN 2016 → 1 Juli 2025, CALIBRATION Juli–September 2025,
THRESHOLD Oktober–Desember 2025, TEST mulai 1 Januari 2026.

1. Model, kalibrasi, dan threshold dibekukan memakai data ≤ 2025.
2. Untuk setiap candle 2026: fitur dihitung dari data sampai candle tersebut → probabilitas → keputusan.
3. Prediksi dicatat dengan timestamp, probabilitas, dan reason code
   (`runs/<id>/backtest/<model>/final_test/predictions.csv`).
4. Label dihitung dengan aturan yang sama. Hanya label dengan horizon lengkap yang dinilai.
5. BUY dinilai benar **hanya** jika label aktual BUY. BUY pada label SELL atau NO_TRADE adalah false positive. Aturan yang sama berlaku untuk SELL.

Periode uji yang benar-benar dinilai dicatat di `test_period.json`.
Perintah ini wajib diberi `--confirm` agar tidak dijalankan berulang kali sebagai alat tuning.

> **Penting (SoT A-05, D-12).** Pipeline generasi sebelumnya sudah pernah dilatih
> sampai Maret 2026, dan hasil 2026 sudah dilihat sebelum desain ini dibuat.
> Karena itu hasil 2026 harus disebut **retrospective out-of-sample**. Uji yang benar-benar
> buta membutuhkan periode forward baru dengan konfigurasi yang dikunci lebih dulu.

## 13. Metrik evaluasi

Dihitung oleh `technical/evaluation.py` per blok evaluasi:

- **Precision BUY dan SELL** (prioritas utama) beserta interval Wilson 95% [41];
- **Recall dan F1** per kelas, **macro F1**, dan **directional macro F1** (rata-rata F1 BUY dan SELL);
- **Confusion matrix** 3×3, distribusi label aktual dan prediksi;
- **Signal coverage** (proporsi BUY + SELL), coverage BUY dan SELL;
- **False positive** per arah, dan **kesalahan arah berlawanan** (BUY saat label SELL dan sebaliknya);
- **Kalibrasi**: log loss, Brier multikelas, ECE top-label;
- Pembanding: baseline always-NO_TRADE, persistence, dan logistic regression; hasil tanpa threshold (argmax) juga dilaporkan.

NO_TRADE yang dominan tidak dianggap keberhasilan. Model yang tidak pernah
memberi sinyal memiliki precision BUY/SELL "tidak terdefinisi" (bukan 100%) dan F1 = 0.

## 14. Alur keputusan Technical Agent

```text
candle M5 tutup
   → 41 fitur kausal
   → XGBoost: [p_SELL, p_NO_TRADE, p_BUY]
   → temperature scaling (T dari blok CALIBRATION)
   → arah kandidat = BUY jika p_BUY ≥ p_SELL, selain itu SELL
   → p_arah ≥ threshold_arah ?  ya → BUY/SELL   tidak → NO_TRADE
```

Threshold dipilih per arah pada blok THRESHOLD: F1 tertinggi dengan syarat
precision ≥ **1,2 × prevalensi kelas** dan minimal **30 sinyal**. Model acak
memiliki precision sebesar prevalensi, sehingga syarat ini menuntut sinyal yang lebih
baik dari acak. Jika tidak ada threshold yang lolos, arah tersebut **dinonaktifkan**
dan dilaporkan (SoT D-09). Nilai 1,2 dan 30 adalah *design decision*.

Output untuk agent lain (`TechnicalSignal` di `technical/decision.py`):

```json
{
  "timestamp": "2026-...T..:..:00+00:00",
  "symbol": "XAUUSD",
  "timeframe": "M5",
  "model_version": "technical-xgboost-<config_hash>",
  "signal": "BUY | SELL | NO_TRADE",
  "confidence": 0.0,
  "probabilities": {"SELL": 0.0, "NO_TRADE": 0.0, "BUY": 0.0},
  "horizon_minutes": 30,
  "reason_code": "BUY_PROBABILITY_ABOVE_THRESHOLD | BELOW_THRESHOLD | SELL_DISABLED_NO_VALID_THRESHOLD | ...",
  "market_regime": {"trend_strength": "trending | ranging", "volatility": "low | normal | high"},
  "feature_snapshot": {"rsi_14": 0.0, "adx_14": 0.0}
}
```

`confidence` adalah probabilitas terkalibrasi untuk keputusan yang dipilih.
`market_regime` adalah ringkasan deskriptif (ADX ≥ 25 = trending; rasio ATR) dan
tidak dipakai model untuk mengambil keputusan.

## 15. Backtesting dan simulasi trading

Backtesting memakai prediksi out-of-sample dari langkah 11 dan 12 (`technical/backtest.py`)
dan memisahkan **tiga ukuran yang berbeda**:

| Ukuran | Pertanyaan | Definisi |
|---|---|---|
| Akurasi klasifikasi | Apakah sinyal sama dengan label? | metrik bagian 13 terhadap label triple-barrier |
| Ketepatan arah | Apakah harga t+H bergerak searah sinyal? | tanda `close[t+H] − close[t]`, tanpa ambang (HIT/MISS/FLAT) |
| Simulasi trading | Berapa hasilnya setelah spread? | `technical/simulation.py`, terpisah dari evaluasi model |

Hasil per sinyal BUY/SELL (`prediction_result`) mengikuti definisi label:

| Hasil | Arti |
|---|---|
| CORRECT | barrier searah tersentuh lebih dulu (label aktual = sinyal) |
| INCORRECT | barrier berlawanan tersentuh lebih dulu |
| NEUTRAL | tidak ada barrier tersentuh (label NO_TRADE); kenaikan kecil **bukan** BUY benar |
| PENDING | horizon belum selesai |
| UNAVAILABLE | horizon melewati gap pasar/data, tidak dapat dinilai |

**Asumsi harga simulasi** (tingkat candle, bukan eksekusi tick):

- OHLC MT5 adalah harga **bid**; ask = bid + spread candle keputusan.
- BUY masuk di ask = `close + spread`; SELL masuk di bid = `close`.
- TP/SL = barrier label (`close ± (k·ATR + spread)` pada harga bid). Bila tidak tersentuh,
  posisi ditutup di close candle t+H. Sentuhan dua sisi di candle yang sama dianggap SL.
- Spread saat keluar dianggap sama dengan saat masuk; slippage, komisi, dan swap tidak dimodelkan.
- Dengan aturan ini TP memberi **+k·ATR** bersih dan SL memberi **−(k·ATR + 2·spread)**.
- P&L = gerak bersih × lot × ukuran kontrak (default 0,01 lot × 100 oz); margin = notional ÷ leverage.
  Lot, leverage, dan ukuran kontrak dapat diubah di dashboard; nilainya hanya mengubah skala
  P&L, bukan win rate atau profit factor.

Simulasi ini mengasumsikan satu posisi per sinyal dan **mengizinkan posisi tumpang tindih**
(setiap candle bersinyal dihitung). Angka simulasi karena itu adalah ukuran kualitas sinyal
setelah biaya, bukan proyeksi akun nyata. Modul ini tidak pernah mengirim order.

## 16. Keterbatasan

- **Backtesting masih sederhana** (bagian 15): tingkat candle, tanpa slippage/komisi, posisi boleh
  tumpang tindih. Precision dan F1 adalah metrik klasifikasi, bukan profit; hasil simulasi
  rentan terhadap backtest overfitting bila konfigurasi dipilih berulang kali dari hasilnya [31], [32].
- Belum ada hasil eksperimen data riil dengan pipeline baru (SoT F.3).
- Data 2026 adalah **retrospective** out-of-sample (lihat bagian 12).
- Satu sumber data (akun demo MetaQuotes); spread demo dapat berbeda dari live.
  Spread saat exit diasumsikan sama dengan saat entry.
- Urutan sentuhan barrier di dalam satu candle M5 tidak diketahui (diselesaikan secara konservatif dengan NO_TRADE).
- Label dengan horizon tumpang tindih membuat sampel tidak independen, sehingga interval Wilson cenderung terlalu sempit.
- Kalibrasi dan early stopping memakai blok yang sama (SoT D-13), dan threshold hanya dipilih dari 3 bulan data.
- Hanya informasi teknikal; peristiwa makro dan berita tidak terlihat oleh model.

## 17. Rencana lanjutan

**Backtesting lanjutan.** Menambahkan slippage, komisi, larangan posisi tumpang tindih,
position sizing berbasis risiko, dan spread aktual saat keluar (dari data tick).

**Integrasi Fundamental Agent (LLM).** Technical Agent bertanggung jawab atas
sinyal teknikal jangka pendek beserta probabilitas dan alasannya; ia tidak membaca
berita atau data makro dan tidak mengeksekusi order. Fundamental Agent nantinya
memberi konteks makro/berita dengan timestamp ketersediaan yang jelas. Komponen
keputusan dan manajemen risiko akan menggabungkan `TechnicalSignal` dengan
output Fundamental Agent, misalnya memveto sinyal teknikal menjelang rilis data penting.

**Eksperimen berikutnya**: hasil EXP-R01–R05 (SoT F.3); M1 sebagai ablation;
pendekatan dua tahap atau meta-labeling [3], [40]; uji forward baru setelah konfigurasi dikunci.

## 18. Referensi utama

Daftar lengkap (41 referensi, termasuk status verifikasinya) ada di SoT bagian H.
Referensi yang dikutip di dokumen ini:

[1] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD*, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.

[3] M. López de Prado, *Advances in Financial Machine Learning*. Hoboken, NJ, USA: Wiley, 2018.

[4] P. Sadorsky, "Predicting gold and silver price direction using tree-based classifiers," *J. Risk Financial Manag.*, vol. 14, no. 5, Art. 198, 2021, doi: 10.3390/jrfm14050198.

[5] C. Krauss, X. A. Do, and N. Huck, "Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500," *Eur. J. Oper. Res.*, vol. 259, no. 2, pp. 689–702, 2017, doi: 10.1016/j.ejor.2016.10.031.

[7] L. Grinsztajn, E. Oyallon, and G. Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?," in *NeurIPS 35 Datasets and Benchmarks*, 2022, doi: 10.52202/068431-0037.

[8] R. Shwartz-Ziv and A. Armon, "Tabular data: Deep learning is not all you need," *Inf. Fusion*, vol. 81, pp. 84–90, 2022, doi: 10.1016/j.inffus.2021.11.011.

[11] C. K. Chow, "On optimum recognition error and reject tradeoff," *IEEE Trans. Inf. Theory*, vol. 16, no. 1, pp. 41–46, 1970, doi: 10.1109/TIT.1970.1054406.

[13] A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning," in *Proc. 22nd ICML*, 2005, pp. 625–632, doi: 10.1145/1102351.1102430.

[15] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. 34th ICML*, PMLR vol. 70, 2017, pp. 1321–1330.

[18] Y. Aït-Sahalia, P. A. Mykland, and L. Zhang, "How often to sample a continuous-time process in the presence of market microstructure noise," *Rev. Financ. Stud.*, vol. 18, no. 2, pp. 351–416, 2005, doi: 10.1093/rfs/hhi016.

[19] T. G. Andersen, T. Bollerslev, F. X. Diebold, and P. Labys, "The distribution of realized exchange rate volatility," *J. Amer. Statist. Assoc.*, vol. 96, no. 453, pp. 42–55, 2001, doi: 10.1198/016214501750332965.

[20] P. R. Hansen and A. Lunde, "Realized variance and market microstructure noise," *J. Bus. Econ. Statist.*, vol. 24, no. 2, pp. 127–161, 2006, doi: 10.1198/073500106000000071.

[21] T. G. Andersen and T. Bollerslev, "Intraday periodicity and volatility persistence in financial markets," *J. Empir. Finance*, vol. 4, no. 2–3, pp. 115–158, 1997, doi: 10.1016/S0927-5398(97)00004-2.

[25] J. W. Wilder Jr., *New Concepts in Technical Trading Systems*. Greensboro, NC, USA: Trend Research, 1978.

[26] G. Appel, *Technical Analysis: Power Tools for Active Investors*. FT Press, 2005.

[27] J. Bollinger, *Bollinger on Bollinger Bands*. New York, NY, USA: McGraw-Hill, 2001.

[28] T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," *PLoS ONE*, vol. 10, no. 3, e0118432, 2015, doi: 10.1371/journal.pone.0118432.

[29] D. G. Baur and B. M. Lucey, "Is gold a hedge or a safe haven? An analysis of stocks, bonds and gold," *Financ. Rev.*, vol. 45, no. 2, pp. 217–229, 2010, doi: 10.1111/j.1540-6288.2010.00244.x.

[30] F. A. O'Connor, B. M. Lucey, J. A. Batten, and D. G. Baur, "The financial economics of gold — A survey," *Int. Rev. Financ. Anal.*, vol. 41, pp. 186–205, 2015, doi: 10.1016/j.irfa.2015.07.005.

[31] D. H. Bailey, J. M. Borwein, M. López de Prado, and Q. J. Zhu, "The probability of backtest overfitting," *J. Comput. Finance*, vol. 20, no. 4, 2017, doi: 10.21314/JCF.2016.322.

[32] D. H. Bailey, J. M. Borwein, M. López de Prado, and Q. J. Zhu, "Pseudo-mathematics and financial charlatanism," *Notices Amer. Math. Soc.*, vol. 61, no. 5, pp. 458–471, 2014, doi: 10.1090/noti1105.

[34] A. Amini and R. Kalantari, "Gold price prediction by a CNN-Bi-LSTM model along with automatic parameter tuning," *PLoS ONE*, vol. 19, no. 3, e0298426, 2024, doi: 10.1371/journal.pone.0298426.

[35] M. Arif *et al.*, "Multivariate LSTM-based intraday gold price prediction with rolling time series validation," *J. Masyarakat Informatika*, vol. 17, no. 1, pp. 26–38, 2026, doi: 10.14710/jmasif.17.1.78091.

[37] N. Chalkidis and R. Savani, "Trading via selective classification," in *Proc. 2nd ACM ICAIF*, 2021, pp. 1–9, doi: 10.1145/3490354.3494379.

[40] J. F. Joubert, "Meta-labeling: Theory and framework," *J. Financial Data Sci.*, vol. 4, no. 3, pp. 31–44, 2022, doi: 10.3905/jfds.2022.1.098.

[41] E. B. Wilson, "Probable inference, the law of succession, and statistical inference," *J. Amer. Statist. Assoc.*, vol. 22, no. 158, pp. 209–212, 1927, doi: 10.1080/01621459.1927.10502953.

