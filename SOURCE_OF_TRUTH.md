# SOURCE OF TRUTH — Technical Agent XAUUSD

Dokumen ini adalah pusat pertanggungjawaban ilmiah dan teknis untuk Technical
Agent. `README.md` menjelaskan **apa** yang dibangun dan **bagaimana** cara
menjalankannya, `docs/METHODOLOGY.md` menjelaskan metodologinya; dokumen ini menjelaskan **mengapa** setiap keputusan diambil,
**apa buktinya**, dan **sejauh mana** klaim tersebut sudah diverifikasi.

Terakhir diperbarui: 30 Oktober 2026.

## Kategori bukti

Setiap klaim diberi salah satu label berikut.

| Label | Arti |
|---|---|
| **Literature-backed** | Didukung publikasi yang sudah diperiksa (abstrak/teks penuh/metadata DOI) |
| **Dataset-derived** | Diperoleh dari analisis data proyek; sumber artefak disebutkan |
| **Experimentally validated** | Dibuktikan oleh eksperimen/tes yang dapat dijalankan ulang |
| **Design decision** | Pilihan rancangan yang beralasan, bukan fakta ilmiah universal |
| **Unverified** | Belum ada bukti memadai; tidak boleh disajikan sebagai fakta |

Status verifikasi referensi (bagian H): **VERIFIED** = metadata dikonfirmasi
lewat Crossref/penerbit/proceedings; **PARTIAL** = publikasi ada, tetapi
sebagian data bibliografi atau isi spesifik belum dikonfirmasi.

---

## A. Research Assumptions

| ID | Asumsi / batasan | Kategori | Catatan |
|---|---|---|---|
| A-01 | Masalah dirumuskan sebagai klasifikasi tiga kelas BUY / SELL / NO_TRADE pada setiap candle M5 yang sudah tutup. | Design decision | Bukan regresi harga. Penelitian emas yang memakai regresi level harga menghasilkan R² tinggi yang tidak bermakna untuk keputusan arah [34], [35]. |
| A-02 | Sistem bersifat **selektif**: NO_TRADE boleh dominan, BUY/SELL hanya dikeluarkan bila confidence cukup. | Literature-backed | Kerangka reject option [11], selective classification [12], dan penerapannya pada trading futures logam termasuk emas [37]. |
| A-03 | Data pengembangan: **1 Januari 2016 – 31 Desember 2025** (development_end = `2026-01-01T00:00:00Z`, eksklusif). | Design decision | Sekitar 10 tahun memberi ~750 ribu candle M5 dan mencakup beberapa kondisi pasar berbeda. Panjang 10 tahun bukan angka yang "dibuktikan" literatur. |
| A-04 | Tahun 2026 adalah periode uji out-of-sample. Hanya candle yang horizon labelnya lengkap yang dinilai. | Design decision | Periode aktual dicatat otomatis di `runs/<id>/test_period.json`. |
| A-05 | **Status 2026 = retrospective out-of-sample, bukan blind test.** | Dataset-derived | Pipeline generasi sebelumnya (dihapus, tersedia di riwayat git commit `0a849ab`) sudah pernah dilatih sampai Maret 2026 dan hasil 2026 sudah dilihat beberapa kali sebelum desain ini dibuat. Desain baru tidak memakai data 2026 untuk fitting, tetapi peneliti sudah pernah melihat perilaku pasar 2026. Uji yang benar-benar buta memerlukan periode forward baru (mis. mulai November 2026) dengan konfigurasi yang dikunci terlebih dahulu (D-12). |
| A-06 | Hasil klasifikasi **bukan** hasil backtesting dan tidak membuktikan profitabilitas. | Literature-backed | Prediksi arah yang benar belum tentu menguntungkan setelah biaya; risiko overfitting backtest dibahas di [31], [32]. |
| A-07 | Hanya data teknikal (OHLC, tick volume, spread, waktu). Tidak ada data fundamental, makro, atau sentimen. | Design decision | Data fundamental akan ditangani Fundamental Agent (di luar cakupan). |
| A-08 | Harga MT5 diasumsikan berbasis Bid. Biaya BUY didekati dengan spread pada candle keputusan. | Unverified | Konvensi umum MT5 untuk CFD/forex; perlu dikonfirmasi per broker. Spread saat exit diasumsikan sama dengan saat entry. |
| A-09 | Spread berasal dari akun **demo** MetaQuotes. | Dataset-derived | Spread demo belum tentu sama dengan akun live; ini keterbatasan langsung pada definisi label. |

---

## B. Dataset Provenance

### B.1 Yang benar-benar tersedia

| Atribut | Nilai | Kategori |
|---|---|---|
| Sumber | MetaTrader 5, server `MetaQuotes-Demo`, broker MetaQuotes Ltd (`configs/brokers/metaquotes_demo.toml`) | Dataset-derived |
| Simbol | `XAUUSD` (dataset_symbol = requested_symbol) | Dataset-derived |
| Akuisisi | `copy_rates_range` (bar native) dan tick historis, oleh collector `market/collector.py` | Dataset-derived |
| Penyimpanan | PostgreSQL, view `market.processed_m5` (migrasi `20260929_0002`) | Dataset-derived |
| Timezone | Semua timestamp `TIMESTAMPTZ` UTC | Dataset-derived |
| Timeframe native yang dikumpulkan | M5 (utama), M15, H1. **M1 belum dikumpulkan.** | Dataset-derived |
| Jumlah candle M5 valid | 754.660 | Dataset-derived (artefak lama `technical/artifacts/data_quality.json`, commit `0a849ab`) |
| Candle pertama | 2016-01-04 (UTC) | Dataset-derived (artefak lama) |
| Candle terakhir saat audit lama | 2026-09-30 06:45 UTC | Dataset-derived (artefak lama) |
| Gap terdeteksi | 2.836 (termasuk akhir pekan dan libur) | Dataset-derived (artefak lama) |
| Duplikat / OHLC tidak valid | 0 / 0 | Dataset-derived (artefak lama) |

> Angka di atas berasal dari run pipeline lama sebelum refactor. Validasi ulang
> dengan pipeline baru akan tercatat di `runs/<id>/run.json` → `data_report`
> setelah `label-scan` dijalankan pada data PostgreSQL. Belum dijalankan.

### B.2 Proses validasi data (`technical/data.py`)

1. OHLC wajib ada, positif, `high ≥ max(open, close, low)`, `low ≤ min(open, close, high)`.
2. Durasi candle tepat 5 menit dan `open_time` berada di grid 5 menit.
3. Duplikat timestamp: dipertahankan observasi terakhir.
4. Gap ditandai (`gap_before`), **tidak** diisi atau diinterpolasi.
5. Spread: `native_spread_price` dipakai lebih dulu; jika tidak ada, rata-rata spread tick pada candle yang sama.

### B.3 Yang masih direncanakan

- Data M1 untuk ablation timeframe (collector sudah mendukung `TIMEFRAME_M1`).
- Data akun live untuk mengukur selisih spread demo vs live.

---

## C. Literature Evidence Matrix

| Keputusan | Bukti ilmiah (yang benar-benar diklaim sumber) | Referensi | DOI/URL | Keterbatasan untuk proyek ini |
|---|---|---|---|---|
| XGBoost sebagai model utama | Sistem tree boosting dengan objektif teregularisasi orde-2, sparsity-aware split, weighted quantile sketch | Chen & Guestrin, 2016 [1] | 10.1145/2939672.2939785 | Paper sistem; bukan bukti performa di pasar keuangan |
| Model pohon untuk data tabular | Pada 45 dataset tabular berukuran menengah, model berbasis pohon tetap unggul dari deep learning | Grinsztajn dkk., 2022 [7] | 10.52202/068431-0037 | Bukan data time-series keuangan |
| Model pohon vs DL | XGBoost umumnya mengungguli model deep tabular baru | Shwartz-Ziv & Armon, 2022 [8] | 10.1016/j.inffus.2021.11.011 | Bukan data keuangan |
| Tree ensemble untuk prediksi arah aset | Gradient-boosted trees dan random forest kompetitif terhadap DNN pada S&P 500 harian | Krauss dkk., 2017 [5] | 10.1016/j.ejor.2016.10.031 | Data harian saham, bukan emas intraday |
| LSTM tidak dipilih sekarang | LSTM mengungguli RF/DNN/LR pada S&P 500 harian | Fischer & Krauss, 2018 [6] | 10.1016/j.ejor.2017.11.054 | Bukti tandingan: LSTM bisa unggul; ditunda karena biaya komputasi dan fitur tabular (D-05) |
| Indikator teknikal + tree classifier untuk arah emas | 13 indikator (RSI, stochastic, ADX, MACD, ROC, OBV, MA) + RF/boosting; akurasi 85–90% pada horizon 20 hari | Sadorsky, 2021 [4] | 10.3390/jrfm14050198 | Data **harian** ETF GLD, split **acak**, label overlap → angka tidak dapat dijadikan pembanding M5 |
| Selective classification pada emas | Klasifikasi biner/ternary (up/down/flat) dengan abstain pada futures logam termasuk emas, bar 30 menit, walk-forward; trade-off akurasi–coverage | Chalkidis & Savani, 2021 [37] | 10.1145/3490354.3494379 | Futures CME 2011–2019, bukan spot XAUUSD M5 |
| Reject option | Aturan reject optimal dan trade-off error–reject | Chow, 1970 [11] | 10.1109/TIT.1970.1054406 | Teori umum pengenalan pola |
| Selective classification modern | Selective classifier dengan risiko terkontrol untuk model terlatih | Geifman & El-Yaniv, 2017 [12] | NeurIPS 2017 | Konteks deep learning citra |
| Kalibrasi probabilitas | Boosted trees cenderung tidak terkalibrasi; Platt/isotonic memperbaiki | Niculescu-Mizil & Caruana, 2005 [13] | 10.1145/1102351.1102430 | Dataset non-keuangan |
| Temperature scaling | Temperature scaling (Platt satu parameter) efektif untuk kalibrasi multikelas | Guo dkk., 2017 [15] | PMLR 70 | Dibuktikan pada neural network, diterapkan di sini ke XGBoost (asumsi) |
| Triple-barrier labeling | Label berbasis barrier atas/bawah dan barrier vertikal (waktu) | López de Prado, 2018 [3] §3.4 | ISBN 978-1-119-48208-6 | Buku metodologi; parameter tidak spesifik emas |
| Triple-barrier empiris | Triple-barrier + bar berbasis informasi lebih baik dari label next-bar | Grądzki dkk., 2025 [39] | 10.1186/s40854-025-00866-w | Kripto, bukan emas |
| Purging & embargo | Purged K-fold dan embargo untuk label overlap | López de Prado, 2018 [3] §7.4 | ISBN 978-1-119-48208-6 | — |
| Validasi kronologis | Evaluasi rolling-origin / out-of-sample | Tashman, 2000 [17]; Bergmeir & Benítez, 2012 [16] | 10.1016/S0169-2070(00)00065-0; 10.1016/j.ins.2011.12.028 | Literatur forecasting umum |
| Expanding-window pada emas intraday | Validasi time-series expanding window 5 split pada XAUUSD H1 | Arif dkk., 2026 [35] | 10.14710/jmasif.17.1.78091 | Regresi level harga H1 tahun 2024 saja |
| Pemilihan M5 (bukan M1) | Return 5 menit dipakai dalam realized volatility FX untuk mengurangi noise mikrostruktur | Andersen dkk., 2001 [19] | 10.1198/016214501750332965 | Estimasi volatilitas, bukan klasifikasi arah |
| Pemilihan M5 (bukan M1) | Dengan noise mikrostruktur yang diabaikan, frekuensi sampling optimal bersifat terbatas (tidak setinggi mungkin) | Aït-Sahalia dkk., 2005 [18]; Hansen & Lunde, 2006 [20] | 10.1093/rfs/hhi016; 10.1198/073500106000000071 | Bukti tidak langsung untuk klasifikasi |
| Fitur waktu/sesi | Volatilitas intraday memiliki pola periodik kuat menurut jam | Andersen & Bollerslev, 1997 [21] | 10.1016/S0927-5398(97)00004-2 | Data FX dan saham, bukan emas |
| Indikator teknikal sebagai fitur | Analisis teknikal dapat diformalkan secara statistik; aturan sederhana memiliki sifat prediktif pada data historis | Lo dkk., 2000 [22]; Brock dkk., 1992 [23] | 10.1111/0022-1082.00265; 10.1111/j.1540-6261.1992.tb04681.x | Data saham harian; hasil bervariasi antarperiode [24] |
| RSI, ATR, ADX (periode 14) | Definisi indikator | Wilder, 1978 [25] | Buku (PARTIAL) | Periode 14 = **konvensi**, bukan nilai optimal terbukti |
| MACD (12, 26, 9) | Definisi indikator | Appel, 2005 [26] | Buku (PARTIAL) | Parameter = konvensi |
| Bollinger Bands (20, 2) | Definisi indikator | Bollinger, 2001 [27] | Buku (PARTIAL) | Parameter = konvensi |
| Precision/F1 lebih informatif dari accuracy | Plot precision-recall lebih informatif dari ROC pada data tidak seimbang | Saito & Rehmsmeier, 2015 [28] | 10.1371/journal.pone.0118432 | Klasifikasi biner |
| Interval Wilson untuk precision | Interval skor untuk proporsi binomial | Wilson, 1927 [41] | 10.1080/01621459.1927.10502953 | Asumsi independen dilanggar oleh label overlap |
| Seleksi fitur terstruktur | Kombinasi metode dengan seleksi variabel terstruktur | Taneva-Angelova dkk., 2025 [36] | 10.3390/ijfs13020102 | Data frekuensi rendah |
| Banyak konfigurasi indikator → perlu seleksi | Klasifikasi sinyal MACD/Ichimoku/MA palsu vs asli dengan 112 fitur | Mamoudan dkk., 2023 [33] | 10.1093/jcde/qwad039 | Timeframe tidak disebut; inkonsistensi internal di paper |

**Research gap (analisis sendiri, bukan klaim referensi):** dari sumber yang
diperiksa, studi klasifikasi **arah** XAUUSD intraday dengan **abstain
terkalibrasi**, label yang memperhitungkan **spread**, dan validasi
**walk-forward dengan purging** masih jarang. Kebanyakan penelitian emas memakai
data harian, split acak, atau regresi level harga [4], [34], [35], [36], [38].
Rujukan metodologis terdekat adalah [37] (futures emas, 30 menit, selektif).

---

## D. Feature Justification

Prinsip umum (berlaku untuk semua fitur):

- **Kausal**: hanya `rolling`, `ewm`, dan `shift(+n)`; tidak ada `shift(-n)` di
  `technical/features.py`. *Experimentally validated* oleh
  `test_features_do_not_change_when_future_candles_change`.
- **Skala relatif**: dibagi ATR14 atau harga agar 2016 dan 2026 sebanding. *Design decision*.
- **Warm-up**: nilai awal NaN. XGBoost menangani NaN secara native; Logistic
  Regression memakai median **dari TRAIN saja**.
- **Gap**: indikator dihitung melintasi gap akhir pekan (seperti chart trader);
  fitur `bars_since_gap` memberi tahu model bahwa baru saja ada gap. *Design decision*.
- **Efektivitas**: belum ada fitur yang terbukti efektif pada data proyek ini.
  Kolom "Ablation" akan diisi setelah `technical.run ablation` dijalankan.

Notasi: `C, O, H, L` = close/open/high/low candle t; `ATR` = ATR14 Wilder.

| Kelompok | Fitur | Definisi | Parameter | Referensi metode | Alasan relevansi untuk XAUUSD | Risiko leakage | Ablation |
|---|---|---|---|---|---|---|---|
| price_action | `log_return_n` | ln(C_t / C_{t−n}) | n = 1, 3, 6, 12 | — | Momentum jangka sangat pendek (5–60 menit) | Tidak ada (shift positif) | Belum dijalankan |
| price_action | `return_n_atr` | (C_t − C_{t−n}) / ATR | n = 6, 24 | — | Gerak relatif terhadap volatilitas; sebanding dengan barrier label | Tidak ada | Belum |
| price_action | `candle_body_atr` | (C − O) / ATR | — | — | Kekuatan arah candle terakhir | Tidak ada | Belum |
| price_action | `candle_range_atr` | (H − L) / ATR | — | — | Ekspansi range | Tidak ada | Belum |
| price_action | `upper/lower_wick_ratio` | wick / (H − L) | — | — | Penolakan harga | Tidak ada | Belum |
| price_action | `close_position_in_range` | (C − L)/(H − L) | — | — | Posisi close dalam candle | Tidak ada | Belum |
| trend | `close_vs_ema_n_atr` | (C − EMA_n)/ATR | n = 12, 48, 288 | EMA standar | Trend 1 jam, 4 jam, 24 jam (dalam M5) | Tidak ada | Belum |
| trend | `ema_12_vs_48_atr` | (EMA12 − EMA48)/ATR | — | — | Persilangan MA | Tidak ada | Belum |
| trend | `ema_48_slope_atr` | (EMA48_t − EMA48_{t−6})/ATR | — | — | Kemiringan trend | Tidak ada | Belum |
| trend | `adx_14`, `di_difference_14` | ADX Wilder; +DI − −DI | 14 | [25] | Kekuatan trend vs ranging | Tidak ada | Belum |
| momentum | `rsi_14`, `rsi_14_change_3` | RSI Wilder; RSI_t − RSI_{t−3} | 14 | [25] | Kondisi jenuh beli/jual | Tidak ada | Belum |
| momentum | `macd_histogram_atr` | (MACD − signal)/ATR | 12, 26, 9 | [26] | Perubahan momentum | Tidak ada | Belum |
| momentum | `stochastic_k_14` | 100·(C − min L₁₄)/(max H₁₄ − min L₁₄) | 14 | — | Posisi dalam range 70 menit | Tidak ada | Belum |
| momentum | `donchian_position_48` | posisi C dalam range 48 bar | 48 (4 jam) | — | Breakout/posisi range | Tidak ada | Belum |
| volatility | `atr_14_pct` | ATR / C | 14 | [25] | Rezim volatilitas absolut | Tidak ada | Belum |
| volatility | `atr_14_vs_daily_mean` | ATR / mean(ATR, 288) | 288 = 24 jam | — | Volatilitas relatif vs hari terakhir | Tidak ada | Belum |
| volatility | `realized_vol_n` | std(log return, n) | 12, 48 | [19] | Volatilitas terealisasi | Tidak ada | Belum |
| volatility | `bollinger_width_20`, `bollinger_position_20` | lebar dan posisi band ±2σ | 20, 2 | [27] | Squeeze dan ekstrem | Tidak ada | Belum |
| volume | `tick_volume_z_48`, `tick_volume_ratio_12_288` | z-score log tick volume; rasio rata-rata | 48; 12/288 | — | Aktivitas pasar (jumlah tick, bukan volume kontrak) | Tidak ada | Belum |
| spread | `spread_atr`, `spread_z_288` | spread/ATR; z-score spread | 288 | — | Biaya relatif dan likuiditas | Tidak ada; spread candle t sudah diketahui saat close | Belum |
| time | `hour_sin/cos`, `day_of_week` | encoding siklik jam UTC; hari | — | [21] | Pola intraday | Tidak ada | Belum |
| time | `session_*` | jam lokal sesi (Tokyo 9–18, London 8–17, New York 8–17; DST via IANA) | — | [21] | Likuiditas antarsesi | Tidak ada | Belum |
| time | `bars_since_gap` | jumlah candle sejak gap terakhir (maks 288) | — | — | Efek pembukaan pasar | Tidak ada | Belum |

Total: **41 fitur dalam 7 kelompok** (lihat output `build_features`).
Pipeline lama memiliki 223 kandidat dengan banyak redundansi (mis. `return_n`,
`log_return_n`, dan `roc_n` adalah transformasi monoton dari besaran yang sama);
lihat D-06.

---

## E. Mathematical Foundations (XGBoost)

Referensi utama: Chen & Guestrin [1], Friedman [2]. Kode: `technical/xgboost_scratch.py`.

### E-1 Model aditif

$$\hat{y}_i^{(t)} = F_0 + \sum_{k=1}^{t} \eta\, f_k(x_i)$$

`f_k` adalah pohon regresi, `η` learning rate (shrinkage). Untuk K kelas,
setiap ronde menambah K pohon (satu per kelas) pada margin masing-masing.
Kode: `ScratchXGBoostClassifier.fit` → `margins[:, class_index] += self.learning_rate * ...`.

### E-2 Objektif teregularisasi

$$\mathcal{L}=\sum_{i=1}^{n}l(y_i,\hat{y}_i)+\sum_{k=1}^{K}\Omega(f_k)$$

$$\Omega(f)=\gamma T+\frac{1}{2}\lambda\sum_{j=1}^{T}w_j^2 \;\; (+\,\alpha\sum_j |w_j| \text{ untuk L1})$$

`T` = jumlah daun, `w_j` = bobot daun, `γ` = penalti per daun (`gamma`),
`λ` = L2 (`reg_lambda`), `α` = L1 (`reg_alpha`).

### E-3 Aproksimasi orde kedua

Ekspansi Taylor loss di sekitar prediksi ronde sebelumnya:

$$\tilde{\mathcal{L}}^{(t)}=\sum_i\left[g_i f_t(x_i)+\frac{1}{2}h_i f_t(x_i)^2\right]+\Omega(f_t)$$

dengan $g_i=\partial_{\hat y} l$ dan $h_i=\partial^2_{\hat y} l$. Untuk
daun j dengan himpunan sampel $I_j$, definisikan $G_j=\sum_{i\in I_j} g_i$ dan
$H_j=\sum_{i\in I_j} h_i$. Objektif menjadi jumlah fungsi kuadrat yang terpisah
per daun: $G_j w_j + \tfrac12 (H_j+\lambda) w_j^2 + \alpha|w_j|$.

### E-4 Bobot daun optimal

Turunkan terhadap $w_j$ dan samakan dengan nol (α = 0):

$$w_j^*=-\frac{G_j}{H_j+\lambda}$$

Dengan L1, solusinya memakai soft-thresholding
$T_\alpha(G)=\operatorname{sign}(G)\max(|G|-\alpha,0)$:
$w_j^*=-T_\alpha(G_j)/(H_j+\lambda)$.
Kode: `leaf_weight`, `soft_threshold`.
Tes: `test_leaf_weight_is_regularized_newton_step`,
`test_leaf_weight_minimizes_second_order_objective` (minimum dicek dengan grid numerik).

### E-5 Split gain

Substitusi $w^*$ menghasilkan skor struktur $-\tfrac12 \sum_j T_\alpha(G_j)^2/(H_j+\lambda) + \gamma T$.
Penurunan loss akibat memecah satu daun menjadi kiri (L) dan kanan (R):

$$Gain=\frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma$$

Split hanya dibuat jika Gain > 0 (ini sekaligus menjadi **pruning**), dan
$H_L, H_R \geq$ `min_child_weight`.
Kode: `split_gain`, `structure_score`, `GradientTree.find_best_split`.
Tes: `test_split_gain_matches_hand_calculation`, `test_gamma_prunes_weak_split`.

> **Catatan implementasi (Experimentally validated, EXP-S05):** kode ini
> mengikuti rumus paper (dengan faktor ½). Library XGBoost 3.4.1 berperilaku
> seolah membandingkan selisih skor **tanpa** faktor ½ dengan `gamma`.
> Akibatnya `gamma = g` di library menghasilkan model yang sama dengan
> `gamma = g/2` di implementasi ini. Bila `gamma = 0` (konfigurasi proyek),
> keduanya identik. Perintah `compare-scratch` melakukan konversi ini otomatis.

### E-6 Softmax multikelas

$$p_{ik}=\frac{e^{z_{ik}-\max_j z_{ij}}}{\sum_{j} e^{z_{ij}-\max_j z_{ij}}}$$

Pengurangan `max` menjaga stabilitas numerik (mencegah overflow `exp`).
Kode: `softmax`. Tes: `test_softmax_rows_sum_to_one_and_is_stable_for_large_margins`.

### E-7 Gradien cross-entropy

Loss $l_i=-\sum_k y_{ik}\log p_{ik}$ (y one-hot). Turunan terhadap margin kelas k:

$$g_{ik}=\frac{\partial l_i}{\partial z_{ik}}=p_{ik}-y_{ik}$$

### E-8 Hessian diagonal

$$\frac{\partial^2 l_i}{\partial z_{ik}^2}=p_{ik}(1-p_{ik})$$

Hessian penuh juga memiliki elemen silang $-p_{ik}p_{ij}$; XGBoost (dan kode
ini) **hanya memakai diagonal**, sehingga setiap kelas diperlakukan sebagai
masalah regresi terpisah per ronde. XGBoost mengalikan diagonal dengan **2**:
$h_{ik}=\max(2p_{ik}(1-p_{ik}), 10^{-16})$. Faktor yang sama dipakai di sini
agar sebanding. Batas bawah $10^{-16}$ mencegah pembagian dengan nol.
Kode: `softmax_gradient_hessian`.
Tes: `test_gradient_and_hessian_match_finite_differences` (gradien = beda hingga;
Hessian = 2 × beda hingga orde kedua).

### E-9 Prediksi awal (base score)

Default: $F_{0k}=\log\pi_k-\frac1K\sum_j\log\pi_j$, dengan $\pi_k$ proporsi kelas
di TRAIN, sehingga $\operatorname{softmax}(F_0)=\pi$. Opsi `base_score="zero"`
memberi probabilitas awal 1/K.
**Experimentally validated:** intercept bawaan XGBoost 3.4.1 untuk
`multi:softprob` cocok dengan rumus ini (EXP-S02). Kode: `compute_initial_margin`.

### E-10 Kandidat split dan nilai hilang

- Kandidat split: kuantil fitur pada TRAIN (`QuantileBinner`, maksimal
  `max_bin` bin). XGBoost "hist" memakai weighted quantile sketch (berbobot
  Hessian) [1], jadi titik potong bisa berbeda pada fitur kontinu.
- Nilai NaN masuk bin khusus. Setiap split mencoba NaN ke kiri dan ke kanan
  lalu menyimpan arah terbaik (default direction ala sparsity-aware split [1]).
- Histogram G dan H dihitung dengan `np.bincount`.

### E-11 Yang **tidak** diimplementasikan (tidak setara penuh dengan XGBoost)

Weighted quantile sketch, subsampling baris/kolom, paralelisasi, cache-aware
access, out-of-core, monotone/interaction constraint, `max_delta_step`, dan trik
subtraksi histogram. Implementasi ini **bukan pengganti** XGBoost resmi; ia
dipakai untuk verifikasi pemahaman matematis dan perbandingan terkendali.

---

## F. Experimental Evidence

### F.1 Eksperimen yang sudah dijalankan

| ID | Data & periode | Model | Konfigurasi | Seed | Protokol | Hasil | Artefak | Kesimpulan & keterbatasan |
|---|---|---|---|---|---|---|---|---|
| EXP-S01 | Sintetis: 600 baris, 3 fitur diskret (nilai 0–9), 3 kelas | XGBoost 3.4.1 vs scratch | 20 ronde, depth 3, η 0,3, λ 1, min_child_weight 1, max_bin 64, tanpa subsampling | 0 | Prediksi pada data training yang sama | Selisih probabilitas maksimum < 1,1 × 10⁻⁷ (base_score 0,5/zero dan default/prior) | `tests/test_xgboost_scratch.py::test_matches_official_xgboost_when_candidate_splits_are_identical` | Saat kandidat split sama, formula gradien, Hessian, gain, bobot daun, dan base score identik dengan library. Hanya data sintetis. |
| EXP-S02 | Sama dengan EXP-S01 | XGBoost 3.4.1 | default | 0 | Membaca `base_score` dari konfigurasi booster | Intercept = log(prior) yang di-center: [−0,3619; 0,2218; 0,1400] vs rumus [−0,3619; 0,2218; 0,1400] | Dijalankan manual, 29 Okt 2026 | Dasar E-9. |
| EXP-S03 | Sintetis: 20.000 baris, 8 fitur kontinu Normal(0,1), 5% NaN; 15.000 train / 5.000 uji | XGBoost 3.4.1 vs scratch | 100 ronde, depth 4, η 0,1, λ 2, min_child_weight 5, max_bin 64 | 1 | Train/uji terpisah | Rata-rata selisih probabilitas 0,0106; maksimum 0,162; argmax sama 97,86%; akurasi uji 0,6400 (library) vs 0,6428 (scratch) | Dijalankan manual, 29 Okt 2026 | Perbedaan berasal dari titik potong kuantil (E-10). Hanya data sintetis. |
| EXP-S05 | Sama dengan EXP-S01 | XGBoost 3.4.1 vs scratch | gamma library = 2 dan 5 | 0 | Bandingkan scratch dengan gamma = g dan g/2 | gamma = g: selisih maks 0,12 dan 0,091; gamma = g/2: selisih maks ≤ 1,0 × 10⁻⁷ | `tests/test_xgboost_scratch.py::test_gamma_convention_matches_library_with_half_factor` | Konvensi gamma berbeda faktor 2 (catatan E-5). |
| EXP-S04 | Sintetis: 700.000 × 41, label acak | Scratch vs library | depth 4, max_bin 64, 3 ronde | 0 | Waktu training per ronde, 1 mesin (12 core; library n_jobs = 4) | Scratch ≈ 3,4 s/ronde; library ≈ 0,32 s/ronde | Dijalankan manual, 29 Okt 2026 | Estimasi rekayasa saja: 150 ronde scratch ≈ 9 menit per fold. Bukan benchmark formal. |

### F.2 Bukti dari pipeline generasi sebelumnya (dihapus, ada di riwayat git)

| ID | Sumber | Temuan | Kategori | Dampak pada desain |
|---|---|---|---|---|
| EXP-L01 | `technical/artifacts/direction_v3/audit_remediation_2026-10-10.md` (commit `0a849ab`) | Dengan label "spread + 1 bps" dan horizon 15 menit, distribusi label 2026: BUY 23.392, SELL 23.307, NO_TRADE 3.954 (≈ 8%) | Dataset-derived | Label lama memaksa model menebak arah hampir di setiap candle → D-03 |
| EXP-L02 | Sumber sama | Policy berlapis (classifier + regressor + quantile interval + target precision 90%) menghasilkan **0 sinyal BUY/SELL** pada fold 2024, 2025, dan 2026 | Dataset-derived | Decision layer disederhanakan → D-07, D-08 |
| EXP-L03 | `technical/artifacts/selective_v2/experiment_report.md` (commit `0a849ab`) | RMSE regressor return 2026 = 19,662 bps vs baseline zero-return 19,660 bps | Dataset-derived | Regressor tidak menambah nilai → dihapus (D-07) |

### F.3 Eksperimen data riil yang direncanakan (belum dijalankan)

| ID | Perintah | Tujuan | Status |
|---|---|---|---|
| EXP-R01 | `python -m technical.run label-scan` | Memilih horizon (3/6/12) dan k (0,5–2,0) berdasarkan distribusi label 2016–2025 | **Belum dijalankan** |
| EXP-R02 | `python -m technical.run walk-forward --models always_no_trade,persistence,logistic,xgboost` | Perbandingan baseline vs XGBoost, fold 2021–2025 | **Belum dijalankan** |
| EXP-R03 | `python -m technical.run compare-scratch --year 2025` | XGBoost resmi vs scratch pada data riil | **Belum dijalankan** |
| EXP-R04 | `python -m technical.run ablation` | Kontribusi tiap kelompok fitur | **Belum dijalankan** |
| EXP-R05 | `python -m technical.run final-test --model xgboost --confirm` | Uji retrospective 2026 setelah desain dikunci | **Belum dijalankan** |

Setiap run menulis `runs/<waktu>_<perintah>_<hash>/run.json` (config lengkap,
hash config, commit git, versi paket, laporan data) dan metrik. Hasil baru boleh
ditambahkan ke tabel F.1 hanya setelah file tersebut benar-benar ada.

---

## G. Decision Log

Tanggal semua keputusan: 29 Oktober 2026 (refactor), kecuali disebut lain.

| ID | Keputusan | Alternatif | Bukti | Alasan | Konsekuensi | Status |
|---|---|---|---|---|---|---|
| D-01 | Timeframe utama **M5** | M1, M15, kombinasi | [18], [19], [20]; B.1 (M1 belum dikumpulkan) | M1 lebih bising dan biaya spread relatif lebih besar terhadap gerak per bar; M5 adalah frekuensi yang lazim dipakai literatur realized volatility. M15 mengurangi sampel tanpa alasan kuat. | M1 hanya untuk ablation masa depan | implemented |
| D-02 | Horizon default **6 bar (30 menit)** | 3 bar (15 menit, dipakai pipeline lama), 12 bar (60 menit) | EXP-L01 (15 menit + threshold kecil → NO_TRADE ≈ 8%) | Horizon lebih panjang membuat gerak signifikan terhadap spread; dipilih ulang dari EXP-R01 **sebelum** model dilatih | Konfigurasi dapat berubah setelah EXP-R01 | proposed (menunggu EXP-R01) |
| D-03 | Label **triple-barrier** dengan barrier = k·ATR14 + spread, k default 1,0 | Fixed return threshold (lama), fixed horizon + ATR | [3], [39]; EXP-L01 | Menyesuaikan volatilitas, memperhitungkan spread, dan NO_TRADE muncul secara alami bila tidak ada gerak berarti. Sentuhan dua sisi di candle yang sama → NO_TRADE (konservatif karena urutan intra-candle tidak diketahui). | Fixed horizon tetap tersedia sebagai pembanding | implemented (k: proposed) |
| D-04 | XGBoost resmi sebagai model referensi utama + XGBoost from scratch | LightGBM [9], CatBoost [10], Random Forest | [1], [5], [7], [8] | Data berupa fitur tabular; model pohon kuat pada tabular; mandat penelitian. LightGBM/CatBoost tidak ditambah untuk menghindari dependency baru tanpa bukti manfaat. | — | implemented |
| D-05 | LSTM/GRU **tidak** diimplementasikan pada tahap ini | LSTM [6], CNN-BiLSTM [34] | [6] (bukti tandingan), [7], [8] | Fitur sudah merangkum sekuens; biaya komputasi lebih besar; baseline tabular harus kuat dulu | Dapat diuji sebagai pembanding setelah EXP-R02 | rejected (untuk tahap ini) |
| D-06 | 41 fitur ringkas dalam 7 kelompok, seleksi lewat **ablation kelompok** | 223 kandidat + Quality Filter + ReliefF + RFECV (lama) | [33], [36] (seleksi penting); D.1 | Pipeline lama mahal dan tidak stabil, banyak fitur redundan. Ablation kelompok lebih mudah ditafsirkan. | Belum ada seleksi otomatis tingkat fitur | implemented |
| D-07 | Hapus regressor return dan quantile interval dari decision layer | Pertahankan (lama) | EXP-L02, EXP-L03 | Tidak menambah nilai prediktif, menambah kompleksitas, dan memicu 0 sinyal | Decision layer = satu classifier + kalibrasi + threshold | implemented |
| D-08 | Satu model multikelas + **temperature scaling** + threshold per arah | Dua tahap (trade eligibility → arah) / meta-labeling [3], [40]; ensemble | [11], [12], [13], [15], [37] | Paling sederhana dan dapat diaudit; dua tahap dan ensemble dapat ditambahkan setelah baseline terbukti | Dua tahap = future work | implemented |
| D-09 | Threshold dipilih pada blok THRESHOLD: F1 tertinggi dengan syarat precision ≥ 1,2 × prevalensi dan ≥ 30 sinyal | Target precision absolut 90% (lama) | EXP-L02 | Model acak memiliki precision = prevalensi kelas; syarat lift memastikan sinyal lebih baik dari acak. Nilai 1,2 dan 30 adalah pilihan rancangan. | Arah yang tidak memenuhi syarat dinonaktifkan dan dilaporkan | implemented (nilai: design decision) |
| D-10 | Walk-forward expanding window, tahun validasi 2021–2025; blok TRAIN / CALIBRATION (Jul–Sep) / THRESHOLD (Okt–Des) / EVALUATION | Random split; K-fold | [3], [16], [17] | Urutan waktu dipertahankan; setiap fitting hanya melihat masa lalu | Kalibrasi dan threshold hanya memakai 3 bulan data | implemented |
| D-11 | Purging berdasarkan `label_end_time` + embargo 6 bar | Tanpa purging | [3] §7.4 | Label horizon H overlap dengan blok berikutnya bila tidak di-purge | Sedikit data di batas blok dibuang | implemented |
| D-12 | 2026 dilabeli **retrospective OOS**; final-test wajib `--confirm`; usulan uji forward baru setelah config dikunci | Menyebut 2026 sebagai blind test | A-05 | Kejujuran metodologis | Klaim generalisasi harus hati-hati | implemented |
| D-13 | Early stopping XGBoost memakai blok CALIBRATION | Holdout terpisah dari akhir TRAIN | — | Menjaga TRAIN tetap utuh dan tidak menyentuh THRESHOLD/EVALUATION | Kalibrasi sedikit optimistis karena blok yang sama dipakai untuk memilih jumlah pohon | implemented (design decision) |
| D-14 | Gaya kode eksplisit (loop dan if/else jelas, tanpa one-liner padat); aturan ruff SIM102/108/114 dinonaktifkan untuk `technical/` | Gaya ringkas | Permintaan pemilik proyek | Keterbacaan untuk audit akademik | — | implemented |
| D-15 | Hapus forecast XGBoost rekursif harga harian (`analytics`), folder `docs/`, MkDocs, dan generator referensi kode; dependency `mkdocs` dan `plotly` dibuang | Pertahankan | Forecast rekursif level harga tidak sesuai rumusan masalah klasifikasi (A-01) | Satu sumber dokumentasi: README + SoT + README per modul | Panduan collector dipindah ke `market/README.md`, panduan EDA ke `analytics/README.md` | implemented |
| D-16 | Backtesting memakai prediksi out-of-sample per fold walk-forward (2021–2025) dan fold final (2026), disimpan di `runs/<id>/backtest/`; akurasi klasifikasi, ketepatan arah, dan simulasi P&L dilaporkan terpisah | Melatih ulang model untuk backtest; satu metrik gabungan | A-06, [31], [32] | Setiap prediksi dibuat model yang tersedia pada waktu itu; metrik yang berbeda tidak dicampur | Simulasi tingkat candle: bid OHLC, ask = bid + spread, spread keluar = spread masuk, tanpa slippage/komisi, posisi boleh tumpang tindih (docs/METHODOLOGY.md §15) | implemented |
| D-17 | Hasil sinyal dinilai dengan definisi label (CORRECT/INCORRECT/NEUTRAL), bukan sekadar tanda perubahan harga | Kenaikan harga apa pun = BUY benar | D-03 | Konsisten dengan target training; kenaikan di bawah barrier tidak cukup menutup biaya | Ketepatan arah tanpa ambang tetap ditampilkan sebagai ukuran terpisah | implemented |
| D-18 | Horizon tetap 6 bar (30 menit), bukan 15 menit | 3 bar (15 menit) | D-02, EXP-L01 | Pada 15 menit gerak tipikal terlalu kecil terhadap spread; horizon final tetap ditentukan EXP-R01 (`label-scan`) | Dashboard membaca horizon dari `run.json`, tidak di-hardcode | implemented (menunggu EXP-R01) |
| D-19 | Dashboard live: MT5 → LiveService → WebSocket; tanpa MT5 tampil "disconnected" dan grafik historis PostgreSQL berlabel, tanpa data dummy; prediksi live memakai `live_bundle.joblib` dari final-test dan dicatat terpisah (`runs/live/`) | Data simulasi saat MT5 tidak ada | — | Tidak menyesatkan pengguna; prediksi live tidak tercampur dengan backtest | Paket MetaTrader5 hanya Windows; epoch MT5 diperlakukan sebagai UTC seperti collector | implemented |
| D-20 | Ensemble soft voting berbobot: `xgboost` + `random_forest` + `logistic`; tiap anggota dikalibrasi (temperature) lalu bobot dipilih dari grid simpleks 0,05 yang meminimalkan log loss pada blok CALIBRATION; `xgboost_balanced` sebagai pembanding class weight | Hard voting; stacking dengan meta-learner; SMOTE | Materi IF2116 Pertemuan 5 (`materi/W05`) | Soft voting butuh probabilitas sebanding; bobot dari blok di luar TRAIN mencegah kebocoran stacking in-sample; grid memuat anggota tunggal sehingga campuran tidak lebih buruk dari anggota terbaik di blok itu; SMOTE tidak dipakai karena label deret waktu tumpang tindih | CALIBRATION kini dipakai untuk early stopping, temperatur, dan bobot; threshold dan evaluasi tetap bersih | implemented (belum dijalankan pada data riil) |
| D-21 | Feature selection di dalam setiap fold (`xgboost_selected`): filter (nilai hilang, konstan, audit kebocoran, redundansi \|Spearman\| > 0,95 dengan mutual information) di TRAIN, gain XGBoost sebagai pembanding, permutation importance (kenaikan log loss) di CALIBRATION; ablation kelompok tetap sebagai wrapper | Seleksi sekali di awal pada seluruh data; ReliefF/RFECV (pipeline lama); gain XGBoost sebagai dasar keputusan | Materi IF2116 Pertemuan 7 (`materi/W07`) | Seleksi di luar fold membocorkan informasi; gain bias pada fitur berkardinalitas tinggi dan fitur yang berkorelasi, sehingga keputusan memakai permutation importance pada data di luar TRAIN | Melengkapi D-06; CALIBRATION dipakai juga untuk permutation importance; threshold dan evaluasi tetap bersih | implemented (belum dijalankan pada data riil) |

---

## H. References

[1] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785–794, doi: 10.1145/2939672.2939785. — VERIFIED

[2] J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *Ann. Statist.*, vol. 29, no. 5, pp. 1189–1232, 2001, doi: 10.1214/aos/1013203451. — VERIFIED

[3] M. López de Prado, *Advances in Financial Machine Learning*. Hoboken, NJ, USA: Wiley, 2018, ISBN 978-1-119-48208-6 (Ch. 3 §3.4 Triple-Barrier, §3.6 Meta-Labeling; Ch. 7 §7.4 Purging, Embargo). — VERIFIED

[4] P. Sadorsky, "Predicting gold and silver price direction using tree-based classifiers," *J. Risk Financial Manag.*, vol. 14, no. 5, Art. 198, 2021, doi: 10.3390/jrfm14050198. — VERIFIED

[5] C. Krauss, X. A. Do, and N. Huck, "Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500," *Eur. J. Oper. Res.*, vol. 259, no. 2, pp. 689–702, 2017, doi: 10.1016/j.ejor.2016.10.031. — VERIFIED

[6] T. Fischer and C. Krauss, "Deep learning with long short-term memory networks for financial market predictions," *Eur. J. Oper. Res.*, vol. 270, no. 2, pp. 654–669, 2018, doi: 10.1016/j.ejor.2017.11.054. — VERIFIED

[7] L. Grinsztajn, E. Oyallon, and G. Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?," in *Adv. Neural Inf. Process. Syst. 35 (Datasets and Benchmarks Track)*, 2022, doi: 10.52202/068431-0037. — VERIFIED

[8] R. Shwartz-Ziv and A. Armon, "Tabular data: Deep learning is not all you need," *Inf. Fusion*, vol. 81, pp. 84–90, 2022, doi: 10.1016/j.inffus.2021.11.011. — VERIFIED

[9] G. Ke *et al.*, "LightGBM: A highly efficient gradient boosting decision tree," in *Adv. Neural Inf. Process. Syst. 30*, 2017. — VERIFIED (halaman belum dikonfirmasi)

[10] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, "CatBoost: Unbiased boosting with categorical features," in *Adv. Neural Inf. Process. Syst. 31*, 2018. — VERIFIED (halaman belum dikonfirmasi)

[11] C. K. Chow, "On optimum recognition error and reject tradeoff," *IEEE Trans. Inf. Theory*, vol. 16, no. 1, pp. 41–46, 1970, doi: 10.1109/TIT.1970.1054406. — VERIFIED

[12] Y. Geifman and R. El-Yaniv, "Selective classification for deep neural networks," in *Adv. Neural Inf. Process. Syst. 30*, 2017. — VERIFIED (halaman belum dikonfirmasi)

[13] A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning," in *Proc. 22nd Int. Conf. Machine Learning (ICML)*, 2005, pp. 625–632, doi: 10.1145/1102351.1102430. — VERIFIED

[14] J. C. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," in *Advances in Large Margin Classifiers*, A. J. Smola, P. Bartlett, B. Schölkopf, and D. Schuurmans, Eds., 1999, pp. 61–74. — PARTIAL (penerbit dan halaman dari sumber sekunder)

[15] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. 34th Int. Conf. Machine Learning (ICML)*, PMLR vol. 70, 2017, pp. 1321–1330. — VERIFIED

[16] C. Bergmeir and J. M. Benítez, "On the use of cross-validation for time series predictor evaluation," *Inf. Sci.*, vol. 191, pp. 192–213, 2012, doi: 10.1016/j.ins.2011.12.028. — VERIFIED

[17] L. J. Tashman, "Out-of-sample tests of forecasting accuracy: An analysis and review," *Int. J. Forecast.*, vol. 16, no. 4, pp. 437–450, 2000, doi: 10.1016/S0169-2070(00)00065-0. — VERIFIED

[18] Y. Aït-Sahalia, P. A. Mykland, and L. Zhang, "How often to sample a continuous-time process in the presence of market microstructure noise," *Rev. Financ. Stud.*, vol. 18, no. 2, pp. 351–416, 2005, doi: 10.1093/rfs/hhi016. — VERIFIED

[19] T. G. Andersen, T. Bollerslev, F. X. Diebold, and P. Labys, "The distribution of realized exchange rate volatility," *J. Amer. Statist. Assoc.*, vol. 96, no. 453, pp. 42–55, 2001, doi: 10.1198/016214501750332965. — VERIFIED (penggunaan return 5 menit dikonfirmasi pada versi working paper)

[20] P. R. Hansen and A. Lunde, "Realized variance and market microstructure noise," *J. Bus. Econ. Statist.*, vol. 24, no. 2, pp. 127–161, 2006, doi: 10.1198/073500106000000071. — VERIFIED

[21] T. G. Andersen and T. Bollerslev, "Intraday periodicity and volatility persistence in financial markets," *J. Empir. Finance*, vol. 4, no. 2–3, pp. 115–158, 1997, doi: 10.1016/S0927-5398(97)00004-2. — VERIFIED

[22] A. W. Lo, H. Mamaysky, and J. Wang, "Foundations of technical analysis: Computational algorithms, statistical inference, and empirical implementation," *J. Finance*, vol. 55, no. 4, pp. 1705–1765, 2000, doi: 10.1111/0022-1082.00265. — VERIFIED

[23] W. Brock, J. Lakonishok, and B. LeBaron, "Simple technical trading rules and the stochastic properties of stock returns," *J. Finance*, vol. 47, no. 5, pp. 1731–1764, 1992, doi: 10.1111/j.1540-6261.1992.tb04681.x. — VERIFIED

[24] C.-H. Park and S. H. Irwin, "What do we know about the profitability of technical analysis?," *J. Econ. Surveys*, vol. 21, no. 4, pp. 786–826, 2007, doi: 10.1111/j.1467-6419.2007.00519.x. — VERIFIED

[25] J. W. Wilder Jr., *New Concepts in Technical Trading Systems*. Greensboro, NC, USA: Trend Research, 1978. — PARTIAL (ISBN dan parameter 14 dari sumber sekunder)

[26] G. Appel, *Technical Analysis: Power Tools for Active Investors*. Upper Saddle River, NJ, USA: FT Press/Prentice Hall, 2005, ISBN 978-0-13-147902-9. — PARTIAL (parameter 12/26/9 belum dikonfirmasi di sumber)

[27] J. Bollinger, *Bollinger on Bollinger Bands*. New York, NY, USA: McGraw-Hill, 2001, ISBN 978-0-07-137368-5. — PARTIAL (parameter 20/2 belum dikonfirmasi di sumber)

[28] T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," *PLoS ONE*, vol. 10, no. 3, e0118432, 2015, doi: 10.1371/journal.pone.0118432. — VERIFIED

[29] D. G. Baur and B. M. Lucey, "Is gold a hedge or a safe haven? An analysis of stocks, bonds and gold," *Financ. Rev.*, vol. 45, no. 2, pp. 217–229, 2010, doi: 10.1111/j.1540-6288.2010.00244.x. — VERIFIED

[30] F. A. O'Connor, B. M. Lucey, J. A. Batten, and D. G. Baur, "The financial economics of gold — A survey," *Int. Rev. Financ. Anal.*, vol. 41, pp. 186–205, 2015, doi: 10.1016/j.irfa.2015.07.005. — VERIFIED

[31] D. H. Bailey, J. M. Borwein, M. López de Prado, and Q. J. Zhu, "The probability of backtest overfitting," *J. Comput. Finance*, vol. 20, no. 4, 2017, doi: 10.21314/JCF.2016.322. — PARTIAL (halaman belum dikonfirmasi)

[32] D. H. Bailey, J. M. Borwein, M. López de Prado, and Q. J. Zhu, "Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance," *Notices Amer. Math. Soc.*, vol. 61, no. 5, pp. 458–471, 2014, doi: 10.1090/noti1105. — VERIFIED (halaman akhir belum dikonfirmasi)

[33] M. M. Mamoudan, A. Ostadi, N. Pourkhodabakhsh, A. M. Fathollahi-Fard, and F. Soleimani, "Hybrid neural network-based metaheuristics for prediction of financial markets: A case study on global gold market," *J. Comput. Des. Eng.*, vol. 10, no. 3, pp. 1110–1125, 2023, doi: 10.1093/jcde/qwad039. — VERIFIED

[34] A. Amini and R. Kalantari, "Gold price prediction by a CNN-Bi-LSTM model along with automatic parameter tuning," *PLoS ONE*, vol. 19, no. 3, e0298426, 2024, doi: 10.1371/journal.pone.0298426. — VERIFIED

[35] M. Arif *et al.*, "Multivariate LSTM-based intraday gold price prediction with rolling time series validation," *J. Masyarakat Informatika*, vol. 17, no. 1, pp. 26–38, 2026, doi: 10.14710/jmasif.17.1.78091. — VERIFIED

[36] G. Taneva-Angelova, S. Raychev, and G. Ilieva, "A framework for gold price prediction combining classical and intelligent methods with financial, economic, and sentiment data fusion," *Int. J. Financial Stud.*, vol. 13, no. 2, Art. 102, 2025, doi: 10.3390/ijfs13020102. — VERIFIED

[37] N. Chalkidis and R. Savani, "Trading via selective classification," in *Proc. 2nd ACM Int. Conf. AI in Finance (ICAIF '21)*, 2021, pp. 1–9, doi: 10.1145/3490354.3494379. — VERIFIED

[38] S. Devarajula, V. Milke, and C. Luca, "Using neural network architectures for intraday trading in the gold market," in *Proc. 15th Int. Conf. Agents and Artificial Intelligence (ICAART)*, vol. 3, 2023, pp. 885–892, doi: 10.5220/0011794400003393. — VERIFIED

[39] P. Grądzki, P. Wójcik, and S. Lessmann, "Algorithmic crypto trading using information-driven bars, triple barrier labeling and deep learning," *Financial Innovation*, vol. 11, Art. 136, 2025, doi: 10.1186/s40854-025-00866-w. — VERIFIED

[40] J. F. Joubert, "Meta-labeling: Theory and framework," *J. Financial Data Sci.*, vol. 4, no. 3, pp. 31–44, 2022, doi: 10.3905/jfds.2022.1.098. — PARTIAL (abstrak belum dibaca)

[41] E. B. Wilson, "Probable inference, the law of succession, and statistical inference," *J. Amer. Statist. Assoc.*, vol. 22, no. 158, pp. 209–212, 1927, doi: 10.1080/01621459.1927.10502953. — VERIFIED
