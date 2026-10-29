# Technical — XAUUSD M5

Pipeline riset dengan XGBoost untuk **SELL=0 / NO_TRADE=1 / BUY=2**.
Data bersumber dari `market.processed_m5` melalui transaksi PostgreSQL read-only.
Seluruh hasil ditulis lokal. Base timeframe **M5**, sedangkan horizon default
**3 candle = 15 menit**; model tidak memakai resampling M15/H1.

Bahan laporan eksperimen bagian 4.1–4.6 tersedia pada
[README laporan baseline](README_laporan_baseline.md), dengan konfigurasi,
strategi training/validasi, hasil prediksi, dan pembandingan yang sudah tersedia.
Lanjutannya tersedia pada [README laporan optimasi](README_laporan_optimasi.md),
berisi metrik, hasil tuning, interpretasi singkat, dan temuan awal.

## Struktur ringkas

```text
technical/
  config.py               parameter fitur, horizon, seleksi dan biaya komputasi
  data_preparation.py     loader, preparation, quality report, label, folds dan CV
  xgboost_model.py        fitur, seleksi, classifier/regressor, evaluation dan CLI
  dashboard.py            builder HTML offline
  dashboard.html          template UI broker; buka HTML hasil di artifacts
  artifacts/
    direction_v3/
      data_quality.json
      fold_2020/ ... fold_2025/ final_2026/
    walk_forward/         hasil versi awal untuk pembanding
  legacy_teknikal.zip      salinan workflow/artefak/test lama untuk pemulihan
```

Kedua file workflow berada langsung di `technical/`, tanpa subfolder `data/`
dan tanpa file `modeling.py`. Keduanya memakai penanda `# %%`
untuk section/cell seperti notebook yang bisa dinavigasi atau dijalankan melalui
editor yang mendukung Python cells. Import modul hanya mendefinisikan fungsi;
training dijalankan melalui `run_pipeline()` atau CLI.

Section `data_preparation.py`: import → load data → preparation/quality report →
label → walk-forward folds → purged CV. Section `xgboost_model.py`: import →
feature engineering → feature selection → modeling/tuning/calibration/policy →
prediction/evaluation → pipeline/artifacts/dashboard → CLI.

`xgboost_model.py` menggunakan XGBoost untuk klasifikasi dan regresi.
Konfigurasi dan dashboard tetap berada di `technical/`. Artefak model dan audit
tetap dipisahkan per fold. Untuk membuka bundle model lama maupun baru gunakan:

```python
from pathlib import Path
from technical.xgboost_model import load_model_bundle

bundle = load_model_bundle(
    Path("technical/artifacts/walk_forward/final_2026/model_classifier.joblib")
)
```

Pembaca ini menyelesaikan path class kalibrasi lama di memory; artifact tidak
ditulis ulang. Bundle baru memakai path class `technical.xgboost_model`.

## Klasifikasi dan regresi

- **Klasifikasi (`XGBClassifier`)** memprediksi probabilitas SELL/NO_TRADE/BUY.
  Label berasal dari future return relatif terhadap threshold spread + minimum
  edge + slippage konfigurasi. Probability dikalibrasi pada CALIBRATION; threshold confidence dipilih
  pada POLICY secara terpisah per arah. BUY/SELL harus searah dengan regressor,
  melampaui biaya, dan lolos interval uncertainty. Target precision default 90%
  dikonfirmasi pada bagian POLICY berikutnya; arah yang gagal menjadi NO_TRADE.
- **Regresi (`XGBRegressor`)** memprediksi `future_return = future_close/close - 1`.
  Target fitting dikalikan 10.000 (basis points); hasil prediksi dibagi 10.000
  kembali menjadi return desimal. `predicted_future_close = close * (1 + return)`.
  Ini prediksi direct return H candle M5, bukan recursive forecast harian.

Horizon default keduanya 3 candle M5 (15 menit). Tuning classifier memakai
macro F1; regressor memakai RMSE return pada TUNE. Keduanya di-refit TRAIN+TUNE
sebelum frozen TEST. Model disimpan sebagai `model_classifier.joblib` dan
`model_regressor.joblib` per fold; actual pending/gap dikeluarkan dari metrik.

Seleksi fitur regressor terpisah dari classifier: quality filter dan ranking gain
TRAIN untuk target return, maksimal 40 fitur. Regresi membandingkan squared loss
dengan pseudo-Huber loss, memakai early stopping dan return shrinkage yang dipilih
pada TUNE. Quantile regressor memberi interval yang diperlebar bila diperlukan
oleh residual CALIBRATION. Penjelasan lengkap, trade-off precision/F1, dan sumber
primer ada di [metodologi selective XGBoost](../docs/technical-selective.md).

## Membaca proses dan feature engineering

Komentar di source memakai penanda `FE-*` untuk kelompok feature engineering,
`SELECT-*` untuk seleksi fitur, `MODEL-*` untuk fitting/tuning/calibration/policy,
`EVAL-*` untuk evaluasi, dan `PIPE-*`/`FOLD-*` untuk orkestrasi. Pada
`data_preparation.py`, penanda `PREP-*` menjelaskan validasi dan pembersihan data.

Feature engineering mengubah candle OHLC, tick volume, spread dan waktu yang
sudah dipersiapkan menjadi kolom input numerik `X`. Contoh:
`return_3 = close / close.shift(3) - 1` mengukur perubahan harga selama tiga
candle M5 (15 menit jika tidak ada gap). `build_features()` menghasilkan
matriks fitur `float32` dan manifest rumus/parameter/provenance. Setiap fitur
memakai informasi candle saat ini yang sudah tutup atau masa lalu.

| Penanda | Kelompok fitur | Yang diukur dan contoh output |
|---|---|---|
| FE-01 | Return | Perubahan harga relatif/logaritmik dan jumlah return rolling: `return_3`, `log_return_12`, `rolling_return_24` |
| FE-02 | Bentuk candle | Range, body, wick, arah candle, posisi close dan perbandingan dengan candle sebelumnya: `body_to_range`, `candle_direction` |
| FE-03 | SMA/EMA | Jarak harga dari moving average, slope dan perubahan MA: `close_vs_ema_24`, `sma_slope_48` |
| FE-04 | MACD | Selisih EMA cepat/lambat, signal, histogram dan perubahannya relatif ke harga: `macd_12_26`, `macd_hist_12_26_9` |
| FE-05 | RSI dan ATR | Momentum RSI, jarak RSI dari 50, delta RSI; ATR relatif terhadap close, perubahan dan rasio median: `rsi_14`, `atr_pct_14` |
| FE-06 | Stochastic | Posisi close terhadap rentang high-low (%K) dan rata-ratanya (%D): `stoch_k_14`, `stoch_d_14_3` |
| FE-07 | ROC | Momentum dalam persentase perubahan close: `roc_6`, `roc_24` |
| FE-08 | Volatilitas | Standar deviasi return dan volatilitas high-low Parkinson: `rolling_std_24`, `high_low_volatility_24` |
| FE-09 | Bollinger Bands | Lebar band, posisi close dalam band dan jarak ke mid: `bb_width_20_2`, `bb_position_20_2` |
| FE-10 | Donchian/breakout | Posisi dalam channel, jarak ke high/low dan breakout channel sebelumnya: `donchian_position_24`, `range_breakout_24` |
| FE-11 | Aktivitas tick | Perubahan tick volume, OBV, z-score dan rasio volume terhadap MA: `obv`, `volume_zscore_24` |
| FE-12 | Spread | Biaya spread absolut/relatif, perubahan, median, z-score dan rasio ke ATR: `spread_pct`, `spread_to_atr` |
| FE-13 | Waktu dan sesi | Jam/hari siklik UTC, sesi Asia/London/New York dan overlap London-New York: `hour_sin`, `london_session` |
| FE-14 | Lag | Nilai fitur terpilih dari candle sebelumnya: `rsi_14_lag_3`, `return_1_lag_6` |

Periode yang dapat dikonfigurasi berada di `config.py`: SMA/EMA default
5/8/12/24/48/96/144/288 bar; RSI/ATR 7/14/21/28; volatilitas
6/12/24/48/96/288; MACD `(8,21,5)`, `(12,26,9)`, `(19,39,9)`;
lag 1/3/6 bar. Kelompok lain memakai periode yang tertulis pada komentar source.
Rolling menghitung observed bars; periode bar dapat melintasi market closure.
Tick volume adalah jumlah tick, bukan volume kontrak transaksi. Warm-up dan
data spread/volume yang hilang tetap `NaN`.

Feature selection dilakukan setelah feature engineering: Quality Filter
menyingkirkan kolom bermasalah, ReliefF memberi ranking relevansi, kemudian
RFECV memilih subset berdasarkan purged CV dan macro F1. Seluruh tahap seleksi
fit hanya TRAIN. Target `y` dibuat terpisah oleh `make_labels()`; harga masa depan
dan kelas target tidak menjadi fitur `X`.

## Menjalankan

```powershell
uv run python -B -m technical.xgboost_model `
  --mode walk-forward-m5 `
  --source-key metaquotes-demo-xauusd `
  --symbol XAUUSD `
  --start 2016-01-01T00:00:00Z `
  --final-tune-start 2025-01-01T00:00:00Z `
  --final-calibration-start 2025-07-01T00:00:00Z `
  --final-policy-start 2025-10-01T00:00:00Z `
  --final-test-start 2026-01-01T00:00:00Z `
  --evaluation-end 2026-10-01T00:00:00Z `
  --target-precision 0.90 `
  --min-policy-signals 50 `
  --horizon-bars 3 `
  --minimum-edge-bps 1 `
  --relieff-top-k 40 `
  --rfecv-min-features 10
```

`--horizon-bars 1/3/6` berarti 5/15/30 menit. Default threshold label:
`spread_price / close + (minimum_edge_bps + slippage_bps) / 10000`. BUY bila return > threshold,
SELL bila return < -threshold, sisanya NO_TRADE. Spread native diutamakan,
dengan fallback tick spread dari candle yang sama. Spread/volume negatif atau
tidak tersedia tidak diisi dengan nilai palsu. Spread hilang menyebabkan label
tidak tersedia; saat inference menghasilkan NO_TRADE dengan alasan
`spread_unavailable`.

Quality filter dan ReliefF hanya melihat prefix training internal pertama yang sudah dipurge.
Daftar kandidat dibekukan sebelum validation internal pertama; RFECV kemudian
menilai daftar tersebut dengan expanding CV pada TRAIN. Ini mencegah label
validation internal masuk ke screening, tetapi kandidat bisa kurang peka terhadap
regime TRAIN yang lebih baru. ReliefF memakai sampling deterministik dari prefix ini: maksimum 10.000 reference
rows, 512 anchors, 10 neighbors/class. Ini bukan random train/test split.
Sampling tidak membatasi data model utama. `--relieff-max-samples` dan
`--relieff-anchors` dapat dinaikkan; semakin besar semakin mahal. RFECV memakai
3 expanding splits dan step 5, serta macro F1 dengan tiga kelas tetap.
`--n-estimators`, `--cv-splits`, `--rfecv-step`, `--n-jobs` mengatur biaya training.
Menjalankan ulang pada output yang sama mengganti artefak turunan tersebut.
`--folds fold_2024 fold_2025 final_2026` menjalankan tiga fold terakhir;
tanpa flag ini semua tujuh fold dijalankan. `--slippage-bps` menambahkan asumsi
biaya eksekusi; nilai default nol bukan perkiraan biaya riil. Interval default
nominal 80% (`--interval-alpha 0.20`) adalah rentang return, bukan target precision
90%. `--no-require-interval-agreement` tersedia untuk eksperimen development;
kesesuaian arah regressor tetap wajib.

## Batas informasi dan waktu

Alur: preparation → validation → causal features → label biaya spread →
walk-forward/purge → quality → ReliefF → RFECV → XGBoost → calibration → policy
→ TEST → metrics/dashboard.

Timestamp candle sumber adalah **open time**. Waktu keputusan adalah **close
time**, lima menit setelah open. Model hanya memakai candle yang sudah tutup;
OHLC, volume, dan spread candle saat ini baru diketahui pada close time.
Naive timestamp, campuran source/symbol, atau timeframe selain M5 ditolak.
Duplicate deterministik memakai baris valid terakhir dalam urutan input stabil.
Gaps dicatat termasuk market closure; harga tidak diinterpolasi atau di-fill.

Rolling fitur memakai jumlah observed bars. EMA, RSI dan ATR memakai EWM
backward-looking dengan warm-up, bukan window yang melihat masa depan. Target
harus mencapai H bar dan H×5 menit sekaligus; target melintasi gap tidak mendapat
actual/label. H bar terakhir tetap mendapat prediksi, actual-nya pending.
Tidak ada forecast candle buatan untuk tanggal yang belum tersedia.

## Walk-forward

| TEST | TRAIN | TUNE | CALIBRATION | POLICY |
|---|---|---|---|---|
| 2020 | 2016–2018 | Jan–Jun 2019 | Jul–Sep 2019 | Oct–Dec 2019 |
| 2021 | 2016–2019 | Jan–Jun 2020 | Jul–Sep 2020 | Oct–Dec 2020 |
| 2022 | 2016–2020 | Jan–Jun 2021 | Jul–Sep 2021 | Oct–Dec 2021 |
| 2023 | 2016–2021 | Jan–Jun 2022 | Jul–Sep 2022 | Oct–Dec 2022 |
| 2024 | 2016–2022 | Jan–Jun 2023 | Jul–Sep 2023 | Oct–Dec 2023 |
| 2025 | 2016–2023 | Jan–Jun 2024 | Jul–Sep 2024 | Oct–Dec 2024 |
| Jan–Sep 2026 | 2016–2024 | Jan–Jun 2025 | Jul–Sep 2025 | Oct–Dec 2025 |

Semua interval half-open UTC; label development harus selesai **sebelum** akhir
bloknya. Internal RFECV juga purge target yang mencapai awal validation.
Feature selection hanya fit TRAIN. TUNE memilih parameter classifier (macro F1)
dan regressor (return RMSE), lalu model refit TRAIN+TUNE. CALIBRATION fit sigmoid
OVR dari log odds dan normalisasi multiclass. Separuh awal POLICY memilih rule
per arah berdasarkan F1 dengan syarat precision dan minimum support; separuh
berikutnya mengonfirmasi rule yang sudah dikunci. Label TEST tidak
dipakai pada tahap-tahap ini. Blok yang tidak cukup data/kelas ditolak eksplisit.

Default final model tidak belajar dari Januari–September 2026. Evaluasi berhenti di actual
yang tersedia; price/classification metrics hanya memakai actual diketahui.
Price regression memprediksi return langsung; return dikalikan 10.000 selama
fitting untuk stabilitas numerik, lalu dikonversi ke return desimal. Harga
prediksi = current close × (1 + predicted return), diplot pada **target time**.
Saat candle aktual berikutnya selesai, fitur dihitung ulang dari histori aktual
yang telah tersedia. Bobot model, fitur terpilih, kalibrasi dan rule tetap beku.
Contoh harian tanggal 2 boleh memakai candle aktual tanggal 1; implementasi ini
menerapkan prinsip yang sama pada setiap candle M5 dengan horizon 15 menit.

## Dashboard

Buka `artifacts/direction_v3/final_2026/dashboard_2026.html` langsung
di browser. File self-contained, termasuk Plotly dan seluruh prediksi, tanpa CDN.

- Candle/line, wheel zoom, drag pan, reset/double-click.
- Pilih tanggal UTC, rentang 288–4.000 bar, geser slider historis, Older/Newer.
- BUY/SELL/NO TRADE pada waktu keputusan dan harga close aktual; hover dan click untuk detail probabilitas serta actual.
- Titik forecast dan garis emas dari current close ke target prediksi yang dipilih,
  beserta interval return. Kemiringan antar-forecast tidak dipakai sebagai arah sinyal.
- Panel probabilitas dan rolling precision BUY/SELL/net move, window default 288 bar.
- Ledger default BUY+SELL; filter tanggal, class, berhasil/gagal/harga tetap/netral/status data;
  pagination dan export filtered CSV; click row memusatkan chart.
- Hasil arah **biru berhasil**, **merah gagal**, **abu-abu netral/belum dinilai**.
  Garis pilihan menghubungkan harga sinyal dengan close aktual di akhir horizon.
- Tab walk-forward, ReliefF/RFECV/importance, stability dan audit data/model.
- Total BUY/SELL per arah: jumlah sinyal/evaluasi, gross, spread, slippage dan net.
- Replay satu posisi: delayed entry, harga fill masuk/keluar, komisi dan net.
  Hasil dalam unit harga XAUUSD; belum dikonversi ke laba akun tanpa ukuran posisi.

Chart merender window terbatas agar ratusan ribu candle tidak membebani browser.
Semua prediksi final tetap tertanam dan bisa ditelusuri. KPI melaporkan seluruh
final TEST; filter ledger tidak mengubah KPI. Precision tanpa sinyal ditampilkan
sebagai tidak tersedia. Ledger otomatis menampilkan semua kelas bila seluruhnya
NO_TRADE. Rolling precision mengecualikan NO_TRADE dan actual yang tidak tersedia.
BUY berhasil bila actual target > harga sinyal, SELL bila actual target < harga
sinyal. Harga tetap adalah FLAT (tidak berhasil). NO_TRADE selalu NETRAL;
pergerakan selama horizon tidak dinilai sebagai keberhasilan NO_TRADE.
Yang dibandingkan adalah close pada akhir 15 menit, bukan sekadar sempat
naik/turun intrabar. Hasil arah terpisah dari hasil setelah biaya.
Trading proxy menjumlahkan sinyal yang overlap dan bukan backtest akun:
belum memperhitungkan lot, contract size, leverage, swap, commission,
stop loss/take profit ataupun posisi bersamaan. NO_TRADE tidak dianggap transaksi.
Slippage hanya mengikuti asumsi bps konfigurasi.

## Audit dan verifikasi

Per fold: classifier/calibrator/policy bundle, regressor, selected features,
inference contract/hash, execution replay, manifest, quality reasons, ReliefF ranking, RFECV scores, hyperparameters,
calibration, policy, predictions CSV, metrics dan confusion matrix PNG.
`regression_feature_selection.json` mencatat seleksi fitur regressor. Metrik
memisahkan raw classifier dari action setelah abstain, precision/support per arah,
interval coverage serta return RMSE terhadap baseline zero-return.
Consolidated: metrics CSV/JSON, feature selection/stability CSV, feature audits.
Per bulan: `monthly_metrics.csv` dan `monthly_diagnostics.png`; evaluasi anchor
tanpa horizon overlap tersimpan di `nonoverlapping_metrics.json`.
Final: `predictions.csv`, `dashboard_metrics.json` dan HTML.
Audit penyempurnaan dan hasil evaluasi nyata ada di
[laporan direction_v3](artifacts/direction_v3/audit_remediation_2026-10-10.md).
Kontrak inference, status actual dan asumsi replay dijelaskan pada
[metodologi](../docs/technical-selective.md#kontrak-arah-dan-perbaikan-audit-direction_v3).

```powershell
uv run pytest tests/test_technical.py -q
uv run ruff check technical tests/test_technical.py
uv run python scripts/generate_code_reference.py
```

Jika hanya mengubah tampilan HTML, rebuild tanpa training/database:

```powershell
uv run python -B -m technical.dashboard
```

Untuk menampilkan tahun lain gunakan `--fold fold_2020` (hingga `fold_2025`)
dan `--output path/hasil.html`.

Tests memakai dataset kecil deterministik dan model XGBoost sungguhan, termasuk
mutation TEST actual yang harus meninggalkan semua fitted state identik.
Docstrings/type hints menjadi sumber generated reference di
`docs/reference/code-reference.md`.

## Referensi dan batas adopsi

- [Sadorsky (2021)](https://www.mdpi.com/1911-8074/14/5/198): directional
  classification dengan tree models dan technical indicators. Studi ETF/daily;
  performanya tidak dapat dianggap sebagai target akurasi M5 XAUUSD.
- [Mamoudan et al. (2023)](https://academic.oup.com/jcde/article/10/3/1110/7148134):
  inspirasi candidate indicators/feature selection. PDF penerbit mengonfirmasi
  penggunaan MFO untuk seleksi fitur sebelum CNN-BiGRU. Pipeline ini memakai
  ReliefF/RFECV sesuai `refactor.md`, bukan replikasi MFO atau neural network.
- [Amini & Kalantari (2024)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0298426):
  pemisahan train/validation/test dan tuning model sebelum out-of-sample evaluation.
  Arsitektur CNN-Bi-LSTM dan data daily tidak direplikasi.
- [Taneva-Angelova et al. (2025)](https://www.mdpi.com/2227-7072/13/2/102):
  framework gold prediction multi-source; menjadi konteks penelitian, tidak
  menambahkan variabel makro/sentiment yang tidak tersedia di dataset M5.
- [Arif et al. (2026)](https://ejournal.undip.ac.id/index.php/jmasif/article/view/78091):
  rolling-origin expanding time-series validation. Paper menggunakan H1;
  yang diadopsi prinsip validasinya, sementara pipeline ini tetap M5.

Desain kalender, cost-aware neutral class, calibration dan confidence policy
di sini adalah keputusan penelitian berdasarkan brief; kelima paper bukan
validasi bahwa strategi profitable atau jaminan prediksi akurat.
