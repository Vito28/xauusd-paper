# REFACTOR PIPELINE TEKNIKAL XAUUSD M5 — XGBOOST WALK-FORWARD 2016–2026

Saya ingin Anda melakukan refactor besar pada folder `teknikal/` untuk penelitian forecasting/signal prediction XAUUSD berbasis candle M5.

JANGAN sekadar menambahkan kode ke `pipeline.py` yang sudah besar.

Rapikan arsitektur folder, pisahkan data preparation, validation, feature engineering, labeling, feature selection, model training, calibration, policy, evaluation, dan dashboard.

Sebelum mengubah kode:
1. Inspect seluruh folder `teknikal/`.
2. Inspect loader/database access yang sudah tersedia.
3. Inspect schema/view PostgreSQL yang saat ini digunakan.
4. Jangan mengubah atau menulis data sumber PostgreSQL.
5. Reuse fungsi yang masih layak.
6. Jangan menghapus workflow lama sebelum workflow baru berhasil.
7. Workflow lama daily recursive forecast boleh dipertahankan sebagai legacy/diagnostic, tetapi jangan dicampur dengan pipeline utama M5.

---

# 1. TUJUAN UTAMA

Bangun pipeline penelitian XAUUSD berbasis timeframe M5 dengan:

- XGBoost sebagai model utama.
- Multiclass prediction:
  - SELL
  - NO_TRADE
  - BUY
- Auxiliary regression untuk memprediksi future return / future price hanya sebagai analisis tambahan.
- Walk-forward / expanding-window time-series evaluation.
- Tidak boleh random train-test split.
- Tidak boleh shuffle data time-series.
- Tidak boleh data leakage.
- Feature selection harus dilakukan hanya menggunakan data TRAIN.
- Model final untuk evaluasi April–Oktober 2026 hanya boleh belajar dari data sampai 31 Maret 2026.

Base timeframe tetap M5.

JANGAN resample model utama menjadi M15 atau H1.

Target default boleh 15 menit ke depan, tetapi diperoleh dari:

    horizon_bars = 3

karena:

    3 × M5 = 15 menit

Artinya:
- data/input tetap M5;
- feature engineering tetap M5;
- hanya prediction horizon yang 15 menit.

Parameter horizon harus configurable agar nanti dapat diuji:

    horizon_bars = 1  -> 5 menit
    horizon_bars = 3  -> 15 menit
    horizon_bars = 6  -> 30 menit

Default penelitian saat ini:

    horizon_bars = 3

---

# 2. REFERENSI METODOLOGI

Gunakan prinsip metodologis dari penelitian berikut sebagai inspirasi, bukan menyalin implementasinya mentah-mentah:

1. Perry Sadorsky (2021)
   "Predicting Gold and Silver Price Direction Using Tree-Based Classifiers"

   Gunakan sebagai dasar:
   - directional classification;
   - technical indicators;
   - tree-based classifier;
   - time-series validation;
   - membandingkan hasil secara kronologis, bukan random CV.

2. Mamoudan et al. (2023)
   "Hybrid neural network-based metaheuristics for prediction of financial markets: a case study on global gold market"

   Gunakan sebagai dasar:
   - technical indicator mempunyai banyak konfigurasi;
   - tidak semua fitur harus masuk model;
   - feature selection penting;
   - MACD, moving average, dan indikator teknikal dapat menghasilkan banyak kandidat fitur.

   Paper tersebut menggunakan MFO untuk feature selection.
   Penelitian ini TIDAK harus menggunakan MFO.
   Pipeline ini menggunakan:
       Quality Filter -> ReliefF -> RFECV.

3. Amini & Kalantari (2024)
   "Gold price prediction by a CNN-Bi-LSTM model along with automatic parameter tuning"

   Ambil prinsip:
   - chronological split;
   - parameter tuning hanya menggunakan data sebelum test;
   - test tidak boleh digunakan untuk pemilihan model.

4. Taneva-Angelova et al. (2025)

   Ambil prinsip:
   - structured variable selection;
   - jangan memasukkan seluruh candidate variable tanpa evaluasi.

5. Arif et al. (2026)
   "Multivariate LSTM-Based Intraday Gold Price Prediction with Rolling Time Series Validation"

   Ambil prinsip:
   - rolling-origin / expanding-window Time Series Cross Validation;
   - evaluasi intraday secara kronologis.

CATATAN:
Paper Arif menggunakan H1.
Pipeline penelitian ini tetap M5.
Yang diadopsi adalah METODE VALIDASI WAKTUNYA, bukan timeframe H1.

---

# 3. MASALAH PIPELINE LAMA YANG HARUS DIHINDARI

Pipeline lama mempunyai mode forecast harga jangka panjang menggunakan:

    lag daily close
        -> XGBoost
        -> recursive forecast
        -> hasil prediksi dimasukkan kembali sebagai lag
        -> forecast berikutnya

Pada hasil saat ini, forecast XGBoost menjadi hampir garis lurus.

JANGAN menggunakan recursive multi-month forecasting tersebut sebagai model utama penelitian M5.

Untuk pipeline M5 gunakan DIRECT WALK-FORWARD PREDICTION:

Pada timestamp t:

    data <= t
        -> feature engineering
        -> XGBoost
        -> prediksi t+h

Setelah actual t+h tersedia:

    prediction
        vs
    actual

Lalu lanjut ke timestamp berikutnya.

Jangan memasukkan prediction sebelumnya sebagai pengganti harga actual untuk membuat feature M5 berikutnya.

Dalam TEST, feature boleh menggunakan candle actual sebelumnya karena pada kondisi nyata candle tersebut memang sudah diketahui pada waktu prediction dilakukan.

Tetapi LABEL TEST tidak boleh digunakan untuk:
- feature selection;
- tuning;
- calibration;
- policy;
- retraining.

---

# 4. STRUKTUR FOLDER BARU

Refactor menjadi kira-kira:

teknikal/
    __init__.py
    config.py
    pipeline.py

    data/
        __init__.py
        loader.py
        preparation.py
        validation.py

    features/
        __init__.py
        technical.py
        price_action.py
        volatility.py
        volume.py
        spread.py
        time_features.py
        feature_builder.py

    labels/
        __init__.py
        direction.py

    selection/
        __init__.py
        quality_filter.py
        relieff_selector.py
        rfecv_selector.py

    validation/
        __init__.py
        walk_forward.py
        purge.py

    models/
        __init__.py
        classifier.py
        regressor.py
        calibration.py
        policy.py

    evaluation/
        __init__.py
        classification.py
        regression.py
        trading_proxy.py

    dashboard/
        __init__.py
        builder.py

    artifacts/
        walk_forward/

    tests/

Boleh menyesuaikan nama jika struktur existing mempunyai convention berbeda.

Yang penting jangan kembali membuat satu `pipeline.py` dengan ribuan baris.

---

# 5. DATA PREPARATION

Gunakan data XAUUSD M5 canonical dari database yang memang sudah digunakan proyek.

Jika `market.processed_m5` masih menjadi canonical clean dataset Model 1, gunakan itu.

Jika implementasi existing sudah mempunyai loader canonical lain, inspect dahulu sebelum mengganti.

Wajib:

- satu `source_key/source_id`;
- satu symbol;
- timeframe M5;
- UTC;
- sort ascending;
- timestamp unique;
- OHLC valid;
- high >= open/close/low;
- low <= open/close/high;
- price > 0;
- spread tidak negatif;
- infinity -> NaN;
- duplicate timestamp dibuang secara deterministic;
- gap dicatat;
- JANGAN interpolation;
- JANGAN forward fill;
- JANGAN backward fill data harga.

Buat data-quality report:

    total_rows
    duplicate_rows
    invalid_ohlc_rows
    missing_rows
    detected_gaps
    spread_missing
    tick_volume_missing
    first_timestamp
    last_timestamp

Simpan:

    artifacts/data_quality.json

---

# 6. FEATURE ENGINEERING M5

Semua feature pada timestamp t hanya boleh menggunakan:

    candle <= t

Tidak boleh menggunakan future candle.

Buat candidate feature pool.

Jangan hard-code bahwa final model harus menggunakan 168 fitur.

168 adalah kandidat awal/build saat ini.
Jumlah kandidat boleh berubah sedikit jika implementasi yang lebih benar menghasilkan jumlah berbeda.

Tetapi buat `feature_manifest.json` sehingga seluruh candidate feature dapat diaudit.

## A. RETURN

Candidate:

    return_1
    return_2
    return_3
    return_6
    return_12
    return_24

    log_return_1
    log_return_3
    log_return_6
    log_return_12

## B. CANDLE / PRICE ACTION

    range_pct
    body_pct
    upper_wick_pct
    lower_wick_pct
    close_position_in_range
    candle_direction
    body_to_range
    upper_to_range
    lower_to_range

Gunakan division-safe implementation.

## C. MOVING AVERAGE

SMA candidate:

    5
    12
    24
    48
    96
    288

EMA candidate:

    5
    12
    24
    48
    96
    288

Buat feature seperti:

    close_vs_sma_N
    close_vs_ema_N
    sma_slope_N
    ema_slope_N

Jangan hanya memasukkan raw SMA karena level harga dapat berubah drastis antar regime.

Utamakan relative/distance feature.

## D. MACD

MACD WAJIB ada.

Jangan hanya menggunakan satu konfigurasi default.

Buat kandidat:

    MACD_FAST:
        fast=8
        slow=21
        signal=5

    MACD_STANDARD:
        fast=12
        slow=26
        signal=9

    MACD_SLOW:
        fast=19
        slow=39
        signal=9

Untuk masing-masing:

    macd_line
    macd_signal
    macd_histogram

Beri nama jelas, contoh:

    macd_12_26
    macd_signal_12_26_9
    macd_hist_12_26_9

Biarkan feature selection menentukan konfigurasi mana yang berguna.

## E. RSI

Candidate:

    RSI 7
    RSI 14
    RSI 21

Tambahkan bila berguna:

    rsi_delta
    distance_from_50

Tetapi semua harus past-only.

## F. MOMENTUM

    ROC 3
    ROC 6
    ROC 12
    ROC 24

    stochastic 14/3
    stochastic 21/5

Jika indikator membutuhkan smoothing, smoothing hanya backward-looking.

## G. ATR / VOLATILITY

ATR:

    ATR 7
    ATR 14
    ATR 21

Gunakan terutama relative form:

    atr_pct_7
    atr_pct_14
    atr_pct_21

Rolling volatility:

    6
    12
    24
    48
    96

Bollinger:

    BB 20, 2
    BB 40, 2

Candidate:

    bb_width
    close_vs_bb_mid
    bb_position

## H. VOLUME

Jika tick_volume valid:

    tick_volume_change
    tick_volume_return
    tick_volume_zscore_24
    tick_volume_zscore_48
    tick_volume_zscore_96
    volume_vs_ma_24

OBV dapat menjadi candidate apabila tick volume valid secara historis.

Jika volume tidak valid:
jangan mengisi dengan nilai palsu.

## I. SPREAD

Spread sangat penting karena target trading memperhitungkan biaya.

Candidate:

    spread_price
    spread_pct
    spread_change
    spread_median_12
    spread_median_24
    spread_median_48
    spread_zscore_24
    spread_zscore_48
    spread_to_atr

## J. TIME / SESSION

    hour_sin
    hour_cos
    weekday_sin
    weekday_cos

Tambahkan session indicator yang configurable:

    asia_session
    london_session
    new_york_session
    london_newyork_overlap

Semua timezone harus eksplisit dan tidak ambigu.

## K. FEATURE LAG

Untuk beberapa feature penting boleh dibuat:

    lag 1
    lag 3
    lag 6

Tetapi jangan secara membabi-buta membuat lag semua feature sampai ribuan kolom.

Candidate generation harus mempunyai config.

---

# 7. TARGET / LABEL

Default:

    horizon_bars = 3

sehingga target adalah perubahan 15 menit ke depan dari data M5.

Hitung:

    future_close = close.shift(-horizon_bars)

    future_return =
        (future_close - close) / close

Biaya transaksi harus memperhitungkan spread yang diketahui pada timestamp t.

Gunakan:

    spread_return = spread_price / close

Tambahkan configurable:

    minimum_edge_bps

Threshold:

    threshold =
        spread_return + minimum_edge_return

Label:

    BUY
        jika future_return > threshold

    SELL
        jika future_return < -threshold

    NO_TRADE
        jika -threshold <= future_return <= threshold

Simpan juga:

    future_return
    future_close
    spread_return
    threshold_return

Label mapping:

    SELL = 0
    NO_TRADE = 1
    BUY = 2

Jangan drop NO_TRADE hanya untuk memperbagus accuracy.

---

# 8. WALK-FORWARD SPLIT

Implement expanding-window walk-forward.

Test 2020:

    DEVELOPMENT = 2016-2019

Internal:

    TRAIN:
        2016-01-01 s/d 2018-12-31

    TUNE:
        2019-01-01 s/d 2019-06-30

    CALIBRATION:
        2019-07-01 s/d 2019-09-30

    POLICY:
        2019-10-01 s/d 2019-12-31

    TEST:
        2020-01-01 s/d 2020-12-31

Test 2021:

    TRAIN:
        2016-2019

    TUNE:
        Jan-Jun 2020

    CALIBRATION:
        Jul-Sep 2020

    POLICY:
        Okt-Des 2020

    TEST:
        2021

Lanjutkan pola expanding window untuk:

    TEST 2022
    TEST 2023
    TEST 2024
    TEST 2025

Jadi secara konseptual:

    history 2016-2019 -> test 2020
    history 2016-2020 -> test 2021
    history 2016-2021 -> test 2022
    history 2016-2022 -> test 2023
    history 2016-2023 -> test 2024
    history 2016-2024 -> test 2025

Untuk setiap development history,
tahun terakhir sebelum TEST dibagi menjadi:

    TUNE       Jan-Jun
    CALIBRATE  Jul-Sep
    POLICY     Oct-Dec

dan periode sebelumnya menjadi TRAIN utama.

---

# 9. FINAL 2026 OUT-OF-SAMPLE

Ini sangat penting.

Untuk final research evaluation:

    TRAIN:
        2016-01-01 s/d 2025-12-31

    TUNE:
        2026-01-01 s/d 2026-01-31

    CALIBRATION:
        2026-02-01 s/d 2026-02-28

    POLICY:
        2026-03-01 s/d 2026-03-31

    FINAL TEST:
        2026-04-01 s/d 2026-10-31

Model tidak boleh belajar dari label April–Oktober.

Jika database saat program dijalankan belum mempunyai actual sampai 31 Oktober:

    actual_eval_end =
        min(2026-10-31, latest_available_actual)

Jangan mengarang actual future.

Metrics hanya dihitung untuk timestamp yang actual t+h-nya sudah tersedia.

---

# 10. PURGE / EMBARGO

Karena target menggunakan future price:

    horizon_bars = H

maka H bar terakhir sebelum setiap boundary tidak boleh digunakan jika labelnya melewati boundary berikutnya.

Contoh horizon=3:

tiga bar yang labelnya membutuhkan harga dari periode TUNE/TEST harus dipurge dari periode sebelumnya.

Implementasikan boundary purge secara eksplisit.

Buat unit test untuk memastikan tidak ada:

    max(label_timestamp TRAIN)
        >=
    start(TUNE)

dan seterusnya.

---

# 11. FEATURE SELECTION

Feature selection WAJIB fit hanya pada TRAIN.

Tidak boleh memakai TUNE, CALIBRATION, POLICY, atau TEST untuk menentukan feature.

Pipeline:

    Candidate Features
        ↓
    Quality Filter
        ↓
    ReliefF
        ↓
    RFECV
        ↓
    Final Features

## STEP 1 — QUALITY FILTER

Pada TRAIN saja:

buang:

- all-null;
- terlalu banyak missing;
- infinity;
- constant;
- near-constant;
- exact duplicate feature;
- invalid numerical feature.

Catat alasan feature dibuang.

Output:

    quality_filter_report.json

## STEP 2 — ReliefF

Gunakan ReliefF untuk ranking relevance terhadap:

    SELL / NO_TRADE / BUY

Default:

    top_k = 40

Jika dataset terlalu besar untuk ReliefF penuh:

gunakan deterministic sampling dari TRAIN saja.

Contoh configurable:

    relieff_max_samples = 100000

Sampling tidak boleh mengambil TEST.

Simpan:

    relieff_ranking.csv

dengan:

    feature
    score
    rank

## STEP 3 — RFECV

Masukkan maksimal 40 feature dari ReliefF.

Gunakan RFECV dengan:

    scoring = f1_macro

Minimum:

    min_features_to_select = 10

Cross-validation RFECV harus time-series/purged.

JANGAN KFold random.

Gunakan step >1 jika computational cost terlalu tinggi.

Simpan:

    rfecv_results.csv
    selected_features.json

Jumlah final feature BOLEH berbeda pada setiap fold.

Ini justru perlu dicatat untuk melihat feature stability antar-regime.

---

# 12. XGBOOST CLASSIFIER

Model utama:

    XGBClassifier

Objective:

    multi:softprob

Classes:

    SELL
    NO_TRADE
    BUY

Gunakan baseline reasonable lalu lakukan tuning pada TUNE.

Candidate parameter:

    n_estimators
    max_depth
    learning_rate
    min_child_weight
    subsample
    colsample_bytree
    gamma
    reg_alpha
    reg_lambda

JANGAN tuning menggunakan TEST.

Primary tuning metric:

    macro F1

Jika class imbalance besar:

gunakan class/sample weighting.

Jangan menggunakan SMOTE sebagai default pada financial time-series.

Setelah best parameter ditemukan:

    fit model menggunakan TRAIN + TUNE

dengan selected feature dari TRAIN.

Kemudian model tersebut masuk tahap calibration.

---

# 13. PROBABILITY CALIBRATION

XGBoost menghasilkan:

    P(SELL)
    P(NO_TRADE)
    P(BUY)

Kalibrasikan hanya menggunakan CALIBRATION block.

Gunakan metode seperti:

    sigmoid / Platt calibration

atau implementation multiclass yang kompatibel dengan sklearn version proyek.

Jangan memakai TEST untuk calibration.

Simpan calibration diagnostic:

    pre_calibration_logloss
    post_calibration_logloss
    probability calibration metrics

---

# 14. POLICY / ABSTAIN

Gunakan POLICY block untuk menentukan confidence threshold.

Contoh:

    confidence = max(class_probability)

Jika:

    confidence < policy_threshold

maka prediction final:

    NO_TRADE

Policy threshold dicari hanya di POLICY block.

Objective utama tetap:

    macro F1

Simpan:

    policy.json

berisi:

    threshold
    metric
    signal_coverage
    class_distribution

---

# 15. AUXILIARY XGBOOST REGRESSOR

Untuk visualisasi prediksi harga, JANGAN menggunakan recursive daily forecast lama.

Gunakan direct future-return regression.

Target:

    future_return_h

Gunakan:

    XGBRegressor

Prediksi:

    predicted_future_return

Kemudian:

    predicted_future_close =
        current_close *
        (1 + predicted_future_return)

Actual:

    actual_future_close =
        close pada t+h

Dengan demikian setiap prediction dibuat langsung dari snapshot M5 pada t.

Tidak:

    prediction t+1
        -> menjadi feature prediction t+2.

Regressor ini secondary/auxiliary.

Classifier tetap model utama penelitian.

Regression metrics:

    MAE
    MSE
    RMSE
    MAPE

---

# 16. METRIK CLASSIFICATION

Primary:

    F1 MACRO

Wajib juga:

    precision SELL
    recall SELL
    F1 SELL

    precision NO_TRADE
    recall NO_TRADE
    F1 NO_TRADE

    precision BUY
    recall BUY
    F1 BUY

    F1 weighted
    balanced accuracy
    confusion matrix
    support per class

Jangan hanya menampilkan accuracy.

---

# 17. TRADING PROXY EVALUATION

Ini BUKAN full trading backtest.

Hitung diagnostic:

    signal_count
    BUY_count
    SELL_count
    NO_TRADE_count
    signal_coverage

Untuk BUY:

    realized_move =
        actual_future_close - current_close

Untuk SELL:

    realized_move =
        current_close - actual_future_close

Net proxy:

    realized_net =
        realized_move - spread_price

NO_TRADE tidak dihitung sebagai transaksi.

Hitung:

    wins
    losses
    win_rate
    mean_net_move
    median_net_move
    mean_move_in_spread_multiple
    cumulative_proxy_PL

Tegaskan dalam dashboard:

"Trading proxy only — bukan simulasi akun riil.
Belum memperhitungkan lot, contract size, leverage,
swap, commission, slippage, stop loss, atau take profit."

---

# 18. OUTPUT PER WALK-FORWARD FOLD

Contoh:

artifacts/walk_forward/
    fold_2020/
        model_classifier.joblib
        model_regressor.joblib
        selected_features.json
        feature_manifest.json
        hyperparameters.json
        calibration.json
        policy.json
        metrics.json
        predictions.csv
        confusion_matrix.png

    fold_2021/
    fold_2022/
    fold_2023/
    fold_2024/
    fold_2025/

    final_2026/
        ...

Buat juga consolidated:

    walk_forward_metrics.csv
    walk_forward_metrics.json
    feature_selection_by_fold.csv
    feature_stability.csv

Feature stability harus menunjukkan contoh:

    feature                fold_count
    rsi_14                 6
    atr_pct_14             6
    macd_hist_12_26_9      5
    spread_pct             5
    ...

---

# 19. FINAL 2026 PREDICTION FILE

Buat:

    predictions_2026_apr_oct.csv

Minimal memiliki:

    prediction_timestamp
    target_timestamp

    open
    high
    low
    close

    predicted_signal
    probability_sell
    probability_no_trade
    probability_buy
    confidence

    predicted_future_return
    predicted_future_close

    actual_future_return
    actual_future_close

    spread_price
    spread_pct
    threshold_return

    actual_signal

    prediction_correct

    realized_gross_move
    realized_net_move
    spread_multiple

    fold
    model_version

Jika actual belum tersedia:

    actual_* = null
    prediction_correct = null

dan row tidak dihitung ke metrics.

---

# 20. HTML DASHBOARD APRIL–OKTOBER 2026

Buat satu self-contained Plotly HTML:

    teknikal/artifacts/walk_forward/final_2026/
    dashboard_2026_apr_oct.html

Harus dapat dibuka offline.

UI harus modern, clean, research-friendly.

WAJIB memiliki zoom, pan, hover, reset, dan date range slider.

## HEADER KPI

Tampilkan:

    Evaluation Period
    Total Predictions
    Macro F1
    Weighted F1
    BUY F1
    SELL F1
    NO_TRADE F1
    Price MAE
    Price RMSE
    MAPE
    Signal Coverage
    Win Rate
    Average Spread

## PRICE CHART

Chart utama:

    Actual XAUUSD Close
    Predicted Future Close

Prediction harus diplot terhadap TARGET TIMESTAMP agar tidak bergeser secara visual.

Tambahkan marker:

    BUY
    SELL
    NO_TRADE

Marker harus clickable/hoverable.

Tooltip minimal:

    Prediction time
    Target time
    Current close
    Predicted signal
    Confidence
    P(BUY)
    P(SELL)
    P(NO_TRADE)
    Predicted future close
    Actual future close
    Predicted return
    Actual return
    Spread
    Threshold
    Correct / Wrong

## CONFIDENCE PANEL

Plot probability/confidence sepanjang waktu.

## PERFORMANCE PANEL

Tampilkan rolling metric, misalnya:

    rolling accuracy
    rolling macro correctness
    rolling realized net move

Gunakan window configurable.

## TABLE

Buat interactive table.

Default table menampilkan signal rows.

Kolom:

    Timestamp
    Target
    Close
    Prediction
    Confidence
    Predicted Price
    Actual Price
    Actual Direction
    Spread
    Threshold
    Correct?
    Net Move
    Spread Multiple

Tambahkan filter:

    Date range
    BUY
    SELL
    NO_TRADE
    Correct
    Wrong

Contoh logika:

Jika:

    model = BUY

tetapi:

    actual_future_return < threshold

maka prediction salah.

Jika:

    model = SELL

dan:

    actual_future_return < -threshold

maka prediction benar.

NO_TRADE benar jika actual berada dalam neutral zone yang sama.

---

# 21. WARNA TABLE SEPERTI YANG DIMINTA

Pada kolom SPREAD:

Jika prediction benar:

    warna spread = BIRU

Jika prediction salah:

    warna spread = MERAH

Jika actual belum tersedia:

    warna spread = ABU-ABU

Jangan menggunakan warna hijau untuk status benar pada kolom spread karena saya ingin gaya yang mendekati MetaTrader:

    BLUE = correct
    RED = wrong
    GRAY = pending

Contoh:

Timestamp    Signal    Close    Future    Spread    Result
10:05        BUY       4320     4327      0.35      CORRECT
10:10        BUY       4327     4319      0.40      WRONG
10:15        SELL      4319     4310      0.32      CORRECT

Pada table:

spread 0.35 -> blue
spread 0.40 -> red
spread 0.32 -> blue

Untuk BUY/SELL sendiri boleh memakai icon/shape berbeda agar tidak bergantung hanya pada warna.

---

# 22. FOLD PERFORMANCE DI DASHBOARD

Tambahkan tab/section:

    WALK-FORWARD PERFORMANCE

Table:

    Fold Test
    Train Range
    Tune Range
    Calibration Range
    Policy Range
    Test Range
    Number Features
    Macro F1
    BUY F1
    SELL F1
    NO_TRADE F1
    RMSE
    Win Rate

Sehingga saya dapat melihat apakah model:

    bagus pada 2020
    turun pada 2021
    bagus pada 2022
    dst.

Tujuannya untuk melihat MARKET REGIME sensitivity.

---

# 23. FEATURE SELECTION DASHBOARD

Tampilkan:

    Top ReliefF Features
    Final RFECV Features
    XGBoost Feature Importance

Buat table per fold.

Saya ingin dapat mengetahui misalnya:

    MACD 12/26/9 terpilih pada 5 fold
    RSI 14 terpilih pada 6 fold
    ATR 14 terpilih pada 6 fold

sehingga pemilihan indikator dapat dipertanggungjawabkan secara ilmiah.

---

# 24. DATA LEAKAGE TEST

Tambahkan unit/integration test untuk membuktikan:

1. Feature t tidak membaca t+1.
2. Label t boleh membaca t+h tetapi label tidak menjadi feature.
3. Feature selector fit hanya TRAIN.
4. ReliefF tidak membaca test.
5. RFECV tidak membaca test.
6. Hyperparameter tuning tidak membaca test.
7. Calibration tidak membaca test.
8. Policy tidak membaca test.
9. Purge boundary benar.
10. Mengubah nilai actual TEST tidak boleh mengubah:
    - selected feature;
    - hyperparameter;
    - calibration;
    - policy.

Ini sangat penting.

---

# 25. COMMAND LINE

Target akhir harus dapat dijalankan kira-kira:

uv run python -B -m teknikal.pipeline \
    --mode walk-forward-m5 \
    --source-key metaquotes-demo-xauusd \
    --symbol XAUUSD \
    --start 2016-01-01T00:00:00Z \
    --final-train-cutoff 2026-04-01T00:00:00Z \
    --evaluation-end 2026-11-01T00:00:00Z \
    --horizon-bars 3 \
    --relieff-top-k 40 \
    --rfecv-min-features 10

Sesuaikan PowerShell syntax jika dibutuhkan.

---

# 26. LOGGING

Terminal harus menunjukkan progress jelas:

[DATA]
Loaded ...
Validated ...

[FOLD 2020]
TRAIN ...
TUNE ...
CALIBRATION ...
POLICY ...
TEST ...

[FEATURE SELECTION]
Candidates: 168
After quality filter: ...
After ReliefF: 40
After RFECV: ...

[MODEL]
Best XGBoost parameters ...

[EVALUATION]
Macro F1 ...
BUY F1 ...
SELL F1 ...
NO_TRADE F1 ...
RMSE ...

Lakukan hal sama untuk setiap fold.

Jangan membuat user menunggu tanpa mengetahui proses yang berjalan.

---

# 27. README

Update README `teknikal/README.md`.

Jelaskan secara sederhana:

Raw XAUUSD M5
    ↓
Data Preparation
    ↓
Data Validation
    ↓
Feature Engineering
    ↓
Target 15-minute future return
    ↓
BUY / SELL / NO_TRADE
    ↓
Walk-forward split
    ↓
Quality Filter
    ↓
ReliefF
    ↓
RFECV
    ↓
XGBoost
    ↓
Probability Calibration
    ↓
Confidence Policy
    ↓
Out-of-sample Test
    ↓
Metrics + Dashboard

Jelaskan juga secara eksplisit:

M5 adalah BASE TIMEFRAME.

15 menit adalah PREDICTION HORIZON.

Keduanya jangan ditulis seolah-olah model menggunakan candle M15.

---

# 28. ACCEPTANCE CRITERIA

Refactor dianggap selesai hanya jika:

- pipeline dapat dijalankan dari 2016 sampai 2026;
- database read-only;
- tidak ada random train/test split;
- walk-forward 2020–2025 berhasil;
- final April–Oktober 2026 berhasil;
- no leakage;
- purge bekerja;
- MACD tersedia dalam beberapa konfigurasi;
- RSI/ATR/SMA/EMA/Bollinger/volatility/volume/spread/time feature tersedia;
- quality filter bekerja;
- ReliefF bekerja;
- RFECV bekerja;
- XGBClassifier menghasilkan probability;
- calibration bekerja;
- policy threshold bekerja;
- auxiliary XGBRegressor menggunakan direct forecast, bukan recursive forecast;
- metrics per fold tersedia;
- predictions 2026 tersedia;
- HTML dashboard tersedia;
- HTML dapat zoom/pan;
- HTML mempunyai interactive prediction table;
- spread berwarna biru jika prediction benar;
- spread berwarna merah jika prediction salah;
- pending berwarna abu-abu;
- selected features dapat diaudit;
- seluruh output reproducible dengan random_state tetap;
- unit test leakage dan time boundaries lolos.

Sebelum implementasi, tampilkan terlebih dahulu:
1. struktur folder saat ini;
2. masalah arsitektur yang ditemukan;
3. file yang akan dibuat/diubah;
4. dependency tambahan yang benar-benar diperlukan.

Setelah itu baru lakukan refactor dan jalankan test.