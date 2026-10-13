# 📊 XAUUSD M5 — Feature Engineering Pipeline

> **Technical feature-engineering pipeline for XAUUSD Machine Learning research using M5 candle data.**

Pipeline ini bertanggung jawab untuk membentuk **candidate feature pool** dari data pasar **XAUUSD timeframe M5** sebelum digunakan oleh model Machine Learning.

Tujuan utama feature engineering bukan memasukkan seluruh indikator secara langsung ke model, tetapi membangun kumpulan kandidat fitur yang cukup luas, kemudian melakukan seleksi secara sistematis.

```text
XAUUSD M5
    │
    ▼
Candidate Feature Engineering
    │
    ▼
150–220+ Candidate Features
    │
    ▼
Quality Filter
    │
    ▼
ReliefF
    │
    ▼
Top-K Features
    │
    ▼
RFECV
    │
    ▼
10–40 Final Features
    │
    ▼
XGBoost
```

---

# 📑 Table of Contents

- [Overview](#-overview)
- [Core Principles](#-core-principles)
- [Feature Groups](#-feature-groups)
- [1. Return Features](#1-return-features)
- [2. Candle & Price Action](#2-candle--price-action)
- [3. SMA](#3-simple-moving-average--sma)
- [4. EMA](#4-exponential-moving-average--ema)
- [5. MACD](#5-macd)
- [6. RSI](#6-rsi)
- [7. Stochastic](#7-stochastic-oscillator)
- [8. ROC](#8-rate-of-change--roc)
- [9. ATR](#9-average-true-range--atr)
- [10. Rolling Volatility](#10-rolling-volatility)
- [11. Bollinger Bands](#11-bollinger-bands)
- [12. Donchian / Range](#12-donchian--range-structure)
- [13. Tick Volume](#13-tick-volume)
- [14. Spread](#14-spread-features)
- [15. Time Features](#15-time-features)
- [16. Trading Session](#16-trading-session)
- [17. Feature Delta](#17-feature-delta)
- [18. Selected Feature Lag](#18-selected-feature-lag)
- [Candidate Feature Count](#-estimated-candidate-feature-count)
- [Feature Manifest](#-feature-manifest)
- [Feature Selection](#-feature-selection)
- [Feature Stability](#-feature-stability)
- [Expected Output](#-expected-output)
- [Research Principles](#-research-principles)

---

# 🎯 Overview

Base timeframe:

```text
M5
```

Seluruh feature engineering utama dibentuk dari candle **5 menit**.

Target awal candidate feature pool:

```text
150–220+ numerical features
```

Jumlah tersebut bukan angka wajib.

Jumlah aktual dapat berubah tergantung:

- ketersediaan `spread`;
- ketersediaan `tick_volume`;
- panjang historical data;
- kualitas feature;
- konfigurasi indikator;
- hasil feature selection.

Model **tidak otomatis menggunakan seluruh candidate feature**.

Sebagai contoh:

```text
188 Candidate Features
        │
        ▼
153 Features
Quality Filter
        │
        ▼
40 Features
ReliefF
        │
        ▼
18 Features
RFECV
        │
        ▼
XGBoost
```

---

# 🛡️ Core Principles

## 1. No Future Leakage

Semua feature pada timestamp:

```text
t
```

hanya boleh menggunakan informasi:

```text
<= t
```

Dilarang menggunakan:

```text
t + 1
t + 2
t + 3
...
```

atau informasi masa depan lainnya.

> **Feature engineering harus sepenuhnya aman terhadap data leakage.**

---

## 2. Base Timeframe Tetap M5

Jika target menggunakan:

```text
horizon_bars = 3
```

maka:

```text
3 × M5 = 15 menit
```

Artinya model menggunakan kondisi pasar pada candle M5 saat ini untuk memprediksi kondisi sekitar **15 menit ke depan**.

Tidak perlu mengubah candle menjadi M15.

---

## 3. Prefer Relative Features

Untuk indikator yang memiliki skala mengikuti harga, utamakan representasi relatif.

Contoh:

```text
close_vs_ema_24
atr_pct_14
spread_pct
body_pct
range_pct
```

dibanding hanya menggunakan nilai absolut indikator.

Tujuannya agar feature lebih stabil ketika level harga XAUUSD berubah sepanjang periode penelitian.

---

## 4. Safe Numerical Operations

Semua operasi pembagian harus menggunakan **safe division**.

Harus menghindari:

```text
NaN
Infinity
division by zero
```

Feature dengan nilai numerik tidak valid nantinya harus terdeteksi oleh **Quality Filter**.

---

# 🧩 Feature Groups

Candidate feature pool dibagi menjadi beberapa kelompok utama:

| # | Feature Group |
|---:|---|
| 1 | Return |
| 2 | Price Action |
| 3 | Candle Structure |
| 4 | SMA |
| 5 | EMA |
| 6 | MACD |
| 7 | RSI |
| 8 | Stochastic |
| 9 | Rate of Change |
| 10 | ATR |
| 11 | Rolling Volatility |
| 12 | Bollinger Bands |
| 13 | Donchian / Range Structure |
| 14 | Tick Volume |
| 15 | Spread |
| 16 | Time Features |
| 17 | Trading Session |
| 18 | Feature Delta |
| 19 | Selected Feature Lag |
| 20 | Relative / Normalized Features |

---

# 1. Return Features

Return digunakan untuk menggambarkan perubahan harga pada beberapa horizon historis.

## Simple Return

```text
return_1
return_2
return_3
return_6
return_12
return_24
return_48
```

## Log Return

```text
log_return_1
log_return_2
log_return_3
log_return_6
log_return_12
log_return_24
log_return_48
```

## Rolling Return

```text
rolling_return_3
rolling_return_6
rolling_return_12
rolling_return_24
```

Perkiraan:

```text
±18 features
```

---

# 2. Candle & Price Action

Feature ini digunakan untuk menggambarkan struktur candle dan posisi harga.

```text
range_pct
body_pct

upper_wick_pct
lower_wick_pct

body_to_range
upper_to_range
lower_to_range

close_position_in_range

open_to_previous_close
high_to_previous_close
low_to_previous_close

close_to_previous_high
close_to_previous_low
```

Contoh:

```text
range_pct =
    (high - low) / close
```

```text
body_pct =
    abs(close - open) / close
```

Perkiraan:

```text
±13 features
```

> Semua pembagian wajib menggunakan safe division.

---

# 3. Simple Moving Average — SMA

Gunakan beberapa window karena karakter trend pada M5 tidak selalu dapat direpresentasikan oleh satu periode moving average.

## Period

```text
5
8
12
24
48
96
144
288
```

Untuk masing-masing periode:

```text
close_vs_sma_N
sma_slope_N
sma_change_N
```

Contoh:

```text
close_vs_sma_24 =
    close / sma_24 - 1
```

Raw SMA boleh dihitung secara internal.

Namun feature utama sebaiknya berupa transformasi relatif terhadap harga.

Perkiraan:

```text
8 periods × 3 transformations
= ±24 features
```

---

# 4. Exponential Moving Average — EMA

## Period

```text
5
8
12
24
48
96
144
288
```

Untuk masing-masing periode:

```text
close_vs_ema_N
ema_slope_N
ema_change_N
```

Contoh:

```text
close_vs_ema_24
ema_slope_24
ema_change_24
```

Perkiraan:

```text
8 periods × 3 transformations
= ±24 features
```

---

# 5. MACD

MACD dibuat dalam beberapa konfigurasi.

Jangan hanya menggunakan konfigurasi default:

```text
12 / 26 / 9
```

Tujuannya adalah memberikan beberapa representasi momentum dengan sensitivitas yang berbeda agar **feature selection** menentukan konfigurasi yang paling informatif.

---

## Fast MACD

```text
fast   = 8
slow   = 21
signal = 5
```

Feature:

```text
macd_8_21
macd_signal_8_21_5
macd_hist_8_21_5
macd_hist_delta_8_21_5
```

---

## Standard MACD

```text
fast   = 12
slow   = 26
signal = 9
```

Feature:

```text
macd_12_26
macd_signal_12_26_9
macd_hist_12_26_9
macd_hist_delta_12_26_9
```

---

## Slow MACD

```text
fast   = 19
slow   = 39
signal = 9
```

Feature:

```text
macd_19_39
macd_signal_19_39_9
macd_hist_19_39_9
macd_hist_delta_19_39_9
```

Total:

```text
3 configurations × 4 features
= ±12 features
```

Feature selection nantinya menentukan konfigurasi MACD yang paling informatif untuk membedakan:

```text
SELL
NO_TRADE
BUY
```

---

# 6. RSI

Gunakan beberapa periode RSI.

```text
RSI 7
RSI 14
RSI 21
RSI 28
```

Untuk setiap periode:

```text
rsi_N
rsi_distance_50_N
rsi_delta_N
```

Contoh:

```text
rsi_14
rsi_distance_50_14
rsi_delta_14
```

Dengan:

```text
rsi_distance_50 =
    rsi - 50
```

Perkiraan:

```text
4 periods × 3 transformations
= ±12 features
```

---

# 7. Stochastic Oscillator

Candidate configuration:

```text
9 / 3
14 / 3
21 / 5
```

Untuk masing-masing:

```text
stoch_k
stoch_d
```

Contoh:

```text
stoch_k_14
stoch_d_14_3
```

Perkiraan:

```text
±6 features
```

---

# 8. Rate of Change — ROC

Candidate:

```text
roc_3
roc_6
roc_12
roc_24
roc_48
```

ROC digunakan untuk menggambarkan momentum perubahan harga pada beberapa horizon historis.

Perkiraan:

```text
±5 features
```

---

# 9. Average True Range — ATR

Candidate period:

```text
7
14
21
28
```

Untuk masing-masing:

```text
atr_pct_N
atr_change_N
atr_vs_median_N
```

Contoh:

```text
atr_pct_14 =
    atr_14 / close
```

ATR relatif digunakan agar nilai feature lebih stabil ketika harga XAUUSD mengalami perubahan level yang besar sepanjang tahun.

Perkiraan:

```text
4 periods × 3 transformations
= ±12 features
```

---

# 10. Rolling Volatility

Rolling window:

```text
6
12
24
48
96
288
```

Candidate:

```text
rolling_std_N
high_low_volatility_N
```

Contoh:

```text
rolling_std_24
high_low_volatility_24
```

Perkiraan:

```text
6 windows × 2 transformations
= ±12 features
```

---

# 11. Bollinger Bands

Candidate configuration:

```text
20 periods / 2 std
40 periods / 2 std
60 periods / 2 std
```

Feature:

```text
bb_width
bb_position
close_vs_bb_mid
```

Contoh:

```text
bb_width_20_2
bb_position_20_2
close_vs_bb_mid_20
```

Perkiraan:

```text
3 configurations × 3 features
= ±9 features
```

---

# 12. Donchian / Range Structure

Feature ini digunakan untuk melihat posisi harga terhadap range historis.

Period:

```text
12
24
48
96
```

Candidate:

```text
donchian_position_N
distance_to_high_N
distance_to_low_N
range_breakout_N
```

Contoh:

```text
donchian_position_24
distance_to_high_24
distance_to_low_24
```

---

# 13. Tick Volume

Feature volume hanya digunakan jika:

```text
tick_volume
```

tersedia dan valid.

Candidate:

```text
tick_volume_change
tick_volume_return

volume_vs_ma_12
volume_vs_ma_24
volume_vs_ma_48

volume_zscore_12
volume_zscore_24
volume_zscore_48
volume_zscore_96

obv
obv_change
```

Perkiraan:

```text
±11 features
```

Jika volume historis tidak tersedia atau tidak valid:

```text
JANGAN
```

melakukan:

```text
forward fill
backfill
synthetic volume
artificial volume
```

Feature harus ditandai sebagai unavailable dan ditangani oleh **Quality Filter**.

---

# 14. Spread Features

Spread merupakan feature penting karena keputusan BUY dan SELL harus mempertimbangkan biaya transaksi.

Candidate:

```text
spread_price
spread_pct
spread_change

spread_vs_median_12
spread_vs_median_24
spread_vs_median_48

spread_zscore_12
spread_zscore_24
spread_zscore_48

spread_to_atr
```

Contoh:

```text
spread_pct =
    spread_price / close
```

```text
spread_to_atr =
    spread_price / atr_14
```

Perkiraan:

```text
±10 features
```

---

# 15. Time Features

Waktu tidak sebaiknya diberikan langsung sebagai nilai linear.

Contoh yang dihindari:

```text
hour = 23
```

karena:

```text
23 → 0
```

sebenarnya memiliki jarak waktu yang sangat dekat.

Gunakan **cyclical encoding**:

```text
hour_sin
hour_cos

weekday_sin
weekday_cos
```

Contoh:

```text
hour_sin =
    sin(2π × hour / 24)

hour_cos =
    cos(2π × hour / 24)
```

---

# 16. Trading Session

Tambahkan indikator sesi trading.

Candidate:

```text
asia_session
london_session
new_york_session
london_newyork_overlap
```

Timezone harus ditentukan secara eksplisit.

Timestamp sumber tetap disimpan dalam:

```text
UTC
```

Session feature boleh dibuat berdasarkan timezone trading yang telah dikonfigurasi.

---

# 17. Feature Delta

Model tidak hanya perlu mengetahui nilai indikator saat ini, tetapi juga arah perubahannya.

Contoh:

```text
rsi_delta
atr_delta
macd_hist_delta
spread_change
volume_change
ema_change
sma_change
```

Sebagai contoh, model tidak hanya mengetahui:

```text
RSI sekarang = 65
```

tetapi juga apakah RSI sedang:

```text
naik
turun
stabil
```

Informasi perubahan dapat membantu model memahami momentum indikator.

---

# 18. Selected Feature Lag

Lag **tidak diterapkan ke seluruh candidate feature secara otomatis**.

Hindari:

```text
200 features
×
lag 1, 3, 6
```

karena dapat menghasilkan banyak feature redundant.

Lag hanya diberikan kepada kelompok feature yang dianggap penting.

Candidate utama:

```text
return
RSI
MACD histogram
ATR
rolling volatility
spread
volume
EMA distance
SMA distance
```

Candidate lag:

```text
lag_1
lag_3
lag_6
```

Contoh:

```text
rsi_14_lag_1
rsi_14_lag_3
rsi_14_lag_6

macd_hist_12_26_9_lag_1
macd_hist_12_26_9_lag_3

atr_pct_14_lag_1
spread_pct_lag_1
```

> Seluruh lag wajib menggunakan nilai masa lalu.

---

# 📊 Estimated Candidate Feature Count

Perkiraan jumlah candidate feature sebelum feature selection:

| Feature Group | Estimated Features |
|---|---:|
| Return | 18 |
| Candle / Price Action | 13 |
| SMA | 24 |
| EMA | 24 |
| MACD | 12 |
| RSI | 12 |
| Stochastic | 6 |
| ROC | 5 |
| ATR | 12 |
| Rolling Volatility | 12 |
| Bollinger Bands | 9 |
| Tick Volume | 11 |
| Spread | 10 |
| Time / Session | 8+ |
| Donchian / Range | 10+ |
| Selected Lag | 20+ |
| **Total** | **≈150–220+** |

Jumlah tersebut boleh berubah sesuai konfigurasi.

Targetnya **bukan** menghasilkan angka tertentu seperti:

```text
168 features
```

tetapi menghasilkan candidate feature pool yang:

- luas;
- relevan;
- dapat dijelaskan;
- dapat diaudit;
- aman dari leakage;
- kemudian diseleksi secara sistematis.

---

# 📋 Feature Manifest

Setiap feature yang dibuat harus diregistrasikan ke:

```text
feature_manifest.json
```

Contoh:

```json
{
  "feature_name": "macd_hist_12_26_9",
  "feature_group": "momentum",
  "base_indicator": "MACD",
  "parameters": {
    "fast": 12,
    "slow": 26,
    "signal": 9
  },
  "transformation": "histogram",
  "lag": 0,
  "lookback_bars": 26,
  "requires_volume": false,
  "requires_spread": false
}
```

Manifest digunakan agar setiap feature dapat:

```text
ditelusuri
↓
dijelaskan
↓
direproduksi
↓
diaudit
```

terutama untuk kebutuhan penelitian.

---

# 🔍 Feature Selection

Candidate feature yang berjumlah ratusan **tidak langsung diberikan ke XGBoost**.

Gunakan tiga tahap utama:

```text
Candidate Features
        │
        ▼
Quality Filter
        │
        ▼
ReliefF
        │
        ▼
RFECV
        │
        ▼
Final Features
```

---

## Stage 1 — Quality Filter

Quality Filter hanya dijalankan menggunakan:

```text
TRAIN DATA
```

Feature dibuang jika memiliki kondisi seperti:

```text
all null
too many missing values
infinity
constant
near constant
exact duplicate
invalid numerical values
```

Output:

```text
quality_filter_report.json
```

Contoh:

```text
Initial Features       : 184
Removed All Null       : 5
Removed Near Constant  : 11
Removed Duplicate      : 4
Remaining              : 164
```

---

## Stage 2 — ReliefF

Setelah Quality Filter, gunakan:

```text
ReliefF
```

untuk melakukan ranking berdasarkan kemampuan feature membedakan:

```text
SELL
NO_TRADE
BUY
```

Default:

```text
top_k = 40
```

Contoh:

```text
164 Features
     │
     ▼
   ReliefF
     │
     ▼
Top 40 Features
```

ReliefF hanya boleh di-fit menggunakan:

```text
TRAIN
```

Dilarang menggunakan:

```text
TUNE
CALIBRATION
POLICY
TEST
```

Output:

```text
relieff_ranking.csv
```

Contoh:

```csv
rank,feature,score
1,atr_pct_14,0.0831
2,macd_hist_12_26_9,0.0782
3,rsi_14,0.0744
4,spread_pct,0.0692
```

---

## Stage 3 — RFECV

Setelah ReliefF, gunakan:

```text
Recursive Feature Elimination
with Cross Validation
```

atau:

```text
RFECV
```

Input maksimal:

```text
40 features
```

Minimum final feature:

```text
min_features_to_select = 10
```

Scoring utama:

```text
f1_macro
```

Cross-validation **wajib time-series aware**.

Jangan gunakan:

```text
KFold random
StratifiedKFold random
shuffle=True
```

Gunakan:

```text
TimeSeriesSplit
```

atau:

```text
Purged Time-Series Split
```

yang sesuai dengan prediction horizon.

Contoh:

```text
ReliefF Top 40
      │
      ▼
    RFECV
      │
      ▼
17 Final Features
```

Jumlah final feature dapat berbeda pada setiap walk-forward fold.

Contoh:

```text
Fold 2020 → 19 features
Fold 2021 → 15 features
Fold 2022 → 22 features
Fold 2023 → 14 features
Fold 2024 → 18 features
Fold 2025 → 16 features
```

Hal tersebut diperbolehkan karena market regime dapat berubah.

---

# 📈 Feature Stability

Feature yang terpilih pada setiap fold harus disimpan.

Output:

```text
feature_selection_by_fold.csv
feature_stability.csv
```

Contoh:

```csv
feature,selected_fold_count
atr_pct_14,6
rsi_14,6
macd_hist_12_26_9,5
spread_pct,5
volatility_24,5
close_vs_ema_48,4
```

Feature stability dapat digunakan untuk mengetahui indikator mana yang konsisten memberikan informasi pada berbagai kondisi pasar.

---

# 📦 Expected Output

Feature Engineering Pipeline minimal menghasilkan:

```text
features/
│
├── feature_manifest.json
│
├── quality_filter_report.json
│
├── relieff_ranking.csv
│
├── feature_selection_by_fold.csv
└── feature_stability.csv
```

Alur hasil:

```text
Raw M5 Data
    │
    ▼
Candidate Features
    │
    ├── Feature Manifest
    │
    ▼
Quality Filter
    │
    ├── Quality Report
    │
    ▼
ReliefF
    │
    ├── Feature Ranking
    │
    ▼
RFECV
    │
    ├── Selected Features by Fold
    │
    ▼
Feature Stability
    │
    ▼
Final Features
```

---

# 🧪 Example Full Process

Misalnya feature engineering menghasilkan:

```text
Candidate Features
188
```

Setelah Quality Filter:

```text
188
 │
 ▼
153
```

Setelah ReliefF:

```text
153
 │
 ▼
40
```

Setelah RFECV:

```text
40
 │
 ▼
18
```

Maka XGBoost hanya dilatih menggunakan:

```text
18 final features
```

bukan seluruh:

```text
188 candidate features
```

---

# 🧠 Research Principles

Feature engineering yang baik **bukan berarti**:

```text
semakin banyak feature
=
semakin bagus model
```

Pendekatan penelitian yang digunakan adalah:

```text
Build Wide Candidate Feature Pool
              │
              ▼
Remove Invalid / Low-Quality Features
              │
              ▼
Measure Feature Relevance
              │
              ▼
Reduce Redundancy
              │
              ▼
Validate Using Time-Series CV
              │
              ▼
Use Final Selected Features
```

Pipeline akhir:

```text
XAUUSD M5
    │
    ▼
150–220+ Candidate Features
    │
    ▼
Quality Filter
    │
    ▼
ReliefF Top 40
    │
    ▼
RFECV
    │
    ▼
10–40 Final Features
    │
    ▼
XGBoost
```

Jumlah final feature **tidak ditentukan secara manual**.

Feature selection menentukan feature mana yang benar-benar berguna untuk membedakan:

```text
SELL
NO_TRADE
BUY
```

---

## ✅ Feature Engineering Checklist

Sebelum feature dianggap valid, pastikan:

- [ ] Feature hanya menggunakan data `<= t`
- [ ] Tidak terdapat future leakage
- [ ] Tidak terdapat `Infinity`
- [ ] Tidak terdapat pembagian dengan nol
- [ ] Missing value dapat dijelaskan
- [ ] Tidak membuat volume sintetis
- [ ] Lag hanya menggunakan historical value
- [ ] Timezone session didefinisikan secara eksplisit
- [ ] Setiap feature terdaftar di `feature_manifest.json`
- [ ] Quality Filter hanya fit pada TRAIN
- [ ] ReliefF hanya fit pada TRAIN
- [ ] RFECV menggunakan time-series validation
- [ ] Feature selection dilakukan di dalam masing-masing walk-forward fold
- [ ] Final feature dapat direproduksi dan diaudit

---

<div align="center">

### XAUUSD Technical ML Research

**M5 Feature Engineering → Feature Selection → XGBoost**

`SELL` • `NO_TRADE` • `BUY`

</div>