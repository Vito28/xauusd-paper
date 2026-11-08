Tabel 6.3 Daftar Fitur, Rumus, dan Stabilitas Seleksi
<!-- widths: 6,30,14,38,12 -->
| No. | Fitur | Kelompok | Rumus atau deskripsi | Terpilih (fold) |
|---|---|---|---|---|
| 1 | `log_return_1` | price_action | ln(close[t] / close[t−1]) | 4/5 |
| 2 | `log_return_3` | price_action | ln(close[t] / close[t−3]) | 1/5 |
| 3 | `log_return_6` | price_action | ln(close[t] / close[t−6]) | 4/5 |
| 4 | `log_return_12` | price_action | ln(close[t] / close[t−12]), return 1 jam | 5/5 |
| 5 | `return_6_atr` | price_action | (close[t] − close[t−6]) ÷ ATR14 | 1/5 |
| 6 | `return_24_atr` | price_action | (close[t] − close[t−24]) ÷ ATR14, perubahan 2 jam | 4/5 |
| 7 | `candle_body_atr` | price_action | (close − open) ÷ ATR14, badan candle bertanda | 0/5 |
| 8 | `candle_range_atr` | price_action | (high − low) ÷ ATR14 | 5/5 |
| 9 | `upper_wick_ratio` | price_action | Sumbu atas ÷ rentang candle | 4/5 |
| 10 | `lower_wick_ratio` | price_action | Sumbu bawah ÷ rentang candle | 2/5 |
| 11 | `close_position_in_range` | price_action | (close − low) ÷ (high − low) | 3/5 |
| 12 | `close_vs_ema_12_atr` | trend | (close − EMA12) ÷ ATR14 | 4/5 |
| 13 | `close_vs_ema_48_atr` | trend | (close − EMA48) ÷ ATR14 | 5/5 |
| 14 | `close_vs_ema_288_atr` | trend | (close − EMA288) ÷ ATR14, jarak ke tren 24 jam | 3/5 |
| 15 | `ema_12_vs_48_atr` | trend | (EMA12 − EMA48) ÷ ATR14 | 4/5 |
| 16 | `ema_48_slope_atr` | trend | (EMA48[t] − EMA48[t−6]) ÷ ATR14 | 1/5 |
| 17 | `adx_14` | trend | Average Directional Index 14 (kekuatan tren) | 2/5 |
| 18 | `di_difference_14` | trend | +DI14 − −DI14 (arah tren) | 4/5 |
| 19 | `rsi_14` | momentum | Relative Strength Index 14 | 5/5 |
| 20 | `rsi_14_change_3` | momentum | RSI14[t] − RSI14[t−3] | 3/5 |
| 21 | `macd_histogram_atr` | momentum | Histogram MACD (12, 26, 9) ÷ ATR14 | 5/5 |
| 22 | `stochastic_k_14` | momentum | Stochastic %K 14 | 5/5 |
| 23 | `donchian_position_48` | momentum | Posisi close dalam kanal high/low 48 candle | 4/5 |
| 24 | `atr_14_pct` | volatility | ATR14 ÷ close | 4/5 |
| 25 | `atr_14_vs_daily_mean` | volatility | ATR14 ÷ rata-rata ATR14 288 candle (rezim volatilitas) | 5/5 |
| 26 | `realized_vol_12` | volatility | Simpangan baku log return 12 candle | 4/5 |
| 27 | `realized_vol_48` | volatility | Simpangan baku log return 48 candle | 5/5 |
| 28 | `bollinger_width_20` | volatility | Lebar Bollinger Bands (20, 2) ÷ middle band | 5/5 |
| 29 | `bollinger_position_20` | volatility | Posisi close di dalam Bollinger Bands | 1/5 |
| 30 | `tick_volume_z_48` | volume | Z-score ln(1 + tick volume) terhadap 48 candle | 5/5 |
| 31 | `tick_volume_ratio_12_288` | volume | Rata-rata tick volume 12 ÷ 288 candle | 5/5 |
| 32 | `spread_atr` | spread | Spread ÷ ATR14 (porsi biaya) | 5/5 |
| 33 | `spread_z_288` | spread | Z-score spread terhadap 288 candle | 4/5 |
| 34 | `hour_sin` | time | sin(2π · jam / 24) | 5/5 |
| 35 | `hour_cos` | time | cos(2π · jam / 24) | 5/5 |
| 36 | `day_of_week` | time | Hari dalam minggu (0 = Senin) | 5/5 |
| 37 | `session_asia` | time | 1 bila jam 09–18 waktu Tokyo | 2/5 |
| 38 | `session_london` | time | 1 bila jam 08–17 waktu London | 5/5 |
| 39 | `session_newyork` | time | 1 bila jam 08–17 waktu New York | 2/5 |
| 40 | `session_london_newyork_overlap` | time | 1 bila sesi London dan New York sama-sama buka | 5/5 |
| 41 | `bars_since_gap` | time | Jumlah candle sejak gap terakhir (maksimum 288) | 5/5 |

Sumber: `20261108T025726Z_walk-forward_2a8bb1db55/run.json` dan `feature_selection_xgboost_selected.csv`. Kolom terakhir adalah jumlah fold walk-forward yang mempertahankan fitur tersebut pada xgboost_selected (bab 8.6). Jam dan sesi dihitung dari timestamp tersimpan, yang sebenarnya jam server broker (bab 3.4).
