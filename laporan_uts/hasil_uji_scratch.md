Sumber: `laporan_uts/uji_scratch_bin_2025.json` dari `laporan_uts/uji_scratch_bin.py` (XGBoost 3.4.1). Data: 353.301 candle 2021-01-01 s.d. 2025 dari `runs/20261107T144502Z_walk-forward_13dc74b30d/backtest/candles/validation_*.csv`. TRAIN 2021-01 s.d. 2024-07 (241.310 candle), CALIBRATION 17.755, THRESHOLD 17.448, EVALUATION 2025 68.819 candle.

Tabel 10.21 XGBoost Library vs From Scratch dengan Hyperparameter yang Disamakan (fold 2025)
<!-- widths: 34,22,22,22 -->
| Ukuran | Library | Scratch (kuantil sendiri) | Scratch (batas bin library) |
|---|---|---|---|
| Pohon per kelas (early stopping) | 150 | 150 | 150 |
| Waktu training (detik) | 2,6 | 127,0 | 127,3 |
| Temperatur kalibrasi | 0,933 | 0,933 | 0,933 |
| Threshold BUY / SELL | nonaktif / 0,52 | 0,52 / 0,50 | nonaktif / 0,52 |
| Sinyal BUY / SELL | 0 / 146 | 202 / 897 | 0 / 146 |
| Precision BUY | – | 53,47% | – |
| Precision SELL | 41,10% | 45,04% | 41,10% |
| Coverage | 0,21% | 1,60% | 0,21% |
| Accuracy argmax (tanpa threshold) | 41,62% | 41,83% | 41,62% |
| Macro F1 argmax (tanpa threshold) | 0,3745 | 0,3760 | 0,3745 |
| Log loss EVALUATION | 0,9958 | 0,9957 | 0,9958 |

Tabel 10.22 Kesesuaian From Scratch dengan Library pada Blok EVALUATION 2025
<!-- widths: 50,25,25 -->
| Ukuran | Scratch (kuantil sendiri) | Scratch (batas bin library) |
|---|---|---|
| Kelas argmax sama, semua baris | 92,79% | 100,00% |
| Kelas argmax sama, baris dengan selisih dua probabilitas teratas ≥ 0,01 (78,4% baris) | 98,25% | 100,00% |
| Kelas argmax sama, selisih ≥ 0,02 (59,4% baris) | 99,44% | 100,00% |
| Rata-rata selisih absolut probabilitas | 0,0037 | 2,4 × 10⁻⁶ |
| Korelasi Pearson P(SELL) / P(NO_TRADE) / P(BUY) | 0,9952 / 0,9993 / 0,9962 | 1,0000 / 1,0000 / 1,0000 |
| Keputusan akhir sama (threshold masing-masing) | 98,59% | 100,00% |
| Keputusan akhir sama bila memakai threshold library | 99,89% | 100,00% |

Batas bin hasil `np.quantile` sama dengan batas bin XGBoost hanya pada 2 dari 41 fitur. Varian batas bin library memakai batas dari `xgboost.QuantileDMatrix(...).get_quantile_cut()` pada data TRAINING yang sama dan membulatkan fitur ke float32 seperti XGBoost; rumus gradien, Hessian, bobot daun, gain, dan base score tidak diubah.
