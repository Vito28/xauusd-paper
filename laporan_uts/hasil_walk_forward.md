Tabel 10.2 Data dan Run Walk-Forward yang Dipakai
<!-- widths: 34,66 -->
| Atribut | Nilai |
|---|---|
| Run | `20261107T042522Z_walk-forward_13dc74b30d`<br>`20261107T144502Z_walk-forward_13dc74b30d` |
| Sumber data | `postgres:market.processed_m5:metaquotes-demo-xauusd` |
| Candle masuk / valid / dipakai | 703.425 / 703.425 / 703.425 |
| Rentang | 2016-01-04T01:05 – 2026-01-01T00:00 UTC |
| Transisi gap | 2.642 |
| Fitur | 41 fitur, 7 kelompok |
| Commit kode / config hash | `1b62d62` / `13dc74b30d` |
| Versi paket | python 3.12.13, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, xgboost 3.4.1 |

Tabel 10.3 Ringkasan Walk-Forward 2021–2025 per Model (rata-rata ± simpangan baku lima fold)
<!-- widths: 22,14,14,18,18,14 -->
| Model | Macro F1 | Directional F1 | Precision BUY | Precision SELL | Coverage sinyal |
|---|---|---|---|---|---|
| always_no_trade | 0,1025 ± 0,0076 | 0,0000 ± 0,0000 | – | – | 0,00% ± 0,00% |
| persistence | 0,3043 ± 0,0050 | 0,3035 ± 0,0074 | 41,50% ± 0,95% | 42,52% ± 1,01% | 46,33% ± 1,67% |
| logistic | 0,1244 ± 0,0183 | 0,0307 ± 0,0304 | 47,51% ± 0,68% (4 fold) | 46,05% ± 3,09% (3 fold) | 3,05% ± 3,25% |
| xgboost | 0,2031 ± 0,0523 | 0,1371 ± 0,0709 | 47,82% ± 1,58% | 47,24% ± 1,35% (4 fold) | 15,97% ± 8,83% |
| xgboost_balanced | 0,1969 ± 0,0364 | 0,1284 ± 0,0472 | 48,52% ± 1,91% | 48,22% ± 2,01% | 14,77% ± 6,60% |
| random_forest | 0,1491 ± 0,0503 | 0,0633 ± 0,0599 | 48,42% ± 2,48% (4 fold) | 45,84% ± 1,62% | 6,76% ± 6,88% |
| ensemble | 0,1944 ± 0,0434 | 0,1253 ± 0,0580 | 47,84% ± 1,54% | 47,73% ± 1,77% (4 fold) | 14,14% ± 7,21% |

Keterangan: "(n fold)" berarti precision hanya terdefinisi pada n fold karena pada fold lain arah tersebut dinonaktifkan atau tidak menghasilkan sinyal.

Catatan reproduksi: model xgboost juga ada di `20261107T144502Z_walk-forward_13dc74b30d` dan ringkasannya identik (hasil dapat direproduksi).

Tabel 10.4 Directional F1 per Tahun Evaluasi
| Model | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|
| always_no_trade | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| persistence | 0,2990 | 0,3057 | 0,3006 | 0,2954 | 0,3166 |
| logistic | 0,0373 | 0,0867 | 0,0000 | 0,0156 | 0,0141 |
| xgboost | 0,1452 | 0,1902 | 0,0092 | 0,2132 | 0,1277 |
| xgboost_balanced | 0,1552 | 0,1686 | 0,0525 | 0,1717 | 0,0941 |
| random_forest | 0,1035 | 0,0197 | 0,0130 | 0,1624 | 0,0178 |
| ensemble | 0,1313 | 0,1565 | 0,0200 | 0,1942 | 0,1246 |

Tabel 10.5 Coverage Sinyal per Tahun Evaluasi
| Model | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|
| always_no_trade | 0,00% | 0,00% | 0,00% | 0,00% | 0,00% |
| persistence | 46,69% | 48,09% | 45,33% | 43,65% | 47,90% |
| logistic | 3,50% | 9,14% | 0,00% | 1,32% | 1,30% |
| xgboost | 15,16% | 22,84% | 0,81% | 26,45% | 14,59% |
| xgboost_balanced | 16,94% | 18,88% | 4,75% | 23,30% | 9,97% |
| random_forest | 10,58% | 1,82% | 1,14% | 18,59% | 1,66% |
| ensemble | 13,52% | 17,04% | 1,81% | 24,10% | 14,24% |

Tabel 10.6 Hasil per Tahun Evaluasi Model xgboost
<!-- widths: 9,10,18,18,11,14,10,10 -->
| Tahun | Macro F1 | BUY: precision (sinyal) | SELL: precision (sinyal) | Coverage | Threshold BUY / SELL | Temperatur | Pohon |
|---|---|---|---|---|---|---|---|
| 2021 | 0,2105 | 48,16% (8.439) | 49,34% (1.968) | 15,16% | 0,48 / 0,54 | 0,831 | 300 |
| 2022 | 0,2320 | 48,06% (12.966) | 46,11% (2.828) | 22,84% | 0,46 / 0,50 | 0,933 | 300 |
| 2023 | 0,1149 | 46,15% (559) | – (0) | 0,81% | 0,54 / nonaktif | 0,831 | 299 |
| 2024 | 0,2717 | 46,26% (15.066) | 47,51% (3.359) | 26,45% | 0,44 / 0,48 | 1,023 | 295 |
| 2025 | 0,1863 | 50,47% (1.058) | 45,99% (8.981) | 14,59% | 0,52 / 0,48 | 0,851 | 295 |

Pohon = best_iteration + 1, yaitu jumlah ronde yang dipakai setelah early stopping (maksimal 300).

Tabel 10.7 Hasil per Tahun Evaluasi Model ensemble
<!-- widths: 9,10,18,18,11,14,10,10 -->
| Tahun | Macro F1 | BUY: precision (sinyal) | SELL: precision (sinyal) | Coverage | Threshold BUY / SELL | Temperatur | Pohon |
|---|---|---|---|---|---|---|---|
| 2021 | 0,2008 | 48,17% (7.825) | 50,41% (1.452) | 13,52% | 0,48 / 0,54 | 0,955 | – |
| 2022 | 0,2067 | 48,16% (8.577) | 46,21% (3.205) | 17,04% | 0,48 / 0,50 | 0,977 | – |
| 2023 | 0,1227 | 46,28% (1.249) | – (0) | 1,81% | 0,52 / nonaktif | 0,977 | – |
| 2024 | 0,2576 | 46,22% (14.527) | 48,25% (2.263) | 24,10% | 0,44 / 0,48 | 1,000 | – |
| 2025 | 0,1841 | 50,39% (901) | 46,06% (8.902) | 14,24% | 0,52 / 0,48 | 0,955 | – |

Pohon = best_iteration + 1, yaitu jumlah ronde yang dipakai setelah early stopping (maksimal 300).

Tabel 10.8 Lift Precision terhadap Prevalensi Kelas pada Blok EVALUATION
<!-- widths: 22,13,13,13,13,13,13 -->
| Model / tahun | Prevalensi BUY | Precision BUY | Lift BUY | Prevalensi SELL | Precision SELL | Lift SELL |
|---|---|---|---|---|---|---|
| persistence / 2021 | 40,1% | 40,0% | 1,00 | 41,2% | 42,0% | 1,02 |
| persistence / 2022 | 41,5% | 42,3% | 1,02 | 42,2% | 41,5% | 0,98 |
| persistence / 2023 | 39,9% | 41,3% | 1,03 | 40,6% | 42,2% | 1,04 |
| persistence / 2024 | 40,3% | 41,1% | 1,02 | 39,6% | 42,6% | 1,08 |
| persistence / 2025 | 42,1% | 42,7% | 1,02 | 41,5% | 44,4% | 1,07 |
| xgboost / 2021 | 40,1% | 48,2% | 1,20 | 41,2% | 49,3% | 1,20 |
| xgboost / 2022 | 41,5% | 48,1% | 1,16 | 42,2% | 46,1% | 1,09 |
| xgboost / 2023 | 39,9% | 46,2% | 1,16 | 40,6% | – | – |
| xgboost / 2024 | 40,3% | 46,3% | 1,15 | 39,6% | 47,5% | 1,20 |
| xgboost / 2025 | 42,1% | 50,5% | 1,20 | 41,5% | 46,0% | 1,11 |
| ensemble / 2021 | 40,1% | 48,2% | 1,20 | 41,2% | 50,4% | 1,22 |
| ensemble / 2022 | 41,5% | 48,2% | 1,16 | 42,2% | 46,2% | 1,09 |
| ensemble / 2023 | 39,9% | 46,3% | 1,16 | 40,6% | – | – |
| ensemble / 2024 | 40,3% | 46,2% | 1,15 | 39,6% | 48,3% | 1,22 |
| ensemble / 2025 | 42,1% | 50,4% | 1,20 | 41,5% | 46,1% | 1,11 |

Lift = precision ÷ prevalensi. Lift 1 setara tebakan acak; kriteria keberhasilan proyek lift ≥ 1,2.

Tabel 10.9 Distribusi Label per Fold (blok TRAIN dan EVALUATION)
<!-- widths: 10,15,15,15,15,15,15 -->
| Tahun | TRAIN SELL | TRAIN NO_TRADE | TRAIN BUY | EVAL SELL | EVAL NO_TRADE | EVAL BUY |
|---|---|---|---|---|---|---|
| 2021 | 33,8% | 32,7% | 33,4% | 41,2% | 18,7% | 40,1% |
| 2022 | 35,2% | 30,1% | 34,7% | 42,2% | 16,2% | 41,5% |
| 2023 | 36,2% | 28,2% | 35,7% | 40,6% | 19,4% | 39,9% |
| 2024 | 36,9% | 26,7% | 36,4% | 39,6% | 20,1% | 40,3% |
| 2025 | 37,3% | 26,0% | 36,8% | 41,5% | 16,4% | 42,1% |

Tabel 10.10 Confusion Matrix Gabungan Lima Fold Model xgboost (baris aktual, kolom prediksi)
| Aktual \ Prediksi | SELL | NO_TRADE | BUY |
|---|---|---|---|
| SELL | 8.001 | 116.355 | 17.364 |
| NO_TRADE | 1.134 | 59.014 | 2.668 |
| BUY | 8.001 | 114.764 | 18.056 |

Total sinyal BUY/SELL 55.224; kesalahan arah berlawanan (BUY saat label SELL atau sebaliknya) 25.365, atau 45,9% dari sinyal.

Tabel 10.11 Bobot dan Log Loss Ensemble per Fold (dipilih pada blok CALIBRATION)
| Tahun | Bobot xgboost | Bobot random_forest | Bobot logistic | Log loss ensemble | Anggota terbaik (log loss) |
|---|---|---|---|---|---|
| 2021 | 0,80 | 0,15 | 0,05 | 0,9895 | xgboost (0,9900) |
| 2022 | 0,35 | 0,65 | 0,00 | 0,9922 | random_forest (0,9927) |
| 2023 | 0,50 | 0,45 | 0,05 | 0,9726 | xgboost (0,9734) |
| 2024 | 0,55 | 0,45 | 0,00 | 0,9950 | xgboost (0,9956) |
| 2025 | 0,80 | 0,10 | 0,10 | 1,0107 | xgboost (1,0110) |

Tabel 10.12 Keragaman Kesalahan Antaranggota Ensemble (rata-rata lima fold, blok CALIBRATION)
| Pasangan anggota | Disagreement | Korelasi kesalahan |
|---|---|---|
| xgboost – random_forest | 28,9% | 0,511 |
| xgboost – logistic | 38,1% | 0,370 |
| random_forest – logistic | 44,2% | 0,266 |

Tabel 10.13 Simulasi Transaksi Gabungan 2021–2025 per Model (lot 0,01, setelah spread)
<!-- widths: 22,11,10,9,9,11,11,17 -->
| Model / periode | Transaksi | Win rate | TP | SL | Waktu habis | Profit factor | Net P&L USD |
|---|---|---|---|---|---|---|---|
| logistic | 10.533 | 48,7% | 46,5% | 48,9% | 4,6% | 0,89 | -1.058,06 |
| xgboost | 55.224 | 49,8% | 47,2% | 46,9% | 5,9% | 0,91 | -4.700,39 |
| xgboost_balanced | 51.050 | 49,8% | 47,4% | 47,2% | 5,4% | 0,90 | -4.170,06 |
| random_forest | 23.402 | 48,8% | 46,6% | 48,7% | 4,7% | 0,86 | -2.908,33 |
| ensemble | 48.901 | 49,6% | 47,1% | 47,4% | 5,5% | 0,90 | -4.652,72 |

Tabel 10.14 Laporan Spread Transaksi 2021–2025 per Model
<!-- widths: 20,13,16,17,15,19 -->
| Model | Transaksi | Spread / harga buka | Spread / jarak TP | Biaya spread USD | P&L kotor → bersih USD |
|---|---|---|---|---|---|
| logistic | 10.533 | 0,0026% | 3,4% | 560,30 | -497,76 → -1.058,06 |
| xgboost | 55.224 | 0,0039% | 6,1% | 5.016,13 | 315,74 → -4.700,39 |
| xgboost_balanced | 51.050 | 0,0039% | 6,2% | 4.506,39 | 336,33 → -4.170,06 |
| random_forest | 23.402 | 0,0041% | 6,2% | 2.179,48 | -728,85 → -2.908,33 |
| ensemble | 48.901 | 0,0039% | 5,9% | 4.469,27 | -183,45 → -4.652,72 |

Tabel 10.15 Simulasi Transaksi XGBoost per Tahun
<!-- widths: 22,11,10,9,9,11,11,17 -->
| Model / periode | Transaksi | Win rate | TP | SL | Waktu habis | Profit factor | Net P&L USD |
|---|---|---|---|---|---|---|---|
| 2021 | 10.407 | 50,9% | 48,4% | 46,4% | 5,3% | 0,97 | -177,73 |
| 2022 | 15.794 | 50,2% | 47,7% | 46,7% | 5,6% | 0,93 | -711,80 |
| 2023 | 559 | 49,0% | 46,2% | 46,7% | 7,2% | 0,85 | -50,80 |
| 2024 | 18.425 | 49,3% | 46,5% | 47,0% | 6,5% | 0,84 | -2.680,73 |
| 2025 | 10.039 | 48,8% | 46,5% | 47,8% | 5,7% | 0,93 | -1.079,33 |

Simulasi tingkat candle: buka di close candle keputusan, tutup pada TP/SL atau close candle ke-6, spread tutup = spread buka, posisi boleh tumpang tindih, tanpa slippage/komisi/swap. Angka mengukur kualitas sinyal setelah biaya, bukan proyeksi saldo akun.
