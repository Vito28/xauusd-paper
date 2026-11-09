Sumber: `20261103T150145Z_final-test_062047d030` (model xgboost, dibuat 2026-11-03T15:01 UTC, commit `01172c4`, config hash `062047d030`).

Tabel 10.18 Hasil Uji Out-of-Sample Retrospektif 2026 Model xgboost
<!-- widths: 36,64 -->
| Atribut | Nilai |
|---|---|
| Periode keputusan | 2026-01-01T00:00 – 2026-09-30T06:45 UTC |
| Candle dinilai (label lengkap) | 50.071 (GAP 1.159, PENDING 6) |
| Candle TRAIN / CALIBRATION / THRESHOLD | 652.935 / 17.442 / 17.298 |
| Pohon setelah early stopping | 300 dari maksimal 300 |
| Temperatur kalibrasi | 0,933 (log loss 0,9899 → 0,9896) |
| Threshold BUY | nonaktif (prevalensi blok THRESHOLD 42,6%; syarat precision 51,1% tidak tercapai) |
| Threshold SELL | 0,52 (59 sinyal, precision 52,5% pada Okt–Des 2025; syarat 50,7%) |
| Sinyal BUY / SELL | 0 / 136 (coverage 0,27%) |
| Precision BUY | – (tidak ada sinyal) |
| Precision SELL | 51,47% (Wilson 95%: 43,1%–59,7%) |
| Prevalensi BUY / SELL 2026 | 41,1% / 43,8% |
| Macro F1 / directional F1 / accuracy | 0,0897 / 0,0032 / 15,22% |
| Log loss / Brier / ECE | 0,9824 / 0,6047 / 0,0046 |

Tabel 10.19 Simulasi Transaksi 2026 (lot 0,01, setelah spread)
<!-- widths: 22,11,10,9,9,11,11,17 -->
| Model / periode | Transaksi | Win rate | TP | SL | Waktu habis | Profit factor | Net P&L USD |
|---|---|---|---|---|---|---|---|
| xgboost / 2026 | 136 | 54,4% | 51,5% | 44,9% | 3,7% | 1,21 | 116,76 |

Tabel 10.20 Frekuensi Sinyal Model xgboost per Periode
<!-- widths: 10,10,10,8,10,11,10,13,18 -->
| Periode | Candle | BUY | SELL | Rata-rata 1 sinyal tiap | Hari ada sinyal | Median sinyal per hari aktif | Sinyal berurutan | Jam tersibuk UTC (WIB) |
|---|---|---|---|---|---|---|---|---|
| 2021 | 70.224 | 8.467 | 2.111 | 33 menit | 258 / 258 | 40 | 54% | 06 (13), 05 (12) |
| 2022 | 70.709 | 12.986 | 2.851 | 22 menit | 258 / 258 | 61 | 62% | 06 (13), 05 (12) |
| 2023 | 70.671 | 559 | 0 | 10,5 jam | 162 / 257 | 2 | 33% | 05 (12), 06 (13) |
| 2024 | 71.209 | 15.092 | 3.363 | 19 menit | 259 / 259 | 73 | 72% | 12 (19), 13 (20) |
| 2025 | 70.487 | 1.071 | 9.020 | 35 menit | 258 / 259 | 37 | 56% | 13 (20), 12 (19) |
| 2026 (final-test) | 51.236 | 0 | 140 | 30,5 jam | 56 / 194 | 2 | 36% | 13 (20), 12 (19) |

Model menilai setiap candle M5 yang sudah tutup. Kolom candle mencakup candle yang labelnya belum lengkap (GAP/PENDING), sehingga jumlah sinyal di sini dapat sedikit lebih besar daripada jumlah sinyal yang dinilai. Sinyal berurutan adalah proporsi sinyal yang berjarak tepat 5 menit dari sinyal sebelumnya. Jam ditulis dalam UTC setelah koreksi zona waktu server (bab 3.4), dengan WIB (UTC+7) di dalam kurung.
