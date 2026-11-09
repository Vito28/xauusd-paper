Sumber: candle `runs/20261107T144502Z_walk-forward_13dc74b30d/backtest/candles/` (2021–2025) dan `runs/20261103T150145Z_final-test_062047d030/backtest/candles/` (2026, sampai September); label aktual dari predictions.csv run yang sama. Dihitung oleh `documents/laporan_uts/eda_m5.py`.

Tabel 4.1 Ringkasan Statistik Candle M5 XAU/USD 2021–2026
<!-- widths: 20,11,9,8,8,9,8,10,8,9 -->
| Variabel | n | Rata-rata | Simp. baku | Min | Median | P99 | Maks | Skew. | Kurt. |
|---|---|---|---|---|---|---|---|---|---|
| Log return M5 (bps) | 402.997 | 0,01 | 6,63 | -211,82 | 0,04 | 17,81 | 286,65 | -0,47 | 48,8 |
| Absolute log return (bps) | 402.997 | 4,10 | 5,21 | 0,00 | 2,65 | 23,65 | 286,65 | 6,24 | 109,1 |
| Rentang high − low (USD) | 404.536 | 2,31 | 3,11 | 0,00 | 1,40 | 14,11 | 212,33 | 8,25 | 181,8 |
| ATR14 (USD) | 404.523 | 2,32 | 2,53 | 0,24 | 1,49 | 11,75 | 61,84 | 5,17 | 50,8 |
| Spread (USD) | 404.536 | 0,09 | 0,08 | 0,00 | 0,07 | 0,33 | 5,95 | 5,80 | 214,5 |
| Spread ÷ ATR14 (%) | 404.523 | 6,9 | 8,0 | 0,0 | 4,9 | 34,1 | 1.142,3 | 18,71 | 1.605,0 |
| Tick volume | 404.536 | 632 | 701 | 1 | 387 | 3.473 | 12.957 | 3,07 | 14,8 |

Return dihitung hanya antara dua candle yang berjarak tepat 5 menit, sehingga lompatan harga melewati akhir pekan atau libur tidak dihitung sebagai return M5. Skew. = skewness; Kurt. = excess kurtosis (distribusi normal = 0).

Tabel 4.2 Karakteristik Candle dan Distribusi Label per Tahun
<!-- widths: 10,11,9,9,9,10,10,9,13,10 -->
| Tahun | Candle | Median close (USD) | Median ATR14 (USD) | Median spread (USD) | Median spread ÷ ATR14 | Rata-rata abs. return (bps) | SELL | NO_TRADE | BUY |
|---|---|---|---|---|---|---|---|---|---|
| 2021 | 70.224 | 1.794 | 1,06 | 0,05 | 4,7% | 3,50 | 41,2% | 18,7% | 40,1% |
| 2022 | 70.709 | 1.804 | 1,15 | 0,05 | 4,2% | 3,76 | 42,2% | 16,2% | 41,5% |
| 2023 | 70.671 | 1.945 | 1,03 | 0,07 | 7,0% | 3,08 | 40,6% | 19,4% | 39,9% |
| 2024 | 71.210 | 2.381 | 1,48 | 0,14 | 9,1% | 3,50 | 39,6% | 20,1% | 40,3% |
| 2025 | 70.487 | 3.345 | 2,68 | 0,15 | 4,8% | 4,50 | 41,5% | 16,4% | 42,1% |
| 2026* | 51.235 | 4.523 | 5,34 | 0,04 | 0,8% | 7,07 | 43,8% | 15,1% | 41,1% |

Tanda * pada 2026: data hanya Januari–September. Proporsi label dihitung dari candle berstatus AVAILABLE (horizon 6 candle lengkap dan tidak melintasi gap).

Tabel 4.3 Ukuran Ketergantungan pada Candle M5 2021–2026
<!-- widths: 60,40 -->
| Ukuran | Nilai |
|---|---|
| Autokorelasi log return, lag 1 / lag 12 candle | -0,002 / -0,002 |
| Autokorelasi absolute return, lag 1 / 12 / 288 candle | 0,353 / 0,258 / 0,184 |
| Korelasi Spearman absolute return dengan tick volume | 0,411 |
| Korelasi Spearman absolute return dengan spread | -0,117 |
| Proporsi return M5 bernilai nol | 0,58% |
| Proporsi return lebih dari 4 simpangan baku | 0,74% (normal: 0,01%) |
| Candle dengan spread tercatat 0 | 5,35% (21.643 candle) |
