# Hasil eksperimen selective XGBoost

Run selesai 9 Oktober 2026. Source: metaquotes-demo-xauusd; 754.660 candle aktual M5; horizon tiga bar (15 menit).
Model memakai candle aktual yang sudah tutup untuk memperbarui fitur; bobot, seleksi, kalibrasi dan rule tetap beku pada TEST.
Target precision 90% per arah, minimum 50 sinyal pada seleksi dan konfirmasi POLICY, interval nominal 80%, minimum edge 1 bps, slippage asumsi 0 bps.

## Hasil periode TEST

| Fold | Keputusan | Raw macro F1 | Action macro F1 | BUY/SELL | Return RMSE (bps) | Zero-return RMSE (bps) | Coverage interval |
|---|---:|---:|---:|---:|---:|---:|---:|
| fold_2024 | 71,209 | 0.4050 | 0.1248 | 0/0 | 9.1455 | 9.1452 | 79.65% |
| fold_2025 | 70,487 | 0.2997 | 0.0871 | 0/0 | 11.7652 | 11.7665 | 75.32% |
| final_2026 | 51,236 | 0.2190 | 0.0483 | 0/0 | 19.6623 | 19.6599 | 66.92% |

Seluruh fold pada run ini tidak menghasilkan rule BUY/SELL yang memenuhi syarat. Precision arah tidak dapat dihitung (null); target 90% belum tercapai. Nol konflik arah pada action terutama karena seluruh action abstain, bukan bukti classifier memiliki kemampuan arah sempurna.

## Pembanding versi sebelumnya pada timestamp yang sama

| Fold | Baris sama | RMSE return lama → baru (bps) | Raw macro F1 lama → baru | Coverage lama → baru | Konflik arah action lama → baru |
|---|---:|---:|---:|---:|---:|
| fold_2024 | 71,209 | 9.1489 → 9.1455 | 0.4050 → 0.4050 | 71.40% → 0.00% | 15,298 → 0 |
| fold_2025 | 70,487 | 11.7784 → 11.7652 | 0.3197 → 0.2997 | 86.10% → 0.00% | 25,006 → 0 |
| final_2026 | 34,020 | 15.1866 → 15.1462 | 0.2978 → 0.2157 | 85.28% → 0.00% | 8,747 → 0 |

Untuk final_2026, pembanding lama menggunakan development Januari–Maret 2026 dan TEST mulai April. Versi baru membekukan development sebelum Januari. Tabel hanya membandingkan timestamp bersama, tetapi kalender training berbeda; perubahan tidak dapat diatribusikan ke satu teknik saja.

## Interpretasi

- Menekan transaksi mengurangi paparan sinyal salah, tetapi tidak meningkatkan kemampuan prediksi secara otomatis. Nilai F1 action turun ketika recall BUY/SELL menjadi nol.
- Return RMSE perlu dibandingkan dengan zero-return pada setiap fold dan bulan. Perbaikan kecil terhadap versi lama belum membuktikan keunggulan ekonomi atau hasil yang stabil.
- Classifier belum membaik: pada timestamp 2026 yang sama, raw macro F1 turun dari 0,2978 menjadi 0,2157. Pada seluruh Januari–September 2026, return RMSE juga sedikit lebih buruk daripada zero-return. Versi ini memperbaiki aturan abstain dan audit, tetapi belum layak disebut peningkatan akurasi model yang konsisten.
- Interval nominal 80% adalah target rentang return. Coverage aktual dapat menyimpang karena perubahan distribusi; bukan tingkat keberhasilan BUY/SELL.
- Tidak ada retuning menggunakan hasil TEST run ini. Karena hasil 2026 sudah pernah dilihat, ini evaluasi retrospektif; validasi selanjutnya memerlukan data forward baru dan konfigurasi terkunci.
- Belum tersedia fitur eksternal USD/real yield/kalender rilis. Integrasi berikutnya harus memakai waktu ketersediaan historis dan vintage yang benar sebelum menguji nilai tambahnya.

## Artefak

- [Dashboard Januari–September 2026](final_2026/dashboard_2026.html)
- [Grafik diagnostik bulanan 2026](final_2026/monthly_diagnostics.png)
- [Metrik bulanan 2026](final_2026/monthly_metrics.csv)
- [Metrik pada horizon tidak overlap](final_2026/nonoverlapping_metrics.json)
- [Perbandingan timestamp sama](matched_baseline_comparison.json)
- [Metodologi dan referensi primer](../../../docs/technical-selective.md)

Verifikasi: 79 tes lolos, satu tes integrasi database dilewati; tes tambahan pada seleksi regressor, quantile state dan metrik bulanan juga lolos. Ruff, JavaScript syntax check dan MkDocs strict berhasil. Preview browser HTML lokal diblokir kebijakan file URL; grafik PNG diperiksa secara visual.
