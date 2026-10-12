# Hasil XGBoost matematis — 11 Oktober 2026

Runtime satu file tanpa paket tambahan sudah diuji memakai `python -S -B`:
site-packages tidak dimuat. Semua import adalah standard library; program baru
tidak memanggil technical, XGBoost, NumPy, pandas, sklearn atau plugin.

## Kesamaan inference

| Fold | Baris aktual diuji | Raw class berbeda | Sinyal berbeda | Alasan policy berbeda | Selisih probabilitas maksimum |
|---|---:|---:|---:|---:|---:|
| 2024, sampel sepanjang tahun | 513 | 0 | 0 | 0 | 3,246 × 10⁻⁸ |
| 2025, sampel sepanjang tahun | 515 | 0 | 0 | 0 | 2,571 × 10⁻⁸ |
| 2026, seluruh TEST tersimpan | 51.236 | 0 | 0 | 0 | 3,549 × 10⁻⁸ |

Final 2026: selisih return maksimum **1,202 × 10⁻¹¹**, interval
**1,163 × 10⁻¹⁰**, harga prediksi terhadap reference **0**. Reference juga
dibandingkan dengan CSV technical sebelumnya: probabilitas/return/class/policy
cocok; selisih harga akibat round-trip CSV ≤9,10 × 10⁻¹³.
Toleransi yang ditetapkan sebelum pengujian: probabilitas 2×10⁻⁶,
return/interval 2×10⁻⁹, harga 0,002. Semua lulus.

BUY=0, SELL=0, NO_TRADE=51.236 pada 2026, sesuai model asal. Pengujian tidak
hanya membandingkan NO_TRADE: seluruh probabilitas, raw class, return, interval
dan harga dibandingkan. Unit tests juga menjalankan policy aktif BUY/SELL.

## Batas klaim

Pohon/bobot diekspor dari model technical yang sudah dilatih; perhitungan
inference ditulis manual. Input fitur diekspor dari pipeline lama setelah
mengolah 754.660 candle aktual penuh. Ekspor awal memakai environment lama;
setelah JSON/CSV tersedia, program baru berjalan mandiri tanpa library tambahan.

Training manual `ScratchClassifier`/`ScratchRegressor` juga tersedia, memakai
exact-split. **Belum ada bukti hasil training ulang identik dengan trainer
histogram technical.** Quantile training, fitting kalibrasi, seleksi fitur,
tuning dan pembuatan 223 fitur OHLC belum direplikasi dari nol di file baru.
Kesamaan di atas berlaku untuk inference pada bobot dan fitur yang sama.

## Menjalankan

```powershell
python -S -B analisis_teknikall/xgboost_matematis.py demo
python -S -B analisis_teknikall/xgboost_matematis.py verify
python -S -B analisis_teknikall/xgboost_matematis.py predict
```

- [Kode satu file](../xgboost_matematis.py)
- [Output manual](predictions_manual.csv)
- [Angka verifikasi 2026](final_2026/parity_report.json)
- [Angka verifikasi 2025](fold_2025/parity_report.json)
- [Angka verifikasi 2024](fold_2024/parity_report.json)
- [Provenance pembanding](provenance.json)
- [Matematika dan cara kerja](../../docs/xgboost-matematis.md)

Kesamaan dengan existing tidak meningkatkan akurasi atau membuktikan kelayakan
trading live. Demo sintetis menampilkan metrik TRAIN, bukan kinerja pasar.

Validasi tambahan: **94 tests passed, 1 skipped** pada full suite repository
(termasuk 9 pengujian matematika baru). Ruff, generated code reference, dan
MkDocs strict build lulus. File output manual berisi 51.236 baris prediksi.
