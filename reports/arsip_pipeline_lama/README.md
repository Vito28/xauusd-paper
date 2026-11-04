# Arsip laporan pipeline lama

Folder ini menyimpan laporan eksperimen dari pipeline Technical Agent generasi
sebelumnya (`selective_v2` / `direction_v3`, horizon 15 menit, label "spread + 1 bps").
Pipeline tersebut sudah diganti oleh modul `technical/` saat ini (label triple-barrier,
walk-forward dengan purging/embargo). Laporan disimpan karena memuat temuan yang
dirujuk `SOURCE_OF_TRUTH.md` bagian F.2.

| File | Isi |
|---|---|
| `laporan_baseline.md` | Konfigurasi, strategi training/validasi, dan baseline (bagian 4.1–4.6) |
| `laporan_optimasi.md` | Metrik, tuning, dan temuan awal |
| `laporan_metodologi_data.md` | Metodologi data untuk XGBoost pipeline lama |
| `klasifikasi_manual.md` | Penjelasan XGBoost classifier manual (versi lama) |

Kode dan artefak yang dirujuk tautan di dalam laporan (misalnya
`artifacts/direction_v3/...` atau `xgboost_matematis.py`) sudah dihapus dari branch
`main`. Semuanya masih bisa dibuka dari riwayat git:

```bash
git show a0e31c0:technical/artifacts/direction_v3/final_2026/metrics.json
git show a0e31c0:analisis_teknikall/xgboost_matematis.py
```

Angka di laporan ini **tidak** dapat dibandingkan langsung dengan hasil pipeline baru
karena definisi label dan horizon berbeda.
