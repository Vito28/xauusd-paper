# XGBoost matematis dalam satu file

Kode: `analisis_teknikall/xgboost_matematis.py`, dengan section `# %%` dan
komentar rumus. Runtime hanya memakai standard library Python; tidak mengimpor
XGBoost, NumPy, pandas, sklearn, technical, atau plugin. JSON/CSV adalah data.

Penjelasan bertahap beserta potongan kode classifier dan contoh hitungan satu
ronde tersedia di `analisis_teknikall/README_klasifikasi_manual.md`.

## Dua pekerjaan yang berbeda

1. **Training dari nol:** `ScratchRegressor` dan `ScratchClassifier` membangun
   pohon sendiri dari gradient/Hessian, exact split, leaf dan boosting.
2. **Kesamaan output existing:** `TechnicalMath` membaca pohon/bobot technical
   dalam JSON dan menghitung prediksinya sendiri: traversal, margin, softmax,
   sigmoid calibration, return, interval dan policy. Ini inference manual
   model beku, bukan hasil training ulang dari nol.

Exact-split tidak diklaim identik dengan trainer histogram XGBoost yang memakai
quantile sketch, subsampling, sampling kolom, early stopping dan operasi paralel.
Fitting kalibrasi, quantile training, seleksi fitur dan tuning pipeline lama
belum direplikasi sebagai trainer murni. Kesamaan diuji **pada pohon yang sama
dan fitur yang sama**, dengan toleransi numerik; bukan klaim bit-identical untuk
semua input masa depan.

## Section dan matematika

| Section | Isi |
|---|---|
| 01 | Softmax stabil, sigmoid, float32 dan validasi |
| 02 | Pohon Newton, gradient/Hessian, L1/L2, gain, missing-value routing |
| 03 REGRESSION | Training scalar squared-loss/pseudo-Huber |
| 04 CLASSIFIER | Training tiga kelas, cross entropy, tree per kelas |
| 05 | Pembacaan JSON dan perhitungan pohon technical manual |
| 06 | Kalibrasi classifier, regresi/quantile dan policy |
| 07 | Evaluasi arah, gross/net dan pembandingan reference |
| 08 | Demo dan CLI predict/verify |

Untuk satu leaf: `G=sum(g_i)`, `H=sum(h_i)`,
`S(G)=sign(G)*max(abs(G)-alpha,0)`, dan `w=-S(G)/(H+lambda)`.
Split gain: `0.5*(S(GL)^2/(HL+lambda) + S(GR)^2/(HR+lambda)
- S(G)^2/(H+lambda)) - gamma`. Pilih gain positif terbesar.

Regresi squared loss memakai `g=prediction-target`, `h=1`.
Pseudo-Huber: residual `r`, `g=r/sqrt(1+(r/delta)^2)`,
`h=(1+(r/delta)^2)^(-3/2)`. Boosting menambah `eta*tree(x)` setiap ronde.

Classifier: `p_k=exp(z_k-max(z))/sum(exp(z_j-max(z)))`,
`g_k=p_k-1[y=k]`, Hessian diagonal upper bound
`h_k=max(2*p_k*(1-p_k),1e-16)`. Tiga tree per ronde; gradient seluruh kelas
dihitung sebelum margin ronde itu diperbarui. Label: SELL=0, NO_TRADE=1, BUY=2.

Leaf JSON sudah memasukkan learning rate; jangan mengalikan eta lagi.
`tree_info` menentukan kelas/quantile. Nilai sama dengan threshold masuk kanan;
NaN mengikuti arah default. Kolom fitur yang tidak ada ditolak, bukan diisi nol.
Kalibrasi: `logit=log(p/(1-p))`, `q=sigmoid(a*logit+b)`, normalisasi `q`.
Regressor bps dibagi 10.000 dan dikali shrinkage; quantile diperluas residual
kalibrasi. Policy tidak menerima actual masa depan.

Rumus merujuk [tutorial resmi boosted trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html).
Pemisahan pohon terlatih dan konfigurasi training mengikuti
[dokumentasi model IO](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html).

## Menjalankan tanpa library tambahan

```powershell
python -S -B analisis_teknikall/xgboost_matematis.py demo
python -S -B analisis_teknikall/xgboost_matematis.py verify
python -S -B analisis_teknikall/xgboost_matematis.py predict
```

`-S` menonaktifkan site-packages. Demo adalah data sintetis, metrik TRAIN;
bukan kinerja emas/holdout. `verify` memeriksa probabilitas, confidence, return,
interval, harga, raw class, final class dan alasan policy. Kode keluar nonzero
jika toleransi/kesamaan gagal. `predict` menulis
`analisis_teknikall/artifacts/predictions_manual.csv`.

Default model final 2026; `--model`, `--input`, `--reference`, `--output`
dapat diubah. Folder model memuat `classifier.json`, `regressor.json`,
`quantile.json`, `contract.json`; hash dan urutan fitur diperiksa.

**Input adalah CSV fitur numerik**, semua kolom model ditambah `close` dan
`spread_price`. `actual_future_close` opsional hanya untuk evaluasi sesudah
prediksi. File ini belum membangun ulang 223 fitur OHLC dari nol. CSV pembanding
diekspor sekali dari pipeline existing agar input persis sama. Ekspor awal
memakai environment library existing; inference manual berjalan mandiri.

## Bukti kesamaan dan batasnya

`parity_report.json` per fold memuat jumlah baris, selisih maksimum, toleransi,
mismatch dan hash input/reference. Numeric tetap dibandingkan meskipun policy
seluruhnya NO_TRADE; unit tests juga menguji rule yang menghasilkan BUY/SELL.
Hasil nyata ada di `analisis_teknikall/artifacts/hasil_verifikasi.md`.

Sama dengan existing tidak berarti akurasi meningkat. Training scratch dapat
diuji sebagai eksperimen terpisah dengan input TRAIN dan validasi waktu yang
terpisah; ia tidak otomatis mewarisi kinerja ataupun policy technical.
