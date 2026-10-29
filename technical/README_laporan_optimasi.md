# Optimasi Baseline XGBoost — Metrik, Hasil, Interpretasi, dan Temuan Awal

Dokumen ini melanjutkan [laporan eksperimen baseline](README_laporan_baseline.md)
dan disusun pada 21 Oktober 2026 dari artifact `direction_v3/final_2026`.
Tidak ada training ulang atau perubahan model dalam penyusunan laporan ini.

Hasil yang tersedia sudah melalui seleksi fitur, tuning, kalibrasi, dan
selective policy. Karena hasil XGBoost default sebelum optimasi tidak
tersimpan sebagai eksperimen terpisah, laporan ini tidak mengklaim peningkatan
sebelum–sesudah optimasi pada TEST. Pembandingan yang dapat dilakukan adalah
antar-kandidat pada TUNE, kalibrasi pada blok CALIBRATION, dan regresi terhadap
baseline nol return pada periode yang sama.

## 1. Metrik Evaluasi

| Komponen | Metrik | Tujuan dan cara membaca |
|---|---|---|
| Pemilihan classifier | Macro F1 | Rata-rata F1 tiga kelas dengan bobot sama; lebih besar lebih baik |
| Classifier | Precision per kelas | Proporsi prediksi suatu kelas yang sesuai actual class |
| Classifier | Recall per kelas | Proporsi actual class yang berhasil dikenali |
| Classifier | Accuracy | Proporsi klasifikasi benar; dibaca bersama metrik per kelas |
| Regresi | RMSE return | Besar error return dengan penalti lebih besar pada error besar; lebih kecil lebih baik |
| Regresi | MAE return/harga | Rata-rata absolute error; lebih kecil lebih baik |
| Regresi | Accuracy arah | Kesesuaian tanda prediksi return dengan tanda actual return |
| Kalibrasi probabilitas | Log loss dan multiclass Brier score | Menilai probabilitas terhadap actual class; lebih kecil lebih baik |
| Interval return | Coverage empiris | Proporsi actual return yang berada dalam interval prediksi |
| Keputusan trading | Directional precision BUY/SELL | Ketepatan arah pada sinyal trade yang diterima policy |
| Keputusan trading | Trade coverage dan jumlah sinyal | Seberapa sering model mengizinkan trade |

Macro F1, precision, recall, dan accuracy classifier mengacu pada label
cost-aware: SELL=0, NO_TRADE=1, BUY=2. Directional precision policy menggunakan
definisi berbeda: BUY sesuai bila actual return positif dan SELL bila negatif.
NO_TRADE selalu netral pada evaluasi arah. Ketepatan arah dan profit setelah
biaya juga merupakan ukuran yang terpisah.

Return regresi dilaporkan dalam basis point: `1 bps = 0,01%`, sehingga return
dikalikan 10.000 untuk ditampilkan sebagai bps. F1 pada tabel menggunakan
skala 0–1, sedangkan precision/recall/accuracy ditampilkan dalam persen.

## 2. Hasil Optimasi yang Tersedia

### 2.1 Rancangan dan konfigurasi terpilih

Eksperimen memakai input XAUUSD M5 dengan target tiga candle atau tepat
15 menit. Sebanyak 223 kandidat fitur dibentuk, kemudian dipilih 15 fitur
classifier dan 40 fitur regressor secara terpisah pada TRAIN.

| Blok | Periode UTC | Jumlah baris | Fungsi |
|---|---|---:|---|
| TRAIN | 2016–2024 | 625.909 | Seleksi fitur dan fitting awal |
| TUNE | Januari–Juni 2025 | 34.465 | Pemilihan parameter, early stopping, shrinkage |
| CALIBRATION | Juli–September 2025 | 17.657 | Fitting kalibrasi probabilitas dan interval |
| POLICY | Oktober–Desember 2025 | 17.507 | Seleksi serta konfirmasi aturan trade |
| TEST | Januari–September 2026 | 51.236 | Evaluasi model beku |

Setelah tuning, tree di-fit kembali pada TRAIN+TUNE. Semua komponen kemudian
dibekukan sebelum TEST. Input TEST diperbarui memakai candle aktual yang
telah selesai, tanpa menggantinya dengan candle hasil prediksi.

| Parameter terpilih | Classifier | Regressor |
|---|---|---|
| Objective | `multi:softprob` | `reg:squarederror` |
| `max_depth` | 4 | 4 |
| `learning_rate` | 0,03 | 0,03 |
| `min_child_weight` | 24 | 24 |
| `reg_lambda` | 12,0 | 12,0 |
| Boosting rounds | 160 | 21 |
| Balanced sample weights | Ya | Tidak digunakan |
| Shrinkage keluaran regresi | Tidak berlaku | 1,0 |

Jumlah boosting rounds multiclass tidak sama dengan jumlah seluruh tree
individual. Early stopping menggunakan patience 20 rounds. Tidak ada
bukti dari artifact ini bahwa menambah jumlah round di atas batas eksperimen
akan memperbaiki hasil.

### 2.2 Hasil tuning classifier

Terdapat enam kandidat classifier: tiga kombinasi parameter, masing-masing
dengan dan tanpa balanced sample weights. Kandidat dipilih berdasarkan
macro F1 pada TUNE. Early stopping memantau multiclass log loss.

| Kandidat | Depth | Learning rate | Min child weight | Lambda | Balanced weights | Macro F1 TUNE |
|---|---:|---:|---:|---:|---|---:|
| C1 | 2 | 0,04 | 24 | 12 | Tidak | 0,3394 |
| C2 | 2 | 0,04 | 24 | 12 | Ya | 0,3452 |
| C3 | 3 | 0,04 | 16 | 8 | Tidak | 0,3434 |
| C4 | 3 | 0,04 | 16 | 8 | Ya | 0,3492 |
| C5 | 4 | 0,03 | 24 | 12 | Tidak | 0,3435 |
| C6 — terpilih | 4 | 0,03 | 24 | 12 | Ya | **0,3525** |

Pada ketiga pasangan parameter yang diuji, balanced sample weights memberikan
macro F1 TUNE yang lebih tinggi. Pada parameter terpilih, selisih terhadap
versi tanpa weighting adalah sekitar **0,0090 poin F1**. Hasil ini berlaku
pada kandidat dan periode TUNE tersebut, bukan bukti peningkatan universal
atau peningkatan TEST sebelum–sesudah optimasi.

### 2.3 Hasil tuning regresi dan pembanding nol return

Regresi menguji tiga kombinasi parameter, dua objective (squared error dan
pseudo-Huber), serta empat faktor shrinkage keluaran (0,1; 0,25; 0,5; 1,0).
Ini menghasilkan 24 konfigurasi evaluasi; sebagian memakai model hasil
fitting yang sama dengan faktor shrinkage berbeda. Pemilihan berdasarkan
RMSE return pada TUNE.

| Periode | XGBoost terpilih: RMSE return (bps) | Baseline nol return: RMSE (bps) | Selisih XGBoost − baseline |
|---|---:|---:|---:|
| TUNE Jan–Jun 2025 | 11,4216 | 11,4226 | −0,0010 bps |
| TEST Jan–Sep 2026 | 19,6623 | 19,6599 | +0,0024 bps |

Baseline memprediksi return nol, ekuivalen dengan mempertahankan harga close
keputusan sebagai prediksi harga target. Pada TUNE, pengurangan RMSE terhadap
baseline hanya sekitar **0,0089% secara relatif**. Pada TEST, XGBoost justru
memiliki RMSE sedikit lebih tinggi. Keunggulan TUNE yang sangat kecil belum
memberikan bukti manfaat praktis dan tidak bertahan pada TEST.

### 2.4 Hasil kalibrasi probabilitas

| Metrik pada CALIBRATION | Sebelum kalibrasi | Sesudah kalibrasi |
|---|---:|---:|
| Log loss | 1,0342 | 1,0307 |
| Multiclass Brier score | 0,6296 | 0,6275 |

Kalibrasi sigmoid one-vs-rest memberikan penurunan kedua metrik pada blok
CALIBRATION. Namun, parameter kalibrasi juga di-fit pada blok yang sama.
Angka ini merupakan hasil pada blok fitting kalibrasi, bukan bukti peningkatan
probabilitas pada data TEST yang terpisah. Artifact yang dibahas belum
menyediakan tabel log loss TEST sebelum–sesudah kalibrasi.

### 2.5 Hasil akhir pada TEST 2026

Sebanyak 50.653 dari 51.236 keputusan memiliki actual tersedia untuk
evaluasi. Sebanyak 580 target melintasi gap dan tiga masih pending.

**Classifier setelah kalibrasi, sebelum selective policy:**

| Kelas | Precision | Recall | F1-score | Actual support |
|---|---:|---:|---:|---:|
| BUY | 46,47% | 94,18% | 0,6223 | 23.392 |
| SELL | 48,64% | 6,74% | 0,1185 | 23.307 |
| NO_TRADE | 41,67% | 0,13% | 0,0025 | 3.954 |

Accuracy classifier sebesar **46,61%**, macro F1 **0,2478**, dan weighted F1
**0,3421**. Nama metrik pada artifact adalah `raw_classifier`, tetapi
keluaran ini sudah memakai probabilitas terkalibrasi, hanya belum melalui
selective policy.

Macro F1 TUNE 0,3525 berasal dari kandidat sebelum refit dan sebelum
kalibrasi. Macro F1 TEST 0,2478 berasal dari model yang telah refit dan
kalibrasi pada periode berbeda. Keduanya tidak boleh diperlakukan sebagai
perbandingan terkontrol yang hanya berbeda periode atau sebagai pengukuran
langsung efek tuning.

**Regresi dan interval:**

| Metrik TEST | Hasil |
|---|---:|
| MAE harga | 5,6251 satuan harga XAUUSD |
| RMSE harga | 9,1810 satuan harga XAUUSD |
| MAE return | 12,2541 bps |
| RMSE return | 19,6623 bps |
| Accuracy arah | 50,30% |
| Coverage interval nominal | 80% |
| Coverage interval empiris | 66,92% |

**Sinyal setelah selective policy:**

| Indikator | Hasil TEST |
|---|---:|
| BUY | 0 |
| SELL | 0 |
| NO_TRADE | 51.236 |
| Trade coverage | 0% |
| Directional precision BUY | Tidak terdefinisi; tidak ada trade |
| Directional precision SELL | Tidak terdefinisi; tidak ada trade |
| Win rate trade | Tidak terdefinisi; tidak ada trade |

Policy memakai target precision arah 90%, minimal 50 sinyal per arah pada
setiap tahap selection/confirmation, dan persyaratan kesesuaian return serta
interval terhadap biaya. Tidak ada aturan selection yang lolos, sehingga
kedua arah dinonaktifkan. Gate audit menunjukkan nol kandidat dengan interval
yang sepenuhnya melewati biaya pada kedua tahap. Ini berbeda dari adanya
sekumpulan trade yang cukup besar dengan precision terukur mencapai 90%.

## 3. Interpretasi Singkat

Optimasi classifier menemukan konfigurasi terbaik di antara enam kandidat
yang diuji. Balanced sample weights meningkatkan macro F1 TUNE pada pasangan
parameter yang tersedia, tetapi hasil TEST masih memperlihatkan kecenderungan
prediksi BUY. Recall BUY tinggi belum disertai precision BUY tinggi, sementara
kemampuan mengenali SELL dan kelas NO_TRADE masih rendah.

Optimasi regresi belum memberikan peningkatan yang konsisten terhadap
baseline nol return. Selisih RMSE pada TUNE sangat kecil, sedangkan pada TEST
RMSE XGBoost sedikit lebih buruk. Accuracy arah sekitar 50% belum menunjukkan
bukti ketepatan arah yang memenuhi sasaran trading penelitian.

Kalibrasi memperbaiki log loss dan Brier score pada blok tempat kalibrasi
di-fit. Hasil tersebut belum cukup untuk menyatakan kualitas probabilitas
masa depan sudah baik. Coverage interval TEST sebesar 66,92% juga berada
13,08 poin persentase di bawah nominal 80%.

Selective policy menahan seluruh sinyal. Perilaku abstain ini sesuai aturan
ketika persyaratan tidak terpenuhi, tetapi belum memenuhi tujuan menghasilkan
sejumlah trade yang sedikit dan akurat. NO_TRADE seluruhnya bukan akurasi
100% dan tidak membuktikan profitabilitas.

**Paragraf siap pakai:**

> Optimasi XGBoost dilakukan melalui seleksi fitur, pengujian parameter,
> pembobotan kelas, early stopping, kalibrasi probabilitas, serta pemilihan
> selective policy. Kandidat classifier terbaik mencapai macro F1 sebesar
> 0,3525 pada periode tuning. Pada TEST 2026, keluaran classifier sebelum
> selective policy memiliki accuracy 46,61% dan macro F1 0,2478. Regresi
> menghasilkan RMSE return 19,6623 bps, sedikit lebih tinggi daripada baseline
> nol return sebesar 19,6599 bps. Seluruh keputusan akhir menjadi NO_TRADE
> karena aturan trade tidak lolos persyaratan pada development. Hasil tersebut
> menunjukkan bahwa konfigurasi terpilih belum memberikan bukti peningkatan
> prediksi yang konsisten maupun precision BUY/SELL yang ditargetkan.

## 4. Temuan Awal

| Temuan | Bukti tersimpan | Implikasi untuk eksperimen berikutnya |
|---|---|---|
| Pemilihan parameter dapat membedakan kandidat pada TUNE | Macro F1 kandidat berkisar 0,3394–0,3525 | Perlu tetap menilai konfigurasi pada periode lain, bukan memilih dari TEST |
| Classifier condong ke BUY | Recall BUY 94,18%, SELL 6,74% pada TEST | Perlu pemeriksaan error per kelas dan perperiode; skor agregat belum cukup |
| Prediksi regresi belum unggul konsisten | RMSE TEST sedikit lebih tinggi dari baseline nol return | Tambahan kompleksitas belum terbukti memberi manfaat pada rancangan saat ini |
| Kalibrasi belum tervalidasi sebelum–sesudah pada TEST | Tabel log loss/Brier yang tersedia berasal dari blok fitting kalibrasi | Perlu evaluasi kalibrasi pada data berikutnya yang tidak dipakai fitting |
| Interval belum mempertahankan coverage nominal | Coverage TEST 66,92%, nominal 80% | Perlu audit coverage perperiode dan kondisi pasar |
| Aturan trade tidak menghasilkan sinyal | BUY=0, SELL=0, coverage=0% | Target trade sedikit namun akurat belum tercapai; perlu membedakan kelemahan prediksi dari efek gerbang policy |
| Efek tiap komponen optimasi belum terisolasi | Tidak ada eksperimen TEST before/after yang terkontrol | Diperlukan ablation terpisah untuk mengukur kontribusi fitur, weighting, kalibrasi, dan policy |

Perubahan rezim pasar, kelemahan fitur, overfitting, atau rancangan label dapat
menjadi hipotesis penelitian lanjutan, tetapi belum terbukti sebagai penyebab
tunggal hasil ini. Angka tersebut juga tidak cukup untuk menyimpulkan bahwa
menambah fitur, melonggarkan policy, atau mengganti algoritma pasti memperbaiki
kinerja. Eksperimen berikutnya perlu memakai protokol development yang jelas
dan evaluasi baru setelah periode yang telah ditinjau.

## Sumber Angka

- [Trial tuning dan parameter terpilih](artifacts/direction_v3/final_2026/hyperparameters.json).
- [Metrik TEST](artifacts/direction_v3/final_2026/metrics.json).
- [Audit kalibrasi](artifacts/direction_v3/final_2026/calibration.json).
- [Audit selective policy](artifacts/direction_v3/final_2026/policy.json).
- [Prediksi per candle](artifacts/direction_v3/final_2026/predictions.csv).
- [Konfigurasi eksperimen](artifacts/direction_v3/config.json).
- [Laporan baseline](README_laporan_baseline.md).

Dummy Classifier, Logistic Regression sebagai classifier trading, dan
Random Forest belum menjadi pembanding terukur pada artifact ini. Tidak ada
angka baru untuk algoritma tersebut yang ditambahkan dalam laporan.
