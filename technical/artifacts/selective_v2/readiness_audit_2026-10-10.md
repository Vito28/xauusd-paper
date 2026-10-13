# Audit kesiapan technical — 10 Oktober 2026

Kesimpulan: pipeline dapat dipakai untuk eksperimen riset, tetapi belum siap sebagai sistem sinyal trading operasional. Audit ini tidak mengubah model atau melatih ulang. Pemeriksaan mencakup source, artifact tiga fold, reproduksi perilaku biaya/bundle, dan 15 tes technical yang semuanya lolos.

## Arti evaluasi 2024–2026

Tiga model/fold terpisah. Model 2024 memakai TRAIN 2016–2022 dan development 2023; model 2025 memakai TRAIN 2016–2023 dan development 2024; model 2026 memakai TRAIN 2016–2024 dan development 2025. Pada setiap tahun development: TUNE Jan–Jun, CALIBRATION Jul–Sep, POLICY Oct–Dec. Model akhir di-refit pada TRAIN+TUNE. TEST 2024 dan 2025 setahun penuh; TEST 2026 Januari–September (actual terakhir 30 September 06:45 UTC).
Model final 2026 tidak diperbarui bobotnya selama TEST. Input menggunakan candle aktual yang telah selesai; horizon prediksi tetap 15 menit, bukan prediksi sekali pada Januari untuk seluruh tahun.

## Temuan kode dan kesiapan

### P1 operasional: tidak ada rule trading yang aktif
Semua 81 kandidat rule per arah di bagian seleksi POLICY mempunyai nol sinyal pada ketiga fold. Jadi tahap precision belum memperoleh sampel BUY/SELL untuk dinilai; kegagalan bukan semata angka precision yang kurang dari 90%. Pada TEST 2026, seluruh interval return mencakup nol dan hanya 0,814% point forecast melampaui biaya keputusan secara absolut. Interval agreement di xgboost_model.py:867/920 menolak semua arah. Penolakan ini konsisten dengan kode, tetapi konfigurasi sekarang belum menyediakan strategi trading yang berfungsi. Diperlukan audit jumlah lolos setiap gate pada development, bukan sekadar menurunkan threshold memakai TEST.

### P2 metodologi: seleksi ReliefF berada di luar CV internal
Di xgboost_model.py:604 ranking ReliefF menggunakan seluruh label TRAIN, kemudian RFECV pada baris 626 membagi TRAIN menjadi inner train/validation. Label inner validation sudah ikut menentukan daftar kandidat fitur. Akibatnya skor RFECV internal berpotensi optimistis. TEST tahunan tetap terpisah: temuan ini tidak berarti harga TEST dipakai melatih model. Perbaikan: evaluasi rangkaian seleksi pada setiap inner-training split atau gunakan blok seleksi terpisah sebelum validation. Prinsip ini dijelaskan dalam [dokumentasi cross-validation sklearn](https://scikit-learn.org/stable/modules/cross_validation.html#data-transformation-with-held-out-data).

### P2 biaya: label precision tidak memasukkan slippage
make_labels() di data_preparation.py:204 memakai spread + minimum edge. Keputusan di xgboost_model.py:899/1254 menambahkan slippage, dan P/L di baris 1316 menguranginya. Karena label tidak ikut berubah, saat slippage positif suatu BUY dapat dihitung benar secara label tetapi rugi setelah biaya. Reproduksi: close 100, future close 100,05, spread 0,01, minimum edge 1 bps menghasilkan label BUY; slippage 10 bps menghasilkan net move -0,06. Default run ini slippage nol sehingga contoh tersebut tidak mengubah metrik historis run. Pilih definisi label net-cost atau tampilkan precision arah dan precision profit-net secara terpisah.

### P2 integrasi: bundle belum menjadi antarmuka inference operasional
load_model_bundle() pada baris 711 membaca satu dictionary. Artifact classifier tidak berisi regressor; artifact regressor tidak berisi classifier. Keduanya memiliki key features dengan daftar berbeda (35 dan 40 fitur pada final). Menggabungkan dictionary secara sembarang dapat menimpa daftar fitur classifier. predict_test() memerlukan state gabungan dan tabel label evaluasi. run_pipeline bekerja karena memakai bundle lengkap di memory; belum ada loader gabungan tervalidasi dan entrypoint predict-latest yang memvalidasi pasangan model, config fitur, versi schema dan freshness candle. Tambahkan round-trip replay test sebelum pemakaian operasional.

### P2 evaluasi: proxy belum merupakan backtest eksekusi
Pergerakan close-to-close dikurangi spread/asumsi slippage bukan simulasi fill bid/ask, latency, posisi overlap, ukuran posisi, commission, swap atau stop loss. nonoverlapping_metrics.json mengurangi horizon yang berulang tetapi tetap bukan account backtest. Dibutuhkan replay eksekusi dan pengujian forward dengan konfigurasi terkunci. Dataset TEST 2026 telah dilihat pada eksperimen terdahulu.

### P3 pelaporan: beberapa status masih tercampur
pending_rows menghitung seluruh actual_signal yang null. Dari 583 baris final, 580 berada dalam rentang waktu teramati dan tidak punya label karena aturan gap; hanya 3 berada setelah observasi terakhir. Pisahkan gap-invalid, biaya tidak tersedia, dan outcome belum tiba. classification.per_class menggunakan zero_division=0 sehingga precision arah tanpa sinyal tertulis 0, sedangkan selective.BUY/SELL.precision adalah null. Untuk pembaca, precision tanpa sinyal harus ditampilkan tidak tersedia dan recall/F1 tetap nol.

## Bagian yang sudah berjalan sesuai desain

- Candle harus tutup, UTC/grid M5 dan OHLC divalidasi; data tidak diinterpolasi.
- Fitur menggunakan masa lalu/saat candle selesai; uji mutasi masa depan menjaga prefix fitur tetap identik.
- Label dipisah dari input; target melintasi gap tidak dievaluasi.
- Development/TEST dipisah dengan purge target pada batas waktu; classifier, regressor dan quantile fitted state tetap identik ketika data TEST dimutasi.
- Seleksi fitur regressor terpisah; probabilitas dan interval dikalibrasi pada blok tersendiri.
- Guard arah menolak BUY dengan return negatif atau SELL dengan return positif; artefak model dapat dibaca kembali.

## Metrik classifier sebelum filter keputusan

Raw di sini berarti argmax probabilitas SETELAH kalibrasi, SEBELUM policy abstain. Support adalah jumlah label aktual kelas tersebut, bukan jumlah prediksi.

| Fold | Kelas | Precision | Recall | F1 | Support aktual |
|---|---|---:|---:|---:|---:|
| fold_2024 | BUY | 43.34% | 31.90% | 36.75% | 27,858 |
| fold_2024 | SELL | 41.24% | 45.80% | 43.40% | 26,347 |
| fold_2024 | NO_TRADE | 36.90% | 46.99% | 41.34% | 16,227 |
| fold_2025 | BUY | 44.78% | 83.97% | 58.41% | 30,841 |
| fold_2025 | SELL | 42.17% | 15.64% | 22.81% | 28,319 |
| fold_2025 | NO_TRADE | 39.29% | 4.89% | 8.70% | 10,471 |
| final_2026 | BUY | 46.24% | 98.80% | 62.99% | 23,392 |
| final_2026 | SELL | 44.12% | 1.24% | 2.41% | 23,307 |
| final_2026 | NO_TRADE | 54.55% | 0.15% | 0.30% | 3,954 |

Pada 2026 terdapat 50.653 outcome berlabel dari 51.236 keputusan. Prediksi raw seluruh keputusan: BUY 50.538, SELL 684, NO_TRADE 14. Recall BUY tinggi terjadi bersamaan dengan kecenderungan menebak BUY hampir sepanjang waktu; precision BUY tetap hanya 46,24%.

## Setelah filter keputusan

Semua fold menghasilkan NO_TRADE saja. Untuk BUY/SELL: precision tidak tersedia, recall 0, F1 0, coverage 0%. Pada 2026, macro F1 action 0,04827; precision/recall/F1 kelas aktual NO_TRADE adalah 7,81% / 100% / 14,48%. Kelas aktual NO_TRADE (return berada di zona biaya) berbeda dari keputusan abstain karena model tidak yakin.

## Regresi dan kestabilan

| Fold | Raw macro F1 | RMSE return model (bps) | RMSE zero-return (bps) | Ketepatan arah regressor | Coverage interval nominal 80% |
|---|---:|---:|---:|---:|---:|
| fold_2024 | 0.4050 | 9.1455 | 9.1452 | 51.45% | 79.65% |
| fold_2025 | 0.2997 | 11.7652 | 11.7665 | 50.61% | 75.32% |
| final_2026 | 0.2190 | 19.6623 | 19.6599 | 50.30% | 66.92% |

Hasil belum menunjukkan keunggulan return yang konsisten terhadap baseline. Interval 2026 hanya mencakup 66,92% aktual; nominal 80% tidak dapat diperlakukan sebagai confidence yang terjamin. Seluruh interval mencakup nol masih dapat bersamaan dengan undercoverage, karena sebagian actual berada di luar batas atas/bawah.

## Urutan pekerjaan sebelum penggunaan operasional

1. Benahi validasi internal seleksi fitur, definisi biaya/label dan status outcome.
2. Buat loader gabungan serta inference candle terbaru yang tervalidasi, disertai replay kesetaraan batch/streaming.
3. Audit distribusi prediksi/calibration dan berapa sampel lolos setiap filter pada development; ukur perubahan tanpa menyetel ulang terhadap TEST.
4. Jalankan backtest dengan asumsi eksekusi jelas dan evaluasi forward baru; pantau data stale, drift, interval coverage dan precision/support per arah.

Artefak sumber: metrics.json, policy.json, predictions.csv, model_classifier.joblib dan model_regressor.joblib pada setiap folder fold. Angka lama tetap dipertahankan; audit ini tidak mengubah hasil training.
