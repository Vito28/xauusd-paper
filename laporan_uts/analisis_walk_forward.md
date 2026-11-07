Analisis di bawah ini membaca Tabel 10.3–10.17. Semua angka berasal dari run 7 November 2026 (walk-forward) dan 3 November 2026 (final-test).

### Temuan 1: XGBoost memiliki keunggulan kecil tetapi konsisten

Precision XGBoost sekitar 47–48% untuk BUY dan SELL (Tabel 10.3), sedangkan prevalensi kedua kelas pada blok EVALUATION sekitar 40–42% (Tabel 10.8). Lift BUY berada pada 1,15–1,20 di kelima tahun, dan lift SELL 1,09–1,20 di empat tahun (SELL dinonaktifkan pada 2023). Artinya, sinyal XGBoost lebih baik daripada tebakan acak di setiap tahun yang dinilai, tetapi keunggulannya kecil: lift pada data evaluasi umumnya sedikit di bawah syarat 1,2 yang dipakai saat memilih threshold.

### Temuan 2: Macro F1 menyesatkan untuk sistem selektif

Persistence memiliki macro F1 tertinggi (0,3043) karena mengeluarkan sinyal pada 46,33% candle, sehingga recall-nya besar. Namun lift-nya hanya 0,98–1,08 (Tabel 10.8), yaitu setara tebakan acak. Sebaliknya, XGBoost dengan coverage 15,97% memiliki precision 4,7–6,3 poin persentase lebih tinggi (47,82% dibanding 41,50% untuk BUY dan 47,24% dibanding 42,52% untuk SELL). Temuan ini mendukung keputusan memakai precision, lift, dan coverage sebagai ukuran utama, bukan macro F1 atau accuracy.

### Temuan 3: Model linear dan random forest terlalu jarang memberi sinyal

Regresi logistik hanya memberi sinyal pada 3,05% candle, dan precision SELL-nya hanya terdefinisi pada tiga fold; random forest 6,76% dengan BUY terdefinisi pada empat fold (Tabel 10.3 dan 10.5). Setelah kalibrasi, probabilitas kedua model jarang melewati threshold yang memenuhi syarat precision, sehingga arah sering dinonaktifkan. Boosting XGBoost menghasilkan pemisahan kelas yang lebih tajam pada data yang sama.

### Temuan 4: Ensemble belum memperbaiki XGBoost

Bobot ensemble didominasi XGBoost (0,35–0,80) dan random forest (0,10–0,65), sedangkan bobot regresi logistik 0–0,10 (Tabel 10.11). Log loss campuran hanya 0,0003–0,0008 lebih kecil daripada anggota terbaik. Pada blok EVALUATION, directional F1 ensemble 0,1253 dibanding 0,1371 untuk XGBoost, dengan precision hampir sama dan coverage lebih rendah (Tabel 10.3 dan 10.7). Syarat keragaman hanya terpenuhi sebagian: korelasi kesalahan XGBoost–random forest 0,511, sedangkan regresi logistik lebih beragam (0,370 dan 0,266) tetapi terlalu lemah sebagai anggota (Tabel 10.12). Sesuai aturan pada bab 8.5, ensemble belum dipakai sebagai model final karena tidak lebih baik secara konsisten dan melatih tiga model.

### Temuan 5: Class weight memberi hasil yang mirip dengan threshold moving

xgboost_balanced memiliki precision sedikit lebih tinggi (48,52% dan 48,22%) dan directional F1 yang lebih stabil antartahun (simpangan baku 0,0472 dibanding 0,0709), tetapi rata-ratanya lebih rendah (0,1284 dibanding 0,1371). Perbedaannya kecil dibandingkan variasi antartahun, sehingga keduanya tetap menjadi kandidat pada tahap tuning.

### Temuan 6: Kinerja berubah antartahun

Tahun 2023 adalah tahun terlemah: coverage XGBoost hanya 0,81%, threshold SELL yang dipilih pada Oktober–Desember 2022 tidak lolos syarat sehingga SELL dinonaktifkan, dan hanya 559 sinyal BUY yang dikeluarkan (Tabel 10.6). Sebaliknya, 2024 menghasilkan coverage 26,45%. Threshold yang dipilih dari tiga bulan terakhir belum selalu cocok untuk tahun berikutnya; hal ini menjadi fokus analisis threshold setelah UTS.

### Temuan 7: Kesalahan didominasi arah berlawanan

Dari 55.224 sinyal XGBoost selama lima tahun, 25.365 (45,9%) adalah kesalahan arah berlawanan, yaitu BUY ketika barrier bawah tersentuh lebih dulu atau sebaliknya (Tabel 10.10). Sinyal NEUTRAL relatif sedikit karena label NO_TRADE pada tahun evaluasi hanya 16–20%, jauh lebih kecil daripada pada blok TRAIN (26–33%, Tabel 10.9). Pergeseran distribusi ini menunjukkan bahwa harga pada 2021–2025 lebih sering menyentuh barrier dalam 30 menit dibandingkan rata-rata data training.

### Temuan 8: Keunggulan klasifikasi belum menutup biaya spread

Simulasi XGBoost pada 2021–2025 menghasilkan 55.224 transaksi dengan win rate 49,8% dan profit factor 0,91 (Tabel 10.13). Sebelum spread, P&L kotornya positif (+315,74 USD), tetapi biaya spread 5.016,13 USD membuat P&L bersihnya −4.700,39 USD (Tabel 10.14). Semua model lain juga negatif setelah spread. Penyebabnya dapat dihitung dari aturan transaksi: TP menghasilkan +k·ATR, sedangkan SL kehilangan k·ATR + 2 × spread. Dengan spread rata-rata 6,1% dari barrier b, berlaku k·ATR ≈ 0,939 b dan spread ≈ 0,061 b, sehingga proporsi TP di antara penutupan TP dan SL yang dibutuhkan agar impas adalah:

$$ p_impas = (k·ATR + 2s) / (2·k·ATR + 2s) ≈ (0,939 + 0,122) / (1,878 + 0,122) ≈ 53,0% $$ (11.1)

Proporsi yang dicapai XGBoost adalah 47,2% / (47,2% + 46,9%) ≈ 50,2%. Keunggulan precision sebesar 3,9–8,4 poin persentase di atas prevalensi masih menyisakan kekurangan sekitar 2,8 poin dari titik impas. Karena posisi boleh tumpang tindih, angka P&L ini mengukur kualitas sinyal setelah biaya, bukan saldo akun nyata.

### Temuan 9: Probabilitas terkalibrasi baik, tetapi batas jumlah pohon tersentuh

Temperatur kalibrasi berada pada 0,83–1,02 (Tabel 10.6). Nilai di bawah 1 berarti probabilitas mentah XGBoost sedikit terlalu lunak dan dipertajam. Pada uji 2026, expected calibration error hanya 0,0046 (Tabel 10.16). Early stopping memakai 295–300 pohon di setiap fold dan 300 pohon pada model final, sehingga batas n_estimators = 300 hampir selalu tersentuh. Model kemungkinan belum konvergen; jumlah pohon maksimum dan learning rate perlu diuji ulang pada tahap tuning.

### Temuan 10: Uji 2026 sangat selektif dan sampelnya kecil

Pada uji 2026, model hanya mengeluarkan 136 sinyal SELL (coverage 0,27%) dengan precision 51,47% terhadap prevalensi SELL 43,8% (lift ≈ 1,18), dan simulasinya positif (profit factor 1,21, +116,76 USD; Tabel 10.16 dan 10.17). Namun interval Wilson 95% untuk precision SELL masih lebar (43,1%–59,7%), sehingga hasil ini belum dapat disimpulkan sebagai keunggulan yang nyata. BUY dinonaktifkan karena prevalensi BUY pada Oktober–Desember 2025 mencapai 42,6% sehingga syarat precision naik menjadi 51,1%. Final-test dijalankan pada 3 November 2026 dengan konfigurasi XGBoost yang sama dengan walk-forward (perbedaan config hash hanya karena bagian random_forest dan ensemble ditambahkan kemudian), dan statusnya tetap out-of-sample retrospektif.

Tabel 11.2 Pemenuhan Kriteria Keberhasilan Tahap UTS (model XGBoost)
<!-- widths: 28,50,22 -->
| Kriteria (Tabel 1.3) | Hasil | Status |
|---|---|---|
| Lift ≥ 1,2 | Lift BUY 1,15–1,20 dan SELL 1,09–1,20 pada blok EVALUATION | Belum konsisten |
| Minimal 30 sinyal per arah | Terpenuhi, kecuali SELL 2023 yang dinonaktifkan | Sebagian besar |
| Konsisten antarperiode | Lift > 1 pada semua arah yang aktif di setiap tahun; coverage 0,81%–26,45% | Sebagian |
| Kualitas probabilitas | Temperatur 0,83–1,02; ECE 2026 = 0,0046 | Terpenuhi |
| Bernilai setelah biaya | Profit factor 0,91 pada 2021–2025; 1,21 pada 136 transaksi 2026 | Belum |
| Implementasi benar | Seluruh tes lolos dan XGBoost from scratch identik pada data uji; compare-scratch data riil belum dijalankan | Sebagian |

Tabel 11.3 Jawaban Rumusan Masalah pada Tahap UTS
<!-- widths: 28,72 -->
| Rumusan masalah (bab 1.2) | Jawaban berdasarkan bukti awal |
|---|---|
| 1) Sinyal lebih presisi daripada acak setelah spread? | Lebih presisi daripada acak (lift 1,09–1,20), tetapi belum menguntungkan setelah spread dengan aturan transaksi saat ini (profit factor 0,91) |
| 2) XGBoost vs baseline | Precision dan lift XGBoost lebih tinggi daripada persistence; coverage dan directional F1 lebih tinggi daripada regresi logistik. Macro F1 persistence lebih tinggi hanya karena coverage-nya besar |
| 3) Ensemble dan feature selection | Ensemble tidak lebih baik secara konsisten. xgboost_selected belum dijalankan pada data riil |
| 4) XGBoost from scratch vs library | Identik pada data uji (selisih < 1,1 × 10⁻⁷); perbandingan pada data XAU/USD (compare-scratch) belum dijalankan |
