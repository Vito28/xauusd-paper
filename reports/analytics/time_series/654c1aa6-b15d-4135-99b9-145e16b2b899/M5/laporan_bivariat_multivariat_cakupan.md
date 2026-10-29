# Analisis bivariat, multivariat, dan cakupan observasi XAUUSD

## Ruang lingkup

Laporan ini membaca tiga grafik pada artifact analisis sumber
`metaquotes-demo-xauusd`: `bi_activity_return.png`, `multi_return_features.png`,
dan `data_observed_coverage.png`. Cakupan aktual adalah 4 Januari 2016
01:05 UTC–30 September 2026 06:40 UTC, dengan 754.660 candle M5.
Grafik bivariat menggunakan candle native M5, matriks korelasi menggunakan
agregat sesi harian UTC, dan heatmap cakupan menghitung candle native per bulan UTC.
Ketiga grafik memiliki unit analisis berbeda dan tidak boleh dibandingkan
seolah-olah menggunakan horizon yang sama.

## 1. Analisis bivariat: aktivitas tick dan spread terhadap besar return

### Cara membaca grafik

Grafik terdiri dari dua panel hexbin. Panel kiri membandingkan
`log_tick_volume = ln(1 + tick_volume)` dengan absolute log return M5.
Panel kanan membandingkan `relative_spread = spread_price / close` dengan
absolute log return M5. Sumbu vertikal menampilkan `100 × abs(log_return)`.
Dengan demikian, nilainya mengukur besar perubahan harga tanpa membedakan
kenaikan atau penurunan.

Setiap panel menggunakan 751.823 pasangan observasi yang bernilai finite dan
selaras pada candle yang sama. Return dihitung hanya ketika candle sebelumnya
berjarak tepat lima menit. Transisi gap tidak diperlakukan sebagai return M5.
Warna pada hexagon menunjukkan jumlah pasangan dalam bin dengan skala warna
logaritmik: warna kuning lebih padat daripada ungu. Warna ini bukan tingkat
keberhasilan transaksi atau probabilitas prediksi.

### Aktivitas tick dan besar perubahan harga

Sebagian besar observasi terkonsentrasi pada absolute return yang kecil.
Pada aktivitas tick yang lebih tinggi, sebaran return secara visual menjadi
lebih lebar dan mencakup lebih banyak gerakan besar. Namun, banyak observasi
dengan aktivitas tinggi tetap memiliki return kecil. Pola tersebut menunjukkan
asosiasi deskriptif antara aktivitas tick dan besarnya gerakan, bukan kepastian
bahwa setiap kenaikan aktivitas diikuti pergerakan besar.

Tick volume merupakan ukuran aktivitas pada feed broker yang digunakan,
bukan total volume transaksi pasar emas global. Karena sumbu vertikal memakai
nilai absolut, grafik tidak menentukan arah BUY atau SELL. Aktivitas dan return
di sini juga berasal dari candle yang sama; hubungan ini belum membuktikan
kemampuan aktivitas saat ini memprediksi return candle berikutnya.

**Narasi laporan:** “Analisis bivariat memperlihatkan bahwa pengamatan dengan
aktivitas tick lebih tinggi memiliki rentang besar perubahan harga yang lebih
luas. Meskipun demikian, mayoritas observasi tetap berada pada return kecil.
Aktivitas tick berpotensi menjadi variabel deskriptif kondisi pasar, tetapi
hubungannya dengan arah dan return masa depan masih memerlukan pengujian.”

### Spread relatif dan besar perubahan harga

Sebagian besar observasi berada pada spread relatif kecil dan absolute return
kecil. Gerakan besar juga muncul pada spread relatif kecil; pengamatan dengan
spread relatif lebih besar tidak memperlihatkan kenaikan besar return yang
konsisten. Dari visual ini, spread relatif tinggi tidak dapat dianggap sebagai
indikator tunggal gerakan besar atau arah harga.

Spread tetap relevan untuk menghitung biaya transaksi, tetapi relevansi biaya
berbeda dari kemampuan memprediksi arah. Spread relatif dibagi dengan close,
sehingga perubahan level harga dapat mengubah rasio meskipun spread dalam
satuan harga tidak berubah. Efek periode harga, sesi, dan perubahan feed perlu
dipisahkan sebelum menafsirkan asosiasi sebagai kondisi likuiditas.

**Narasi laporan:** “Hubungan spread relatif dengan besar return tidak
memperlihatkan pola peningkatan yang konsisten pada visualisasi. Spread perlu
dipertimbangkan sebagai komponen biaya eksekusi, sedangkan perannya sebagai
prediktor memerlukan evaluasi tersendiri.”

## 2. Analisis multivariat deskriptif: matriks korelasi Spearman

### Variabel dan pengukuran

Matriks membandingkan lima variabel pada tingkat sesi harian UTC:

| Variabel | Pengukuran |
|---|---|
| `log_return` | Log perubahan close terhadap sesi terobservasi sebelumnya |
| `absolute_return` | Nilai absolut log return sesi |
| `relative_range` | Rata-rata `(high − low) / close` candle M5 dalam sesi |
| `relative_spread` | Rata-rata `spread_price / close` candle M5 dalam sesi |
| `tick_volume` | Jumlah tick volume candle M5 dalam sesi |

Relative range pada matriks bukan range dari high dan low keseluruhan hari.
Korelasi Spearman mengukur hubungan berdasarkan peringkat, dengan nilai antara
−1 dan +1. Nilai diagonal 1,00 merupakan hubungan variabel dengan dirinya sendiri.
Setiap sel menggunakan pasangan data yang tersedia, sebanyak 2.772 atau 2.773
observasi. Perbedaan ini terjadi karena return sesi pertama tidak tersedia.

Matriks disebut multivariat karena menampilkan beberapa variabel sekaligus,
tetapi setiap koefisien tetap merupakan korelasi dua variabel. Ini bukan hasil
model multivariat yang mengendalikan pengaruh variabel lain.

### Temuan utama

| Pasangan | Spearman rho | Interpretasi deskriptif |
|---|---:|---|
| Relative range–tick volume | +0,53 | Sesi lebih aktif cenderung memiliki rata-rata range M5 relatif lebih besar |
| Absolute return–relative range | +0,40 | Besar perubahan close antarsesi berasosiasi positif dengan rata-rata range M5 relatif |
| Absolute return–tick volume | +0,22 | Asosiasi positif lebih lemah antara aktivitas sesi dan besar perubahan close |
| Relative range–relative spread | −0,41 | Kedua ukuran relatif cenderung memiliki peringkat berlawanan dalam sampel |
| Relative spread–tick volume | −0,32 | Sesi lebih aktif cenderung memiliki spread relatif lebih rendah dalam sampel |
| Absolute return–relative spread | −0,16 | Asosiasi negatif relatif lemah |
| Log return–relative range | −0,01 | Hubungan monoton dengan return bertanda dekat nol |
| Log return–relative spread | −0,02 | Hubungan monoton dengan return bertanda dekat nol |
| Log return–tick volume | −0,02 | Hubungan monoton dengan return bertanda dekat nol |

Angka mengikuti pembulatan dua desimal pada grafik. Hubungan paling menonjol
berkaitan dengan aktivitas dan besar gerakan, sedangkan hubungan ketiga fitur
dengan return bertanda sangat lemah. Temuan ini menunjukkan bahwa membaca
seberapa aktif atau bergejolak pasar tidak sama dengan mengetahui arah harga.

Korelasi negatif spread relatif tidak membuktikan bahwa spread menyebabkan
penurunan volatilitas atau bahwa strategi tertentu menguntungkan. Penggabungan
seluruh tahun, perubahan level harga, dan variasi kondisi broker dapat berperan.
Grafik belum memiliki uji signifikansi atau pengendalian faktor tersebut.
Korelasi yang dekat nol juga tidak meniadakan kemungkinan hubungan nonlinear.

**Narasi laporan:** “Matriks korelasi Spearman menunjukkan hubungan positif
antara aktivitas tick dan relative range sebesar 0,53, serta hubungan antara
absolute return dan relative range sebesar 0,40. Namun, korelasi relative range,
relative spread, dan aktivitas tick dengan log return bertanda mendekati nol.
Variabel tersebut lebih jelas berkaitan dengan karakteristik aktivitas dan
besar pergerakan dalam sampel daripada dengan arah perubahan harga.”

## 3. Cakupan observasi: jumlah candle per bulan

### Cara membaca grafik

Heatmap menampilkan tahun pada sumbu vertikal, bulan pada sumbu horizontal,
dan jumlah candle M5 yang tersimpan pada skala warna. Biru lebih gelap
menunjukkan lebih banyak candle. Ini merupakan jumlah observasi, bukan
persentase coverage terhadap kalender perdagangan yang seharusnya.

Pada cakupan 2016–2025, seluruh bulan terlihat memiliki observasi. Jumlah
candle berubah antarbulan; perbedaan dapat berkaitan dengan jumlah hari
perdagangan, panjang bulan, libur, jeda sesi, atau kehilangan data.
Heatmap saja tidak membedakan penyebab-penyebab tersebut. Sel yang lebih terang
tetap dapat memiliki ribuan candle; jangan membacanya sebagai data kosong.

Oktober–Desember 2026 kosong karena berada di luar cakupan artifact, bukan
bukti kegagalan ingestion. September 2026 berakhir pada 30 September 06:40 UTC,
sehingga sesi terakhir belum mewakili satu hari UTC penuh.

### Angka pendukung dari insights.json

| Indikator | Hasil |
|---|---:|
| Candle yang dimuat dan dipertahankan | 754.660 |
| Bar berstatus valid | 754.660 |
| Bar dengan harga invalid | 0 |
| Bar belum tertutup saat analisis | 0 |
| Transisi antarbar melebihi interval M5 | 2.836 |
| Return M5 yang dapat dihitung | 751.823 |
| Candle tahun 2024 | 71.210 |
| Candle tahun 2025 | 70.487 |
| Candle tahun 2026 sampai batas observasi | 51.235 |

Return M5 lebih sedikit daripada candle karena candle pertama belum memiliki
pembanding dan return pada transisi gap tidak dihitung. Sebanyak 2.836 transisi
gap bukan berarti 2.836 candle hilang: satu transisi dapat meliputi jeda pasar
normal atau rentang dengan beberapa candle yang tidak tersedia. Jumlah candle
2026 yang lebih kecil juga tidak dapat dibandingkan sebagai tahun penuh.

**Narasi laporan:** “Heatmap cakupan menunjukkan tersedianya observasi bulanan
pada periode 2016–2025 serta observasi Januari–September 2026. Sebanyak 754.660
candle yang dimuat berstatus valid dan tidak ditemukan harga invalid dalam
pemeriksaan yang dilakukan. Namun, kelengkapan data belum dapat dinyatakan
100% karena jumlah observasi belum dibandingkan dengan kalender perdagangan
broker, dan transisi gap dapat mencakup penutupan pasar maupun data yang hilang.”

## Implikasi untuk pengembangan model

Aktivitas tick, relative range, dan spread dapat diuji sebagai fitur kondisi
pasar dan biaya transaksi. Pengujian prediksi perlu menggunakan fitur yang
sudah tersedia pada waktu keputusan untuk target masa depan, bukan menganggap
korelasi pada waktu yang sama sebagai akurasi forecasting. Hubungan perlu
dievaluasi perperiode dan pada horizon target model. Audit kelengkapan perlu
membandingkan timestamp aktual dengan jadwal perdagangan broker dan riwayat
ingestion. Ketiga grafik ini belum mengukur precision BUY/SELL, profit strategi,
atau kemampuan prediksi model.

**Sumber:** tiga PNG yang disebutkan, `insights.json`, serta definisi transformasi
dan plot pada `analytics/data.py` dan `analytics/analytics.py`. Tidak ada data
tambahan atau penyebab ekonomi yang diasumsikan dalam penjelasan ini.
