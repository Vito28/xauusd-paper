# Analisis univariat XAUUSD pada periode observasi 2016–2026

## Data dan ruang lingkup

Analisis menggunakan 754.660 candle XAUUSD dengan timeframe M5 dari sumber
`metaquotes-demo-xauusd`. Cakupan aktual dalam artifact adalah 4 Januari 2016
01:05 UTC sampai 30 September 2026 06:40 UTC. Grafik univariat merangkum candle
tersebut menjadi sesi berdasarkan tanggal UTC; harga penutupan sesi merupakan
close candle terakhir yang tersedia pada tanggal tersebut. Dengan demikian,
grafik ini menjelaskan karakteristik antarsesi harian, bukan langsung perubahan
setiap candle M5 atau candle harian resmi broker.

Univariat berarti mempelajari karakteristik satu variabel beserta transformasinya.
Harga, return, volatilitas, dan drawdown pada grafik berikut berasal dari seri
harga yang sama. Beberapa panel dalam satu gambar tidak otomatis menjadikannya
analisis bivariat.

Tahun 2026 hanya mencakup 193 sesi terobservasi. Sesi terakhir dan bulan terakhir
dapat parsial. Seluruh bar yang dimuat berstatus valid, tetapi terdapat 2.836
transisi gap, termasuk kemungkinan penutupan pasar. Validitas bar tidak membuktikan
kelengkapan kalender perdagangan. Grafik dan angka di bawah merupakan hasil
deskriptif dataset ini, bukan estimasi profit strategi atau penjelasan sebab ekonomi.

## 1. Tren harga, indeks harga, dan drawdown

**Gambar:** `uni_trend_drawdown.png`.

Panel pertama menunjukkan level harga penutupan sesi. Panel kedua menormalkan
harga awal menjadi indeks 100 dan memakai skala log agar perubahan relatif
lebih mudah dibandingkan. Panel ketiga menunjukkan penurunan terhadap puncak
harga penutupan sesi yang telah dicapai:

`drawdown_t = 100 × (close_t / maksimum(close sampai t) − 1)`.

Harga penutupan sesi terakhir meningkat 288,61% dibandingkan sesi pertama,
sehingga indeks akhir sekitar 388,61. Grafik memperlihatkan kenaikan yang lebih
tajam pada bagian akhir periode, terutama sekitar 2024–2025, disertai koreksi
besar pada 2026. Tren naik tersebut tidak berlangsung secara terus-menerus:
terdapat 1.284 sesi dengan log return negatif dan drawdown maksimum sebesar
26,52% dari puncak close sesi.

Temuan ini menunjukkan bahwa kenaikan jangka panjang tetap disertai risiko
penurunan yang besar. Angka 288,61% merupakan perubahan harga selama cakupan
observasi, bukan return tahunan rata-rata, profit trading, atau hasil setelah
biaya transaksi. Drawdown ini juga bukan drawdown intraday atau drawdown akun.

## 2. Return sesi dan volatilitas rolling

**Gambar:** `uni_return_volatility.png`.

Panel atas menampilkan log return antarsesi terobservasi:

`r_t = ln(close_t / close_sesi_sebelumnya)`.

Nilai pada grafik dikalikan 100 untuk ditampilkan sebagai persen log return.
Panel bawah menunjukkan standar deviasi return dalam jendela 20 observasi sesi.
Jendela ini bukan 20 hari kalender dan volatilitasnya tidak diannualisasi.

Secara visual, fluktuasi relatif kecil pada sebagian periode 2017–2019. Lonjakan
volatilitas terlihat sekitar 2020, kembali meningkat pada beberapa bagian
2025, dan paling menonjol pada 2026. Pada 2026, grafik memperlihatkan pergerakan
negatif ekstrem mendekati −9% sampai −10% log return serta puncak volatilitas
rolling sekitar 3,6%; angka ini adalah perkiraan visual, bukan statistik presisi
yang tersimpan dalam `insights.json`.

Volatilitas berubah sepanjang waktu. Besarnya gerakan harga pada masa terbaru
tidak dapat diasumsikan sama dengan periode awal. Hal ini menjadi dasar untuk
menguji fitur volatilitas dan membandingkan kinerja model perperiode. Grafik
ini belum membuktikan bahwa volatilitas tinggi membuat arah BUY atau SELL
lebih mudah diprediksi.

## 3. Heatmap return bulanan

**Gambar:** `uni_monthly_returns.png`.

Heatmap menunjukkan perubahan close akhir bulan yang terobservasi terhadap
close bulan sebelumnya. Hijau menunjukkan kenaikan, merah menunjukkan penurunan,
dan sel kosong menunjukkan nilai yang tidak tersedia. Return bulanan pada
grafik ini merupakan persentase perubahan harga biasa, berbeda dari log return
sesi pada grafik sebelumnya.

Tahun 2025 didominasi bulan positif, termasuk Maret sekitar +9,3% dan September
sekitar +11,9%. Sebaliknya, 2026 memperlihatkan pembalikan arah yang kuat: Januari
sekitar +13,3%, Februari +7,9%, Maret −11,3%, Juni −11,6%, dan Agustus +9,9%.
September tercatat sekitar −6,2%, dengan cakupan akhir bulan yang dapat parsial.
Angka tersebut mengikuti pembulatan satu desimal pada gambar.

Perbedaan ini menunjukkan bahwa kenaikan pada satu bulan tidak menjamin
kelanjutan arah pada bulan berikutnya. Warna yang berubah antartahun juga belum
membuktikan pola musiman yang konsisten. Januari 2016 kosong karena tidak tersedia
close Desember 2015 sebagai anchor; Oktober–Desember 2026 kosong karena berada
di luar cakupan observasi. Sel kosong tidak boleh dibaca sebagai return nol.

## 4. Boxplot distribusi return per tahun

**Gambar:** `uni_yearly_returns.png`.

Boxplot membandingkan distribusi log return sesi setiap tahun. Garis di dalam
kotak menunjukkan median, kotak menunjukkan rentang kuartil pertama sampai
ketiga, dan titik di luar whisker menunjukkan observasi ekstrem menurut aturan
boxplot. Titik tersebut tidak otomatis merupakan kesalahan data.

Kotak tahun 2017–2018 relatif sempit, sedangkan 2020, 2025, dan terutama 2026
memperlihatkan sebaran yang lebih besar. Pada 2026 terlihat observasi negatif
yang jauh dari mayoritas return. Median yang dekat nol atau sedikit positif
tetap dapat berdampingan dengan penurunan besar karena median tidak mewakili
ukuran kerugian pada ekor distribusi.

Jumlah sampel juga berbeda: tahun penuh umumnya memiliki sekitar 257–259
return sesi, sedangkan 2026 memiliki 193. Tahun 2016 memiliki 258 sesi tetapi
hanya 257 return karena sesi pertama belum memiliki pembanding. Perbandingan
distribusi perlu mempertimbangkan jumlah sampel dan periode 2026 yang parsial.
Perbedaan visual ini mengindikasikan perubahan distribusi, tetapi belum
merupakan hasil uji statistik normalitas atau kestasioneran.

## 5. Autokorelasi return dan absolute return

**Gambar:** `uni_return_dependence.png`.

ACF mengukur hubungan suatu seri dengan nilainya pada lag sebelumnya. Lag
di sini adalah urutan sesi UTC terobservasi, bukan hari kalender. Panel atas
menampilkan ACF log return, sedangkan panel bawah menampilkan ACF absolute
return atau besar pergerakan tanpa memperhatikan arah.

Autokorelasi return pada lag 1–30 secara visual dekat nol. Ini menunjukkan
hubungan linear yang lemah antara return saat ini dan return sebelumnya pada
lag yang ditampilkan. ACF absolute return cenderung positif, terutama pada
beberapa lag awal. Pola tersebut konsisten dengan pengelompokan volatilitas:
periode gerakan besar cenderung berdekatan dengan periode gerakan besar lainnya.

Grafik tidak memiliki interval kepercayaan atau uji signifikansi. Karena itu,
temuan ini harus ditulis sebagai indikasi deskriptif, bukan bukti signifikansi,
kestasioneran, kausalitas, atau ketidakmungkinan prediksi. Pola besar gerakan
yang berulang juga tidak langsung menentukan arah BUY atau SELL.

## Implikasi untuk rancangan model

Analisis univariat memperlihatkan tren harga jangka panjang, koreksi besar,
perubahan distribusi return, dan indikasi pengelompokan volatilitas. Temuan
tersebut mendukung pengujian fitur return, momentum relatif, volatilitas rolling,
serta jarak terhadap puncak historis. Pembuatan fitur harus menggunakan informasi
yang sudah tersedia pada waktu prediksi. Manfaat setiap fitur tetap perlu
dibuktikan melalui validasi berurutan berdasarkan waktu dan pembandingan baseline.

Untuk model teknikal M5, pemeriksaan serupa perlu dilakukan kembali pada horizon
M5 atau horizon target model. Temuan harian dalam laporan ini tidak otomatis
berlaku untuk prediksi 15 menit. Grafik univariat juga belum mengukur precision
BUY/SELL, tingkat keberhasilan transaksi, ataupun keuntungan setelah spread
dan slippage.

**Sumber lokal:** lima PNG univariat, `insights.json`, dan `analysis.md` pada
direktori yang sama. Tidak ada data tambahan atau penjelasan sebab ekonomi
yang diasumsikan dalam laporan ini.
