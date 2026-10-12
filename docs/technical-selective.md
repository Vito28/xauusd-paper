# XGBoost selektif untuk XAUUSD M5

Tujuan eksperimen ini adalah mengurangi BUY/SELL yang salah sambil mengukur
kemampuan prediksi return secara jujur. Target **precision 90% per arah** adalah
syarat development yang diinginkan, bukan hasil yang dijanjikan. F1 tetap
dipakai untuk memilih model dan memilih rule di antara kandidat yang memenuhi
syarat precision dan jumlah sinyal.

## Arti metrik dan NO_TRADE

| Ukuran | Interpretasi |
|---|---|
| BUY precision | BUY dengan actual target > harga sinyal / semua BUY dengan actual diketahui |
| SELL precision | SELL dengan actual target < harga sinyal / semua SELL dengan actual diketahui |
| Recall | Proporsi seluruh kenaikan/penurunan aktual yang berhasil diambil, termasuk gerakan kecil |
| F1 per arah | Keseimbangan precision dan recall; abstain berlebihan menurunkan recall |
| Raw classifier macro F1 | Kualitas argmax probabilitas terkalibrasi sebelum aturan abstain |
| Action macro F1 | Kualitas tiga kelas setelah filter keputusan |
| Coverage | Proporsi semua keputusan yang menghasilkan BUY atau SELL |
| Return RMSE vs zero | Apakah forecast lebih baik daripada memprediksi return nol |

NO_TRADE adalah keputusan untuk abstain, termasuk saat model tidak cukup yakin.
Ini berbeda dari label aktual NO_TRADE, yaitu return yang berada di zona biaya
spread + minimum edge + slippage konfigurasi. **Jika semua keputusan NO_TRADE, precision BUY/SELL adalah
null, recall dan F1 arah nol. Hasil itu tidak membuktikan akurasi 90%.**
Precision arah juga berbeda dari profitabilitas akun: besarnya kerugian,
biaya eksekusi, ukuran posisi dan posisi yang overlap tetap berpengaruh.

## Protokol waktu dan candle aktual

Default final fold:

| Bagian | Periode UTC | Fungsi |
|---|---|---|
| TRAIN | 2016 sampai sebelum 2025 | Seleksi fitur dan fitting kandidat |
| TUNE | Januari–Juni 2025 | Pilih parameter, jumlah tree dan shrinkage |
| CALIBRATION | Juli–September 2025 | Kalibrasi probabilitas dan interval return |
| POLICY | Oktober–Desember 2025 | Seleksi rule di bagian awal; konfirmasi di bagian akhir |
| TEST | Januari–September 2026 | Inference model beku dan evaluasi retrospektif |

Semua interval half-open; target development yang mencapai blok berikutnya
dipurge. Kalender final dapat diubah melalui empat flag `--final-*-start` dan
`--evaluation-end`. Untuk menilai seluruh 2026 sejak Januari, seluruh fitting
harus berakhir sebelum Januari. Tuning pada Januari–Maret 2026 mengubah bulan
tersebut menjadi development, sehingga tidak boleh sekaligus disebut TEST.

Model memakai candle **aktual yang sudah selesai**. Pada keputusan M5 berikutnya,
fitur dihitung ulang dari histori yang sudah tersedia; bobot model, daftar fitur,
kalibrator dan rule tetap. Contoh tanggal 2 memakai candle aktual tanggal 1
mengikuti prinsip yang sama. Forecast sebelumnya tidak menjadi candle buatan
dan label 2026 tidak masuk fitting. Horizon default tetap tiga M5 = 15 menit,
bukan forecast satu hari. Periode kosong tidak diisi ramalan atau interpolasi.

Data TEST 2026 pernah dilihat pada versi terdahulu. Evaluasi baru ini merupakan
holdout retrospektif yang dipisahkan dari fitting, bukan pembuktian pada data
masa depan yang belum pernah diamati. Validasi lanjutan memerlukan periode
forward baru dengan konfigurasi yang dikunci sebelum melihat hasilnya.

## Perubahan model, bukan hanya threshold

1. **Classifier:** seleksi Quality → ReliefF → purged RFECV pada TRAIN,
   enam kandidat kombinasi parameter dan class weighting, early stopping pada
   TUNE, pilih macro F1, lalu refit TRAIN+TUNE dengan jumlah tree terpilih.
2. **Regressor:** seleksi fitur tersendiri berdasarkan target return TRAIN,
   maksimal 40 fitur dari ranking gain. Gain adalah heuristik model, bukan bukti
   kausalitas. TUNE membandingkan tiga konfigurasi regularisasi/kedalaman dengan
   squared loss dan pseudo-Huber; early stopping membatasi jumlah tree.
3. **Return shrinkage:** faktor 0,1 / 0,25 / 0,5 / 1 dipilih dengan RMSE TUNE.
   Ini dapat menahan forecast berlebihan, tetapi hasil yang mendekati return nol
   tetap harus mengalahkan baseline zero-return untuk menunjukkan nilai prediktif.
4. **Kalibrasi:** probabilitas memakai sigmoid OVR yang dinormalisasi pada
   CALIBRATION. Log loss dan Brier sebelum/sesudah disimpan sebagai diagnosis
   in-sample kalibrasi, bukan bukti generalisasi baru.
5. **Ketidakpastian:** quantile regressor memprediksi quantile 0,1 dan 0,9
   (nominal interval 80%). Batas yang berpotongan diurutkan; residual CALIBRATION
   dapat memperlebar interval. Coverage TEST dilaporkan tersendiri.

Pseudo-Huber dan quantile loss tersedia dalam
[dokumentasi parameter XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html).
Pemakaian eval set dan best iteration mengikuti
[antarmuka sklearn XGBoost](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html).
Pilihan rentang parameter dan shrinkage di atas adalah desain eksperimen repo ini.

## Keputusan BUY/SELL yang dapat diaudit

BUY harus memenuhi semuanya: argmax classifier BUY, probabilitas/margin lolos,
predicted return positif dan melebihi biaya, serta lower interval melebihi biaya
bila interval agreement aktif. SELL memakai syarat simetris: return negatif
dan upper interval lebih kecil daripada negatif biaya. Biaya keputusan adalah
spread/close + minimum edge + asumsi slippage bps.

POLICY dibagi secara kronologis. Bagian pertama mengevaluasi 81 rule per arah:
sembilan threshold probabilitas, tiga margin probabilitas, dan tiga kelipatan
edge. Pilih F1 tertinggi dari rule dengan precision minimal 90% dan minimal
50 sinyal per arah. Rule terpilih dikunci, lalu harus memenuhi kedua syarat itu
lagi pada bagian konfirmasi. Gagal konfirmasi menonaktifkan arah tersebut,
tanpa mencari threshold pengganti dari data konfirmasi.

Penilaian kedua bagian menggunakan anchor horizon yang tidak overlap dan purge
di batasnya. Ini mengurangi pengulangan outcome, tetapi tidak menghilangkan
ketergantungan pasar. Wilson 95% disimpan sebagai diagnosis dengan asumsi
Bernoulli independen; nilai precision 90% dari sampel terbatas tidak berarti
batas bawah confidence interval juga 90%.

Pemisahan probabilitas dan keputusan mengikuti prinsip
[threshold tuning scikit-learn](https://scikit-learn.org/stable/modules/classification_threshold.html).
Trade-off antara risiko kesalahan dan coverage sejalan dengan
[Geifman & El-Yaniv, 2017](https://arxiv.org/abs/1705.08500);
jaminan eksperimen neural-network dalam paper tersebut tidak dipindahkan ke FX.

Interval ini dikalibrasi sekali dan dibekukan. Dependensi waktu dan perubahan
distribusi membatasi klaim coverage. Penelitian
[Gibbs & Candès, 2021](https://arxiv.org/abs/2106.00170) membahas adaptasi terhadap
distribution shift; pipeline ini **belum mengimplementasikan adaptive conformal
inference** dan tidak mengklaim jaminan coverage masa depan.

## Membaca grafik dan mempertimbangkan ekonomi emas

Titik forecast diplot pada waktu target. Garis emas yang dipilih menghubungkan
harga aktual pada waktu keputusan dengan forecast target yang sama. BUY/SELL
dibandingkan dengan perubahan itu, bukan kemiringan garis antar-forecast yang
berasal dari waktu keputusan berbeda. Interval vertikal menampilkan rentang
return terkalibrasi untuk keputusan tersebut. Grafik rolling precision hanya
menghitung BUY/SELL yang outcome-nya diketahui.

Kenaikan harga nominal emas tidak otomatis memberi daya prediksi M5. Return,
jarak relatif terhadap MA, ATR/close, sesi dan perubahan volatilitas lebih cocok
sebagai kandidat terukur daripada menganggap tren naik selalu berlanjut. Di luar
OHLCV, USD, suku bunga/real yield dan berita risiko dapat menjadi calon variabel
tambahan. World Gold Council mengelompokkan pendorong emas menjadi ekspansi
ekonomi, risiko/ketidakpastian, opportunity cost dan momentum dalam
[Gold Return Attribution Model](https://www.gold.org/goldhub/tools/gold-return-attribution-model).
Model atribusi bulanan itu bukan bukti bahwa variabel tersebut memberi sinyal M5.

Data eksternal tersebut belum ditambahkan karena belum tersedia dalam dataset
saat ini. Sebelum digunakan, setiap observasi harus memiliki waktu rilis aktual,
vintage revisi, zona waktu dan as-of join yang benar. Kalender FOMC/CPI/NFP juga
dapat diuji sebagai filter risiko, dengan aturannya dipilih sebelum TEST.

## Menjalankan dan menelusuri hasil

```powershell
uv run python -B -m technical.xgboost_model `
  --source-key metaquotes-demo-xauusd `
  --folds fold_2024 fold_2025 final_2026 `
  --target-precision 0.90 --min-policy-signals 50
```

Output default `technical/artifacts/direction_v3/`; hasil lama di
`technical/artifacts/walk_forward/` tetap tersedia. Per fold, baca
`hyperparameters.json`, `regression_feature_selection.json`, `calibration.json`,
`policy.json`, `metrics.json` dan `predictions.csv`. Dashboard final ada di
`final_2026/dashboard_2026.html`. Target precision, dukungan sampel, alasan
NO_TRADE dan konflik arah dilaporkan eksplisit. Jangan menurunkan syarat dengan
melihat TEST hanya untuk menghasilkan transaksi.

`monthly_metrics.csv` dan `monthly_diagnostics.png` memisahkan hasil per bulan:
return RMSE vs zero-return, raw/action macro F1, jumlah BUY/SELL dan return skill.
`nonoverlapping_metrics.json` mengevaluasi anchor horizon yang tidak overlap.
Keduanya dihitung setelah inference dan tidak memengaruhi fitting/rule.

## Kontrak arah dan perbaikan audit direction_v3

BUY berhasil jika close aktual akhir horizon lebih tinggi dari harga sinyal;
SELL berhasil jika lebih rendah. Harga tetap = FLAT (tidak berhasil).
NO_TRADE selalu NEUTRAL dan tidak masuk penyebut precision BUY/SELL. Penilaian
memakai close setelah 15 menit, bukan high/low intrabar atau menunggu sampai
harga akhirnya sesuai. Label training tetap tiga kelas berbasis biaya;
raw/classification macro F1 mengukur label itu secara terpisah.

Quality filter dan ReliefF sekarang fit hanya prefix training internal pertama
setelah purge. Kandidat dibekukan sebelum validation internal pertama, kemudian
RFECV memakai expanding CV pada TRAIN. Test mutation memastikan perubahan
label validation tidak mengubah ranking screening. Ini prefix screening tetap,
bukan ReliefF yang diulang per fold; kandidat bisa kurang peka pada regime baru.
Slippage masuk threshold label/keputusan tepat sekali.

`policy.json` memiliki `gate_funnel` per arah dan bagian POLICY: argmax arah,
input finite, point melampaui biaya, interval melampaui biaya. Point dan interval
adalah diagnosis gerbang masing-masing, bukan hitungan kumulatif seluruh rule.
Dengan ini nol peluang lolos filter dibedakan dari precision rendah pada sinyal.

`actual_status` terpisah dari outcome: AVAILABLE, PENDING, GAP_INVALID,
COST_UNAVAILABLE, OUTSIDE_EVALUATION. NO_TRADE tetap NEUTRAL ketika actual belum
tersedia. `pending_rows` khusus target melewati histori terakhir; jumlah semua
harga actual hilang berada di `unavailable_actual_rows`. Untuk artifact lama
tanpa status, dashboard memperkirakan pending dari waktu keputusan terakhir.

## Inference dari candle aktual

```python
from pathlib import Path
import pandas as pd
from technical.xgboost_model import load_frozen_models, predict_latest

bundle = load_frozen_models(Path("technical/artifacts/direction_v3/final_2026"))
# history: DataFrame schema loader, dari history_start sampai candle tertutup terbaru.
latest = predict_latest(bundle, history, as_of=pd.Timestamp.now(tz="UTC"))
```

Contract menyimpan konfigurasi, identitas source/symbol, urutan fitur classifier
dan regressor secara terpisah, awal histori, batas keputusan, serta SHA-256
model. Loader memverifikasi hash/urutan fitur; hanya muat joblib lokal terpercaya.
Hash mendeteksi artifact berubah/tercampur, bukan autentikasi penerbit.

Inference menolak awal histori terpotong, identitas berbeda, waktu naive dan
candle lebih lama dari batas staleness (default 10 menit). Fitur dihitung ulang
dari histori aktual lengkap agar EMA/kumulatif konsisten dengan batch; model
tetap beku. Histori September sengaja ditolak sebagai data terkini pada Oktober.
API ini untuk integrasi/paper monitoring; HTML masih snapshot offline.

## Hasil harga, spread dan replay eksekusi

SELL 4.300 ke 4.100 = **+200 unit harga kotor**. BUY memakai keluar dikurangi
masuk; SELL memakai masuk dikurangi keluar. Jika spread 0,5 dan slippage 0,2,
proxy bersih = 200 − 0,5 − 0,2 = **199,3 unit harga**. Minimum edge adalah
syarat sinyal, bukan biaya tambahan yang dipotong dari keuntungan.
NO_TRADE tidak memiliki transaksi atau keuntungan yang dihitung.

Dashboard merangkum total BUY/SELL, jumlah yang dapat dievaluasi, gross, spread,
slippage, dan net per arah serta gabungan. Proxy menjumlahkan sinyal overlap.
`execution_replay.json` terpisah membatasi satu posisi, masuk satu bar M5 setelah
keputusan pada open bar, keluar pada close target semula. OHLC diasumsikan bid;
ask = bid + spread bar. BUY masuk ask/keluar bid; SELL masuk bid/keluar ask.
Gross memakai perpindahan bid, lalu spread dipotong sekali, kemudian biaya
slippage/commission round-trip sesuai konfigurasi. Spread SELL memakai exit.
Harga fill entry/exit sudah termasuk sisi bid/ask; jangan memotong spread lagi.
Entry hilang, exit belum tersedia dan overlap dicatat eksplisit.

Unit hasil adalah unit harga XAUUSD, bukan saldo akun. Konversi untuk kontrak
linear bergantung lot, contract size, biaya akun dan mata uang. Dashboard tidak
menebak ukuran posisi. Replay belum memodelkan tick fill, latency aktual,
margin/leverage, swap, SL/TP atau equity akun; default slippage/commission nol
bukan perkiraan biaya riil.

## Batas kesiapan

Kebenaran kode riset tidak membuktikan precision 90% atau kelayakan trading live.
2024–2026 adalah tiga TEST tahunan dengan development masing-masing. Karena
hasilnya sudah ditinjau, ini evaluasi retrospektif. Forward monitoring candle
baru dan verifikasi biaya/fill broker masih diperlukan; jangan memilih ulang
threshold memakai TEST yang sama hanya agar transaksi muncul.
