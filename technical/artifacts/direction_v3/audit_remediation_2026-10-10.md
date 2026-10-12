# Laporan penyempurnaan audit XGBoost — 10 Oktober 2026

**Kesimpulan: perbaikan kode, kontrak evaluasi dan dashboard sudah diterapkan.
Model belum menunjukkan kemampuan BUY/SELL yang layak untuk trading live.**
Semua keputusan pada tiga TEST yang dijalankan adalah NO_TRADE. Precision
BUY/SELL belum terukur; ini bukan hasil akurasi 90% atau 100%.

## Aturan hasil yang sekarang dipakai

- Marker sinyal ada pada harga close aktual saat keputusan.
- BUY berhasil jika close aktual 15 menit kemudian lebih tinggi; SELL berhasil
  jika lebih rendah. Harga sama diberi FLAT dan tidak dihitung berhasil.
- NO_TRADE selalu NETRAL, tanpa transaksi dan tanpa kontribusi ke akurasi BUY/SELL.
- Harga sempat naik/turun di tengah horizon tidak menentukan hasil. Horizon tetap
  tiga candle M5, tanpa memilih waktu keluar sesudah melihat harga.
- Hasil arah terpisah dari hasil setelah biaya. Gerakan benar yang terlalu kecil
  masih dapat menghasilkan net negatif.

## Hasil evaluasi ulang

| TEST | Keputusan | BUY | SELL | NO_TRADE | Precision BUY/SELL | Net proxy / replay |
|---|---:|---:|---:|---:|---|---:|
| 2024 | 71.209 | 0 | 0 | 71.209 | Belum terukur | 0 / 0 |
| 2025 | 70.487 | 0 | 0 | 70.487 | Belum terukur | 0 / 0 |
| Jan–Sep 2026 | 51.236 | 0 | 0 | 51.236 | Belum terukur | 0 / 0 |

Recall dan F1 arah setelah abstain sama-sama nol. Tidak ada transaksi yang
menang ataupun kalah. Nol hasil adalah akibat tidak bertransaksi, bukan bukti
profitabilitas atau perlindungan modal pada transaksi masa depan.

Untuk melihat kemampuan classifier sebelum filter, berikut metrik **label
tiga kelas berbasis biaya**, yang berbeda dari precision arah sinyal dashboard:

| Raw classifier final 2026 | Precision | Recall | F1 | Support aktual |
|---|---:|---:|---:|---:|
| BUY | 46,47% | 94,18% | 62,23% | 23.392 |
| SELL | 48,64% | 6,74% | 11,85% | 23.307 |
| NO_TRADE | 41,67% | 0,13% | 0,25% | 3.954 |

Raw macro F1: **0,247765**. Regressor final 2026 memiliki return RMSE
**19,662281 bps**, sedikit lebih buruk dari zero-return **19,659872 bps**.
Direction accuracy regressor **50,30%**; interval coverage **66,92%** dengan
nominal 80%. Model belum membuktikan nilai prediktif yang memadai.

### Mengapa seluruhnya NO_TRADE?

Pada POLICY final (Oktober–Desember 2025), tidak ada satu pun anchor BUY atau
SELL yang lolos gerbang interval setelah biaya, baik bagian seleksi maupun
konfirmasi. Point forecast BUY melampaui biaya hanya pada 2 anchor seleksi dan
4 konfirmasi; SELL nol. Setiap arah membutuhkan minimal 50 sinyal per tahap.
Jadi aturan bahkan belum memiliki sampel transaksi cukup untuk mengukur target
precision 90%. Mematikan filter interval saja tidak menyelesaikan kelemahan
point forecast. Threshold tidak diturunkan berdasarkan hasil TEST.

### Cakupan data dan arti tahun evaluasi

Sumber: 754.660 candle M5 tervalidasi, histori mulai 2016 sampai actual terakhir
30 September 2026 pukul 06:45 UTC. Final TEST berisi 50.653 actual yang dapat
dinilai, 580 outcome melintasi gap, dan 3 target masih menunggu data.
2024/2025 masing-masing memiliki 3 target di luar akhir tahun evaluasinya;
statusnya OUTSIDE_EVALUATION, bukan pending.

2024–2026 adalah tiga TEST tahunan dengan model/development masing-masing.
Model final memakai TRAIN 2016–2024, TUNE Jan–Jun 2025, CALIBRATION Jul–Sep
2025, POLICY Okt–Des 2025. Input TEST diperbarui dari candle aktual tertutup;
bobot tetap beku. Semua TEST ini sudah ditinjau: hasilnya evaluasi retrospektif,
bukan forward test baru yang belum pernah dilihat.

## Temuan audit dan tindak lanjut yang diterapkan

| Temuan | Perbaikan | Bukti / batas |
|---|---|---|
| Label validation internal ikut screening ReliefF | Quality/ReliefF hanya fit prefix training internal pertama setelah purge, kandidat dibekukan sebelum RFECV | Mutation test ranking; prefix tetap dapat kurang adaptif pada regime lebih baru |
| Tidak diketahui gerbang mana yang menutup seluruh sinyal | Simpan gate_funnel per arah dan tahap POLICY | policy.json; hitungan point/interval adalah diagnosis masing-masing gerbang |
| Label dan keputusan tidak konsisten terhadap slippage | Masukkan slippage dalam label dan keputusan tepat sekali | Test biaya label dan replay |
| Dua bundle memiliki key fitur yang dapat bertabrakan | Loader gabungan memverifikasi contract, SHA-256, urutan fitur masing-masing estimator | Semua tiga pasangan artifact berhasil dimuat; test model diubah ditolak |
| Belum ada satu-prediksi-terbaru yang teruji | predict_latest memakai histori aktual lengkap, identitas source, batas waktu dan staleness | Fixture batch/latest parity, histori terpotong dan candle basi ditolak |
| Trading proxy menghitung sinyal overlap | Tambah replay satu posisi, delayed entry, bid/ask dan biaya eksplisit | execution_replay.json; bukan simulasi tick fill atau akun |
| Pending bercampur dengan gap; abstain dianggap tepat | Pisahkan actual_status, outcome_status, net_outcome; tanpa sinyal precision=null | Dashboard, CSV dan JSON tersinkronisasi |
| Harga marker/hasil kurang jelas | Marker di close keputusan, garis outcome ke actual target, hasil net terpisah | Template dan JavaScript diperbarui |

## Total keuntungan dan contoh 4.300 ke 4.100

**SELL: 4.300 − 4.100 = +200 unit harga kotor.** Dengan spread 0,5 dan
slippage 0,2, proxy bersih = **199,3 unit harga**. Untuk BUY pada harga yang
sama, kotor = −200 dan bersih = −200,7. Ini contoh aritmetika, bukan transaksi
model. Minimum edge adalah syarat sinyal, bukan biaya yang dipotong lagi.

Dashboard sekarang memiliki tabel total BUY/SELL, sinyal yang dievaluasi,
belum dinilai, gross, spread, slippage dan net per arah/gabungan. Ledger sinyal
menampilkan harga sinyal, harga actual target, gross, biaya dan net per baris.

Panel **replay satu posisi** menampilkan jumlah posisi ditutup, harga fill
masuk/keluar, gross bid, spread, slippage, komisi dan net. Default entry satu
bar M5 setelah keputusan, exit tetap pada target semula. OHLC diasumsikan bid;
BUY masuk ask/keluar bid, SELL masuk bid/keluar ask. Spread BUY memakai entry,
spread SELL memakai exit. Harga fill sudah memasukkan spread; rincian spread
tidak dipotong dua kali. Slippage/komisi round-trip default nol adalah asumsi,
bukan hasil pengukuran biaya broker.

Hasil ini **unit harga XAUUSD**, belum keuntungan uang akun. Lot, contract size,
mata uang, serta biaya akun harus diketahui untuk konversi. Replay belum
memodelkan tick fill, latency broker, leverage/margin, swap, SL/TP atau equity.
Sinyal dengan exit hilang dicatat unresolved, tidak dianggap untung/nol.

## Verifikasi dan batas yang masih terbuka

- Full suite: **85 passed, 1 skipped**. Termasuk isolation TEST, screening
  mutation, neutral outcomes, biaya satu kali, replay overlap, contoh SELL
  +200/BUY −200, bundle hash dan batch/latest parity.
- Ruff lulus, JavaScript syntax check lulus, generated code reference dan
  MkDocs strict build lulus.
- Training/evaluasi nyata ketiga fold selesai; loader/hash ketiga model lulus.
  Penyegaran outcome, versi metadata dan laporan laba tidak melatih ulang atau
  mengubah sinyal yang dihasilkan.
- Grafik diagnosis bulanan diperiksa visual. Interaksi browser dashboard belum
  diverifikasi dalam sesi ini karena akses file lokal browser dibatasi.
- Kinerja live/90% belum terbukti. Tahap berikutnya adalah eksperimen development
  untuk mengatasi point forecast mendekati nol, lalu forward monitoring data
  baru dan verifikasi fill/biaya. Jangan mengubah threshold pada TEST yang sama.

## Artefak

- [Dashboard final 2026](final_2026/dashboard_2026.html)
- [Metrik final](final_2026/metrics.json)
- [Replay eksekusi](final_2026/execution_replay.json)
- [Diagnosis per bulan](final_2026/monthly_diagnostics.png)
- [Audit sebelum perbaikan](../selective_v2/readiness_audit_2026-10-10.md)
- [Metodologi dan kontrak inference](../../../docs/technical-selective.md)
