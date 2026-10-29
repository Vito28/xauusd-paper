# Prediksi Harga End-to-End dengan LSTM

Proyek ini membuat **10.000 data harga harian sintetis**, melatih LSTM dari nol
dengan NumPy, mengevaluasi 2.000 data terakhir, lalu memprediksi hari berikutnya.

Implementasi memperlihatkan seluruh proses internal: normalisasi, sliding window,
forget/input/output gate, cell state, hidden state, forward pass, BPTT, gradient
clipping, optimizer Adam, inverse scaling, evaluasi, dan recursive forecasting.
Tidak ada TensorFlow, Keras, PyTorch, atau modul LSTM siap pakai.

Penjelasan rumus lengkap tersedia di [`MATEMATIKA_LSTM.md`](MATEMATIKA_LSTM.md).
Setelah program dijalankan, buka `visualisasi_proses.html` di browser dan geser
slider timestep untuk melihat nilai gate, cell state, hidden state, serta contoh
substitusi angka pada neuron pertama.

## Menjalankan

```bash
python -m pip install -r requirements.txt
python lstm_price_prediction.py --regenerate --epochs 50
```

Training dapat ditumpuk. Contoh total 100 epoch dalam dua sesi:

```bash
# Sesi pertama: membuat model dan menjalankan epoch 1–50
python lstm_price_prediction.py --regenerate --epochs 50

# Sesi kedua: memuat checkpoint dan menjalankan epoch 51–100
python lstm_price_prediction.py --resume --epochs 50
```

`--epochs` berarti jumlah epoch pada sesi saat ini, bukan target total epoch.
Checkpoint menyimpan bobot, momentum Adam, varians Adam, langkah optimizer,
riwayat loss, dan jumlah epoch selesai. Karena shuffle memakai seed per epoch,
hasil 50+50 sama dengan menjalankan 100 epoch sekaligus.

## Output

- `dataset_harga_10000.csv`: 10.000 baris tanggal dan harga.
- `hasil_prediksi_test_10000.csv`: aktual vs prediksi pada test set.
- `prediksi_masa_depan_10000.csv`: forecast periode yang belum ada.
- `riwayat_training_10000.csv`: loss pada setiap epoch dari seluruh sesi.
- `model_lstm_10000.npz`: bobot model terlatih.
- `checkpoint_lstm.npz`: state lengkap untuk melanjutkan training.
- `grafik_prediksi_10000.svg`: grafik aktual, prediksi test, dan forecast.
- `visualisasi_proses_10000.html`: visualisasi interaktif seluruh alur dan isi LSTM.
- `jejak_internal_lstm_10000.csv`: nilai internal gate/state pada tiap timestep.

Pembagian data dilakukan berurutan (80% train, 20% test), bukan acak. Scaler
hanya di-fit pada train set agar tidak terjadi kebocoran data. Dataset ini cocok
untuk demonstrasi alur LSTM, bukan dasar keputusan keuangan nyata.
