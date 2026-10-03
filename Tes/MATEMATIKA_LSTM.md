# Proses Matematika LSTM dari Nol

Implementasi pada `lstm_price_prediction.py` tidak memakai TensorFlow, Keras,
PyTorch, atau modul LSTM siap pakai. NumPy hanya dipakai untuk penyimpanan dan
operasi matriks; persamaan forward, loss, BPTT, gradient clipping, dan Adam
ditulis sendiri.

## 1. Data dan normalisasi

Dataset mempunyai 10.000 harga harian. Data indeks 0–7.999 menjadi train dan
8.000–9.999 menjadi test. Nilai minimum dan maksimum hanya dihitung dari train
supaya tidak terjadi data leakage.

```text
x_scaled = (x - minimum_train) / (maximum_train - minimum_train)
x_asli   = x_scaled * (maximum_train - minimum_train) + minimum_train
```

Dengan window default 30, input pertama adalah harga hari 1–30 dan targetnya
hari 31. Karena 8.000 titik pertama dipakai untuk train, jumlah sequence train
adalah `8.000 - 30 = 7.970`. Test terdiri dari 2.000 sequence.

Training dapat dibagi menjadi beberapa sesi. `checkpoint_lstm.npz` menyimpan
semua bobot, dua state Adam (`m` dan `v`), langkah optimizer, jumlah epoch, dan
riwayat loss. Karena itu sesi kedua melanjutkan tepat dari keadaan sesi pertama,
bukan menginisialisasi model baru.

## 2. Forward pass satu timestep

Untuk setiap timestep `t`, input harga `x_t` digabungkan dengan hidden state
sebelumnya:

```text
z_t = concat(h_(t-1), x_t)
```

Model menghitung empat gate:

```text
a_f = W_f z_t + b_f        f_t = sigmoid(a_f)   # memori yang dipertahankan
a_i = W_i z_t + b_i        i_t = sigmoid(a_i)   # informasi yang ditulis
a_o = W_o z_t + b_o        o_t = sigmoid(a_o)   # memori yang dikeluarkan
a_g = W_g z_t + b_g        g_t = tanh(a_g)      # kandidat memori baru
```

Fungsi aktivasi dihitung sebagai:

```text
sigmoid(a) = 1 / (1 + exp(-a))
tanh(a)    = (exp(a) - exp(-a)) / (exp(a) + exp(-a))
```

Cell dan hidden state diperbarui dengan operasi elemen-per-elemen:

```text
c_t = f_t * c_(t-1) + i_t * g_t
h_t = o_t * tanh(c_t)
```

Sesudah timestep terakhir, dense layer menghasilkan harga berskala:

```text
y_hat = W_y h_T + b_y
```

## 3. Loss

Model menggunakan Mean Squared Error:

```text
L = (1/N) * sum((y_hat - y)^2)
dL/dy_hat = 2(y_hat - y) / N
```

## 4. Backpropagation Through Time

Cache forward dibaca dari timestep terakhir ke pertama. Untuk setiap timestep:

```text
do = dh * tanh(c_t)
dc = dc_next + dh * o_t * (1 - tanh(c_t)^2)
df = dc * c_(t-1)
di = dc * g_t
dg = dc * i_t
dc_next = dc * f_t
```

Turunan terhadap nilai sebelum aktivasi:

```text
da_f = df * f_t * (1 - f_t)
da_i = di * i_t * (1 - i_t)
da_o = do * o_t * (1 - o_t)
da_g = dg * (1 - g_t^2)
```

Karena setiap gate memiliki bentuk `a = Wz + b`:

```text
dW = da z^T
db = da
dz = W^T da
```

Bagian `dz` yang mengarah ke hidden state menjadi gradien untuk timestep
sebelumnya. Gradient global dipotong pada norm 1 agar training stabil, lalu
bobot diperbarui memakai rumus Adam yang ditulis dalam `adam_update()`.

## 5. Forecast

Prediksi test menggunakan 12 nilai aktual sebelumnya. Untuk forecast masa depan,
hasil prediksi dimasukkan kembali ke window sebagai input periode selanjutnya
(recursive forecasting), kemudian dikembalikan ke satuan rupiah dengan inverse
scaling.

Jalankan program dan buka `visualisasi_proses_10000.html`. Slider timestep menunjukkan
perubahan forget/input/output gate, cell state, dan hidden state untuk satu
sequence test. Angka lengkapnya juga tersedia di `jejak_internal_lstm_10000.csv`.
