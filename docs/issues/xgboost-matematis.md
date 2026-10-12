# Implementasi Algoritma XGBoost Matematis untuk XAUUSD

## Tujuan

Menyediakan implementasi matematis Newton gradient boosting untuk riset XAUUSD yang dapat dijelaskan dari objective sampai prediksi. Runtime modul matematis hanya menggunakan standard library Python.

## Ruang lingkup

- Training dari nol dengan exact split untuk regresi dan klasifikasi SELL / NO_TRADE / BUY.
- Gradient, Hessian, regularisasi L1/L2, bobot leaf, split gain, learning rate, stopping constraints, dan missing-value routing.
- Inference manual terhadap pohon JSON model technical yang sudah terlatih, termasuk margin, softmax, kalibrasi, return, interval, dan policy.
- Dokumentasi yang membedakan training scratch dari inference model beku. Exact-split tidak diklaim menghasilkan model yang identik dengan trainer histogram XGBoost.
- Data fitur numerik sebagai input. Pembuatan ulang seluruh fitur OHLC, tuning, dan fitting kalibrasi merupakan pekerjaan terpisah.

## Formulasi matematis

Dengan f_t(x) = w_{q(x)}, T jumlah leaf, eta learning rate:

$$
\hat y_i^{(t)} = \hat y_i^{(t-1)} + \eta f_t(x_i)
$$

Objective terregularisasi:

$$
\mathcal L = \sum_i \ell(y_i,\hat y_i)
+ \sum_t \left(\gamma T_t + \alpha \sum_j |w_{tj}| + \frac{\lambda}{2}\sum_j w_{tj}^2\right)
$$

Gradient dan Hessian dihitung terhadap margin/prediksi sebelum ronde baru. Ekspansi Taylor orde dua dipakai untuk mencari pohon Newton sebelum shrinkage eta diterapkan:

$$
g_i = \partial_{\hat y_i}\ell,\quad
h_i = \partial_{\hat y_i}^2\ell,\quad
G_j = \sum_{i\in I_j}g_i,\quad H_j = \sum_{i\in I_j}h_i
$$

Definisikan S_alpha(G) = sign(G) max(|G| - alpha, 0):

$$
w_j^* = -\frac{S_\alpha(G_j)}{H_j+\lambda}
$$

$$
Gain = \frac12\left(
\frac{S_\alpha(G_L)^2}{H_L+\lambda}
+\frac{S_\alpha(G_R)^2}{H_R+\lambda}
-\frac{S_\alpha(G_L+G_R)^2}{H_L+H_R+\lambda}
\right)-\gamma
$$

Untuk squared loss ell = (prediction - target)^2 / 2: g = prediction - target dan h = 1.

Untuk multiclass: p_k = exp(z_k - max(z)) / sum_j exp(z_j - max(z)), g_k = p_k - 1[y=k]. Dokumentasikan penggunaan Hessian diagonal upper bound h_k = max(2 p_k (1-p_k), epsilon) serta satu pohon per kelas per ronde; semua gradient ronde dihitung sebelum margin diperbarui.

## File terkait

- analisis_teknikall/xgboost_matematis.py
- tests/test_xgboost_matematis.py
- docs/xgboost-matematis.md
- Model JSON dan input/reference yang diperlukan untuk verifikasi parity dicatat dengan hash, urutan fitur, dan toleransi.

## Kriteria penerimaan

- [ ] Bobot leaf, gain split, dan gradient/Hessian sesuai rumus dengan tes numerik.
- [ ] Classifier dan regressor dilatih secara terpisah; probabilitas ternormalisasi.
- [ ] Missing-value routing, regularisasi, dan batas pertumbuhan pohon diuji.
- [ ] Demo berjalan dengan python -S -B tanpa library tambahan.
- [ ] Inference manual diuji pada pohon dan input yang sama dengan reference; toleransi dan mismatch dilaporkan.
- [ ] Nilai leaf JSON yang sudah mengandung learning rate tidak dikalikan eta dua kali.
- [ ] Data dipisah berdasarkan waktu; target masa depan tidak masuk fitur atau policy saat prediksi.
- [ ] Metrik training sintetis dipisahkan dari evaluasi holdout XAUUSD.
- [ ] Docstring, type hints, dokumentasi matematika, dan referensi kode diperbarui.

## Referensi

- [Tutorial resmi boosted trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)
- [Custom objective multiclass dan Hessian](https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html)
- [Model IO](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html)

