# Perhitungan Manual Klasifikasi XGBoost: Kode dan Penjelasan

Dokumen ini menjelaskan `xgboost_matematis.py` yang diperiksa pada 21 Oktober
2026. Potongan kode berasal dari implementasi yang sudah ada, bukan trainer baru.
File sumber lengkap tetap menjadi acuan untuk import, helper, dan konteks setiap
potongan. Kode memakai standard library Python, tanpa library ML/DL saat runtime.

## 1. Dua jalur yang harus dibedakan

| Jalur | Komponen | Yang dilakukan |
|---|---|---|
| Training dari nol | `ScratchClassifier`, `grow_tree`, `Node` | Menghitung gradient, curvature, split, leaf, dan boosting sendiri |
| Inference model technical | `FrozenTrees`, `TechnicalMath` | Membaca pohon terlatih dari JSON lalu menghitung prediksi, kalibrasi, dan policy sendiri |

Trainer scratch memakai exact split dan tidak mereplikasi training histogram,
row/column subsampling, balanced sample weights, early stopping, seleksi fitur,
dan seluruh fitting kalibrasi pada `technical`. Karena itu, training scratch
tidak dijanjikan menghasilkan pohon atau skor yang sama dengan `technical`.
Kesamaan output pada artifact pembanding menggunakan pohon terlatih dan input
fitur yang sama. Pembentukan ulang 223 fitur dari OHLC belum dilakukan di file ini.

Alur classifier scratch adalah:

`X TRAIN + label → margin awal → softmax → gradient/curvature → tiga tree → update margin → ulang → probabilitas → argmax`.

## 2. Input dan label kelas

Potongan sumber:

```python
LABELS = ("SELL", "NO_TRADE", "BUY")
```

Satu baris X memuat fitur numerik pada satu waktu keputusan. Urutan kolom
harus konsisten pada training dan prediksi. Target y berupa integer:
SELL=0, NO_TRADE=1, BUY=2. Label tersebut disediakan kepada trainer, bukan
ditemukan sendiri tanpa data target.

Pada pipeline technical, target dibentuk dari return tiga candle M5 ke depan
dan ambang spread+minimum edge+slippage. Future return hanya digunakan untuk
label TRAIN/evaluasi; nilai masa depan tidak menjadi fitur saat inference.
Label NO_TRADE sebagai kelas pelatihan juga berbeda dari tindakan abstain
setelah selective policy.

## 3. Margin awal dan softmax

Margin adalah skor riil untuk tiap kelas, bukan probabilitas dan bukan harga.
Pada awal training scratch, ketiga margin diset nol:

```python
self.trees, self.width = [], len(x[0])
margins = [[0.0] * 3 for _ in y]
```

Dengan margin `[0, 0, 0]`, probabilitas awal setiap kelas adalah 1/3.
Ini inisialisasi trainer scratch; jangan menganggap semua model hasil ekspor
technical memakai base margin nol. Jalur FrozenTrees membaca base score JSON.

Fungsi sumber untuk konversi margin:

```python
def softmax(margins: list[float], *, single: bool = False) -> list[float]:
    """Convert additive class margins into probabilities using stable exp(z-max(z))."""
    cast = f32 if single else float
    peak = max(margins)
    weights = [cast(math.exp(cast(value - peak))) for value in margins]
    total = 0.0
    for weight in weights:
        total = cast(total + weight)
    return [cast(weight / total) for weight in weights]
```

Rumusnya `p_k = exp(z_k − max(z)) / sum_j(exp(z_j − max(z)))`.
Pengurangan maksimum mempertahankan nilai probabilitas matematis sekaligus
mengurangi risiko overflow eksponensial. `single=False` dipakai trainer
scratch; `single=True` memakai pembulatan float32 untuk pembandingan model beku.
Ketiga probabilitas berjumlah sekitar satu, sesuai presisi perhitungan.

## 4. Loss, gradient, dan pendekatan Hessian

Loss klasifikasi satu observasi adalah cross entropy: `L_i = −log(p_i,y_i)`.
Untuk margin kelas k, gradient tepatnya `g_ik = p_ik − 1[y_i=k]`.

Potongan sumber dalam `ScratchClassifier.fit()`:

```python
probabilities = [softmax(row) for row in margins]
trees = []
for label in range(3):
    gradients = [
        p[label] - int(target == label)
        for p, target in zip(probabilities, y, strict=True)
    ]
    hessians = [max(2 * p[label] * (1 - p[label]), 1e-16) for p in probabilities]
    trees.append(grow_tree(x, gradients, hessians, self.config))
```

Contoh actual BUY dan probabilitas `[1/3, 1/3, 1/3]` memberi gradient
`[+1/3, +1/3, −2/3]`. Gradient negatif pada BUY akan mendorong koreksi margin
BUY ke atas melalui leaf weight yang bertanda negatif terhadap jumlah gradient.
Gradient kelas lain mendorong margin turun. Bentuk koreksi akhir tetap bergantung
pada observasi yang digabungkan dalam leaf dan regularisasinya.

**Hessian perlu dijelaskan secara tepat.** Hessian cross entropy multiclass
yang lengkap adalah matriks `diag(p) − p p^T`, dengan elemen diagonal
`p_k(1−p_k)` dan elemen silang `−p_k p_l`. Kode menggunakan pendekatan
diagonal upper bound `h_k = max(2 p_k(1−p_k), 1e−16)`, bukan menyatakan
bahwa turunan kedua skalar yang tepat adalah `2p(1−p)`.
Pendekatan dengan faktor dua juga terdapat pada
[implementasi objective multiclass XGBoost v3.0.0](https://github.com/dmlc/xgboost/blob/v3.0.0/src/objective/multiclass_obj.cu).
Versi scratch tidak menerapkan sample weights pada gradient/Hessian.

Semua probabilitas dihitung sebelum tiga tree ronde tersebut diperbarui.
Jadi, gradient kelas kedua dan ketiga memakai snapshot margin ronde yang
sama, bukan margin yang sudah diubah oleh tree kelas pertama.

## 5. Regularisasi dan nilai leaf

Untuk observasi yang masuk ke satu leaf, hitung `G = sum(g_i)` dan
`H = sum(h_i)`. Aproksimasi objective leaf yang dipakai adalah:

`G w + 0.5 (H + lambda) w^2 + alpha |w|`.

Dengan alpha nol, turunannya terhadap w memberi
`G + (H+lambda) w = 0`, sehingga `w = −G/(H+lambda)`.
Untuk alpha nonnol, kode memakai soft threshold:

```python
def soft_threshold(gradient: float, alpha: float) -> float:
    """Apply L1 regularization: sign(G) * max(abs(G)-alpha, 0)."""
    return math.copysign(max(abs(gradient) - alpha, 0.0), gradient)
```

Potongan pembentukan leaf dalam `grow_tree()`:

```python
g, h = sum(gradients[i] for i in indices), sum(hessians[i] for i in indices)
node = Node(
    -soft_threshold(g, config.reg_alpha) / (h + config.reg_lambda)
    if h + config.reg_lambda > 0
    else 0.0
)
```

Nilai leaf adalah koreksi skor kelas, bukan kode kelas, probabilitas, harga,
atau nilai profit. Lambda mengecilkan besarnya koreksi melalui denominator;
alpha dapat menjadikan koreksi nol ketika jumlah gradient cukup kecil.
Prinsip objective regularized, aproksimasi orde dua, dan perhitungan leaf
dibahas dalam [tutorial resmi boosted trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html).

## 6. Mencari split berdasarkan gain

Fungsi score dalam `grow_tree()`:

```python
def score(g: float, h: float) -> float:
    """Return regularized gain numerator divided by curvature."""
    return (
        soft_threshold(g, config.reg_alpha) ** 2 / (h + config.reg_lambda)
        if h + config.reg_lambda > 0
        else 0.0
    )
```

Potongan pengujian kandidat split:

```python
gl, hl = lg + (mg if missing_left else 0), lh + (mh if missing_left else 0)
gr, hr = g - gl, max(0.0, h - hl)
if min(hl, hr) < config.min_child_weight:
    continue
gain = 0.5 * (score(gl, hl) + score(gr, hr) - score(g, h)) - config.gamma
if gain > best_gain:
    best_gain, best = gain, (feature, threshold, missing_left)
```

`lg/lh` adalah prefix sum gradient/curvature kandidat kiri dari nilai fitur
yang diketahui; `mg/mh` adalah jumlah untuk observasi missing. `gl/hl` dan
`gr/hr` menjadi statistik anak kiri dan kanan setelah arah missing dipilih.

Gain mengukur pengurangan aproksimasi objective ketika parent dipecah menjadi
dua anak, setelah penalti gamma. Kode mencoba fitur dan threshold numerik
yang berbeda, serta dua arah missing. Hanya gain positif terbesar yang dipilih.
Ini bukan Gini impurity atau information gain entropy pada decision tree biasa.

Pembatas pertumbuhan mencakup `max_depth`, jumlah observasi minimum untuk
memungkinkan split, gain positif, dan `min_child_weight`. Min child weight
adalah batas jumlah curvature anak, bukan langsung jumlah baris anak.
Jika split tidak layak, node tetap menjadi leaf.

Pencarian split bersifat greedy pada setiap node, bukan pencarian seluruh
kombinasi struktur pohon secara global. Implementasi ini menggunakan exact
split pada nilai terobservasi, berbeda dari histogram trainer technical.

## 7. Traversal pohon dan missing value

Method sumber pada `Node`:

```python
def predict(self, row: list[float]) -> float:
    """Traverse stored splits until a leaf; equality follows the right branch."""
    node = self
    while node.feature >= 0:
        value = row[node.feature]
        left = node.missing_left if math.isnan(value) else value < node.threshold
        child = node.left if left else node.right
        if child is None:
            raise ValueError("Pohon rusak: child tidak tersedia")
        node = child
    return node.value
```

Nilai fitur lebih kecil dari threshold masuk kiri. Nilai sama dengan threshold
masuk kanan. `NaN` mengikuti arah default yang dipilih saat training.
Missing bukan otomatis diganti nol. Traversal berakhir pada leaf dan
mengembalikan koreksi margin untuk tree tersebut.

## 8. Boosting: satu tree per kelas per ronde

Potongan sumber setelah tiga tree ronde selesai:

```python
self.trees.append(trees)
for margin, row in zip(margins, x, strict=True):
    for label, tree in enumerate(trees):
        margin[label] += self.learning_rate * tree.predict(row)
```

Untuk setiap kelas k dan ronde t:

`z_k^(t)(x) = z_k^(t−1)(x) + eta × f_t,k(x)`.

Learning rate eta membatasi besar koreksi tiap ronde. Setelah margin diupdate,
ronde selanjutnya menghitung ulang probabilitas dan gradient. Pohon berikutnya
mengoreksi loss model yang sudah ada. Trainer scratch membuat tiga tree per
ronde; contoh 30 ronde menghasilkan 90 tree individual.

Sebuah tree untuk kelas BUY tetap menggunakan seluruh observasi TRAIN, termasuk
SELL dan NO_TRADE, melalui gradient kelas BUY. Bukan hanya baris berlabel BUY.

## 9. Probabilitas akhir dan pemilihan kelas

Potongan sumber dalam `predict_proba()`:

```python
return [
    softmax(
        [
            self.learning_rate
            * sum(round_trees[k].predict(row) for round_trees in self.trees)
            for k in range(3)
        ]
    )
    for row in x
]
```

Potongan sumber dalam `predict()`:

```python
return [max(range(3), key=p.__getitem__) for p in self.predict_proba(x)]
```

Jumlah leaf semua ronde menghasilkan tiga margin, kemudian softmax menghasilkan
tiga probabilitas yang belum dikalibrasi. Argmax memilih probabilitas terbesar.
Jika tie, implementasi memilih kelas dengan indeks pertama.
Memilih BUY dengan probabilitas terbesar tidak mensyaratkan probabilitas BUY
di atas 50% dan belum berarti trade tersebut layak dieksekusi.

## 10. Contoh hitungan satu ronde yang dijalankan

Contoh ini menggunakan tiga baris sintetis dan satu fitur ilustrasi. Fitur
0/1/2 tidak merepresentasikan harga emas aktual atau hasil feature engineering.

```python
from analisis_teknikall.xgboost_matematis import ScratchClassifier, TreeConfig

x = [[0.0], [1.0], [2.0]]
y = [0, 1, 2]
model = ScratchClassifier(
    rounds=1,
    learning_rate=0.1,
    config=TreeConfig(
        max_depth=1,
        min_child_weight=0.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        gamma=0.0,
    ),
).fit(x, y)

print(model.predict_proba([[2.0]]))
print(model.predict([[2.0]]))
```

`min_child_weight=0` sengaja dipakai pada contoh sangat kecil ini. Default
1,0 akan menolak anak dengan curvature 4/9. Ini parameter demonstrasi, bukan
hasil tuning atau rekomendasi untuk data emas.

### A. Gradient awal

Setiap probabilitas awal 1/3 dan setiap curvature pendekatan sebesar
`2 × (1/3) × (2/3) = 4/9`.

| Baris | Fitur | Label | g SELL | g NO_TRADE | g BUY |
|---|---:|---|---:|---:|---:|
| 1 | 0 | SELL | −2/3 | +1/3 | +1/3 |
| 2 | 1 | NO_TRADE | +1/3 | −2/3 | +1/3 |
| 3 | 2 | BUY | +1/3 | +1/3 | −2/3 |

### B. Split tree BUY

Tree BUY memilih threshold 2, sehingga baris fitur 0 dan 1 masuk kiri,
sedangkan fitur 2 masuk kanan.

- Kiri: `G_L = 2/3`, `H_L = 8/9`.
- Kanan: `G_R = −2/3`, `H_R = 4/9`.
- Parent: `G = 0`, `H = 4/3`.
- Leaf kiri: `−(2/3)/(8/9+1) = −6/17 = −0,352941`.
- Leaf kanan: `−(−2/3)/(4/9+1) = 6/13 = 0,461538`.
- Gain: `0.5 × (4/17 + 4/13 − 0) = 0,271493`.

### C. Tiga tree dan margin observasi fitur 2

Hasil implementasi satu ronde yang diperiksa:

| Tree kelas | Threshold | Leaf kiri | Leaf kanan |
|---|---:|---:|---:|
| SELL | 1 | +0,461538 | −0,352941 |
| NO_TRADE | 1 | −0,230769 | +0,176471 |
| BUY | 2 | −0,352941 | +0,461538 |

Observasi fitur 2 masuk kanan pada ketiga tree. Dengan eta 0,1, margin menjadi:

`[−0,03529412, +0,01764706, +0,04615385]`.

Softmax menghasilkan probabilitas:

| Kelas | Probabilitas |
|---|---:|
| SELL | 0,31855002 |
| NO_TRADE | 0,33586883 |
| BUY | 0,34558115 |

`predict()` mengembalikan kelas 2 atau BUY karena nilainya paling besar.
Probabilitas BUY baru sekitar 34,56%; contoh ini bukan bukti keyakinan tinggi,
akurasi pasar, generalisasi, atau hasil transaksi. Contoh hanya memeriksa
konsistensi perhitungan satu ronde pada data sintetis.

## 11. Inference manual yang menyamai model technical

Training scratch di atas berbeda dari pembacaan model yang sudah dilatih.
Pada `FrozenTrees.margins()`, potongan pentingnya adalah:

```python
result = self.base.copy()
for group, left, right, default, indices, values in self.trees:
    node = 0
    while left[node] != -1:
        value = row[indices[node]]
        go_left = default[node] if math.isnan(value) else value < values[node]
        node = left[node] if go_left else right[node]
    result[group] = f32(result[group] + values[node])
return result
```

`group` menentukan kelas tujuan tree. Nilai leaf JSON sudah memasukkan
learning rate, sehingga tidak dikalikan eta lagi. Jumlah margin dikonversi
ke softmax. Float32 meniru presisi input dan akumulasi model beku.

Pada `TechnicalMath.predict()`, probabilitas mentah kemudian dikalibrasi:

```python
calibrated = []
for p, a, b in zip(
    raw,
    self.meta["calibration_coefficients"],
    self.meta["calibration_intercepts"],
    strict=True,
):
    clipped = min(max(p, 1e-6), 1 - 1e-6)
    calibrated.append(sigmoid(a * math.log(clipped / (1 - clipped)) + b))
total = sum(calibrated)
probabilities = [p / total for p in calibrated]
```

Koefisien a dan b dibaca dari metadata hasil fitting CALIBRATION technical,
bukan dilatih ulang oleh kode ini. Kalibrasi memakai log odds dan sigmoid
per kelas, kemudian dinormalisasi. Ini berbeda dari softmax awal classifier.

Terakhir, `choose_signal()` memeriksa kelas terpilih, aktivasi arah dalam
policy, threshold probabilitas, margin probabilitas, arah/edge regressor,
interval return, dan biaya. Dengan demikian, prediksi kelas BUY dapat menjadi
tindakan NO_TRADE bila syarat trade belum terpenuhi. Argmax classifier,
probabilitas terkalibrasi, dan keputusan final merupakan tiga tahap berbeda.

Artifact `final_2026/parity_report.json` menyimpan pembandingan 51.236 baris:
selisih probabilitas maksimum sekitar 3,55 × 10^-8, serta nol mismatch
raw signal, final signal, dan alasan policy. Kesamaan ini berlaku pada
pohon dan fitur pembanding yang sama dalam toleransi numerik, bukan kesamaan
training dari nol atau bukti peningkatan akurasi pasar.

## 12. Menjalankan kode yang tersedia

Dari root proyek dalam PowerShell:

```powershell
uv run python -S -B analisis_teknikall/xgboost_matematis.py demo
uv run python -S -B analisis_teknikall/xgboost_matematis.py verify
```

`-S` menonaktifkan pemuatan site-packages pada proses Python. Perintah demo
melatih data sintetis dan bukan evaluasi holdout emas. Perintah verify
membandingkan inference manual dengan reference artifact. Menjalankan contoh
bagian 10 juga memakai modul standard-library-only yang sama, bukan XGBoost.

## Penelusuran sumber

> Catatan: file di bawah ini sudah dihapus dari `main` saat refactor Technical
> Agent dan masih tersedia di riwayat git (`git show 9a31d44:<path>`). Versi
> barunya adalah `technical/xgboost_scratch.py`.

- [File program lengkap](xgboost_matematis.py).
- [Penjelasan arsitektur matematis](../docs/xgboost-matematis.md).
- [Hasil verifikasi yang tersimpan](artifacts/final_2026/parity_report.json).
- [Laporan verifikasi lintas-fold](artifacts/hasil_verifikasi.md).

Tidak ada perubahan pada trainer atau hasil prediksi existing dalam
penyusunan penjelasan ini. Contoh sintetis satu ronde dijalankan terpisah
tanpa menulis ulang artifact model.
