"""XGBoost from scratch untuk klasifikasi multikelas (SELL, NO_TRADE, BUY).

Hanya memakai NumPy. Tidak memanggil XGBoost, LightGBM, atau scikit-learn.
Rumus mengikuti Chen & Guestrin (2016); penurunan lengkap ada di
SOURCE_OF_TRUTH.md bagian E. Nomor rumus (E-1, E-2, ...) dicantumkan di
komentar agar kode dan dokumen mudah dicocokkan.

Ringkasan algoritma per ronde boosting t:
  1. p_i = softmax(margin_i)                                    (E-6)
  2. Untuk setiap kelas k:
       g_ik = p_ik - y_ik                                        (E-7)
       h_ik = max(2 * p_ik * (1 - p_ik), 1e-16)                  (E-8)
     lalu bangun satu pohon regresi dengan g dan h tersebut.
  3. Bobot daun  w* = -T_alpha(G) / (H + lambda)                 (E-4)
     Gain split  0.5 * [S(G_L,H_L) + S(G_R,H_R) - S(G,H)] - gamma (E-5)
     dengan S(G,H) = T_alpha(G)^2 / (H + lambda).
  4. margin_ik += eta * pohon_k(x_i)                             (E-9)

Perbedaan dengan XGBoost resmi (dicatat transparan):
  - Kandidat split berasal dari kuantil fitur pada data training (tanpa bobot
    Hessian). XGBoost "hist" memakai weighted quantile sketch.
  - Tidak ada subsampling baris/kolom, monotone constraint, atau paralelisasi.
  - Tidak ada optimasi sistem (cache-aware access, out-of-core, sparse block).
  - Margin awal (base score) default "prior": log(prior kelas) yang di-center
    ke rata-rata nol; ini sama dengan intercept bawaan XGBoost 3.x untuk
    multi:softprob (diverifikasi empiris, lihat SOURCE_OF_TRUTH bagian F).
    Opsi "zero" memberi probabilitas awal seragam 1/3 (XGBoost lama, base_score=0.5).
"""

import math
import time

import numpy as np

HESSIAN_FLOOR = 1e-16
LOG_LOSS_EPSILON = 1e-15


# ---------------------------------------------------------------------------
# Fungsi matematika dasar (diuji langsung di tests/test_xgboost_scratch.py)
# ---------------------------------------------------------------------------


def softmax(margins: np.ndarray) -> np.ndarray:
    """Softmax stabil: exp(z - max(z)) / jumlah exp(z - max(z))."""
    shifted = margins - margins.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def one_hot(labels: np.ndarray, class_count: int) -> np.ndarray:
    encoded = np.zeros((len(labels), class_count))
    encoded[np.arange(len(labels)), labels] = 1.0
    return encoded


def softmax_gradient_hessian(
    probabilities: np.ndarray, targets_one_hot: np.ndarray, class_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Gradien dan Hessian diagonal cross-entropy multikelas untuk satu kelas.

    Turunan pertama terhadap margin kelas k : p_k - y_k
    Turunan kedua (diagonal)                : p_k * (1 - p_k)
    XGBoost resmi mengalikan Hessian diagonal dengan 2 (lihat SOURCE_OF_TRUTH E-8);
    faktor yang sama dipakai di sini agar sebanding.
    """
    probability = probabilities[:, class_index]
    target = targets_one_hot[:, class_index]
    gradient = probability - target
    hessian = 2.0 * probability * (1.0 - probability)
    hessian = np.maximum(hessian, HESSIAN_FLOOR)
    return gradient, hessian


def soft_threshold(gradient_sum: float | np.ndarray, alpha: float) -> float | np.ndarray:
    """T_alpha(G) = sign(G) * max(|G| - alpha, 0). Efek regularisasi L1."""
    return np.sign(gradient_sum) * np.maximum(np.abs(gradient_sum) - alpha, 0.0)


def leaf_weight(
    gradient_sum: float, hessian_sum: float, reg_lambda: float, reg_alpha: float
) -> float:
    """Bobot daun optimal w* = -T_alpha(G) / (H + lambda)."""
    return float(-soft_threshold(gradient_sum, reg_alpha) / (hessian_sum + reg_lambda))


def structure_score(gradient_sum, hessian_sum, reg_lambda: float, reg_alpha: float):
    """S(G, H) = T_alpha(G)^2 / (H + lambda). Bisa menerima skalar atau array."""
    shrunk = soft_threshold(gradient_sum, reg_alpha)
    return shrunk * shrunk / (hessian_sum + reg_lambda)


def split_gain(
    gradient_left,
    hessian_left,
    gradient_right,
    hessian_right,
    reg_lambda: float,
    reg_alpha: float,
    gamma: float,
):
    """Gain = 0.5 * [S(G_L,H_L) + S(G_R,H_R) - S(G_L+G_R, H_L+H_R)] - gamma."""
    score_left = structure_score(gradient_left, hessian_left, reg_lambda, reg_alpha)
    score_right = structure_score(gradient_right, hessian_right, reg_lambda, reg_alpha)
    score_parent = structure_score(
        gradient_left + gradient_right, hessian_left + hessian_right, reg_lambda, reg_alpha
    )
    return 0.5 * (score_left + score_right - score_parent) - gamma


def compute_initial_margin(labels: np.ndarray, class_count: int, method: str) -> np.ndarray:
    """Margin awal F_0 per kelas.

    "prior": F_0k = log(pi_k) - rata-rata_j log(pi_j), pi_k = proporsi kelas k di TRAIN.
             softmax(F_0) = pi, jadi model mulai dari distribusi kelas training.
    "zero" : F_0k = 0, probabilitas awal 1/K.
    """
    if method == "zero":
        return np.zeros(class_count)
    counts = np.bincount(labels, minlength=class_count).astype(float)
    proportions = np.maximum(counts / counts.sum(), 1e-12)
    log_proportions = np.log(proportions)
    return log_proportions - log_proportions.mean()


def multiclass_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    picked = probabilities[np.arange(len(labels)), labels]
    picked = np.clip(picked, LOG_LOSS_EPSILON, 1.0)
    return float(-np.mean(np.log(picked)))


# ---------------------------------------------------------------------------
# Binning: mengubah nilai fitur kontinu menjadi indeks bin kuantil
# ---------------------------------------------------------------------------


class QuantileBinner:
    """Membuat batas bin dari data TRAINING saja, lalu memetakan nilai ke bin.

    Bin 0 .. (jumlah_batas) untuk nilai yang diketahui; bin terakhir
    (missing_bin = max_bin) khusus untuk NaN.
    """

    def __init__(self, max_bin: int):
        if max_bin < 2 or max_bin > 255:
            raise ValueError("max_bin harus di antara 2 dan 255")
        self.max_bin = max_bin
        self.missing_bin = max_bin
        self.cut_points: list[np.ndarray] = []

    def fit(self, features: np.ndarray) -> "QuantileBinner":
        self.cut_points = []
        quantile_levels = np.linspace(0.0, 1.0, self.max_bin + 1)[1:-1]
        for column in range(features.shape[1]):
            values = features[:, column]
            values = values[np.isfinite(values)]
            if len(values) == 0:
                cuts = np.array([], dtype=float)
            else:
                cuts = np.unique(np.quantile(values, quantile_levels))
            self.cut_points.append(cuts)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        row_count, column_count = features.shape
        if column_count != len(self.cut_points):
            raise ValueError("Jumlah fitur berbeda dengan saat fit")
        bins = np.empty((row_count, column_count), dtype=np.uint8)
        for column in range(column_count):
            values = features[:, column]
            # side="right": nilai < cut[0] -> bin 0, cut[0] <= nilai < cut[1] -> bin 1, dst.
            column_bins = np.searchsorted(self.cut_points[column], values, side="right")
            column_bins[~np.isfinite(values)] = self.missing_bin
            bins[:, column] = column_bins
        return bins

    def threshold_value(self, column: int, split_bin: int) -> float:
        """Nilai asli dari split 'bin <= split_bin' -> 'nilai < threshold'."""
        return float(self.cut_points[column][split_bin])


# ---------------------------------------------------------------------------
# Pohon regresi berbasis gradien dan Hessian
# ---------------------------------------------------------------------------


class GradientTree:
    """Satu pohon dengan node yang disimpan sebagai list (mudah dibaca dan diekspor).

    Aturan routing di node internal:
      bin == missing_bin           -> ikut arah default (missing_left)
      bin <= split_bin             -> kiri
      selain itu                   -> kanan
    """

    def __init__(self, max_depth, min_child_weight, reg_lambda, reg_alpha, gamma, missing_bin):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.missing_bin = missing_bin

        self.feature: list[int] = []
        self.split_bin: list[int] = []
        self.missing_left: list[bool] = []
        self.left_child: list[int] = []
        self.right_child: list[int] = []
        self.value: list[float] = []
        self.gain: list[float] = []

    def new_node(self, value: float) -> int:
        self.feature.append(-1)
        self.split_bin.append(-1)
        self.missing_left.append(True)
        self.left_child.append(-1)
        self.right_child.append(-1)
        self.value.append(value)
        self.gain.append(0.0)
        return len(self.value) - 1

    def histograms(self, node_bins: np.ndarray, gradient: np.ndarray, hessian: np.ndarray):
        """Jumlah g dan h per (fitur, bin) untuk baris di sebuah node.

        Semua fitur diproses dengan satu np.bincount: indeks bin digeser sebesar
        fitur * (max_bin + 1) supaya bin setiap fitur menempati rentang sendiri.
        """
        row_count, feature_count = node_bins.shape
        bins_per_feature = self.missing_bin + 1
        offsets = np.arange(feature_count) * bins_per_feature
        flat_index = (node_bins.astype(np.int64) + offsets).ravel()
        total_length = feature_count * bins_per_feature

        repeated_gradient = np.repeat(gradient, feature_count)
        repeated_hessian = np.repeat(hessian, feature_count)
        gradient_histogram = np.bincount(
            flat_index, weights=repeated_gradient, minlength=total_length
        )
        hessian_histogram = np.bincount(
            flat_index, weights=repeated_hessian, minlength=total_length
        )

        gradient_histogram = gradient_histogram.reshape(feature_count, bins_per_feature)
        hessian_histogram = hessian_histogram.reshape(feature_count, bins_per_feature)
        return gradient_histogram, hessian_histogram

    def find_best_split(self, gradient_histogram, hessian_histogram, gradient_total, hessian_total):
        """Cari (fitur, bin, arah missing) dengan gain terbesar.

        Untuk setiap fitur, kandidat split adalah 'bin <= s' untuk s = 0 .. max_bin-2.
        Nilai missing dicoba masuk kiri dan kanan (sparsity-aware default direction).
        """
        best = {"gain": 0.0, "feature": -1, "split_bin": -1, "missing_left": True}
        feature_count = gradient_histogram.shape[0]

        for feature in range(feature_count):
            known_gradient = gradient_histogram[feature, : self.missing_bin]
            known_hessian = hessian_histogram[feature, : self.missing_bin]
            missing_gradient = gradient_histogram[feature, self.missing_bin]
            missing_hessian = hessian_histogram[feature, self.missing_bin]

            # Jumlah kumulatif kiri untuk split setelah bin s.
            left_gradient = np.cumsum(known_gradient)[:-1]
            left_hessian = np.cumsum(known_hessian)[:-1]

            for missing_goes_left in (True, False):
                if missing_goes_left:
                    gradient_left = left_gradient + missing_gradient
                    hessian_left = left_hessian + missing_hessian
                else:
                    gradient_left = left_gradient
                    hessian_left = left_hessian
                gradient_right = gradient_total - gradient_left
                hessian_right = hessian_total - hessian_left

                gains = split_gain(
                    gradient_left,
                    hessian_left,
                    gradient_right,
                    hessian_right,
                    self.reg_lambda,
                    self.reg_alpha,
                    self.gamma,
                )
                allowed = (hessian_left >= self.min_child_weight) & (
                    hessian_right >= self.min_child_weight
                )
                gains = np.where(allowed, gains, -np.inf)
                if len(gains) == 0:
                    continue

                position = int(np.argmax(gains))
                if gains[position] > best["gain"]:
                    best = {
                        "gain": float(gains[position]),
                        "feature": feature,
                        "split_bin": position,
                        "missing_left": missing_goes_left,
                    }
        return best

    def build(self, bins: np.ndarray, gradient: np.ndarray, hessian: np.ndarray) -> "GradientTree":
        root_rows = np.arange(len(gradient))
        self.grow(bins, gradient, hessian, root_rows, depth=0)
        return self

    def grow(self, bins, gradient, hessian, rows, depth) -> int:
        gradient_total = float(gradient[rows].sum())
        hessian_total = float(hessian[rows].sum())
        node = self.new_node(
            leaf_weight(gradient_total, hessian_total, self.reg_lambda, self.reg_alpha)
        )

        # Stopping criteria: kedalaman maksimum atau node terlalu kecil.
        if depth >= self.max_depth:
            return node
        if hessian_total < 2 * self.min_child_weight:
            return node

        node_bins = bins[rows]
        gradient_histogram, hessian_histogram = self.histograms(
            node_bins, gradient[rows], hessian[rows]
        )
        best = self.find_best_split(
            gradient_histogram, hessian_histogram, gradient_total, hessian_total
        )

        # Pruning: split hanya dibuat jika gain (sudah dikurangi gamma) positif.
        if best["feature"] < 0:
            return node

        feature_bins = node_bins[:, best["feature"]]
        is_missing = feature_bins == self.missing_bin
        goes_left = np.where(is_missing, best["missing_left"], feature_bins <= best["split_bin"])

        left_index = self.grow(bins, gradient, hessian, rows[goes_left], depth + 1)
        right_index = self.grow(bins, gradient, hessian, rows[~goes_left], depth + 1)

        self.feature[node] = best["feature"]
        self.split_bin[node] = best["split_bin"]
        self.missing_left[node] = best["missing_left"]
        self.left_child[node] = left_index
        self.right_child[node] = right_index
        self.gain[node] = best["gain"]
        return node

    def predict(self, bins: np.ndarray) -> np.ndarray:
        feature = np.array(self.feature)
        split_bin = np.array(self.split_bin)
        missing_left = np.array(self.missing_left)
        left_child = np.array(self.left_child)
        right_child = np.array(self.right_child)
        value = np.array(self.value)

        row_count = bins.shape[0]
        rows = np.arange(row_count)
        node = np.zeros(row_count, dtype=np.int64)
        for _ in range(self.max_depth):
            is_internal = feature[node] >= 0
            if not is_internal.any():
                break
            safe_feature = np.maximum(feature[node], 0)
            row_bins = bins[rows, safe_feature]
            is_missing = row_bins == self.missing_bin
            go_left = np.where(is_missing, missing_left[node], row_bins <= split_bin[node])
            next_node = np.where(go_left, left_child[node], right_child[node])
            node = np.where(is_internal, next_node, node)
        return value[node]

    def leaf_count(self) -> int:
        count = 0
        for feature in self.feature:
            if feature < 0:
                count += 1
        return count


# ---------------------------------------------------------------------------
# Model boosting multikelas
# ---------------------------------------------------------------------------


class ScratchXGBoostClassifier:
    """Gradient boosting multikelas dengan softmax, satu pohon per kelas per ronde."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        gamma: float = 0.0,
        max_bin: int = 64,
        early_stopping_rounds: int | None = None,
        base_score: str = "prior",
        verbose: bool = False,
    ):
        if n_estimators < 1 or max_depth < 1:
            raise ValueError("n_estimators dan max_depth minimal 1")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate harus di (0, 1]")
        if reg_lambda < 0 or reg_alpha < 0 or gamma < 0 or min_child_weight < 0:
            raise ValueError("Parameter regularisasi tidak boleh negatif")
        if base_score not in ("prior", "zero"):
            raise ValueError("base_score harus 'prior' atau 'zero'")
        self.base_score = base_score
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.max_bin = max_bin
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose

        self.class_count = 0
        self.initial_margin = np.zeros(0)
        self.binner: QuantileBinner | None = None
        self.rounds: list[list[GradientTree]] = []
        self.best_iteration = 0
        self.history: list[dict] = []

    def fit(
        self, features, labels, eval_features=None, eval_labels=None
    ) -> "ScratchXGBoostClassifier":
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if len(features) != len(labels):
            raise ValueError("Jumlah baris fitur dan label berbeda")

        self.class_count = int(labels.max()) + 1
        self.binner = QuantileBinner(self.max_bin).fit(features)
        bins = self.binner.transform(features)
        targets = one_hot(labels, self.class_count)

        # Initial prediction (base score), sama untuk semua baris.
        self.initial_margin = compute_initial_margin(labels, self.class_count, self.base_score)
        margins = np.tile(self.initial_margin, (len(labels), 1))

        use_eval = eval_features is not None and eval_labels is not None
        if use_eval:
            eval_bins = self.binner.transform(np.asarray(eval_features, dtype=float))
            eval_labels = np.asarray(eval_labels, dtype=int)
            eval_margins = np.tile(self.initial_margin, (len(eval_labels), 1))
            best_loss = math.inf
            rounds_without_improvement = 0

        self.rounds = []
        self.history = []
        for round_index in range(self.n_estimators):
            started = time.perf_counter()
            probabilities = softmax(margins)
            round_trees = []
            for class_index in range(self.class_count):
                gradient, hessian = softmax_gradient_hessian(probabilities, targets, class_index)
                tree = GradientTree(
                    self.max_depth,
                    self.min_child_weight,
                    self.reg_lambda,
                    self.reg_alpha,
                    self.gamma,
                    self.binner.missing_bin,
                )
                tree.build(bins, gradient, hessian)
                round_trees.append(tree)

            # Update semua kelas SETELAH seluruh pohon ronde ini dibangun,
            # karena gradien setiap kelas dihitung dari probabilitas yang sama.
            for class_index in range(self.class_count):
                margins[:, class_index] += self.learning_rate * round_trees[class_index].predict(
                    bins
                )
            self.rounds.append(round_trees)

            record = {
                "round": round_index,
                "train_log_loss": multiclass_log_loss(labels, softmax(margins)),
                "seconds": time.perf_counter() - started,
            }

            if use_eval:
                for class_index in range(self.class_count):
                    eval_margins[:, class_index] += self.learning_rate * round_trees[
                        class_index
                    ].predict(eval_bins)
                eval_loss = multiclass_log_loss(eval_labels, softmax(eval_margins))
                record["eval_log_loss"] = eval_loss
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    self.best_iteration = round_index
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
            else:
                self.best_iteration = round_index

            self.history.append(record)
            if self.verbose:
                print(record)

            if use_eval and self.early_stopping_rounds is not None:
                if rounds_without_improvement >= self.early_stopping_rounds:
                    break

        # Buang ronde setelah iterasi terbaik (setara dengan early stopping XGBoost).
        self.rounds = self.rounds[: self.best_iteration + 1]
        return self

    def predict_margin(self, features) -> np.ndarray:
        if self.binner is None:
            raise ValueError("Model belum di-fit")
        bins = self.binner.transform(np.asarray(features, dtype=float))
        margins = np.tile(self.initial_margin, (bins.shape[0], 1))
        for round_trees in self.rounds:
            for class_index in range(self.class_count):
                margins[:, class_index] += self.learning_rate * round_trees[class_index].predict(
                    bins
                )
        return margins

    def predict_proba(self, features) -> np.ndarray:
        return softmax(self.predict_margin(features))

    def predict(self, features) -> np.ndarray:
        return self.predict_proba(features).argmax(axis=1)

    def feature_gain(self, feature_count: int) -> np.ndarray:
        """Total gain split per fitur (analog 'total_gain' di XGBoost)."""
        totals = np.zeros(feature_count)
        for round_trees in self.rounds:
            for tree in round_trees:
                for node_index, feature in enumerate(tree.feature):
                    if feature >= 0:
                        totals[feature] += tree.gain[node_index]
        return totals
