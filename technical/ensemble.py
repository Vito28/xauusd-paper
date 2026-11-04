"""Ensemble soft voting berbobot untuk BUY / SELL / NO_TRADE.

Anggota dipilih agar kesalahannya berbeda (lihat configs/technical.toml [ensemble]):
  xgboost        : boosting, pohon dangkal berurutan -> menurunkan bias.
  random_forest  : bagging, pohon dalam paralel      -> menurunkan variance.
  logistic       : model linear                      -> pola kesalahan lain lagi.

Alur fit(TRAIN, CALIBRATION):
  1. Setiap anggota dilatih pada TRAIN (XGBoost memakai CALIBRATION untuk early stopping).
  2. Probabilitas setiap anggota pada CALIBRATION dikalibrasi dengan temperature scaling,
     karena soft voting hanya adil bila probabilitas antaranggota sebanding.
  3. Bobot w dipilih dari grid simpleks (w >= 0, jumlah w = 1) yang meminimalkan log loss
     campuran pada CALIBRATION:  p_ensemble = sum_m w_m * p_m.
     Bobot dipelajari dari blok yang tidak dipakai untuk training, sehingga tidak terjadi
     kebocoran seperti stacking yang belajar dari prediksi in-sample. Grid juga memuat
     bobot (1, 0, 0), jadi campuran tidak pernah lebih buruk daripada anggota terbaik
     pada blok tersebut.

Threshold BUY/SELL tetap dipilih belakangan di blok THRESHOLD oleh run_fold.
"""

import numpy as np

from technical.decision import apply_temperature, fit_temperature
from technical.xgboost_scratch import multiclass_log_loss


def weight_grid(member_count: int, step: float) -> list[list[float]]:
    """Semua kombinasi bobot kelipatan step yang jumlahnya 1 (grid simpleks)."""
    units = int(round(1 / step))
    combinations = []

    def fill(prefix: list[int], remaining: int) -> None:
        if len(prefix) == member_count - 1:
            combinations.append(prefix + [remaining])
            return
        for value in range(remaining + 1):
            fill(prefix + [value], remaining - value)

    fill([], units)
    grid = []
    for combination in combinations:
        grid.append([value / units for value in combination])
    return grid


def blend(probabilities: list[np.ndarray], weights: list[float]) -> np.ndarray:
    result = np.zeros_like(probabilities[0])
    for member_probabilities, weight in zip(probabilities, weights, strict=True):
        result += weight * member_probabilities
    return result


def search_weights(probabilities: list[np.ndarray], labels: np.ndarray, step: float) -> dict:
    """Pilih bobot dengan log loss campuran terkecil (seri: kandidat pertama dipertahankan)."""
    best_weights = None
    best_loss = None
    for weights in weight_grid(len(probabilities), step):
        loss = multiclass_log_loss(labels, blend(probabilities, weights))
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_weights = weights
    return {"weights": best_weights, "log_loss": best_loss}


def diversity(names: list[str], probabilities: list[np.ndarray], labels: np.ndarray) -> dict:
    """Seberapa berbeda kesalahan antaranggota (syarat ensemble yang efektif).

    disagreement      : proporsi baris dengan kelas argmax berbeda.
    error_correlation : korelasi indikator "salah" kedua anggota; makin kecil makin beragam.
    """
    report = {}
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            first_class = probabilities[first].argmax(axis=1)
            second_class = probabilities[second].argmax(axis=1)
            first_wrong = (first_class != labels).astype(float)
            second_wrong = (second_class != labels).astype(float)
            if first_wrong.std() == 0 or second_wrong.std() == 0:
                correlation = None
            else:
                correlation = float(np.corrcoef(first_wrong, second_wrong)[0, 1])
            report[f"{names[first]}|{names[second]}"] = {
                "disagreement": float((first_class != second_class).mean()),
                "error_correlation": correlation,
            }
    return report


class SoftVotingEnsemble:
    def __init__(self, config):
        self.config = config
        self.member_names = list(config.ensemble.members)
        self.weight_step = config.ensemble.weight_step
        self.members = []
        self.temperatures = []
        self.weights = []
        self.summary = {}

    def fit(self, features, labels, eval_features=None, eval_labels=None):
        # Import di sini karena technical.models juga mengimpor modul ini.
        from technical.models import create_model

        if eval_features is None or eval_labels is None:
            raise ValueError("Ensemble butuh blok CALIBRATION untuk kalibrasi dan bobot")
        eval_labels = np.asarray(eval_labels, dtype=int)

        calibrated = []
        member_report = {}
        for name in self.member_names:
            member = create_model(name, self.config)
            member.fit(features, labels, eval_features, eval_labels)
            raw = member.predict_proba(eval_features)
            calibration = fit_temperature(raw, eval_labels)
            probabilities = apply_temperature(raw, calibration["temperature"])

            self.members.append(member)
            self.temperatures.append(calibration["temperature"])
            calibrated.append(probabilities)
            member_report[name] = {
                "temperature": calibration["temperature"],
                "log_loss": multiclass_log_loss(eval_labels, probabilities),
            }

        chosen = search_weights(calibrated, eval_labels, self.weight_step)
        self.weights = chosen["weights"]
        for name, weight in zip(self.member_names, self.weights, strict=True):
            member_report[name]["weight"] = weight

        best_single = min(member_report, key=lambda name: member_report[name]["log_loss"])
        self.summary = {
            "method": "weighted_soft_voting",
            "fitted_on": "calibration",
            "members": member_report,
            "ensemble_log_loss": chosen["log_loss"],
            "best_single_member": best_single,
            "best_single_log_loss": member_report[best_single]["log_loss"],
            "diversity": diversity(self.member_names, calibrated, eval_labels),
        }
        return self

    def predict_proba(self, features) -> np.ndarray:
        result = None
        for member, temperature, weight in zip(
            self.members, self.temperatures, self.weights, strict=True
        ):
            # Anggota berbobot nol tidak memengaruhi hasil; lewati agar inferensi lebih cepat.
            if weight == 0:
                continue
            weighted = weight * apply_temperature(member.predict_proba(features), temperature)
            result = weighted if result is None else result + weighted
        return result

    def report(self) -> dict:
        return self.summary
