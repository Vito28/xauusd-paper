"""Metrik evaluasi klasifikasi tiga kelas untuk sinyal selektif.

Aturan penilaian (sesuai ketentuan penelitian):
  - Prediksi BUY benar hanya jika label aktual BUY. Jika label aktual
    NO_TRADE atau SELL, prediksi BUY dihitung sebagai false positive.
  - Aturan yang sama berlaku untuk SELL.
  - Overall accuracy dilaporkan, tetapi bukan ukuran utama karena NO_TRADE
    dapat mendominasi.
  - Metrik ini adalah metrik KLASIFIKASI, bukan hasil backtesting/profit.
"""

import math

import numpy as np

from technical.config import BUY, CLASS_NAMES, NO_TRADE, SELL

CLASS_COUNT = 3
PROBABILITY_FLOOR = 1e-15


def confusion_matrix(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Baris = label aktual, kolom = prediksi, urutan SELL, NO_TRADE, BUY."""
    matrix = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=int)
    for actual_value, predicted_value in zip(actual, predicted, strict=True):
        matrix[int(actual_value), int(predicted_value)] += 1
    return matrix


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple:
    """Interval kepercayaan Wilson 95% untuk proporsi (mis. precision).

    Mengasumsikan percobaan independen; label yang horizonnya tumpang tindih
    membuat interval ini cenderung terlalu sempit. Gunakan sebagai indikasi.
    """
    if trials == 0:
        return None, None
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
    margin = margin / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def per_class_metrics(matrix: np.ndarray) -> dict:
    result = {}
    for class_index, name in enumerate(CLASS_NAMES):
        true_positive = int(matrix[class_index, class_index])
        predicted_count = int(matrix[:, class_index].sum())
        actual_count = int(matrix[class_index, :].sum())

        precision = true_positive / predicted_count if predicted_count > 0 else None
        recall = true_positive / actual_count if actual_count > 0 else 0.0
        if precision is not None and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        lower, upper = wilson_interval(true_positive, predicted_count)

        result[name] = {
            "precision": precision,
            "precision_wilson95": [lower, upper],
            "recall": recall,
            "f1": f1,
            "support": actual_count,
            "predicted": predicted_count,
            "false_positive": predicted_count - true_positive,
        }
    return result


def expected_calibration_error(
    actual: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    """ECE top-label: rata-rata |akurasi - confidence| per bin confidence."""
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == actual).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(actual)
    error = 0.0
    for position in range(bins):
        lower = edges[position]
        upper = edges[position + 1]
        if position == bins - 1:
            inside = (confidence >= lower) & (confidence <= upper)
        else:
            inside = (confidence >= lower) & (confidence < upper)
        count = int(inside.sum())
        if count == 0:
            continue
        gap = abs(correct[inside].mean() - confidence[inside].mean())
        error += gap * count / total
    return float(error)


def probability_metrics(actual: np.ndarray, probabilities: np.ndarray) -> dict:
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(actual)), actual] = 1.0
    picked = np.clip(probabilities[np.arange(len(actual)), actual], PROBABILITY_FLOOR, 1.0)
    return {
        "log_loss": float(-np.mean(np.log(picked))),
        "brier_multiclass": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece_top_label": expected_calibration_error(actual, probabilities),
    }


def evaluate_predictions(actual, predicted, probabilities=None) -> dict:
    """Ringkasan metrik untuk satu blok evaluasi.

    actual, predicted : array int 0/1/2.
    probabilities     : array (n, 3) terkalibrasi, opsional.
    """
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    total = len(actual)
    matrix = confusion_matrix(actual, predicted)
    classes = per_class_metrics(matrix)

    buy_signals = int((predicted == BUY).sum())
    sell_signals = int((predicted == SELL).sum())

    # Kesalahan arah berlawanan lebih berat daripada salah abstain.
    buy_when_sell = int(matrix[SELL, BUY])
    sell_when_buy = int(matrix[BUY, SELL])

    macro_f1 = (classes["SELL"]["f1"] + classes["NO_TRADE"]["f1"] + classes["BUY"]["f1"]) / 3
    directional_f1 = (classes["SELL"]["f1"] + classes["BUY"]["f1"]) / 2

    actual_distribution = {}
    predicted_distribution = {}
    for class_index, name in enumerate(CLASS_NAMES):
        actual_distribution[name] = int((actual == class_index).sum())
        predicted_distribution[name] = int((predicted == class_index).sum())

    result = {
        "samples": total,
        "accuracy": float((actual == predicted).mean()) if total > 0 else None,
        "macro_f1": macro_f1,
        "directional_macro_f1": directional_f1,
        "per_class": classes,
        "confusion_matrix": {"rows_actual_cols_predicted": CLASS_NAMES, "values": matrix.tolist()},
        "actual_distribution": actual_distribution,
        "predicted_distribution": predicted_distribution,
        "signal_coverage": (buy_signals + sell_signals) / total if total > 0 else 0.0,
        "buy_coverage": buy_signals / total if total > 0 else 0.0,
        "sell_coverage": sell_signals / total if total > 0 else 0.0,
        "opposite_direction_errors": {
            "BUY_when_SELL": buy_when_sell,
            "SELL_when_BUY": sell_when_buy,
        },
        "no_trade_rate": float((predicted == NO_TRADE).mean()) if total > 0 else None,
    }
    if probabilities is not None:
        result["probability"] = probability_metrics(actual, np.asarray(probabilities, dtype=float))
    return result
