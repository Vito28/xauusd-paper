"""Decision layer Technical Agent: kalibrasi, threshold selektif, dan kontrak output.

Alur keputusan untuk satu candle:
  1. Model menghasilkan probabilitas mentah [p_sell, p_no_trade, p_buy].
  2. Temperature scaling (di-fit pada blok CALIBRATION) -> probabilitas terkalibrasi.
  3. Arah kandidat = BUY jika p_buy >= p_sell, selain itu SELL.
  4. Sinyal dikeluarkan hanya jika p_arah >= threshold arah tersebut
     (threshold dipilih pada blok THRESHOLD). Jika tidak, keputusan NO_TRADE.

Lihat SOURCE_OF_TRUTH.md bagian G (D-07, D-08) untuk alasan desain ini.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

from technical.config import BUY, CLASS_NAMES, NO_TRADE, SELL, DecisionConfig
from technical.xgboost_scratch import multiclass_log_loss

PROBABILITY_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# Kalibrasi: temperature scaling
# ---------------------------------------------------------------------------


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """p_T = softmax(log(p) / T). T > 1 melunakkan, T < 1 menajamkan probabilitas."""
    log_probabilities = np.log(np.clip(probabilities, PROBABILITY_FLOOR, 1.0))
    scaled = log_probabilities / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exponent = np.exp(scaled)
    return exponent / exponent.sum(axis=1, keepdims=True)


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """Pilih T yang meminimalkan log loss pada blok CALIBRATION (grid search 1 dimensi)."""
    candidates = np.exp(np.linspace(np.log(0.25), np.log(4.0), 121))
    best_temperature = 1.0
    best_loss = multiclass_log_loss(labels, probabilities)
    loss_before = best_loss

    for temperature in candidates:
        loss = multiclass_log_loss(labels, apply_temperature(probabilities, temperature))
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)

    return {
        "method": "temperature_scaling",
        "temperature": best_temperature,
        "log_loss_before": loss_before,
        "log_loss_after": best_loss,
    }


# ---------------------------------------------------------------------------
# Threshold selektif per arah
# ---------------------------------------------------------------------------


def direction_candidates(probabilities: np.ndarray, direction: int) -> np.ndarray:
    """True jika arah ini lebih kuat daripada arah lawannya."""
    if direction == BUY:
        return probabilities[:, BUY] >= probabilities[:, SELL]
    return probabilities[:, SELL] > probabilities[:, BUY]


def precision_recall_f1(true_positive: int, signal_count: int, actual_count: int) -> tuple:
    precision = true_positive / signal_count if signal_count > 0 else 0.0
    recall = true_positive / actual_count if actual_count > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1


def select_thresholds(
    probabilities: np.ndarray, labels: np.ndarray, settings: DecisionConfig
) -> dict:
    """Pilih threshold BUY dan SELL secara terpisah pada blok THRESHOLD.

    Kriteria per arah:
      - jumlah sinyal >= min_signals
      - precision     >= min_precision_lift x prevalensi kelas tersebut
      - di antara yang lolos, pilih F1 tertinggi (seri: precision lebih tinggi).
    Jika tidak ada threshold yang lolos, arah tersebut dinonaktifkan.
    """
    grid = np.arange(
        settings.threshold_grid_start,
        settings.threshold_grid_stop + 1e-9,
        settings.threshold_grid_step,
    )
    result = {}

    for direction in [BUY, SELL]:
        name = CLASS_NAMES[direction]
        actual = labels == direction
        actual_count = int(actual.sum())
        prevalence = actual_count / len(labels) if len(labels) > 0 else 0.0
        required_precision = settings.min_precision_lift * prevalence
        candidate = direction_candidates(probabilities, direction)

        trials = []
        best = None
        for threshold in grid:
            predicted = candidate & (probabilities[:, direction] >= threshold)
            signal_count = int(predicted.sum())
            true_positive = int((predicted & actual).sum())
            precision, recall, f1 = precision_recall_f1(true_positive, signal_count, actual_count)

            trial = {
                "threshold": round(float(threshold), 4),
                "signals": signal_count,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            trials.append(trial)

            eligible = signal_count >= settings.min_signals and precision >= required_precision
            if not eligible:
                continue
            if best is None:
                best = trial
            elif f1 > best["f1"]:
                best = trial
            elif f1 == best["f1"] and precision > best["precision"]:
                best = trial

        result[name] = {
            "enabled": best is not None,
            "threshold": best["threshold"] if best is not None else None,
            "prevalence": prevalence,
            "required_precision": required_precision,
            "chosen": best,
            "trials": trials,
        }
    return result


def decide(probabilities: np.ndarray, thresholds: dict) -> tuple[np.ndarray, np.ndarray]:
    """Ubah probabilitas terkalibrasi menjadi keputusan dan reason code."""
    count = len(probabilities)
    decisions = np.full(count, NO_TRADE)
    reasons = np.full(count, "BELOW_THRESHOLD", dtype=object)

    for direction in [BUY, SELL]:
        name = CLASS_NAMES[direction]
        candidate = direction_candidates(probabilities, direction)
        rule = thresholds[name]
        if not rule["enabled"]:
            reasons[candidate] = f"{name}_DISABLED_NO_VALID_THRESHOLD"
            continue
        accepted = candidate & (probabilities[:, direction] >= rule["threshold"])
        decisions[accepted] = direction
        reasons[accepted] = f"{name}_PROBABILITY_ABOVE_THRESHOLD"

    invalid = ~np.isfinite(probabilities).all(axis=1)
    decisions[invalid] = NO_TRADE
    reasons[invalid] = "INVALID_PROBABILITY"
    return decisions, reasons


# ---------------------------------------------------------------------------
# Kontrak output untuk integrasi multi-agent
# ---------------------------------------------------------------------------


@dataclass
class TechnicalSignal:
    """Satu keputusan Technical Agent. Format ini yang akan dibaca agent lain."""

    timestamp: str
    symbol: str
    timeframe: str
    model_version: str
    signal: str
    confidence: float
    probabilities: dict
    horizon_minutes: int
    reason_code: str
    market_regime: dict = field(default_factory=dict)
    feature_snapshot: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def market_regime(feature_row: dict) -> dict:
    """Ringkasan rezim pasar yang mudah diaudit (bukan input keputusan model)."""
    regime = {}
    adx = feature_row.get("adx_14")
    if adx is not None and np.isfinite(adx):
        regime["trend_strength"] = "trending" if adx >= 25 else "ranging"
    volatility_ratio = feature_row.get("atr_14_vs_daily_mean")
    if volatility_ratio is not None and np.isfinite(volatility_ratio):
        if volatility_ratio >= 1.5:
            regime["volatility"] = "high"
        elif volatility_ratio <= 0.67:
            regime["volatility"] = "low"
        else:
            regime["volatility"] = "normal"
    return regime


def build_signal(
    timestamp: str,
    symbol: str,
    model_version: str,
    horizon_bars: int,
    calibrated_probabilities: np.ndarray,
    decision: int,
    reason: str,
    feature_row: dict,
    snapshot_features: list[str],
) -> TechnicalSignal:
    probabilities = {}
    for index, name in enumerate(CLASS_NAMES):
        probabilities[name] = float(calibrated_probabilities[index])

    snapshot = {}
    for name in snapshot_features:
        if name in feature_row:
            snapshot[name] = float(feature_row[name])

    return TechnicalSignal(
        timestamp=timestamp,
        symbol=symbol,
        timeframe="M5",
        model_version=model_version,
        signal=CLASS_NAMES[decision],
        confidence=float(calibrated_probabilities[decision]),
        probabilities=probabilities,
        horizon_minutes=horizon_bars * 5,
        reason_code=reason,
        market_regime=market_regime(feature_row),
        feature_snapshot=snapshot,
    )
