"""Reproducible M5 feature, selection, and walk-forward configuration."""

from dataclasses import dataclass, field
from pathlib import Path

LABELS = ("SELL", "NO_TRADE", "BUY")
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    """Control research cost and causal indicator parameters without changing M5.

    ReliefF uses a bounded TRAIN reference sample and bounded query anchors.
    All model-fitting rows remain chronological; sampling affects ReliefF only.
    Session hours are local wall-clock hours, with DST handled by IANA zones.
    """

    horizon_bars: int = 3
    minimum_edge_bps: float = 1.0
    random_state: int = 42
    relieff_top_k: int = 40
    relieff_max_samples: int = 10000
    relieff_anchors: int = 512
    relieff_neighbors: int = 10
    rfecv_min_features: int = 10
    rfecv_step: int = 5
    cv_splits: int = 3
    n_estimators: int = 160
    early_stopping_rounds: int = 20
    regression_top_k: int = 40
    target_precision: float = 0.90
    min_policy_signals: int = 50
    interval_alpha: float = 0.20
    require_interval_agreement: bool = True
    slippage_bps: float = 0.0
    final_tune_start: str = "2025-01-01T00:00:00Z"
    final_calibration_start: str = "2025-07-01T00:00:00Z"
    final_policy_start: str = "2025-10-01T00:00:00Z"
    final_test_start: str = "2026-01-01T00:00:00Z"
    n_jobs: int = 4
    max_missing: float = 0.4
    near_constant: float = 0.999
    rolling_window: int = 288
    ma_windows: tuple[int, ...] = (5, 8, 12, 24, 48, 96, 144, 288)
    oscillator_windows: tuple[int, ...] = (7, 14, 21, 28)
    volatility_windows: tuple[int, ...] = (6, 12, 24, 48, 96, 288)
    macd_configs: tuple[tuple[int, int, int], ...] = ((8, 21, 5), (12, 26, 9), (19, 39, 9))
    lag_bars: tuple[int, ...] = (1, 3, 6)
    sessions: dict[str, tuple[str, int, int]] = field(
        default_factory=lambda: {
            "asia_session": ("Asia/Tokyo", 9, 18),
            "london_session": ("Europe/London", 8, 17),
            "new_york_session": ("America/New_York", 8, 17),
        }
    )

    def __post_init__(self) -> None:
        """Reject invalid costs, windows, and selection settings before loading data."""
        import math

        positive = (
            self.horizon_bars,
            self.relieff_top_k,
            self.relieff_max_samples,
            self.relieff_anchors,
            self.relieff_neighbors,
            self.rfecv_min_features,
            self.rfecv_step,
            self.n_estimators,
            self.early_stopping_rounds,
            self.regression_top_k,
            self.min_policy_signals,
            self.n_jobs,
            self.rolling_window,
        )
        if min(positive) < 1 or self.cv_splits < 2:
            raise ValueError("Horizon, windows, dan selection settings harus positif")
        if not math.isfinite(self.minimum_edge_bps) or self.minimum_edge_bps < 0:
            raise ValueError("minimum_edge_bps harus finite dan tidak negatif")
        if not 0 <= self.max_missing < 1 or not 0 < self.near_constant <= 1:
            raise ValueError("Quality filter settings tidak valid")
        if not 0.5 < self.target_precision <= 1 or not 0 < self.interval_alpha < 1:
            raise ValueError("Target precision harus >0.5 sampai 1; interval alpha harus (0,1)")
        if not math.isfinite(self.slippage_bps) or self.slippage_bps < 0:
            raise ValueError("Slippage bps harus finite dan tidak negatif")
        from analytics.data import parse_utc

        boundaries = [
            parse_utc(value)
            for value in (
                self.final_tune_start,
                self.final_calibration_start,
                self.final_policy_start,
                self.final_test_start,
            )
        ]
        if any(a >= b for a, b in zip(boundaries, boundaries[1:], strict=False)):
            raise ValueError("Tanggal TUNE/CALIBRATION/POLICY/TEST harus berurutan")
        if any(
            w < 2 for w in (*self.ma_windows, *self.oscillator_windows, *self.volatility_windows)
        ) or any(lag < 1 for lag in self.lag_bars):
            raise ValueError("Lookback minimal 2 dan lag minimal 1")
        if any(not (0 < fast < slow and signal > 0) for fast, slow, signal in self.macd_configs):
            raise ValueError("MACD membutuhkan 0 < fast < slow dan signal positif")
