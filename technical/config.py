"""Membaca dan memvalidasi konfigurasi eksperimen dari configs/technical.toml."""

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "technical.toml"

# Urutan kelas tetap dipakai di seluruh proyek: indeks 0, 1, 2.
SELL = 0
NO_TRADE = 1
BUY = 2
CLASS_NAMES = ["SELL", "NO_TRADE", "BUY"]

ALL_FEATURE_GROUPS = [
    "price_action",
    "trend",
    "momentum",
    "volatility",
    "volume",
    "spread",
    "time",
]


@dataclass
class DataConfig:
    source_key: str = "metaquotes-demo-xauusd"
    symbol: str = "XAUUSD"
    timeframe: str = "M5"
    history_start: str = "2016-01-01T00:00:00Z"
    development_end: str = "2026-01-01T00:00:00Z"
    test_end: str = "2027-01-01T00:00:00Z"


@dataclass
class LabelConfig:
    method: str = "triple_barrier"
    horizon_bars: int = 6
    atr_period: int = 14
    barrier_atr_multiple: float = 1.0


@dataclass
class SplitConfig:
    validation_years: list[int] = field(default_factory=lambda: [2021, 2022, 2023, 2024, 2025])
    embargo_bars: int = 6


@dataclass
class FeatureConfig:
    groups: list[str] = field(default_factory=lambda: list(ALL_FEATURE_GROUPS))


@dataclass
class XGBoostConfig:
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    min_child_weight: float = 20.0
    reg_lambda: float = 5.0
    reg_alpha: float = 0.0
    gamma: float = 0.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    max_bin: int = 64
    early_stopping_rounds: int = 30


@dataclass
class ScratchConfig:
    n_estimators: int = 150
    max_depth: int = 4
    learning_rate: float = 0.05
    min_child_weight: float = 20.0
    reg_lambda: float = 5.0
    reg_alpha: float = 0.0
    gamma: float = 0.0
    max_bin: int = 64


@dataclass
class RandomForestConfig:
    n_estimators: int = 300
    max_depth: int = 12
    min_samples_leaf: int = 100
    max_features: float = 0.33
    max_samples: float = 0.5


@dataclass
class EnsembleConfig:
    members: list[str] = field(default_factory=lambda: ["xgboost", "random_forest", "logistic"])
    weight_step: float = 0.05


@dataclass
class DecisionConfig:
    threshold_grid_start: float = 0.34
    threshold_grid_stop: float = 0.90
    threshold_grid_step: float = 0.02
    min_precision_lift: float = 1.2
    min_signals: int = 30


@dataclass
class RunConfig:
    random_state: int = 42
    n_jobs: int = 4
    output_dir: str = "runs"


@dataclass
class TechnicalConfig:
    data: DataConfig = field(default_factory=DataConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    scratch: ScratchConfig = field(default_factory=ScratchConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> TechnicalConfig:
    """Baca file TOML lalu ubah setiap bagian menjadi dataclass yang tervalidasi."""
    with open(path, "rb") as file:
        raw = tomllib.load(file)

    config = TechnicalConfig(
        data=DataConfig(**raw.get("data", {})),
        labels=LabelConfig(**raw.get("labels", {})),
        splits=SplitConfig(**raw.get("splits", {})),
        features=FeatureConfig(**raw.get("features", {})),
        xgboost=XGBoostConfig(**raw.get("xgboost", {})),
        scratch=ScratchConfig(**raw.get("scratch", {})),
        random_forest=RandomForestConfig(**raw.get("random_forest", {})),
        ensemble=EnsembleConfig(**raw.get("ensemble", {})),
        decision=DecisionConfig(**raw.get("decision", {})),
        run=RunConfig(**raw.get("run", {})),
    )
    validate_config(config)
    return config


def validate_config(config: TechnicalConfig) -> None:
    """Tolak konfigurasi yang melanggar aturan penelitian sebelum data dibaca."""
    if config.data.timeframe != "M5":
        raise ValueError("Pipeline ini dirancang untuk timeframe M5")

    history_start = pd.Timestamp(config.data.history_start)
    development_end = pd.Timestamp(config.data.development_end)
    test_end = pd.Timestamp(config.data.test_end)
    if history_start.tzinfo is None or development_end.tzinfo is None or test_end.tzinfo is None:
        raise ValueError("Semua timestamp konfigurasi wajib memiliki timezone (UTC)")
    if not history_start < development_end < test_end:
        raise ValueError("Urutan wajib: history_start < development_end < test_end")

    if config.labels.method not in ("triple_barrier", "fixed_horizon"):
        raise ValueError("labels.method harus 'triple_barrier' atau 'fixed_horizon'")
    if config.labels.horizon_bars < 1:
        raise ValueError("labels.horizon_bars minimal 1")
    if config.labels.atr_period < 2:
        raise ValueError("labels.atr_period minimal 2")
    if config.labels.barrier_atr_multiple <= 0:
        raise ValueError("labels.barrier_atr_multiple harus positif")

    if config.splits.embargo_bars < 0:
        raise ValueError("splits.embargo_bars tidak boleh negatif")
    for year in config.splits.validation_years:
        validation_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        if validation_start >= development_end:
            raise ValueError(f"Tahun validasi {year} melewati development_end")

    for group in config.features.groups:
        if group not in ALL_FEATURE_GROUPS:
            raise ValueError(f"Kelompok fitur tidak dikenal: {group}")
    if len(config.features.groups) == 0:
        raise ValueError("Minimal satu kelompok fitur")

    allowed_members = ["xgboost", "xgboost_balanced", "random_forest", "logistic"]
    if len(config.ensemble.members) < 2:
        raise ValueError("ensemble.members minimal dua model")
    for member in config.ensemble.members:
        if member not in allowed_members:
            raise ValueError(
                f"Anggota ensemble tidak dikenal: {member}. Pilihan: {allowed_members}"
            )
    units = 1 / config.ensemble.weight_step
    if not 0 < config.ensemble.weight_step <= 0.5 or abs(units - round(units)) > 1e-9:
        raise ValueError("ensemble.weight_step harus membagi 1 habis, misalnya 0.05 atau 0.1")

    if config.decision.min_precision_lift < 1.0:
        raise ValueError("min_precision_lift minimal 1.0 (tidak lebih buruk dari acak)")
    if config.decision.min_signals < 1:
        raise ValueError("decision.min_signals minimal 1")
