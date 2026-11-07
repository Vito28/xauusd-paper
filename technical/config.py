"""Membaca dan memvalidasi konfigurasi eksperimen dari configs/technical.toml."""

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "technical.toml"

# Urutan kelas tetap dipakai di seluruh proyek: indeks 0, 1, 2.
# SELL < NO_TRADE < BUY dipilih agar (label - 1) langsung menjadi arah -1 / 0 / +1
# (dipakai audit kebocoran di technical/feature_selection.py) [teknis].
SELL = 0
NO_TRADE = 1
BUY = 2
CLASS_NAMES = ["SELL", "NO_TRADE", "BUY"]

# Total 41 fitur dalam 7 kelompok; seleksi lewat ablation kelompok [SoT D-06].
ALL_FEATURE_GROUPS = [
    "price_action",
    "trend",
    "momentum",
    "volatility",
    "volume",
    "spread",
    "time",
]


# ---------------------------------------------------------------------------
# Nilai default konfigurasi
#
# configs/technical.toml menimpa nilai di bawah; default ini hanya dipakai jika sebuah
# kunci tidak ada di TOML. Jika nilai atau alasannya diubah, ubah di kedua file.
#
# Penanda sumber alasan pada komentar:
#   [SoT X]  alasan tertulis di SOURCE_OF_TRUTH.md (bagian atau ID X).
#   [design] pilihan rancangan, BELUM hasil tuning. Validasi lewat walk-forward
#            (rata-rata fold 2021-2025), jangan pernah memakai data 2026 untuk memilihnya.
#   [teknis] konsekuensi matematis atau komputasi yang pasti, bukan selera.
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    # Broker demo MetaQuotes; spread akun demo belum tentu sama dengan live [SoT B.1, A-09].
    source_key: str = "metaquotes-demo-xauusd"
    symbol: str = "XAUUSD"
    # M1 lebih bising dan spread relatif lebih besar; M15 mengurangi sampel [SoT D-01].
    timeframe: str = "M5"
    # Sekitar 10 tahun (~750 ribu candle M5), mencakup beberapa kondisi pasar [SoT A-03].
    history_start: str = "2016-01-01T00:00:00Z"
    # Eksklusif. Training, tuning, kalibrasi, dan threshold selesai sebelum ini [SoT A-03].
    development_end: str = "2026-01-01T00:00:00Z"
    # Eksklusif. 2026 = retrospective out-of-sample, BUKAN blind test [SoT A-04, A-05].
    test_end: str = "2027-01-01T00:00:00Z"


@dataclass
class LabelConfig:
    # Barrier atas/bawah + batas waktu; NO_TRADE muncul alami tanpa gerak berarti [SoT D-03].
    method: str = "triple_barrier"
    # 6 x M5 = 30 menit. 15 menit terlalu kecil dibanding spread; nilai final menunggu
    # label-scan (EXP-R01) [SoT D-02, D-18].
    horizon_bars: int = 6
    # Konvensi Wilder, BUKAN nilai optimal yang terbukti [SoT tabel C].
    atr_period: int = 14
    # k pada barrier = k * ATR14 + spread. Usulan awal, menunggu label-scan [SoT D-03].
    # k lebih besar -> barrier lebih jarang tersentuh -> NO_TRADE lebih banyak [teknis].
    barrier_atr_multiple: float = 1.0


@dataclass
class SplitConfig:
    # Expanding window, satu fold per tahun evaluasi [SoT D-10]. Dimulai 2021 agar fold
    # pertama tetap punya ~4,5 tahun data TRAIN (2016-01 s/d 2020-07) [design].
    validation_years: list[int] = field(default_factory=lambda: [2021, 2022, 2023, 2024, 2025])
    # Disamakan dengan horizon_bars karena label "melihat" H candle ke depan [SoT D-11].
    embargo_bars: int = 6


@dataclass
class FeatureConfig:
    groups: list[str] = field(default_factory=lambda: list(ALL_FEATURE_GROUPS))


@dataclass
class XGBoostConfig:
    # Model referensi utama [SoT D-04]. Semua nilai [design] kecuali disebut lain.
    # Default XGBoost sebagai pembanding: learning_rate 0.3, max_depth 6,
    # min_child_weight 1, reg_lambda 1, subsample 1, colsample_bytree 1, max_bin 256.

    # Batas ATAS jumlah ronde; jumlah final dipilih early stopping.
    n_estimators: int = 300
    # Pohon dangkal (weak learner): maksimal 2^4 = 16 daun, interaksi sampai 4 fitur.
    max_depth: int = 4
    # Shrinkage: setiap pohon hanya menyumbang 5% koreksinya. 0.3 cepat menghafal noise;
    # 0.01 butuh ribuan pohon.
    learning_rate: float = 0.05
    # Minimum JUMLAH HESSIAN per daun, bukan jumlah baris. Hessian softprob = 2p(1-p) <= 0.5,
    # jadi minimal ~40 candle per daun (p=0.5), ~63 (p=0.2), ~111 (p=0.1) [teknis].
    min_child_weight: float = 20.0
    # L2: w = -G / (H + lambda). Daun kecil (H=20) dikecilkan ke 80%, daun besar (H=1000)
    # tetap ~99,5% [teknis]; yang ditekan terutama daun kecil yang berisi noise.
    reg_lambda: float = 5.0
    # L1 dimatikan: fitur padat (bukan sparse); reg_lambda + min_child_weight sudah cukup.
    reg_alpha: float = 0.0
    # Pruning cukup lewat max_depth dan min_child_weight. gamma = 0 juga membuat library dan
    # scratch identik meskipun konvensi gamma keduanya beda faktor 2 [SoT E-5, EXP-S05].
    gamma: float = 0.0
    # 80% baris dan 80% fitur acak per pohon -> pohon lebih beragam, variance turun.
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    # 64 bin kuantil (~1,6% data per bin): lebih cepat dan tidak membuat potongan super
    # halus yang hanya cocok dengan noise.
    max_bin: int = 64
    # Berhenti jika mlogloss CALIBRATION tidak membaik 30 ronde (~10% dari n_estimators).
    # Blok yang sama dipakai temperature scaling -> kalibrasi sedikit optimistis [SoT D-13].
    early_stopping_rounds: int = 30


@dataclass
class ScratchConfig:
    # Disamakan dengan XGBoostConfig, tanpa subsampling agar deterministik.
    # 150 (bukan 300) karena waktu: ~3,4 detik per ronde -> ~9 menit per fold [SoT EXP-S04].
    # early_stopping_rounds diambil dari XGBoostConfig.
    n_estimators: int = 150
    max_depth: int = 4
    learning_rate: float = 0.05
    min_child_weight: float = 20.0
    reg_lambda: float = 5.0
    reg_alpha: float = 0.0
    gamma: float = 0.0
    # Maksimal 255: bin disimpan sebagai uint8 dan satu bin dicadangkan untuk NaN [teknis].
    max_bin: int = 64


@dataclass
class RandomForestConfig:
    # Bagging pohon dalam -> menurunkan variance. Semua nilai [design].
    # Menambah pohon tidak membuat random forest overfit, hanya menambah waktu.
    n_estimators: int = 300
    # Pohon dalam (bias rendah), dibatasi agar tidak menghafal dan model tetap wajar.
    max_depth: int = 12
    # Data M5 bising: daun minimal 100 candle.
    min_samples_leaf: int = 100
    # Proporsi fitur acak per split (~13 dari 41), membuat pohon beragam.
    max_features: float = 0.33
    # Proporsi baris per pohon (bootstrap), juga mempercepat training.
    max_samples: float = 0.5


@dataclass
class EnsembleConfig:
    # Kesalahan anggota dibuat berbeda: boosting, bagging, dan linear [SoT D-20].
    members: list[str] = field(default_factory=lambda: ["xgboost", "random_forest", "logistic"])
    # Untuk 3 anggota: 0.05 -> 231 kombinasi bobot; 0.1 -> 66 (kasar); 0.01 -> 5.151
    # (lambat dan rawan overfit pada blok CALIBRATION) [teknis untuk jumlah, design untuk
    # pilihan]. Harus membagi 1 habis (dicek validate_config).
    weight_step: float = 0.05


@dataclass
class FeatureSelectionConfig:
    # Dipakai xgboost_selected, diulang di setiap fold [SoT D-21]. Semua nilai [design].
    # Buang fitur dengan nilai hilang > 40% di TRAIN.
    max_missing: float = 0.4
    # Sampel TRAIN untuk mutual information (k-NN, lambat pada ratusan ribu baris) dan korelasi.
    sample_rows: int = 30000
    # |Spearman| antarfitur > 0.95 = hampir identik -> redundan.
    redundancy_threshold: float = 0.95
    # |Spearman| fitur-label > 0.95 dicurigai bocor; fitur kausal wajar tidak setinggi itu.
    leakage_threshold: float = 0.95
    # Berapa kali setiap fitur diacak; rata-rata beberapa kali mengurangi efek kebetulan.
    permutation_repeats: int = 3
    # Jaring pengaman: minimal 5 fitur teratas tetap dipertahankan.
    min_features: int = 5


@dataclass
class DecisionConfig:
    # 0.34 = sedikit di atas 1/3 (tebakan seragam untuk 3 kelas) [design].
    threshold_grid_start: float = 0.34
    # Di atas 0.90 sinyal biasanya terlalu sedikit untuk lolos min_signals [design].
    threshold_grid_stop: float = 0.90
    # 0.34, 0.36, ..., 0.90 = 29 kandidat threshold [teknis].
    threshold_grid_step: float = 0.02
    # Model acak punya precision = prevalensi; 1.2 = minimal 20% lebih baik dari acak.
    # Angka 1.2 sendiri adalah pilihan rancangan [SoT D-09].
    min_precision_lift: float = 1.2
    # Precision dari segelintir sinyal terlalu tidak stabil. Angka 30 = pilihan rancangan
    # [SoT D-09].
    min_signals: int = 30


@dataclass
class RunConfig:
    # Angka bebas agar hasil bisa diulang; 42 hanya konvensi, tidak punya makna statistik.
    random_state: int = 42
    # Jumlah thread CPU; terutama memengaruhi kecepatan.
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
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
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
        feature_selection=FeatureSelectionConfig(**raw.get("feature_selection", {})),
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

    selection = config.feature_selection
    for name in ["max_missing", "redundancy_threshold", "leakage_threshold"]:
        value = getattr(selection, name)
        if not 0 < value <= 1:
            raise ValueError(f"feature_selection.{name} harus di (0, 1]")
    if selection.sample_rows < 1000:
        raise ValueError("feature_selection.sample_rows minimal 1000")
    if selection.permutation_repeats < 1 or selection.min_features < 1:
        raise ValueError("feature_selection.permutation_repeats dan min_features minimal 1")

    if config.decision.min_precision_lift < 1.0:
        raise ValueError("min_precision_lift minimal 1.0 (tidak lebih buruk dari acak)")
    if config.decision.min_signals < 1:
        raise ValueError("decision.min_signals minimal 1")
