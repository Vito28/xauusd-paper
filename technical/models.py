"""Kandidat model dengan antarmuka yang sama: fit(...) dan predict_proba(X).

Model yang tersedia:
  always_no_trade : baseline yang selalu abstain (coverage 0).
  persistence     : baseline naif "arah H candle terakhir berlanjut", memakai
                    barrier yang sama dengan label sehingga adil dibandingkan.
  logistic        : multinomial logistic regression (baseline linear).
  xgboost         : XGBoost resmi (library) - model referensi utama.
  xgboost_balanced: XGBoost dengan bobot kelas balanced (pembanding threshold moving).
  xgboost_scratch : implementasi NumPy di technical/xgboost_scratch.py.
  random_forest   : bagging pohon dalam (anggota ensemble, menurunkan variance).
  ensemble        : soft voting berbobot beberapa model di atas (technical/ensemble.py).
  xgboost_selected: XGBoost dengan feature selection di dalam fold (technical/feature_selection.py).

Semua preprocessing (imputasi, scaling, binning) di-fit hanya pada data TRAIN.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from technical.config import BUY, NO_TRADE, SELL, TechnicalConfig
from technical.ensemble import SoftVotingEnsemble
from technical.feature_selection import FeatureSelectedModel
from technical.xgboost_scratch import ScratchXGBoostClassifier

MODEL_NAMES = [
    "always_no_trade",
    "persistence",
    "logistic",
    "xgboost",
    "xgboost_balanced",
    "xgboost_scratch",
    "random_forest",
    "ensemble",
    "xgboost_selected",
]


class AlwaysNoTradeModel:
    def fit(self, features, labels, eval_features=None, eval_labels=None):
        return self

    def predict_proba(self, features) -> np.ndarray:
        probabilities = np.zeros((len(features), 3))
        probabilities[:, NO_TRADE] = 1.0
        return probabilities


class PersistenceModel:
    """BUY jika close naik lebih dari barrier selama H candle terakhir, SELL jika turun.

    Membutuhkan kolom 'past_move' dan 'barrier' di DataFrame fitur (diisi oleh run.py),
    keduanya hanya memakai informasi sampai candle t.
    """

    def fit(self, features, labels, eval_features=None, eval_labels=None):
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        past_move = features["past_move"].to_numpy()
        barrier = features["barrier"].to_numpy()
        probabilities = np.zeros((len(features), 3))
        probabilities[:, NO_TRADE] = 1.0

        up = past_move > barrier
        down = past_move < -barrier
        probabilities[up, NO_TRADE] = 0.0
        probabilities[up, BUY] = 1.0
        probabilities[down, NO_TRADE] = 0.0
        probabilities[down, SELL] = 1.0
        return probabilities


class LogisticModel:
    def __init__(self, random_state: int):
        self.pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=1.0, max_iter=500, random_state=random_state)),
            ]
        )

    def fit(self, features, labels, eval_features=None, eval_labels=None):
        self.pipeline.fit(features, labels)
        return self

    def predict_proba(self, features) -> np.ndarray:
        return align_probabilities(self.pipeline.predict_proba(features), self.pipeline.classes_)


class LibraryXGBoostModel:
    def __init__(self, config: TechnicalConfig, balanced: bool = False):
        settings = config.xgboost
        # balanced=True: kelas jarang diberi bobot n / (K * n_kelas) agar kesalahan padanya
        # lebih mahal. Default False karena ketidakseimbangan ditangani threshold per arah.
        self.balanced = balanced
        self.early_stopping_rounds = settings.early_stopping_rounds
        self.model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            tree_method="hist",
            eval_metric="mlogloss",
            n_estimators=settings.n_estimators,
            max_depth=settings.max_depth,
            learning_rate=settings.learning_rate,
            min_child_weight=settings.min_child_weight,
            reg_lambda=settings.reg_lambda,
            reg_alpha=settings.reg_alpha,
            gamma=settings.gamma,
            subsample=settings.subsample,
            colsample_bytree=settings.colsample_bytree,
            max_bin=settings.max_bin,
            early_stopping_rounds=settings.early_stopping_rounds,
            random_state=config.run.random_state,
            n_jobs=config.run.n_jobs,
        )

    def fit(self, features, labels, eval_features=None, eval_labels=None):
        sample_weight = None
        if self.balanced:
            sample_weight = compute_sample_weight("balanced", labels)
        if eval_features is None:
            self.model.set_params(early_stopping_rounds=None)
            self.model.fit(features, labels, sample_weight=sample_weight, verbose=False)
        else:
            self.model.fit(
                features,
                labels,
                sample_weight=sample_weight,
                eval_set=[(eval_features, eval_labels)],
                verbose=False,
            )
        return self

    def predict_proba(self, features) -> np.ndarray:
        return self.model.predict_proba(features)

    def best_iteration(self) -> int | None:
        return getattr(self.model, "best_iteration", None)

    def feature_importance(self) -> dict[str, float]:
        booster = self.model.get_booster()
        return booster.get_score(importance_type="total_gain")


class ScratchModel:
    def __init__(self, config: TechnicalConfig):
        settings = config.scratch
        self.model = ScratchXGBoostClassifier(
            n_estimators=settings.n_estimators,
            max_depth=settings.max_depth,
            learning_rate=settings.learning_rate,
            min_child_weight=settings.min_child_weight,
            reg_lambda=settings.reg_lambda,
            reg_alpha=settings.reg_alpha,
            gamma=settings.gamma,
            max_bin=settings.max_bin,
            early_stopping_rounds=config.xgboost.early_stopping_rounds,
        )

    def fit(self, features, labels, eval_features=None, eval_labels=None):
        if eval_features is not None:
            eval_features = np.asarray(eval_features, dtype=float)
        self.model.fit(np.asarray(features, dtype=float), labels, eval_features, eval_labels)
        return self

    def predict_proba(self, features) -> np.ndarray:
        return self.model.predict_proba(np.asarray(features, dtype=float))

    def best_iteration(self) -> int:
        return self.model.best_iteration


class RandomForestModel:
    """Bagging: banyak pohon dalam pada sampel bootstrap dan subset fitur acak.

    Pohon dalam memiliki bias rendah tetapi variance tinggi; merata-ratakan ratusan pohon
    menurunkan variance. Nilai hilang ditangani langsung oleh scikit-learn >= 1.4.
    """

    def __init__(self, config: TechnicalConfig):
        settings = config.random_forest
        self.model = RandomForestClassifier(
            n_estimators=settings.n_estimators,
            max_depth=settings.max_depth,
            min_samples_leaf=settings.min_samples_leaf,
            max_features=settings.max_features,
            max_samples=settings.max_samples,
            random_state=config.run.random_state,
            n_jobs=config.run.n_jobs,
        )

    def fit(self, features, labels, eval_features=None, eval_labels=None):
        self.model.fit(features, labels)
        return self

    def predict_proba(self, features) -> np.ndarray:
        return align_probabilities(self.model.predict_proba(features), self.model.classes_)


def align_probabilities(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Pastikan kolom probabilitas selalu berurutan SELL, NO_TRADE, BUY."""
    aligned = np.zeros((len(probabilities), 3))
    for column, class_value in enumerate(classes):
        aligned[:, int(class_value)] = probabilities[:, column]
    return aligned


def create_model(name: str, config: TechnicalConfig):
    if name == "always_no_trade":
        return AlwaysNoTradeModel()
    if name == "persistence":
        return PersistenceModel()
    if name == "logistic":
        return LogisticModel(config.run.random_state)
    if name == "xgboost":
        return LibraryXGBoostModel(config)
    if name == "xgboost_balanced":
        return LibraryXGBoostModel(config, balanced=True)
    if name == "xgboost_scratch":
        return ScratchModel(config)
    if name == "random_forest":
        return RandomForestModel(config)
    if name == "ensemble":
        return SoftVotingEnsemble(config)
    if name == "xgboost_selected":
        return FeatureSelectedModel(config, base_model="xgboost")
    raise ValueError(f"Model tidak dikenal: {name}. Pilihan: {MODEL_NAMES}")


def is_rule_baseline(name: str) -> bool:
    """Baseline aturan tidak menghasilkan probabilitas yang perlu dikalibrasi."""
    return name in ("always_no_trade", "persistence")
