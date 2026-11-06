"""Feature selection di dalam setiap fold walk-forward (materi IF2116 Pertemuan 7).

Semua langkah hanya melihat blok TRAIN dan CALIBRATION milik fold tersebut, jadi
seleksi diulang per fold seperti Pipeline scikit-learn dan tidak membocorkan blok
THRESHOLD maupun EVALUATION.

  1. Filter (TRAIN)
     - buang fitur dengan nilai hilang > max_missing atau bernilai konstan;
     - audit kebocoran: |Spearman(fitur, arah label)| > leakage_threshold dicatat;
     - mutual information fitur terhadap label (menangkap hubungan non-linear);
     - redundansi: dari pasangan |Spearman| > redundancy_threshold, fitur dengan
       mutual information lebih tinggi dipertahankan.
  2. Embedded (TRAIN, early stopping di CALIBRATION)
     XGBoost dilatih pada fitur hasil filter; total gain dicatat sebagai pembanding.
  3. Permutation importance (CALIBRATION)
     Acak satu fitur lalu ukur kenaikan log loss. Fitur dipertahankan jika kenaikannya > 0,
     artinya model benar-benar memakai fitur itu untuk data di luar TRAIN.
  4. Model akhir dilatih ulang pada fitur terpilih.

Seleksi berbasis kelompok (wrapper) tetap tersedia lewat perintah `ablation`.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from technical.xgboost_scratch import multiclass_log_loss

# ---------------------------------------------------------------------------
# 1. Filter
# ---------------------------------------------------------------------------


def sample_positions(row_count: int, limit: int, seed: int) -> np.ndarray:
    """Posisi baris sampel (urut waktu dipertahankan) agar filter tetap cepat."""
    if row_count <= limit:
        return np.arange(row_count)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(row_count, size=limit, replace=False))


def leakage_audit(features: pd.DataFrame, labels, threshold: float) -> list[dict]:
    """Fitur yang hampir identik dengan arah label patut dicurigai bocor dari masa depan."""
    direction = pd.Series(np.asarray(labels, dtype=float) - 1, index=features.index)
    correlation = features.rank().corrwith(direction.rank()).abs()
    flags = []
    for name, value in correlation.items():
        if value > threshold:
            flags.append({"feature": name, "spearman_with_label": float(value)})
    return flags


def filter_features(features: pd.DataFrame, labels, settings, seed: int) -> dict:
    columns = list(features.columns)

    missing_share = features.isna().mean()
    dropped_missing = [name for name in columns if missing_share[name] > settings.max_missing]
    remaining = [name for name in columns if name not in dropped_missing]

    spread = features[remaining].std()
    dropped_constant = [name for name in remaining if not spread[name] > 0]
    remaining = [name for name in remaining if name not in dropped_constant]

    positions = sample_positions(len(features), settings.sample_rows, seed)
    sample = features.iloc[positions][remaining]
    sample_labels = np.asarray(labels, dtype=int)[positions]

    leakage = leakage_audit(sample, sample_labels, settings.leakage_threshold)

    # Mutual information tidak menerima NaN; median dihitung dari sampel TRAIN ini saja.
    filled = sample.fillna(sample.median()).fillna(0.0)
    scores = mutual_info_classif(filled.to_numpy(), sample_labels, random_state=seed)
    information = {}
    for name, score in zip(remaining, scores, strict=True):
        information[name] = float(score)

    # Redundansi: periksa fitur dari mutual information tertinggi; fitur yang terlalu
    # mirip dengan fitur yang sudah dipertahankan dibuang.
    correlation = sample.rank().corr().abs()
    kept = []
    dropped_redundant = []
    for name in sorted(remaining, key=lambda column: information[column], reverse=True):
        partner = None
        for other in kept:
            if correlation.loc[name, other] > settings.redundancy_threshold:
                partner = other
                break
        if partner is None:
            kept.append(name)
        else:
            dropped_redundant.append(
                {
                    "feature": name,
                    "kept": partner,
                    "spearman": float(correlation.loc[name, partner]),
                }
            )

    return {
        "kept": [name for name in remaining if name in kept],
        "dropped_missing": dropped_missing,
        "dropped_constant": dropped_constant,
        "dropped_redundant": dropped_redundant,
        "leakage_flags": leakage,
        "mutual_information": information,
    }


# ---------------------------------------------------------------------------
# 3. Permutation importance
# ---------------------------------------------------------------------------


def permutation_importance(model, features: pd.DataFrame, labels, repeats: int, seed: int):
    """Kenaikan log loss saat satu kolom diacak. Mengembalikan (log loss awal, hasil per fitur)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    base_loss = multiclass_log_loss(labels, model.predict_proba(features))

    shuffled = features.copy()
    result = {}
    for name in features.columns:
        original = shuffled[name].to_numpy().copy()
        increases = []
        for _ in range(repeats):
            shuffled[name] = rng.permutation(original)
            loss = multiclass_log_loss(labels, model.predict_proba(shuffled))
            increases.append(loss - base_loss)
        shuffled[name] = original
        result[name] = {"mean": float(np.mean(increases)), "std": float(np.std(increases))}
    return base_loss, result


# ---------------------------------------------------------------------------
# 4. Model dengan seleksi fitur di dalam fold
# ---------------------------------------------------------------------------


class FeatureSelectedModel:
    def __init__(self, config, base_model: str = "xgboost"):
        self.config = config
        self.base_model = base_model
        self.selected = []
        self.model = None
        self.summary = {}

    def fit(self, features, labels, eval_features=None, eval_labels=None):
        # Import di sini karena technical.models juga mengimpor modul ini.
        from technical.models import create_model

        if eval_features is None or eval_labels is None:
            raise ValueError(
                "Feature selection butuh blok CALIBRATION untuk permutation importance"
            )
        settings = self.config.feature_selection
        seed = self.config.run.random_state

        filtering = filter_features(features, labels, settings, seed)
        candidates = filtering["kept"]

        screening = create_model(self.base_model, self.config)
        screening.fit(features[candidates], labels, eval_features[candidates], eval_labels)
        gain = screening.feature_importance()
        base_loss, importance = permutation_importance(
            screening, eval_features[candidates], eval_labels, settings.permutation_repeats, seed
        )

        selected = [name for name in candidates if importance[name]["mean"] > 0]
        if len(selected) < settings.min_features:
            ranked = sorted(candidates, key=lambda name: importance[name]["mean"], reverse=True)
            top = ranked[: settings.min_features]
            selected = [name for name in candidates if name in top]
        self.selected = selected

        if selected == candidates:
            self.model = screening
        else:
            self.model = create_model(self.base_model, self.config)
            self.model.fit(features[selected], labels, eval_features[selected], eval_labels)

        self.summary = build_summary(features.columns, filtering, gain, importance, selected)
        self.summary["base_model"] = self.base_model
        self.summary["calibration_log_loss_before_shuffle"] = base_loss
        return self

    def predict_proba(self, features) -> np.ndarray:
        return self.model.predict_proba(features[self.selected])

    def best_iteration(self):
        return self.model.best_iteration()

    def selection_report(self) -> dict:
        return self.summary


def build_summary(columns, filtering, gain, importance, selected) -> dict:
    redundant = {}
    for item in filtering["dropped_redundant"]:
        redundant[item["feature"]] = item["kept"]

    features = {}
    for name in columns:
        if name in filtering["dropped_missing"]:
            status = "dropped_missing"
        elif name in filtering["dropped_constant"]:
            status = "dropped_constant"
        elif name in redundant:
            status = "dropped_redundant"
        elif name in selected:
            status = "selected"
        else:
            status = "dropped_permutation"
        entry = {
            "status": status,
            "mutual_information": filtering["mutual_information"].get(name),
            "gain": gain.get(name, 0.0) if name in importance else None,
            "permutation_importance": importance[name]["mean"] if name in importance else None,
            "permutation_std": importance[name]["std"] if name in importance else None,
        }
        if name in redundant:
            entry["redundant_with"] = redundant[name]
        features[name] = entry

    # Seberapa sejalan peringkat gain (embedded) dengan permutation importance.
    names = list(importance)
    gain_rank = pd.Series([gain.get(name, 0.0) for name in names]).rank()
    permutation_rank = pd.Series([importance[name]["mean"] for name in names]).rank()
    rank_correlation = float(gain_rank.corr(permutation_rank)) if len(names) > 2 else None

    return {
        "total_features": len(columns),
        "after_filter": len(importance),
        "selected_count": len(selected),
        "selected": list(selected),
        "leakage_flags": filtering["leakage_flags"],
        "gain_permutation_rank_correlation": rank_correlation,
        "features": features,
    }


def summarize_selection(reports: list[dict], feature_group: dict) -> list[dict]:
    """Stabilitas seleksi antarfold: berapa fold sebuah fitur terpilih."""
    rows = []
    for name in reports[0]["features"]:
        selected_folds = 0
        importances = []
        for report in reports:
            entry = report["features"][name]
            if entry["status"] == "selected":
                selected_folds += 1
            if entry["permutation_importance"] is not None:
                importances.append(entry["permutation_importance"])
        rows.append(
            {
                "feature": name,
                "group": feature_group.get(name),
                "selected_folds": selected_folds,
                "total_folds": len(reports),
                "mean_permutation_importance": float(np.mean(importances)) if importances else None,
            }
        )
    rows.sort(
        key=lambda row: (row["selected_folds"], row["mean_permutation_importance"] or -np.inf),
        reverse=True,
    )
    return rows
