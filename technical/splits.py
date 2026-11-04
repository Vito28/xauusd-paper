"""Pembagian data secara kronologis: walk-forward validation dan final test 2026.

Setiap fold terdiri dari empat blok berurutan yang tidak saling tumpang tindih:

  TRAIN       : melatih model.
  CALIBRATION : mempelajari temperature scaling probabilitas.
  THRESHOLD   : memilih threshold confidence BUY/SELL.
  EVALUATION  : mengukur performa. Tidak dipakai untuk fitting apa pun.

Fold validasi untuk tahun Y (contoh Y = 2024):
  TRAIN       = history_start  .. Y-1-07-01
  CALIBRATION = Y-1-07-01      .. Y-1-10-01
  THRESHOLD   = Y-1-10-01      .. Y-01-01
  EVALUATION  = Y-01-01        .. Y+1-01-01

Fold final:
  TRAIN       = history_start  .. 2025-07-01
  CALIBRATION = 2025-07-01     .. 2025-10-01
  THRESHOLD   = 2025-10-01     .. 2026-01-01   (development_end)
  TEST        = 2026-01-01     .. test_end     (hanya label yang horizonnya lengkap)

Purging: baris di sebuah blok hanya dipakai jika label_end_time-nya (ditambah
embargo) masih berada sebelum akhir blok. Dengan begitu, label di blok awal
tidak pernah "melihat" harga yang berada di blok berikutnya.
"""

from dataclasses import dataclass

import pandas as pd

M5_MINUTES = 5
BLOCK_NAMES = ["train", "calibration", "threshold", "evaluation"]


@dataclass
class Fold:
    name: str
    train_start: pd.Timestamp
    calibration_start: pd.Timestamp
    threshold_start: pd.Timestamp
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp

    def block_bounds(self) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        return {
            "train": (self.train_start, self.calibration_start),
            "calibration": (self.calibration_start, self.threshold_start),
            "threshold": (self.threshold_start, self.evaluation_start),
            "evaluation": (self.evaluation_start, self.evaluation_end),
        }

    def describe(self) -> dict:
        bounds = self.block_bounds()
        description = {"name": self.name}
        for block_name in BLOCK_NAMES:
            start, end = bounds[block_name]
            description[block_name] = {"start": start.isoformat(), "end": end.isoformat()}
        return description


def utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"Timestamp wajib memiliki timezone: {value}")
    return timestamp.tz_convert("UTC")


def make_validation_folds(history_start: str, validation_years: list[int]) -> list[Fold]:
    folds = []
    for year in validation_years:
        fold = Fold(
            name=f"validation_{year}",
            train_start=utc(history_start),
            calibration_start=pd.Timestamp(f"{year - 1}-07-01", tz="UTC"),
            threshold_start=pd.Timestamp(f"{year - 1}-10-01", tz="UTC"),
            evaluation_start=pd.Timestamp(f"{year}-01-01", tz="UTC"),
            evaluation_end=pd.Timestamp(f"{year + 1}-01-01", tz="UTC"),
        )
        folds.append(fold)
    return folds


def make_final_fold(history_start: str, development_end: str, test_end: str) -> Fold:
    development_end_time = utc(development_end)
    return Fold(
        name="final_test",
        train_start=utc(history_start),
        calibration_start=development_end_time - pd.DateOffset(months=6),
        threshold_start=development_end_time - pd.DateOffset(months=3),
        evaluation_start=development_end_time,
        evaluation_end=utc(test_end),
    )


def split_fold(labels: pd.DataFrame, fold: Fold, embargo_bars: int) -> dict[str, pd.Index]:
    """Kembalikan indeks baris untuk setiap blok setelah purging dan embargo.

    labels wajib memiliki kolom: prediction_time, label_end_time, label.
    """
    embargo = pd.Timedelta(minutes=embargo_bars * M5_MINUTES)
    has_label = labels["label"].notna()

    blocks = {}
    for block_name, (start, end) in fold.block_bounds().items():
        inside = (labels["prediction_time"] >= start) & (labels["prediction_time"] < end)
        purged = (labels["label_end_time"] + embargo) <= end
        if block_name == "evaluation":
            # Blok evaluasi adalah blok terakhir; embargo tidak diperlukan,
            # tetapi label tetap harus selesai sebelum batas akhir evaluasi.
            purged = labels["label_end_time"] <= end
        mask = inside & purged & has_label
        blocks[block_name] = labels.index[mask]
    return blocks


def assert_no_overlap(labels: pd.DataFrame, blocks: dict[str, pd.Index]) -> None:
    """Pastikan setiap label blok selesai sebelum blok berikutnya dimulai."""
    for position in range(len(BLOCK_NAMES) - 1):
        current_name = BLOCK_NAMES[position]
        next_name = BLOCK_NAMES[position + 1]
        current = blocks[current_name]
        following = blocks[next_name]
        if len(current) == 0 or len(following) == 0:
            continue
        last_label_end = labels.loc[current, "label_end_time"].max()
        first_next_decision = labels.loc[following, "prediction_time"].min()
        if last_label_end > first_next_decision:
            raise AssertionError(
                f"Label blok {current_name} tumpang tindih dengan blok {next_name}"
            )
