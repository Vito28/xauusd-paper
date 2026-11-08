# ruff: noqa: E501  (string tabel Markdown sengaja ditulis utuh per baris)
"""EDA tingkat candle M5 untuk laporan bab 4.6 (ringkasan statistik, profil jam, label per tahun).

Analisis harian di reports/analytics/ tidak otomatis berlaku untuk horizon M5, jadi
bagian ini mengulang pemeriksaan pada candle M5 yang dipakai model. Data diambil dari
candle yang disimpan run walk-forward (2021–2025) dan final-test (2026) di folder runs/;
label aktual diambil dari predictions.csv run yang sama (hanya status AVAILABLE).

Jalankan dari root repo xauusd-paper:

    uv run python <dir>/eda_m5.py --walk-forward runs/<walk-forward> --final-test runs/<final-test> \
        --out <dir>

Menulis <dir>/hasil_eda_m5.md, <dir>/hasil_zona_waktu.md, <dir>/eda_m5.json, dan tiga gambar
di <dir>/gambar/.

Catatan zona waktu: kolom open_time_utc di database sebenarnya jam server MetaQuotes-Demo
(EET/EEST, aturan DST Uni Eropa), karena collector membaca field "time" MT5 sebagai epoch UTC.
Profil per jam di bawah dikonversi ke UTC yang benar dengan server_to_utc().
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from technical.features import average_true_range  # noqa: E402

M5_MINUTES = 5
BPS = 10_000
ATR_PERIOD = 14
WIB_OFFSET_HOURS = 7
SERVER_TIMEZONE = "Europe/Helsinki"  # EET (UTC+2) / EEST (UTC+3), aturan DST Uni Eropa

# Palet kategori (skill dataviz, mode terang): slot 1-3 lolos uji CVD untuk semua pasangan.
SERIES = {"SELL": "#2a78d6", "NO_TRADE": "#eb6834", "BUY": "#1baf7a"}
BAR = "#2a78d6"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
DPI = 200


# ---------------------------------------------------------------------------
# Format angka gaya Indonesia
# ---------------------------------------------------------------------------


def number(value, digits=2) -> str:
    text = f"{value:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def integer(value) -> str:
    return f"{int(round(value)):,}".replace(",", ".")


def percent(value, digits=1) -> str:
    return number(100 * value, digits) + "%"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def read_candles(walk_forward: Path, final_test: Path) -> pd.DataFrame:
    paths = sorted((walk_forward / "backtest" / "candles").glob("validation_*.csv"))
    paths.append(final_test / "backtest" / "candles" / "final_test.csv")
    frames = []
    for path in paths:
        frames.append(pd.read_csv(path))
    candles = pd.concat(frames).drop_duplicates("open_time_utc")
    candles["open_time_utc"] = pd.to_datetime(candles["open_time_utc"], utc=True)
    candles = candles.sort_values("open_time_utc").reset_index(drop=True)
    candles = candles[candles["open_time_utc"] >= pd.Timestamp("2021-01-01", tz="UTC")]
    return candles.reset_index(drop=True)


def server_to_utc(times: pd.Series) -> pd.Series:
    """Jam server broker yang tersimpan berlabel UTC -> UTC yang sebenarnya."""
    naive = times.dt.tz_localize(None)
    local = naive.dt.tz_localize(SERVER_TIMEZONE, ambiguous="NaT", nonexistent="shift_forward")
    return local.dt.tz_convert("UTC")


def add_columns(candles: pd.DataFrame) -> pd.DataFrame:
    data = candles.copy()
    step = data["open_time_utc"].diff()
    data["gap_before"] = step != pd.Timedelta(minutes=M5_MINUTES)
    log_return = np.log(data["close"] / data["close"].shift(1)) * BPS
    # Return yang melintasi gap (akhir pekan, libur) bukan return 5 menit.
    data["log_return_bps"] = log_return.where(~data["gap_before"])
    data["abs_return_bps"] = data["log_return_bps"].abs()
    data["range_usd"] = data["high"] - data["low"]
    data["atr_14"] = average_true_range(data["high"], data["low"], data["close"], ATR_PERIOD)
    data["spread_to_atr"] = data["spread"] / data["atr_14"]
    data["year"] = data["open_time_utc"].dt.year
    data["true_utc"] = server_to_utc(data["open_time_utc"])
    data["hour"] = data["true_utc"].dt.hour
    data["month"] = data["open_time_utc"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    return data


def read_labels(walk_forward: Path, final_test: Path) -> pd.DataFrame:
    paths = sorted((walk_forward / "backtest" / "xgboost").glob("validation_*/predictions.csv"))
    paths.append(final_test / "backtest" / "xgboost" / "final_test" / "predictions.csv")
    frames = []
    for path in paths:
        frame = pd.read_csv(path, usecols=["candle_open_time", "label_status", "actual_label"])
        frames.append(frame)
    labels = pd.concat(frames)
    labels["candle_open_time"] = pd.to_datetime(labels["candle_open_time"], utc=True)
    labels["year"] = labels["candle_open_time"].dt.year
    return labels


# ---------------------------------------------------------------------------
# Statistik
# ---------------------------------------------------------------------------

STAT_ROWS = [
    ("Log return M5 (bps)", "log_return_bps", 2),
    ("Absolute log return (bps)", "abs_return_bps", 2),
    ("Rentang high − low (USD)", "range_usd", 2),
    ("ATR14 (USD)", "atr_14", 2),
    ("Spread (USD)", "spread", 2),
    ("Spread ÷ ATR14 (%)", "spread_to_atr", 1),
    ("Tick volume", "tick_volume", 0),
]


def summary_statistics(data: pd.DataFrame) -> list[dict]:
    rows = []
    for label, column, digits in STAT_ROWS:
        values = data[column].dropna()
        scale = 100 if column == "spread_to_atr" else 1
        values = values * scale
        rows.append(
            {
                "label": label,
                "digits": digits,
                "count": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "median": float(values.median()),
                "p99": float(values.quantile(0.99)),
                "max": float(values.max()),
                "skew": float(values.skew()),
                "kurtosis": float(values.kurt()),
            }
        )
    return rows


def autocorrelation(series: pd.Series, lag: int) -> float:
    return float(series.autocorr(lag=lag))


def facts(data: pd.DataFrame) -> dict:
    no_gap = data.dropna(subset=["log_return_bps"])
    by_hour = data.groupby("hour").agg(
        abs_return=("abs_return_bps", "mean"),
        tick_volume=("tick_volume", "mean"),
        spread=("spread", "median"),
        spread_to_atr=("spread_to_atr", "median"),
    )
    zero_spread = data["spread"] == 0
    by_month = zero_spread.groupby(data["open_time_utc"].dt.strftime("%Y-%m")).mean()
    return {
        "zero_spread_share": float(zero_spread.mean()),
        "zero_spread_rows": int(zero_spread.sum()),
        "zero_spread_months_over_20pct": {k: float(v) for k, v in by_month[by_month > 0.2].items()},
        "candles": int(len(data)),
        "first": str(data["open_time_utc"].iloc[0]),
        "last": str(data["open_time_utc"].iloc[-1]),
        "gap_transitions": int(data["gap_before"].iloc[1:].sum()),
        "returns_without_gap": int(len(no_gap)),
        "zero_return_share": float((no_gap["log_return_bps"] == 0).mean()),
        "return_beyond_4_std_share": float(
            (no_gap["log_return_bps"].abs() > 4 * no_gap["log_return_bps"].std()).mean()
        ),
        "acf_return_lag1": autocorrelation(data["log_return_bps"], 1),
        "acf_return_lag12": autocorrelation(data["log_return_bps"], 12),
        "acf_abs_return_lag1": autocorrelation(data["abs_return_bps"], 1),
        "acf_abs_return_lag12": autocorrelation(data["abs_return_bps"], 12),
        "acf_abs_return_lag288": autocorrelation(data["abs_return_bps"], 288),
        "spearman_abs_return_tick_volume": float(
            no_gap["abs_return_bps"].corr(no_gap["tick_volume"], method="spearman")
        ),
        "spearman_abs_return_spread": float(
            no_gap["abs_return_bps"].corr(no_gap["spread"], method="spearman")
        ),
        "by_hour": by_hour.reset_index().to_dict(orient="list"),
    }


def yearly_table(data: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    available = labels[labels["label_status"] == "AVAILABLE"]
    shares = available.groupby("year")["actual_label"].value_counts(normalize=True).unstack()
    grouped = data.groupby("year").agg(
        candles=("close", "size"),
        close=("close", "median"),
        atr=("atr_14", "median"),
        spread=("spread", "median"),
        spread_to_atr=("spread_to_atr", "median"),
        abs_return=("abs_return_bps", "mean"),
    )
    grouped["labelled"] = available.groupby("year").size()
    return grouped.join(shares)


# ---------------------------------------------------------------------------
# Gambar
# ---------------------------------------------------------------------------


def style_axis(axis) -> None:
    axis.set_facecolor("white")
    for side in ["top", "right", "left"]:
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(BASELINE)
    axis.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    axis.yaxis.grid(True, color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)


def hourly_figure(data: pd.DataFrame, path: Path) -> None:
    by_hour = data.groupby("hour").agg(
        abs_return=("abs_return_bps", "mean"),
        tick_volume=("tick_volume", "mean"),
        spread_to_atr=("spread_to_atr", "median"),
    )
    panels = [
        ("abs_return", "Rata-rata |log return| M5 (bps)", 1),
        ("tick_volume", "Rata-rata tick volume per candle", 1),
        ("spread_to_atr", "Median spread ÷ ATR14 (%)", 100),
    ]
    figure, axes = plt.subplots(3, 1, figsize=(6.3, 5.0), sharex=True)
    for axis, (column, title, scale) in zip(axes, panels, strict=True):
        style_axis(axis)
        values = by_hour[column] * scale
        axis.bar(by_hour.index, values, width=0.78, color=BAR, linewidth=0)
        axis.set_title(title, loc="left", fontsize=9, color=INK, pad=4)
        for hour, label in [(7, "London"), (13, "New York"), (21, "Tutup harian")]:
            axis.axvline(hour - 0.5, color=INK_MUTED, linewidth=0.6, linestyle=(0, (2, 2)))
            if column == "abs_return":
                axis.text(
                    hour - 0.35,
                    values.max() * 1.16,
                    label,
                    fontsize=7,
                    color=INK_SECONDARY,
                    va="top",
                )
        if column == "abs_return":
            axis.set_ylim(0, values.max() * 1.18)
    axes[-1].set_xticks(range(0, 24, 2))
    axes[-1].set_xlabel(
        "Jam buka candle (UTC setelah koreksi zona waktu server; WIB = UTC + 7)",
        fontsize=8,
        color=INK_SECONDARY,
    )
    figure.tight_layout(h_pad=1.2)
    figure.savefig(path, dpi=DPI, facecolor="white")
    plt.close(figure)


def label_figure(table: pd.DataFrame, path: Path) -> None:
    years = [str(year) if year != 2026 else "2026*" for year in table.index]
    figure, axis = plt.subplots(figsize=(6.3, 2.9))
    style_axis(axis)
    axis.yaxis.grid(False)
    axis.xaxis.grid(True, color=GRID, linewidth=0.6)
    left = np.zeros(len(table))
    positions = np.arange(len(table))[::-1]
    for label in ["SELL", "NO_TRADE", "BUY"]:
        values = table[label].to_numpy() * 100
        axis.barh(
            positions,
            values,
            left=left,
            height=0.62,
            color=SERIES[label],
            label=label,
            edgecolor="white",
            linewidth=1.0,
        )
        for position, start, value in zip(positions, left, values, strict=True):
            text_color = "#ffffff"
            axis.text(
                start + value / 2,
                position,
                f"{value:.1f}".replace(".", ","),
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color,
            )
        left += values
    axis.set_yticks(positions, years)
    axis.set_xlim(0, 100)
    axis.set_xlabel("Proporsi label aktual (%)", fontsize=8, color=INK_SECONDARY)
    axis.legend(
        ncol=3,
        frameon=False,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        labelcolor=INK_SECONDARY,
        handlelength=1.2,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=DPI, facecolor="white")
    plt.close(figure)


def monthly_figure(data: pd.DataFrame, path: Path) -> None:
    monthly = data.groupby("month").agg(
        atr=("atr_14", "median"),
        spread=("spread", "median"),
        spread_to_atr=("spread_to_atr", "median"),
    )
    panels = [
        ("atr", "Median ATR14 per bulan (USD)", 1),
        ("spread", "Median spread per bulan (USD)", 1),
        ("spread_to_atr", "Median spread ÷ ATR14 per bulan (%)", 100),
    ]
    figure, axes = plt.subplots(3, 1, figsize=(6.3, 4.8), sharex=True)
    for axis, (column, title, scale) in zip(axes, panels, strict=True):
        style_axis(axis)
        axis.plot(monthly.index, monthly[column] * scale, color=BAR, linewidth=1.6)
        axis.set_title(title, loc="left", fontsize=9, color=INK, pad=4)
        axis.set_ylim(bottom=0)
    figure.tight_layout(h_pad=1.2)
    figure.savefig(path, dpi=DPI, facecolor="white")
    plt.close(figure)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def markdown(stats, yearly, found, sources) -> list[str]:
    lines = [
        f"Sumber: candle `{sources['walk_forward']}/backtest/candles/` (2021–2025) dan "
        f"`{sources['final_test']}/backtest/candles/` (2026, sampai September); label aktual dari "
        "predictions.csv run yang sama. Dihitung oleh `laporan_uts/eda_m5.py`.",
        "",
        "Tabel 4.1 Ringkasan Statistik Candle M5 XAU/USD 2021–2026",
        "<!-- widths: 20,11,9,8,8,9,8,10,8,9 -->",
        "| Variabel | n | Rata-rata | Simp. baku | Min | Median | P99 | Maks | Skew. | Kurt. |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in stats:
        digits = row["digits"]
        lines.append(
            f"| {row['label']} | {integer(row['count'])} | {number(row['mean'], digits)} | "
            f"{number(row['std'], digits)} | {number(row['min'], digits)} | {number(row['median'], digits)} | "
            f"{number(row['p99'], digits)} | {number(row['max'], digits)} | {number(row['skew'], 2)} | "
            f"{number(row['kurtosis'], 1)} |"
        )
    lines += [
        "",
        "Return dihitung hanya antara dua candle yang berjarak tepat 5 menit, sehingga lompatan "
        "harga melewati akhir pekan atau libur tidak dihitung sebagai return M5. Skew. = skewness; Kurt. = excess "
        "kurtosis (distribusi normal = 0).",
        "",
        "Tabel 4.2 Karakteristik Candle dan Distribusi Label per Tahun",
        "<!-- widths: 10,11,9,9,9,10,10,9,13,10 -->",
        "| Tahun | Candle | Median close (USD) | Median ATR14 (USD) | Median spread (USD) | Median spread ÷ ATR14 | Rata-rata abs. return (bps) | SELL | NO_TRADE | BUY |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for year, row in yearly.iterrows():
        label = f"{year}*" if year == 2026 else str(year)
        lines.append(
            f"| {label} | {integer(row['candles'])} | {number(row['close'], 0)} | {number(row['atr'], 2)} | "
            f"{number(row['spread'], 2)} | {percent(row['spread_to_atr'])} | {number(row['abs_return'], 2)} | "
            f"{percent(row['SELL'])} | {percent(row['NO_TRADE'])} | {percent(row['BUY'])} |"
        )
    lines += [
        "",
        "Tanda * pada 2026: data hanya Januari–September. Proporsi label dihitung dari candle berstatus AVAILABLE "
        "(horizon 6 candle lengkap dan tidak melintasi gap).",
        "",
        "Tabel 4.3 Ukuran Ketergantungan pada Candle M5 2021–2026",
        "<!-- widths: 60,40 -->",
        "| Ukuran | Nilai |",
        "|---|---|",
        f"| Autokorelasi log return, lag 1 / lag 12 candle | {number(found['acf_return_lag1'], 3)} / {number(found['acf_return_lag12'], 3)} |",
        f"| Autokorelasi absolute return, lag 1 / 12 / 288 candle | {number(found['acf_abs_return_lag1'], 3)} / {number(found['acf_abs_return_lag12'], 3)} / {number(found['acf_abs_return_lag288'], 3)} |",
        f"| Korelasi Spearman absolute return dengan tick volume | {number(found['spearman_abs_return_tick_volume'], 3)} |",
        f"| Korelasi Spearman absolute return dengan spread | {number(found['spearman_abs_return_spread'], 3)} |",
        f"| Proporsi return M5 bernilai nol | {percent(found['zero_return_share'], 2)} |",
        f"| Proporsi return lebih dari 4 simpangan baku | {percent(found['return_beyond_4_std_share'], 2)} (normal: 0,01%) |",
        f"| Candle dengan spread tercatat 0 | {percent(found['zero_spread_share'], 2)} ({integer(found['zero_spread_rows'])} candle) |",
        "",
    ]
    return lines


def time_zone_evidence(data: pd.DataFrame) -> list[str]:
    """Bar pertama setiap minggu: jam server tersimpan vs UTC setelah konversi."""
    week_open = data[data["open_time_utc"].diff() > pd.Timedelta(hours=24)]
    stored = week_open["open_time_utc"].dt.strftime("%a %H").value_counts()
    converted = week_open["true_utc"].dt.strftime("%a %H").value_counts()
    week_close = data["open_time_utc"].shift(1)[
        data["open_time_utc"].diff() > pd.Timedelta(hours=24)
    ]
    closed = week_close.dt.strftime("%a %H").value_counts()
    days = {"Mon": "Senin", "Sun": "Minggu", "Fri": "Jumat", "Tue": "Selasa"}

    def top(counts, n=2):
        parts = []
        for key, value in counts.head(n).items():
            day, clock = key.split()
            parts.append(f"{days.get(day, day)} pukul {clock}.xx ({integer(value)} minggu)")
        return "; ".join(parts)

    total = len(week_open)
    return [
        "Tabel 3.4 Bukti Zona Waktu Timestamp: Bar Pertama dan Terakhir Setiap Minggu 2021–2026",
        "<!-- widths: 38,62 -->",
        "| Ukuran | Hasil |",
        "|---|---|",
        f"| Jumlah minggu perdagangan | {integer(total)} |",
        f"| Bar pertama minggu menurut kolom open_time_utc | {top(stored)} |",
        f"| Bar terakhir minggu menurut kolom open_time_utc | {top(closed)} |",
        f"| Bar pertama minggu setelah dikonversi dari EET/EEST ke UTC | {top(converted)} |",
        "| Jadwal pasar emas sebenarnya | Buka Minggu 22:00 UTC (saat DST AS) atau 23:00 UTC; tutup Jumat 21:00/22:00 UTC |",
        "",
    ]


def normalization_table(data: pd.DataFrame) -> list[str]:
    """Dampak normalisasi ATR: perubahan harga 6 candle dalam USD vs dibagi ATR14 (fitur return_6_atr)."""
    change = data["close"] - data["close"].shift(6)
    valid = ~data["gap_before"].rolling(6).max().fillna(1).astype(bool)
    frame = pd.DataFrame(
        {
            "year": data["year"],
            "usd": change.where(valid),
            "atr": (change / data["atr_14"]).where(valid),
        }
    ).dropna()
    grouped = frame.groupby("year").agg(usd=("usd", "std"), atr=("atr", "std"))
    ratio_usd = grouped["usd"].max() / grouped["usd"].min()
    ratio_atr = grouped["atr"].max() / grouped["atr"].min()
    lines = [
        "Tabel 6.2 Dampak Normalisasi ATR: Simpangan Baku Perubahan Harga 6 Candle per Tahun",
        "<!-- widths: 20,40,40 -->",
        "| Tahun | Perubahan close 6 candle (USD) | Perubahan close 6 candle ÷ ATR14 (fitur return_6_atr) |",
        "|---|---|---|",
    ]
    for year, row in grouped.iterrows():
        label = f"{year}*" if year == 2026 else str(year)
        lines.append(f"| {label} | {number(row['usd'], 2)} | {number(row['atr'], 2)} |")
    lines += [
        f"| Rasio tahun terbesar ÷ terkecil | {number(ratio_usd, 1)} × | {number(ratio_atr, 2)} × |",
        "",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk-forward", required=True)
    parser.add_argument("--final-test", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    walk_forward = Path(arguments.walk_forward)
    final_test = Path(arguments.final_test)
    out = Path(arguments.out)
    (out / "gambar").mkdir(exist_ok=True)

    data = add_columns(read_candles(walk_forward, final_test))
    labels = read_labels(walk_forward, final_test)
    stats = summary_statistics(data)
    found = facts(data)
    yearly = yearly_table(data, labels)

    hourly_figure(data, out / "gambar" / "eda_m5_profil_jam.png")
    label_figure(yearly, out / "gambar" / "eda_m5_label_tahun.png")
    monthly_figure(data, out / "gambar" / "eda_m5_atr_spread_bulanan.png")

    sources = {"walk_forward": walk_forward.as_posix(), "final_test": final_test.as_posix()}
    lines = markdown(stats, yearly, found, sources)
    (out / "hasil_eda_m5.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "hasil_fe_atr.md").write_text("\n".join(normalization_table(data)), encoding="utf-8")
    zone_lines = time_zone_evidence(data)
    (out / "hasil_zona_waktu.md").write_text("\n".join(zone_lines), encoding="utf-8")
    print("\n".join(zone_lines))
    report = {
        "facts": found,
        "statistics": stats,
        "yearly": yearly.reset_index().to_dict(orient="list"),
    }
    (out / "eda_m5.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
