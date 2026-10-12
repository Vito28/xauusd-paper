"""Embed Plotly, frozen predictions and audits into one offline broker-style HTML file."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from plotly.offline import get_plotlyjs

from technical.config import ROOT


def write_dashboard(
    predictions: pd.DataFrame,
    candles: pd.DataFrame,
    metrics: dict[str, Any],
    folds: list[dict[str, Any]],
    feature_audits: list[dict[str, Any]],
    output: Path,
    *,
    rolling_window: int = 288,
) -> Path:
    """Build an offline dashboard with windowed candle/line charts and paginated filters.

    Full M5 predictions remain embedded. Charts render a bounded time window for
    responsiveness; previous/next/date/slider navigation can inspect all rows.
    Signal markers use decision timestamps and observed close; forecasts use
    target timestamps. Selected signals connect to their actual fixed-horizon
    outcome. Rolling directional precision excludes abstentions. The ledger shows all rows when no
    trade passed validation, so rejection reasons remain inspectable.
    """
    from technical.xgboost_model import attach_direction_outcomes, evaluate

    predictions = attach_direction_outcomes(predictions)
    metrics = {**metrics, **evaluate(predictions)}
    rolling = (
        predictions.prediction_correct.astype("Float64")
        .where(predictions.predicted_signal.ne("NO_TRADE"))
        .rolling(rolling_window, min_periods=1)
        .mean()
    )
    net = predictions.realized_net_move.fillna(0).rolling(rolling_window, min_periods=1).sum()
    data = predictions.copy()
    data["rolling_trade_precision"] = rolling
    data["rolling_net_move"] = net
    payload = {
        "predictions": json.loads(data.to_json(orient="records", date_format="iso")),
        "candles": json.loads(
            candles[["prediction_timestamp", "open", "high", "low", "close"]].to_json(
                orient="records", date_format="iso"
            )
        ),
        "metrics": metrics,
        "folds": folds,
        "featureAudits": feature_audits,
        "rollingWindow": rolling_window,
    }
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")
    html = template.replace("/*__PLOTLY__*/", get_plotlyjs()).replace(
        "/*__PAYLOAD__*/null", encoded
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def main() -> None:
    """Rebuild a dashboard from saved fold artifacts without querying or retraining.

    Useful after template edits or to inspect an earlier annual TEST. Root must
    contain walk_forward_metrics.json and feature_audits.json from the pipeline.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts" / "direction_v3")
    parser.add_argument("--fold", default="final_2026")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.fold not in {"final_2026", *(f"fold_{y}" for y in range(2020, 2026))}:
        parser.error("Fold tidak valid")
    folder = arguments.root / arguments.fold
    predictions = pd.read_csv(folder / "predictions.csv")
    for name in ("prediction_timestamp", "target_timestamp", "open_time_utc"):
        predictions[name] = pd.to_datetime(predictions[name], utc=True)
    predictions["prediction_correct"] = predictions.prediction_correct.astype("boolean")
    summary = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    metrics_path = folder / "dashboard_metrics.json"
    quality_path = arguments.root / "data_quality.json"
    if not quality_path.exists():
        quality_path = arguments.root.parent / "data_quality.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists()
        else {
            **summary["metrics"],
            "metadata": {"fold": arguments.fold},
            "data_quality": json.loads(quality_path.read_text(encoding="utf-8")),
        }
    )
    replay_path = folder / "execution_replay.json"
    if replay_path.exists():
        metrics["execution_replay"] = json.loads(replay_path.read_text(encoding="utf-8"))
    folds = json.loads((arguments.root / "walk_forward_metrics.json").read_text(encoding="utf-8"))
    audits = json.loads((arguments.root / "feature_audits.json").read_text(encoding="utf-8"))
    path = arguments.output or folder / "dashboard_2026.html"
    window = metrics.get("metadata", {}).get("config", {}).get("rolling_window", 288)
    write_dashboard(predictions, predictions, metrics, folds, audits, path, rolling_window=window)
    print(path)


if __name__ == "__main__":
    main()
