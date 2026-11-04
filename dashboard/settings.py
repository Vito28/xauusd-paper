"""Pengaturan dashboard dari environment variable GOLD_* atau file .env."""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from technical.config import PROJECT_ROOT


class DashboardSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GOLD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Simbol broker yang dipantau di halaman live (contoh: XAUUSD, XAUUSDm, GOLD).
    live_symbol: str = "XAUUSD"
    # False untuk mematikan polling MT5 (misalnya di Docker/Linux atau saat tes).
    live_enabled: bool = True
    mt5_terminal_path: Path | None = None
    # Folder hasil eksperimen technical.run (backtest/ dan live_bundle.joblib).
    runs_dir: Path = PROJECT_ROOT / "runs"

    @field_validator("mt5_terminal_path", mode="before")
    @classmethod
    def empty_path_is_none(cls, value: object) -> object:
        if value == "":
            return None
        return value


def load_dashboard_settings() -> DashboardSettings:
    return DashboardSettings()
