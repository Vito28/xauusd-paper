"""Manage broker TOML profiles and runtime settings in notebook-style sections.

Flow: configs/brokers/<name>.toml + GOLD_* environment -> validated configuration
-> explicit collector commands. This module reads and writes broker configuration
only; market-data acquisition and reports live in market.collector. Loading a
module does not read settings, initialize MT5, or connect to PostgreSQL.
"""

from __future__ import annotations

# %% 01. Import dan dependensi konfigurasi
import json
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url

from market.contracts import Environment

# %% 02. Profile broker: identitas, timeframe, policy, dan loader TOML
# PROFILE-01 | Validasi identitas source dan kebijakan collection dari configs/brokers.
# Password/login tidak menjadi bagian profile; autentikasi memakai terminal yang sudah login.

class SourceProfile(BaseModel):
    """Stable source identity and expected account attributes."""

    source_key: str = Field(min_length=3)
    broker_name: str = Field(min_length=1)
    server_name: str = Field(min_length=1)
    environment: Environment
    dataset_symbol: str = "XAUUSD"
    requested_symbol: str = Field(min_length=1)


class MarketProfile(BaseModel):
    """Market and timeframe selection for one broker source."""

    base_timeframe: str = "M5"
    additional_native_timeframes: list[str] = Field(default_factory=list)
    timezone: str = "UTC"

    @model_validator(mode="after")
    def require_utc(self) -> MarketProfile:
        """Enforce UTC as the collector's canonical timezone."""

        if self.timezone.upper() != "UTC":
            raise ValueError("broker profile timezone must be UTC")
        return self


class CollectionProfile(BaseModel):
    """Chunking, retry, and candle-finalization behavior."""

    bar_chunk: str = "1month"
    tick_chunk: str = "1day"
    max_attempts: int = Field(default=3, ge=1, le=10)
    retry_delays_seconds: list[float] = Field(default_factory=lambda: [1, 2, 4])
    finalization_delay_seconds: int = Field(default=5, ge=0, le=300)

    @model_validator(mode="after")
    def validate_policy(self) -> CollectionProfile:
        """Require the deterministic partition policy mandated by the platform."""

        if self.bar_chunk != "1month" or self.tick_chunk != "1day":
            raise ValueError("bars must use 1month and ticks must use 1day chunks")
        if len(self.retry_delays_seconds) < self.max_attempts - 1:
            raise ValueError("retry_delays_seconds does not cover max_attempts")
        return self


class BrokerProfile(BaseModel):
    """Complete validated non-secret configuration for one data source."""

    source: SourceProfile
    market: MarketProfile
    collection: CollectionProfile

    def validate_environment(self, environment: Environment) -> None:
        """Fail fast when a profile targets another physical environment."""

        if self.source.environment is not environment:
            raise ValueError(
                f"profile is {self.source.environment.value}, runtime is {environment.value}"
            )


def load_broker_profile(name: str, root: Path = Path("configs/brokers")) -> BrokerProfile:
    """Load a named broker profile without allowing directory traversal.

    Args:
        name: Profile stem, such as ``example_demo``.
        root: Directory containing broker TOML files.

    Returns:
        Parsed and validated broker profile.

    Raises:
        ValueError: If the profile name contains path components.
        FileNotFoundError: If the configured profile does not exist.
    """

    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("profile must be a simple file stem")
    path = root / f"{name}.toml"
    with path.open("rb") as handle:
        return BrokerProfile.model_validate(tomllib.load(handle))


# %% 03. Runtime settings: environment, database, terminal, dan dashboard
# PROFILE-02 | Baca GOLD_* hanya saat load_settings() dipanggil oleh command.
# Database demo/live diperiksa agar konfigurasi lintas environment ditolak.

class Settings(BaseSettings):
    """Runtime settings loaded from ``GOLD_*`` environment variables.

    ``environment`` and ``database_url`` intentionally have no defaults so a
    collector process cannot silently fall back to a live account or database.
    Dashboard URLs are separate, read-only connection targets selected only
    from the ``demo`` and ``live`` environment whitelist.
    """

    model_config = SettingsConfigDict(
        env_prefix="GOLD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Environment
    database_url: str
    dashboard_demo_database_url: str | None = None
    dashboard_live_database_url: str | None = None
    mt5_terminal_path: Path | None = None
    log_level: str = "INFO"
    collector_version: str = "0.1.0"
    safety_lag_seconds: int = Field(default=5, ge=0, le=300)
    store_valid_tick_raw_payload: bool = False

    @field_validator("mt5_terminal_path", mode="before")
    @classmethod
    def empty_terminal_path_is_none(cls, value: object) -> object:
        """Treat an empty optional environment variable as no explicit path."""

        return None if value == "" else value

    @field_validator("store_valid_tick_raw_payload")
    @classmethod
    def valid_tick_raw_payload_must_remain_disabled(cls, value: bool) -> bool:
        """Allow environment parsing of ``false`` while rejecting raw tick payloads."""

        if value:
            raise ValueError("valid tick raw payload storage must remain disabled")
        return value

    @model_validator(mode="after")
    def validate_database_environment(self) -> Settings:
        """Reject obvious cross-environment collector and dashboard DSN mistakes."""

        self._validate_database_for_environment(self.environment, self.database_url)
        for environment, database_url in (
            (Environment.DEMO, self.dashboard_demo_database_url),
            (Environment.LIVE, self.dashboard_live_database_url),
        ):
            if database_url:
                self._validate_database_for_environment(environment, database_url)
        return self

    @staticmethod
    def _validate_database_for_environment(environment: Environment, database_url: str) -> None:
        """Reject a URL whose database name visibly belongs to the opposite environment."""

        database = (make_url(database_url).database or "").lower()
        opposite = "live" if environment is Environment.DEMO else "demo"
        if opposite in database:
            raise ValueError(f"{environment.value} environment cannot use database {database!r}")

    def dashboard_database_url(self, environment: Environment) -> str:
        """Return the configured read-only dashboard database for one environment.

        Args:
            environment: The explicit, server-validated ``demo`` or ``live`` choice.

        Returns:
            SQLAlchemy PostgreSQL URL for the selected dashboard database.

        Raises:
            ValueError: If the selected dashboard URL is absent from configuration.
        """

        database_url = (
            self.dashboard_demo_database_url
            if environment is Environment.DEMO
            else self.dashboard_live_database_url
        )
        if not database_url:
            variable = f"GOLD_DASHBOARD_{environment.value.upper()}_DATABASE_URL"
            raise ValueError(f"{variable} is required to browse {environment.value} data")
        return database_url


def load_settings() -> Settings:
    """Load and validate runtime settings for an explicit command invocation."""

    return Settings()  # type: ignore[call-arg]


# %% 04. Menulis profile broker ke configs/brokers
# PROFILE-03 | Serialisasi konfigurasi non-secret menjadi TOML setelah validasi.
# Pemanggil memilih nama profile; file yang sudah ada dilindungi kecuali overwrite=True.
# Contoh: save_broker_profile("broker_baru", profile) -> configs/brokers/broker_baru.toml.

def save_broker_profile(
    name: str,
    profile: BrokerProfile,
    root: Path = Path("configs/brokers"),
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and save a non-secret broker profile as UTF-8 TOML.

    Args:
        name: Simple profile stem, using the same naming contract as the loader.
        profile: Source identity, market selection and collection policy to save.
        root: Directory for broker configuration; defaults to configs/brokers.
        overwrite: Whether an existing profile may be replaced; defaults to false.

    Returns:
        Path of the saved TOML configuration, readable by load_broker_profile().

    Raises:
        ValueError: For a path-like name, invalid profile or non-finite TOML numbers.
        FileExistsError: When the profile exists and overwrite is false.

    Side effects:
        Creates the configuration directory if needed and writes one TOML file.
        Does not load runtime credentials, initialize MT5, or access PostgreSQL.
    """
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("profile must be a simple file stem")
    # Revalidate serialized fields, including profiles built/copied without validation.
    validated = BrokerProfile.model_validate(profile.model_dump(mode="json"))
    lines: list[str] = []
    for section, values in validated.model_dump(mode="json").items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            # Current fields are strings, numbers and flat lists, also valid as TOML literals.
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
            lines.append(f"{key} = {encoded}")
        lines.append("")
    content = "\n".join(lines)
    # Check that the emitted document round-trips before opening the destination.
    BrokerProfile.model_validate(tomllib.loads(content))
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.toml"
    with path.open("w" if overwrite else "x", encoding="utf-8") as handle:
        handle.write(content)
    return path
