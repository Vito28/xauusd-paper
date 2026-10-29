"""Tests for source identity, symbol resolution, and MT5 preflight safety."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import pytest

from market.collector import MT5PreflightError, MT5Provider, SymbolResolutionError, resolve_symbol
from market.contracts import Environment
from market.profile import Settings, load_broker_profile


class FakeMT5Api:
    """Small namedtuple-based MT5 API fake used only for preflight tests."""

    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_TRADE_MODE_CONTEST = 1
    ACCOUNT_TRADE_MODE_REAL = 2

    def __init__(
        self, *, mode: int = 0, server: str = "Example-Demo", company: str = "Example Broker"
    ) -> None:
        """Configure fake account values without any real terminal access."""

        self.mode = mode
        self.server = server
        self.company = company

    def terminal_info(self) -> object:
        """Return connected fake terminal metadata."""

        info = namedtuple("Terminal", "connected path company maxbars")
        return info(True, str(Path("terminal64.exe")), self.company, 100_000)

    def account_info(self) -> object:
        """Return fake account metadata; no login is present."""

        info = namedtuple("Account", "server company trade_mode currency")
        return info(self.server, self.company, self.mode, "USD")


def provider_with(api: FakeMT5Api) -> MT5Provider:
    """Inject a fake initialized API into the concrete provider."""

    provider = MT5Provider()
    provider._mt5 = api  # type: ignore[assignment]
    provider._initialized = True
    return provider


def test_exact_symbol_is_required() -> None:
    """Related broker symbols are candidates, never automatic substitutions."""

    assert resolve_symbol("XAUUSD", ["XAUUSD", "XAUUSDm"])[0] == "XAUUSD"
    with pytest.raises(SymbolResolutionError, match="candidates"):
        resolve_symbol("XAUUSD", ["XAUUSDm", "XAUUSD.a"])


def test_profile_environment_mismatch_is_rejected() -> None:
    """A live runtime cannot use a demo broker profile."""

    profile = load_broker_profile("example_demo")
    with pytest.raises(ValueError, match="profile is demo"):
        profile.validate_environment(Environment.LIVE)


def test_finex_live_profile_has_a_stable_live_source_identity() -> None:
    """Finex live collection uses a distinct source key and live account policy."""

    profile = load_broker_profile("finex_live")

    assert profile.source.source_key == "finex-bisnis-solusi-real-xauusd"
    assert profile.source.environment is Environment.LIVE
    assert profile.source.server_name == "FinexBisnisSolusi-Real"
    assert profile.source.requested_symbol == "XAUUSD"


def test_empty_optional_terminal_path_is_none() -> None:
    """An empty env path remains optional instead of becoming the current directory."""

    settings = Settings(
        environment="demo",
        database_url="postgresql+psycopg://user:pass@localhost/gold_market_demo",
        mt5_terminal_path="",
    )
    assert settings.mt5_terminal_path is None


def test_dashboard_database_urls_are_selected_by_environment() -> None:
    """The browser can select only the separately configured demo/live database URLs."""

    settings = Settings(
        environment="demo",
        database_url="postgresql+psycopg://user:pass@localhost/gold_market_demo",
        dashboard_demo_database_url="postgresql+psycopg://user:pass@localhost/gold_market_demo",
        dashboard_live_database_url="postgresql+psycopg://user:pass@localhost/gold_market_live",
    )

    assert settings.dashboard_database_url(Environment.DEMO).endswith("gold_market_demo")
    assert settings.dashboard_database_url(Environment.LIVE).endswith("gold_market_live")


def test_valid_tick_raw_payload_storage_cannot_be_enabled() -> None:
    """The parsed boolean setting remains permanently false by policy."""

    with pytest.raises(ValueError, match="must remain disabled"):
        Settings(
            environment="demo",
            database_url="postgresql+psycopg://user:pass@localhost/gold_market_demo",
            store_valid_tick_raw_payload=True,
        )


@pytest.mark.parametrize("environment,mode", [(Environment.DEMO, 2), (Environment.LIVE, 0)])
def test_account_environment_mismatch_is_rejected(environment: Environment, mode: int) -> None:
    """Both demo-to-live and live-to-demo account mismatches fail preflight."""

    profile_name = "example_demo" if environment is Environment.DEMO else "example_live"
    with pytest.raises(MT5PreflightError, match="account is"):
        provider_with(FakeMT5Api(mode=mode)).preflight(
            load_broker_profile(profile_name), environment
        )


def test_broker_and_server_mismatch_are_rejected() -> None:
    """Active terminal company and server must match non-secret profile values."""

    profile = load_broker_profile("example_demo")
    with pytest.raises(MT5PreflightError, match="server mismatch"):
        provider_with(FakeMT5Api(server="Wrong-Demo")).preflight(profile, Environment.DEMO)
    with pytest.raises(MT5PreflightError, match="broker mismatch"):
        provider_with(FakeMT5Api(company="Wrong Broker")).preflight(profile, Environment.DEMO)
