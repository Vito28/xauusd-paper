"""Regression coverage for the consolidated collector and configuration workflow."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from market.collector import (
    IngestionService,
    MT5PreflightError,
    coverage_report,
    run_status,
    validation_report,
)
from market.contracts import TimePartition
from market.profile import BrokerProfile, load_broker_profile, save_broker_profile
from tests.market.test_validation import valid_bar_payload


def test_profile_save_roundtrips_and_requires_explicit_overwrite(tmp_path: Path) -> None:
    """TOML preserves Unicode/escaping and cannot replace an existing profile implicitly."""
    root = tmp_path / "brokers"
    fields = load_broker_profile("example_demo").model_dump()
    fields["source"]["broker_name"] = 'Broker "Baru" — Indonesia\nDemo'
    profile = BrokerProfile.model_validate(fields)
    path = save_broker_profile("new_demo", profile, root)
    assert path == root / "new_demo.toml"
    assert load_broker_profile("new_demo", root) == profile
    contents = path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        save_broker_profile("new_demo", load_broker_profile("example_demo"), root)
    assert path.read_text(encoding="utf-8") == contents
    replacement = load_broker_profile("example_demo")
    save_broker_profile("new_demo", replacement, root, overwrite=True)
    assert load_broker_profile("new_demo", root) == replacement


@pytest.mark.parametrize("name", ["../escape", "nested/escape", "", "."])
def test_profile_save_rejects_path_names_before_writing(tmp_path: Path, name: str) -> None:
    """Invalid profile names do not create a directory or write a file."""
    root = tmp_path / "brokers"
    with pytest.raises(ValueError, match="simple file stem"):
        save_broker_profile(name, load_broker_profile("example_demo"), root)
    assert not root.exists()


@pytest.mark.parametrize("first", ["market.collector", "database.storage"])
def test_import_orders_do_not_load_settings_or_activate_connections(first: str) -> None:
    """Fresh collector/repository imports have no runtime I/O or circular dependencies."""
    code = """
import sys
from unittest.mock import patch
import pydantic_settings
import sqlalchemy
with patch.object(
    pydantic_settings.BaseSettings, '__init__', side_effect=AssertionError('settings loaded')
):
    with patch.object(sqlalchemy, 'create_engine', side_effect=AssertionError('engine created')):
        __import__(sys.argv[1])
        import market.collector
        import market.profile
        import database.storage
        import market.dashboard
        assert 'MetaTrader5' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, first],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_database_reports_keep_source_filters_bound() -> None:
    """Coverage, status and validation reports keep user values out of SQL text."""
    session = MagicMock()
    source_key = "demo'; DROP TABLE market.bar_observation; --"
    for report in (coverage_report, run_status, validation_report):
        report(session, source_key)
        statement, params = session.execute.call_args.args
        assert ":source_key" in str(statement)
        assert source_key not in str(statement)
        assert params["source_key"] == source_key
        assert not any(word in str(statement).upper() for word in ("INSERT ", "UPDATE ", "DELETE "))


class RecordingDatabase:
    """Track transaction completion around mocked repositories without a real database."""

    def __init__(self) -> None:
        """Create the shared session and transaction outcome log."""
        self.active_session = MagicMock()
        self.outcomes: list[str] = []

    @contextmanager
    def transaction(self) -> Iterator[MagicMock]:
        """Record commit/rollback according to whether the caller raises."""
        try:
            yield self.active_session
        except Exception:
            self.outcomes.append("rollback")
            raise
        else:
            self.outcomes.append("commit")


def collector_fixture() -> tuple[IngestionService, RecordingDatabase, MagicMock, TimePartition]:
    """Construct an isolated collector with real normalization and mocked persistence."""
    database = RecordingDatabase()
    provider = MagicMock()
    service = IngestionService(
        database=database,  # type: ignore[arg-type]
        provider=provider,
        source_id=uuid4(),
        source_key="fixture-demo-xauusd",
        actual_symbol="XAUUSD",
        symbol_spec_version_id=None,
        point=Decimal("0.01"),
        collector_version="test",
        max_attempts=3,
        retry_delays_seconds=[1, 2],
        finalization_delay_seconds=0,
        sleeper=MagicMock(),
    )
    service.ingestion_repository = MagicMock()
    service.market_repository = MagicMock()
    service.rejection_repository = MagicMock()
    service.market_repository.insert_bars.return_value = 1
    service.rejection_repository.insert_many.return_value = 1
    partition = TimePartition(datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 2, 1, tzinfo=UTC))
    return service, database, provider, partition


def test_ingestion_routes_accepted_and_rejected_rows_in_one_transaction() -> None:
    """Valid bars and invalid OHLC enter separate repositories with one shared checkpoint."""
    service, database, provider, partition = collector_fixture()
    provider.fetch_bars.return_value = [valid_bar_payload(), {**valid_bar_payload(), "high": 1}]
    counts = service._collect_partition(
        run_id=uuid4(), kind="bars", timeframe="M5", partition=partition
    )
    assert (counts.received, counts.inserted, counts.rejected) == (2, 1, 1)
    bars = service.market_repository.insert_bars.call_args.kwargs["bars"]
    rejected = service.rejection_repository.insert_many.call_args.kwargs["records"]
    assert bars[0].close == Decimal("2605.0")
    assert rejected[0].reason_code == "INVALID_OHLC"
    assert database.outcomes == ["commit"]
    for operation in (
        service.market_repository.insert_bars,
        service.rejection_repository.insert_many,
        service.ingestion_repository.save_checkpoint,
    ):
        assert operation.call_args.args[0] is database.active_session
    assert service.ingestion_repository.save_checkpoint.call_args.kwargs["status"] == "fetched"


def test_ingestion_retries_after_rollback_without_a_premature_checkpoint() -> None:
    """Transient storage failure rolls back and only the successful retry saves a checkpoint."""
    service, database, provider, partition = collector_fixture()
    provider.fetch_bars.return_value = [valid_bar_payload()]
    service.market_repository.insert_bars.side_effect = [RuntimeError("temporary storage error"), 1]
    service.rejection_repository.insert_many.return_value = 0
    counts = service._collect_partition(
        run_id=uuid4(), kind="bars", timeframe="M5", partition=partition
    )
    assert counts.inserted == 1
    assert database.outcomes == ["rollback", "commit"]
    service.sleeper.assert_called_once_with(1)
    assert provider.fetch_bars.call_count == 2
    service.ingestion_repository.save_checkpoint.assert_called_once()


def test_ingestion_preflight_failure_does_not_retry_or_store() -> None:
    """Permanent MT5 identity mismatch never sleeps or reaches persistence."""
    service, database, provider, partition = collector_fixture()
    provider.fetch_bars.side_effect = MT5PreflightError("account mismatch")
    with pytest.raises(MT5PreflightError):
        service._collect_partition(run_id=uuid4(), kind="bars", timeframe="M5", partition=partition)
    assert database.outcomes == []
    provider.fetch_bars.assert_called_once()
    service.sleeper.assert_not_called()
    service.ingestion_repository.save_checkpoint.assert_not_called()
