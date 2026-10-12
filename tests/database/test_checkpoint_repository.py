"""Unit tests for checkpoint-based resume decisions without PostgreSQL."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from database.storage import IngestionRepository
from market.contracts import TimePartition


class ScalarSession:
    """Minimal session fake returning a configured scalar checkpoint status."""

    def __init__(self, value: str | None) -> None:
        """Store the status returned by ``scalar``."""

        self.value = value

    def scalar(self, _statement: Any) -> str | None:
        """Return the configured checkpoint status."""

        return self.value


def test_resume_skips_only_fetched_or_empty_partitions() -> None:
    """Failed and absent checkpoints remain eligible for a later resume run."""

    repository = IngestionRepository()
    partition = TimePartition(datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 2, 1, tzinfo=UTC))
    arguments = {
        "source_id": uuid4(),
        "kind": "bars",
        "timeframe": "M5",
        "partition": partition,
    }
    assert repository.partition_completed(ScalarSession("fetched"), **arguments)  # type: ignore[arg-type]
    assert repository.partition_completed(ScalarSession("empty"), **arguments)  # type: ignore[arg-type]
    assert not repository.partition_completed(ScalarSession("failed"), **arguments)  # type: ignore[arg-type]
    assert not repository.partition_completed(ScalarSession(None), **arguments)  # type: ignore[arg-type]
