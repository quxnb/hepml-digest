from pathlib import Path
from datetime import datetime, timezone

import pytest

from hepml_digest.models import State
from hepml_digest.store import state_lock


def test_state_lock_prevents_overlapping_writers(tmp_path: Path):
    state_path = tmp_path / "state.json"

    with state_lock(state_path):
        with pytest.raises(TimeoutError):
            with state_lock(state_path, timeout=0):
                pass

    with state_lock(state_path, timeout=0):
        pass


def test_legacy_state_loads_without_queue_or_cursor():
    state = State.model_validate({"schema_version": 2, "records": {}})
    assert state.pending_papers == {}
    assert state.last_backfill_at is None


def test_all_operational_timestamps_share_timezone_validation():
    state = State.model_validate({
        "last_backfill_at": "2026-10-29T00:00:00",
        "pending_backfill_since": "2026-10-22T00:00:00",
        "work": {"screening:test:v1": {
            "first_seen_at": "2026-10-22T00:00:00", "fingerprint": "test",
            "next_retry_at": "2026-10-30T00:00:00",
        }},
        "health": {"last_analysis_success_at": "2026-10-29T00:00:00"},
    })
    assert state.last_backfill_at == datetime(2026, 10, 29, tzinfo=timezone.utc)
    assert state.pending_backfill_since.tzinfo is timezone.utc
    assert state.work["screening:test:v1"].first_seen_at.tzinfo is timezone.utc
    assert state.work["screening:test:v1"].next_retry_at.tzinfo is timezone.utc
    assert state.health.last_analysis_success_at.tzinfo is timezone.utc
