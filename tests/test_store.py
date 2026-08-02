from pathlib import Path

import pytest

from hepml_digest.store import state_lock


def test_state_lock_prevents_overlapping_writers(tmp_path: Path):
    state_path = tmp_path / "state.json"

    with state_lock(state_path):
        with pytest.raises(TimeoutError):
            with state_lock(state_path, timeout=0):
                pass

    with state_lock(state_path, timeout=0):
        pass
