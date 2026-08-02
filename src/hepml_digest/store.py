from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import State


@contextmanager
def state_lock(path: Path, timeout: float = 30.0) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.monotonic() + timeout
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            while not locked:
                try:
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                    locked = True
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Timed out waiting for state lock: {lock_path}"
                        )
                    time.sleep(0.1)
        else:
            import fcntl

            while not locked:
                try:
                    fcntl.flock(
                        descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                    locked = True
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Timed out waiting for state lock: {lock_path}"
                        )
                    time.sleep(0.1)
        yield
    finally:
        if locked:
            if os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def load_state(path: Path) -> State:
    if not path.exists():
        return State()
    return State.model_validate_json(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: State) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = state.model_dump_json(indent=2)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def prune_state(state: State, retention_days: int) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    state.records = {
        key: record
        for key, record in state.records.items()
        if record.processed_at >= cutoff
    }
