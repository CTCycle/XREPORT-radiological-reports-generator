"""Single-owner lock for an application's data folder.

Two processes mutating the same SQLite database or checkpoint folder is never
intended, so a second process on the same data root fails closed. The OS drops
the lock when the owner exits or crashes, so no stale lock file can block a start.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import IO

LOCK_FILENAME = ".xreport.lock"


###############################################################################
class DataFolderInUseError(RuntimeError):
    """Raised when another process already owns the data folder."""


###############################################################################
class DataFolderLock:

    # -------------------------------------------------------------------------
    def __init__(self, directory: Path) -> None:
        self.path = directory / LOCK_FILENAME
        self._handle: IO[bytes] | None = None

    # -------------------------------------------------------------------------
    def acquire(self) -> None:
        if self._handle is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            self._lock(handle)
        except OSError as exc:
            handle.close()
            raise DataFolderInUseError(
                f"Another XREPORT process is already using the data folder {self.path.parent}. "
                "Stop it before starting this application."
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()).encode())
        handle.flush()
        self._handle = handle

    # -------------------------------------------------------------------------
    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            self._unlock(handle)
        finally:
            handle.close()

    # -------------------------------------------------------------------------
    @staticmethod
    def _lock(handle: IO[bytes]) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]

    # -------------------------------------------------------------------------
    @staticmethod
    def _unlock(handle: IO[bytes]) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
