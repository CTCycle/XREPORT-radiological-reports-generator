from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from xreport_shared.common.process_lock import DataFolderInUseError, DataFolderLock


###############################################################################
def test_second_owner_is_rejected_until_the_first_releases(tmp_path: Path) -> None:
    first = DataFolderLock(tmp_path)
    first.acquire()
    probe = (
        "import sys; from pathlib import Path\n"
        "from xreport_shared.common.process_lock import DataFolderInUseError, DataFolderLock\n"
        "try:\n    DataFolderLock(Path(sys.argv[1])).acquire()\n"
        "except DataFolderInUseError:\n    sys.exit(3)\n"
    )
    app_dir = Path(__file__).resolve().parents[2]

    def second_owner() -> int:
        return subprocess.run(
            [sys.executable, "-c", probe, str(tmp_path)], cwd=app_dir, env={"PYTHONPATH": str(app_dir)}
        ).returncode

    assert second_owner() == 3
    first.release()
    assert second_owner() == 0


###############################################################################
def test_same_process_cannot_take_the_lock_twice(tmp_path: Path) -> None:
    first = DataFolderLock(tmp_path)
    first.acquire()
    try:
        with pytest.raises(DataFolderInUseError):
            DataFolderLock(tmp_path).acquire()
    finally:
        first.release()
