# PyInstaller onedir/windowed specification for the packaged backend.
import os
from pathlib import Path

from PyInstaller.building.build_main import Analysis, COLLECT, EXE, PYZ
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata


repo_root = Path(os.environ["XREPORT_REPO_ROOT"])
app_root = repo_root / "app"
server_root = app_root / "server"

hiddenimports = [
    "server.desktop_entry",
    "server.common.runtime_layout",
    *collect_submodules("alembic"),
    *collect_submodules("server"),
    *collect_submodules("uvicorn"),
]
datas = [
    *collect_data_files("transformers"),
    *copy_metadata("transformers"),
    *collect_data_files("keras"),
    (str(server_root / "migrations"), "server/migrations"),
]
# PyInstaller's maintained hooks already place these native libraries in the
# onedir `_internal` tree.  Adding collect_dynamic_libs here duplicates the
# largest Torch DLLs and can make a CPU release hundreds of MB larger.
binaries = []
excludes = [
    "pytest",
    "playwright",
    "ruff",
    "pyright",
    "jupyter",
    "notebook",
    "pip",
    "setuptools",
    "uv",
    "tests",
    "test",
    "torch.utils.tensorboard",
    "tensorboard",
    "tensorflow",
    "tkinter",
    "nltk.app",
    "nltk.draw",
    "nltk.test",
    "pysqlite2",
    "MySQLdb",
    "sympy.testing",
]

analysis = Analysis(
    [str(server_root / "desktop_entry.py")],
    pathex=[str(app_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="XREPORT-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.zipfiles,
    analysis.datas,
    strip=False,
    upx=False,
    name="XREPORT-backend",
)
