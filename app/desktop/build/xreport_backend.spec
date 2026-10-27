# PyInstaller onedir/windowed specification for the packaged backend.
import os
from pathlib import Path
from runpy import run_path

from PyInstaller.building.build_main import Analysis, COLLECT, EXE, PYZ
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata


repo_root = Path(os.environ["XREPORT_REPO_ROOT"])
app_root = repo_root / "app"
server_root = app_root / "server"

# XREPORT is inference-only: dataset preparation and training belong to Studio.
STUDIO_ONLY_MODULES = run_path(str(server_root / "common" / "profile_modules.py"))[
    "STUDIO_ONLY_MODULES"
]
studio_excludes = list(STUDIO_ONLY_MODULES)


def is_studio_module(name):
    return any(name == module or name.startswith(module + ".") for module in studio_excludes)


hiddenimports = [
    "server.desktop_entry",
    "server.common.runtime_layout",
    *collect_submodules("alembic"),
    *(name for name in collect_submodules("server") if not is_studio_module(name)),
    *collect_submodules("uvicorn"),
]
datas = [
    *collect_data_files("transformers"),
    *copy_metadata("transformers"),
    *collect_data_files("keras"),
    # Only the XREPORT Inference migration stream ships with the desktop backend.
    (str(server_root / "migrations" / "inference"), "server/migrations/inference"),
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
    *studio_excludes,
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
