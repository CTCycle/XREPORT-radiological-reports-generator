# PyInstaller onedir/windowed specification for the packaged backend.
import os
from pathlib import Path
from PyInstaller.building.build_main import Analysis, COLLECT, EXE, PYZ
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata


repo_root = Path(os.environ["XREPORT_REPO_ROOT"])
app_root = repo_root / "app"
# XREPORT Inference ships only its own package and the shared package; dataset
# preparation and training live in xreport_studio, which is never collected.
inference_root = app_root / "xreport_inference"

hiddenimports = [
    "xreport_inference.desktop_entry",
    "xreport_shared.common.runtime_layout",
    *collect_submodules("alembic"),
    *collect_submodules("xreport_shared"),
    *collect_submodules("xreport_inference"),
    *collect_submodules("uvicorn"),
]
datas = [
    *collect_data_files("transformers"),
    *copy_metadata("transformers"),
    *collect_data_files("keras"),
    # Only the XREPORT Inference migration stream ships with the desktop backend.
    (str(inference_root / "migrations" / "inference"), "xreport_inference/migrations/inference"),
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
    "xreport_studio",
]

analysis = Analysis(
    [str(inference_root / "desktop_entry.py")],
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
