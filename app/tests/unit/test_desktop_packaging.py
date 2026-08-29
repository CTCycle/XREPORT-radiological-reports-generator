from __future__ import annotations

import json
from pathlib import Path
import struct
import subprocess
import sys
import zipfile

import pytest

from server.common.desktop_security import token_matches
from server.common.runtime_layout import RuntimeLayout, ensure_packaged_data

sys.path.insert(0, str(Path(__file__).parents[2] / "desktop" / "build"))
from verify_runtime_bundle import verify_archive, verify_portable  # noqa: E402

###############################################################################
def test_packaged_layout_seeds_data_without_overwriting_user_edits(tmp_path: Path, monkeypatch) -> None:
    runtime = tmp_path / "runtime"
    data = tmp_path / "data"
    (runtime / "settings").mkdir(parents=True)
    (runtime / "client").mkdir()
    (runtime / "settings" / ".env.example").write_text("EMBEDDED_DATABASE=true\n", encoding="utf-8")
    (runtime / "settings" / "configurations.json").write_text("{\"global\": {\"seed\": 42}}", encoding="utf-8")
    monkeypatch.setenv("XREPORT_DESKTOP", "true")
    monkeypatch.setenv("XREPORT_RUNTIME_ROOT", str(runtime))
    monkeypatch.setenv("XREPORT_DATA_ROOT", str(data))
    monkeypatch.setenv("XREPORT_RELEASE_VERSION", "3.0.0")
    monkeypatch.setenv("XREPORT_RUNTIME_VARIANT", "cpu")
    monkeypatch.setenv("XREPORT_CLIENT_DIST_DIR", str(runtime / "client"))

    layout = RuntimeLayout.from_environment()
    ensure_packaged_data(layout)
    env_path = data / ".env"
    config_path = data / "settings" / "configurations.json"
    assert env_path.read_text(encoding="utf-8") == "EMBEDDED_DATABASE=true\n"
    assert config_path.is_file()

    env_path.write_text("CUSTOM_SETTING=kept\n", encoding="utf-8")
    ensure_packaged_data(layout)
    assert env_path.read_text(encoding="utf-8") == "CUSTOM_SETTING=kept\n"

###############################################################################
def test_desktop_token_rejects_missing_or_wrong_values(monkeypatch) -> None:
    token = "a" * 64
    monkeypatch.setenv("XREPORT_DESKTOP_TOKEN", token)

    assert token_matches(token)
    assert not token_matches("b" * 64)
    assert not token_matches(None)

###############################################################################
def test_runtime_bundle_rejects_mutable_or_log_artifacts(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    (staging / "backend").mkdir(parents=True)
    (staging / "client").mkdir()
    (staging / "settings").mkdir()
    (staging / "backend" / "XREPORT-backend.exe").write_bytes(b"stub")
    (staging / "client" / "index.html").write_text("<html></html>", encoding="utf-8")
    (staging / "client" / "error.html").write_text("<html><div id='status'></div></html>", encoding="utf-8")
    (staging / "settings" / ".env.example").write_text("", encoding="utf-8")
    (staging / "settings" / "configurations.json").write_text("{}", encoding="utf-8")
    (staging / "settings" / "inference_models.json").write_text("{}", encoding="utf-8")
    script = Path(__file__).parents[2] / "desktop" / "build" / "create_runtime_bundle.py"

    output = tmp_path / "runtime.zip"
    audit = tmp_path / "audit.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--staging",
            str(staging),
            "--output",
            str(output),
            "--version",
            "3.0.0",
            "--variant",
            "cpu",
            "--architecture",
            "windows-x64",
            "--source-commit",
            "0" * 40,
            "--audit",
            str(audit),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    manifest = json.loads(audit.read_text(encoding="utf-8"))
    assert output.is_file()
    assert manifest["format"] == 2
    assert manifest["architecture"] == "windows-x64"
    assert manifest["source_commit"] == "0" * 40
    assert manifest["payload_sha256"]
    assert "runtime-manifest.json" not in completed.stdout
    verified = verify_archive(
        output,
        expected_version="3.0.0",
        expected_variant="cpu",
        expected_source_commit="0" * 40,
    )
    assert verified["format"] == 2
    with pytest.raises(ValueError, match="architecture"):
        verify_archive(
            output,
            expected_version="3.0.0",
            expected_variant="cpu",
            expected_source_commit="0" * 40,
            expected_architecture="other",
        )
    with pytest.raises(ValueError, match="variant"):
        verify_archive(
            output,
            expected_version="3.0.0",
            expected_variant="cuda",
            expected_source_commit="0" * 40,
        )

    missing_required = tmp_path / "missing-required.zip"
    with zipfile.ZipFile(output) as source, zipfile.ZipFile(missing_required, "w") as target:
        for info in source.infolist():
            if info.filename != "client/error.html":
                target.writestr(info, source.read(info))
    with pytest.raises(ValueError, match="missing required"):
        verify_archive(
            missing_required,
            expected_version="3.0.0",
            expected_variant="cpu",
            expected_source_commit="0" * 40,
        )

    portable = tmp_path / "portable.exe"
    archive_bytes = output.read_bytes()
    portable.write_bytes(b"MZ" + archive_bytes + b"XRPZIP01" + struct.pack("<QQ", 2, len(archive_bytes)))
    assert verify_portable(
        portable,
        expected_version="3.0.0",
        expected_variant="cpu",
        expected_source_commit="0" * 40,
    )["format"] == 2
    invalid_overlay = tmp_path / "invalid-overlay.exe"
    invalid_overlay.write_bytes(b"MZ" + archive_bytes + b"XRPZIP01" + struct.pack("<QQ", 1, len(archive_bytes) + 3))
    with pytest.raises(ValueError, match="bounds"):
        verify_portable(
            invalid_overlay,
            expected_version="3.0.0",
            expected_variant="cpu",
            expected_source_commit="0" * 40,
        )

    missing_backend = tmp_path / "missing-backend-manifest.zip"
    with zipfile.ZipFile(output) as source, zipfile.ZipFile(missing_backend, "w") as target:
        for info in source.infolist():
            payload = source.read(info)
            if info.filename == "runtime-manifest.json":
                manifest_without_backend = json.loads(payload)
                manifest_without_backend.pop("backend_executable")
                payload = json.dumps(manifest_without_backend).encode("utf-8")
            target.writestr(info, payload)
    with pytest.raises(ValueError, match="backend path"):
        verify_archive(
            missing_backend,
            expected_version="3.0.0",
            expected_variant="cpu",
            expected_source_commit="0" * 40,
        )

    (staging / "logs").mkdir()
    (staging / "logs" / "backend.log").write_text("forbidden", encoding="utf-8")
    rejected = subprocess.run(
        [
            sys.executable,
            str(script),
            "--staging",
            str(staging),
            "--output",
            str(tmp_path / "rejected.zip"),
            "--version",
            "3.0.0",
            "--variant",
            "cpu",
            "--architecture",
            "windows-x64",
            "--source-commit",
            "0" * 40,
            "--audit",
            str(tmp_path / "rejected.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "forbidden runtime staging entry" in rejected.stderr

    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("../escape.txt", b"forbidden")
    with pytest.raises(ValueError, match="unsafe member path"):
        verify_archive(
            unsafe,
            expected_version="3.0.0",
            expected_variant="cpu",
        )

###############################################################################
def test_desktop_build_inputs_are_locked_and_placeholder_free() -> None:
    requirements = (Path(__file__).parents[2] / "desktop" / "build" / "cpu-runtime-requirements.txt").read_text(encoding="utf-8")
    assert "--only-binary=:all:" in requirements
    assert "torch==2.10.0+cpu --hash=sha256:" in requirements
    assert "torchvision==0.25.0+cpu --hash=sha256:" in requirements

    build_rs = (Path(__file__).parents[2] / "desktop" / "src-tauri" / "build.rs").read_text(encoding="utf-8")
    assert "Desktop runtime has not been generated" in build_rs
    assert "runtime archive placeholder" not in build_rs

###############################################################################
def test_packaged_desktop_processes_are_windowless() -> None:
    desktop_root = Path(__file__).parents[2] / "desktop"
    spec = (desktop_root / "build" / "xreport_backend.spec").read_text(encoding="utf-8")
    backend = (desktop_root / "src-tauri" / "src" / "backend.rs").read_text(encoding="utf-8")
    shell = (desktop_root / "src-tauri" / "src" / "lib.rs").read_text(encoding="utf-8")

    assert "console=False" in spec
    assert "XREPORT_PYINSTALLER_CONSOLE" not in spec
    assert "creation_flags(&mut command, 0x08000000)" in backend
    assert 'windows_subsystem = "windows"' in shell
