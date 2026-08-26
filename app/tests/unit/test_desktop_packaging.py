from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from server.common.desktop_security import token_matches
from server.common.runtime_layout import RuntimeLayout, ensure_packaged_data


###############################################################################
def test_packaged_layout_seeds_data_without_overwriting_edits(tmp_path: Path, monkeypatch) -> None:
    runtime = tmp_path / "runtime"
    data = tmp_path / "data"
    (runtime / "settings").mkdir(parents=True)
    (runtime / "settings" / ".env.example").write_text("EMBEDDED_DATABASE=true\n", encoding="utf-8")
    (runtime / "settings" / "configurations.json").write_text("{\"global\": {\"seed\": 42}}", encoding="utf-8")
    monkeypatch.setenv("XREPORT_DESKTOP", "true")
    monkeypatch.setenv("XREPORT_RUNTIME_ROOT", str(runtime))
    monkeypatch.setenv("XREPORT_DATA_ROOT", str(data))
    monkeypatch.setenv("XREPORT_RELEASE_VERSION", "3.0.0")
    monkeypatch.setenv("XREPORT_RUNTIME_VARIANT", "cpu")

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
def test_desktop_token_is_constant_time_checked(monkeypatch) -> None:
    token = "a" * 64
    monkeypatch.setenv("XREPORT_DESKTOP_TOKEN", token)
    assert token_matches(token)
    assert not token_matches("b" * 64)
    assert not token_matches(None)


###############################################################################
def test_runtime_bundle_streaming_audit_rejects_forbidden_entries(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    (staging / "backend").mkdir(parents=True)
    (staging / "client").mkdir()
    (staging / "settings").mkdir()
    (staging / "backend" / "XREPORT-backend.exe").write_bytes(b"stub")
    (staging / "client" / "index.html").write_text("<html></html>", encoding="utf-8")
    (staging / "settings" / ".env.example").write_text("", encoding="utf-8")
    (staging / "settings" / "configurations.json").write_text("{}", encoding="utf-8")
    (staging / "settings" / "inference_models.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "runtime.zip"
    audit = tmp_path / "audit.json"
    script = Path(__file__).parents[2] / "desktop" / "build" / "create_runtime_bundle.py"
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
            "--audit",
            str(audit),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    manifest = json.loads(audit.read_text(encoding="utf-8"))
    assert output.is_file()
    assert manifest["payload_sha256"]
    assert "runtime-manifest.json" not in completed.stdout
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
            "--audit",
            str(tmp_path / "rejected.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "forbidden runtime staging entry" in rejected.stderr


###############################################################################
def test_pyinstaller_spec_packages_alembic_migration_files() -> None:
    spec = Path(__file__).parents[2] / "desktop" / "build" / "xreport_backend.spec"
    source = spec.read_text(encoding="utf-8")

    assert 'collect_submodules("alembic")' in source
    assert '"server/migrations"' in source
