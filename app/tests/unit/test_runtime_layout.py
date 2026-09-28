from __future__ import annotations

from pathlib import Path

from server.common.runtime_layout import RuntimeLayout


def test_source_layout_defaults_to_repository_resources(monkeypatch) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("XREPORT_DESKTOP", "false")
    monkeypatch.setenv("XREPORT_RESOURCES_DIR", "")

    layout = RuntimeLayout.from_environment()

    assert layout.mode == "source"
    assert layout.resources_root == repository_root / "resources"


def test_source_layout_accepts_an_explicit_resource_root(monkeypatch, tmp_path) -> None:
    selected_root = tmp_path / "selected-resources"
    monkeypatch.setenv("XREPORT_DESKTOP", "false")
    monkeypatch.setenv("XREPORT_RESOURCES_DIR", str(selected_root))

    layout = RuntimeLayout.from_environment()

    assert layout.resources_root == selected_root.resolve()
