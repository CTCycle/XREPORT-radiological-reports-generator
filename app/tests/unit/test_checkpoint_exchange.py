from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from server.domain.checkpoint_exchange import ExchangeManifest
from server.repositories import checkpoint_archive as archive_module
from server.repositories.checkpoint_archive import (
    ArchiveError,
    CORE_FILES,
    inspect_archive,
)
from server.services import checkpoint_exchange as exchange
from server.services.errors import BadRequestError, ConflictError
from server.services.jobs import JobExecutionError

REVISION = "a" * 40


###############################################################################
class FakeCheckpoints:
    def __init__(self) -> None:
        self.records: dict[str, SimpleNamespace] = {}

    def get_checkpoint(self, name: str):
        return self.records.get(name)

    def verify_artifact(self, name: str):
        return self.records[name]

    def register_completed_checkpoint(self, name: str, path: Path):
        record = SimpleNamespace(
            name=name,
            path=Path(path),
            artifact_complete=all((Path(path) / item).is_file() for item in CORE_FILES),
        )
        self.records[name] = record
        return record


###############################################################################
class Node:
    """One simulated machine: its own checkpoint, encoder and tokenizer roots."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.checkpoints_dir = root / "checkpoints"
        self.encoders_dir = root / "models" / "XRAYEncoder"
        self.tokenizers_dir = root / "models" / "tokenizers"
        self.exports_dir = root / "exports"
        self.checkpoints_dir.mkdir(parents=True)
        for attribute, value in (
            ("CHECKPOINTS_DIR", self.checkpoints_dir),
            ("ENCODERS_DIR", self.encoders_dir),
            ("TOKENIZERS_DIR", self.tokenizers_dir),
            ("EXPORTS_DIR", self.exports_dir),
        ):
            monkeypatch.setattr(exchange, attribute, value)
        monkeypatch.setattr(exchange, "pinned_encoder_revision", lambda: REVISION)
        self.repository = FakeCheckpoints()
        self.jobs = SimpleNamespace(
            should_stop=lambda _job: False, update_progress=lambda *_: None
        )
        self.service = exchange.CheckpointExchangeService(
            self.repository, self.jobs  # type: ignore[arg-type]
        )

    def install_encoder(self) -> None:
        snapshot = exchange.encoder_snapshot_dir(REVISION)
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "config.json").write_text("{}")
        (snapshot / "model.safetensors").write_bytes(b"w")

    def create_checkpoint(self, name: str, payload: bytes = b"keras") -> Path:
        path = self.checkpoints_dir / name
        (path / "configuration").mkdir(parents=True)
        (path / "saved_model.keras").write_bytes(payload)
        (path / "configuration" / "configuration.json").write_text("{}")
        (path / "configuration" / "metadata.json").write_text(
            json.dumps({"tokenizer": "org/tok", "max_report_size": 8})
        )
        (path / "configuration" / "session_history.json").write_text(
            json.dumps({"epochs": 3, "history": {"loss": [2.0, 1.0], "val_loss": [2.5, 1.5]}})
        )
        self.repository.register_completed_checkpoint(name, path)
        return path


@pytest.fixture
def stub_ml(monkeypatch: pytest.MonkeyPatch):
    def stage_tokenizer(_metadata, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer.json").write_text('{"vocab": 1}')
        return [("tokenizer/tokenizer.json", directory / "tokenizer.json")]

    monkeypatch.setattr(
        exchange.CheckpointExchangeService, "_stage_tokenizer", staticmethod(stage_tokenizer)
    )
    monkeypatch.setattr(
        exchange.CheckpointExchangeService, "_smoke_test", staticmethod(lambda _dir: None)
    )


def _export(node: Node, name: str, include_encoder: bool = False) -> Path:
    node.service.run_export(name, include_encoder)
    return node.exports_dir / f"{name}.xreportckpt"


def _import(node: Node, archive: Path, rename_to: str | None = None) -> dict:
    staged = node.root / "upload.xreportckpt"
    staged.write_bytes(archive.read_bytes())
    plan = node.service.assess_archive(staged, archive.name, rename_to)
    return node.service.run_import(str(staged), plan)


###############################################################################
def test_export_import_roundtrip_is_idempotent_and_conflict_aware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_ml
) -> None:
    producer = Node(tmp_path / "studio", monkeypatch)
    producer.install_encoder()
    producer.create_checkpoint("run1")
    archive = _export(producer, "run1")
    manifest = ExchangeManifest.model_validate_json(
        zipfile.ZipFile(archive).read("manifest.json")
    )
    assert manifest.provenance.epochs == 3
    assert manifest.provenance.final_val_loss == 1.5

    consumer = Node(tmp_path / "xreport", monkeypatch)
    consumer.install_encoder()
    result = _import(consumer, archive)["import"]
    assert result["status"] == "imported"
    final = consumer.checkpoints_dir / "run1"
    assert (final / "exchange.json").is_file()
    assert "run1" in consumer.repository.records
    assert (consumer.tokenizers_dir / "_exchange" / "org--tok" / "tokenizer.json").is_file()
    assert not [p for p in consumer.checkpoints_dir.iterdir() if p.name.startswith(".import-")]

    again = _import(consumer, archive)["import"]
    assert again["status"] == "already_imported"

    producer_other = Node(tmp_path / "other", monkeypatch)
    producer_other.install_encoder()
    producer_other.create_checkpoint("run1", payload=b"different")
    other_archive = _export(producer_other, "run1")
    consumer2 = Node(tmp_path / "xreport2", monkeypatch)
    consumer2.install_encoder()
    consumer2.create_checkpoint("run1")
    (consumer2.checkpoints_dir / "run1" / "exchange.json").write_text(
        json.dumps({"content_sha256": "0" * 64})
    )
    staged = consumer2.root / "u.xreportckpt"
    staged.write_bytes(other_archive.read_bytes())
    with pytest.raises(ConflictError, match="run1_2"):
        consumer2.service.assess_archive(staged, "x", None)
    renamed = consumer2.service.assess_archive(staged, "x", "run1_2")
    assert renamed.renamed_from == "run1"


###############################################################################
def test_checksum_mismatch_leaves_no_trace_and_keeps_existing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_ml
) -> None:
    producer = Node(tmp_path / "studio", monkeypatch)
    producer.install_encoder()
    producer.create_checkpoint("run1")
    good = _export(producer, "run1")
    tampered = tmp_path / "tampered.xreportckpt"
    with zipfile.ZipFile(good) as source, zipfile.ZipFile(tampered, "w") as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "saved_model.keras":
                data = b"KERAS"  # same size, different bytes
            target.writestr(info.filename, data)

    consumer = Node(tmp_path / "xreport", monkeypatch)
    consumer.install_encoder()
    known_good = consumer.create_checkpoint("known")
    with pytest.raises(JobExecutionError) as error:
        _import(consumer, tampered)
    assert error.value.code == "checksum_mismatch"
    assert "run1" not in consumer.repository.records
    assert not (consumer.checkpoints_dir / "run1").exists()
    assert known_good.is_dir()
    assert not [p for p in consumer.checkpoints_dir.iterdir() if p.name.startswith(".import-")]


###############################################################################
def test_failed_smoke_test_blocks_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_ml
) -> None:
    producer = Node(tmp_path / "studio", monkeypatch)
    producer.install_encoder()
    producer.create_checkpoint("run1")
    archive = _export(producer, "run1")
    consumer = Node(tmp_path / "xreport", monkeypatch)
    consumer.install_encoder()

    def failing(_directory: Path) -> None:
        raise JobExecutionError("corrupt weights", code="smoke_test_failed")

    monkeypatch.setattr(
        exchange.CheckpointExchangeService, "_smoke_test", staticmethod(failing)
    )
    with pytest.raises(JobExecutionError, match="corrupt weights"):
        _import(consumer, archive)
    assert consumer.repository.records == {}
    assert not (consumer.checkpoints_dir / "run1").exists()


###############################################################################
def test_cancelled_import_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_ml
) -> None:
    producer = Node(tmp_path / "studio", monkeypatch)
    producer.install_encoder()
    producer.create_checkpoint("run1")
    archive = _export(producer, "run1")
    consumer = Node(tmp_path / "xreport", monkeypatch)
    consumer.install_encoder()
    consumer.jobs.should_stop = lambda _job: True
    staged = consumer.root / "u.xreportckpt"
    staged.write_bytes(archive.read_bytes())
    plan = consumer.service.assess_archive(staged, "x", None)
    assert consumer.service.run_import(str(staged), plan, job_id="j") == {}
    assert consumer.repository.records == {}
    assert not staged.exists()
    assert not [p for p in consumer.checkpoints_dir.iterdir()]


###############################################################################
def test_missing_encoder_is_explicit_when_not_bundled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_ml
) -> None:
    producer = Node(tmp_path / "studio", monkeypatch)
    producer.install_encoder()
    producer.create_checkpoint("run1")
    archive_plain = tmp_path / "plain.xreportckpt"
    archive_plain.write_bytes(_export(producer, "run1").read_bytes())
    bundled = _export(producer, "run1", include_encoder=True)
    consumer = Node(tmp_path / "xreport", monkeypatch)  # no encoder installed
    with pytest.raises(BadRequestError, match="encoder"):
        consumer.service.assess_archive(archive_plain, "x", None)

    result = _import(consumer, bundled)["import"]
    assert result["status"] == "imported"
    assert exchange.encoder_installed(REVISION)


###############################################################################
def test_incompatible_versions_and_lone_keras_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_ml
) -> None:
    producer = Node(tmp_path / "studio", monkeypatch)
    producer.install_encoder()
    producer.create_checkpoint("run1")
    good = _export(producer, "run1")
    consumer = Node(tmp_path / "xreport", monkeypatch)
    consumer.install_encoder()

    for field, value, message in (
        ("format_version", 99, "newer"),
        ("custom_objects_api", 99, "custom-layer"),
        ("architecture", "other", "architecture"),
    ):
        edited = tmp_path / f"{field}.xreportckpt"
        with zipfile.ZipFile(good) as source, zipfile.ZipFile(edited, "w") as target:
            for info in source.infolist():
                data = source.read(info.filename)
                if info.filename == "manifest.json":
                    manifest = json.loads(data)
                    manifest[field] = value
                    data = json.dumps(manifest).encode()
                target.writestr(info.filename, data)
        with pytest.raises(BadRequestError, match=message):
            consumer.service.assess_archive(edited, "x", None)

    lone = tmp_path / "lone.keras"
    lone.write_bytes(b"not a zip")
    with pytest.raises(BadRequestError, match="lone .keras"):
        consumer.service.assess_archive(lone, "lone.keras", None)


###############################################################################
def test_legacy_four_file_archive_is_accepted_with_local_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_ml
) -> None:
    producer = Node(tmp_path / "old", monkeypatch)
    old = producer.create_checkpoint("legacy-run")
    legacy = tmp_path / "legacy-run.zip"
    with zipfile.ZipFile(legacy, "w") as target:
        for relative in CORE_FILES:
            target.write(old / relative, relative)

    consumer = Node(tmp_path / "xreport", monkeypatch)
    with pytest.raises(JobExecutionError, match="encoder"):
        _import(consumer, legacy)
    assert consumer.repository.records == {}

    consumer.install_encoder()
    result = _import(consumer, legacy)["import"]
    assert result["status"] == "imported"
    assert result["legacy"] is True
    assert result["checkpoint"] == "legacy-run"


###############################################################################
def _write_zip(path: Path, members: list[tuple[str, bytes]]) -> zipfile.ZipFile:
    with zipfile.ZipFile(path, "w") as target:
        for name, data in members:
            target.writestr(name, data)
    return zipfile.ZipFile(path)


@pytest.mark.parametrize(
    "name",
    [
        "../evil.json",
        "/abs/manifest.json",
        "C:evil.json",
        "tokenizer/../../x",
        "other/file.txt",
        "saved_model.keras/",
    ],
)
def test_hostile_member_names_are_rejected(tmp_path: Path, name: str) -> None:
    archive = _write_zip(tmp_path / "a.zip", [(name, b"x")])
    with pytest.raises(ArchiveError):
        inspect_archive(archive)


def test_duplicate_symlink_and_oversized_archives_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duplicate = tmp_path / "dup.zip"
    with pytest.warns(UserWarning):
        archive = _write_zip(duplicate, [("manifest.json", b"1"), ("manifest.json", b"2")])
    with pytest.raises(ArchiveError, match="duplicate"):
        inspect_archive(archive)

    link = tmp_path / "link.zip"
    with zipfile.ZipFile(link, "w") as target:
        info = zipfile.ZipInfo("saved_model.keras")
        info.external_attr = 0o120777 << 16
        target.writestr(info, "target")
    with pytest.raises(ArchiveError, match="non-regular"):
        inspect_archive(zipfile.ZipFile(link))

    many = tmp_path / "many.zip"
    monkeypatch.setattr(archive_module, "MAX_ENTRIES", 2)
    archive = _write_zip(
        many, [(f"tokenizer/f{index}.json", b"x") for index in range(3)]
    )
    with pytest.raises(ArchiveError, match="more than"):
        inspect_archive(archive)

    monkeypatch.setattr(archive_module, "MAX_TOTAL_BYTES", 10)
    big = _write_zip(tmp_path / "big.zip", [("saved_model.keras", b"x" * 100)])
    with pytest.raises(ArchiveError, match="expands"):
        inspect_archive(big)


def test_streamed_size_is_enforced_against_the_declared_size(tmp_path: Path) -> None:
    archive = _write_zip(tmp_path / "a.zip", [("saved_model.keras", b"x" * 64)])
    with pytest.raises(ArchiveError, match="exceeds"):
        archive_module.extract_member(
            archive, "saved_model.keras", tmp_path / "out" / "f", expected_size=8
        )


###############################################################################
def test_sweep_removes_only_import_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = Node(tmp_path / "xreport", monkeypatch)
    keep = node.create_checkpoint("keep")
    (node.checkpoints_dir / ".import-abc").mkdir()
    (node.checkpoints_dir / ".import-upload-1.xreportckpt").write_bytes(b"x")
    assert exchange.sweep_stale_staging() == 2
    assert keep.is_dir()


###############################################################################
def test_failed_import_removes_the_shared_resources_it_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_ml
) -> None:
    producer = Node(tmp_path / "studio", monkeypatch)
    producer.install_encoder()
    producer.create_checkpoint("run1")
    bundled = _export(producer, "run1", include_encoder=True)
    consumer = Node(tmp_path / "xreport", monkeypatch)  # nothing installed yet

    def failing(_directory: Path) -> None:
        raise JobExecutionError("bad weights", code="smoke_test_failed")

    monkeypatch.setattr(
        exchange.CheckpointExchangeService, "_smoke_test", staticmethod(failing)
    )
    with pytest.raises(JobExecutionError):
        _import(consumer, bundled)

    assert not exchange.encoder_installed(REVISION)
    assert not (consumer.tokenizers_dir / "_exchange" / "org--tok").exists()
    assert consumer.repository.records == {}
