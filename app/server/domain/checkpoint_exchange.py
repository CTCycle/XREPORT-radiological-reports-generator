from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EXCHANGE_FORMAT = "xreport-checkpoint"
EXCHANGE_FORMAT_VERSION = 1
EXCHANGE_ARCHITECTURE = "xreport-beit-transformer"
# Bump when registered Keras custom-object names or signatures change.
CUSTOM_OBJECTS_API = 1
EXCHANGE_FILE_EXTENSION = ".xreportckpt"

ENCODER_REPOSITORY = "microsoft/beit-base-patch16-224"

###############################################################################
class ManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

###############################################################################
class ManifestProducer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    application: str
    version: str
    keras: str | None = None
    torch: str | None = None

###############################################################################
class ManifestTokenizer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    bundled: bool

###############################################################################
class ManifestEncoder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repository: str
    revision: str
    bundled: bool

###############################################################################
class ManifestProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_utc: str
    epochs: int | None = None
    final_loss: float | None = None
    final_val_loss: float | None = None

###############################################################################
class ExchangeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["xreport-checkpoint"] = EXCHANGE_FORMAT
    format_version: int
    architecture: str
    custom_objects_api: int
    checkpoint_name: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    legacy: bool = False
    producer: ManifestProducer
    tokenizer: ManifestTokenizer
    encoder: ManifestEncoder
    provenance: ManifestProvenance
    files: list[ManifestFile]

###############################################################################
class ExportRequest(BaseModel):
    checkpoint: str
    include_encoder: bool = False

###############################################################################
class ExportResult(BaseModel):
    filename: str
    size: int
    content_sha256: str

###############################################################################
class ImportOutcome(BaseModel):
    checkpoint: str
    content_sha256: str
    status: Literal["imported", "already_imported"]
    legacy: bool = False
    renamed_from: str | None = None
