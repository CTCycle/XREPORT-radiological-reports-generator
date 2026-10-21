from __future__ import annotations

import base64
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from threading import RLock
from typing import Any, Literal, Mapping

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from server.common.path import DATABASE_CONFIG_FILE_PATH, DATABASE_FILE_PATH

###############################################################################
LEGACY_DATABASE_VARIABLES = frozenset(
    {
        "EMBEDDED_DATABASE",
        "DATABASE_URL",
        "DATABASE_ENGINE",
        "DATABASE_HOST",
        "DATABASE_PORT",
        "DATABASE_NAME",
        "DATABASE_USERNAME",
        "DATABASE_PASSWORD",
        "DATABASE_SSL",
        "DATABASE_SSL_CA",
        "DATABASE_CONNECT_TIMEOUT",
        "DATABASE_INSERT_BATCH_SIZE",
    }
)
CONFIG_SCHEMA_VERSION = 1
_CONFIG_LOCK = RLock()
_BOOTSTRAP_ERROR: str | None = None

###############################################################################
class DatabaseConfigurationError(RuntimeError):
    """Raised when database configuration cannot be safely loaded or saved."""

###############################################################################
class DatabaseConfiguration(BaseModel):
    """Validated in-memory connection configuration; secrets are never serialized directly."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        hide_input_in_errors=True,
    )

    backend: Literal["sqlite", "postgresql"]
    sqlite_path: Path | None = None
    engine: str | None = None
    host: str | None = None
    port: int = Field(default=5432, ge=1, le=65535)
    database_name: str | None = None
    username: str | None = None
    password: str | None = Field(default=None, repr=False)
    ssl: bool = False
    ssl_ca: str | None = None
    connect_timeout: int = Field(default=30, ge=1, le=300)
    insert_batch_size: int = Field(default=1000, ge=1)

    # -------------------------------------------------------------------------
    @field_validator("sqlite_path", mode="before")
    @classmethod
    def parse_sqlite_path(cls, value: Any) -> Any:
        if isinstance(value, str):
            return Path(value)
        return value

    # -------------------------------------------------------------------------
    @field_validator("password", mode="before")
    @classmethod
    def normalize_empty_password(cls, value: Any) -> Any:
        return None if value == "" else value

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_backend_fields(self) -> DatabaseConfiguration:
        if self.backend == "sqlite":
            if self.sqlite_path is None or not str(self.sqlite_path).strip():
                raise ValueError("A SQLite database path is required.")
            if not self.sqlite_path.is_absolute():
                raise ValueError("The SQLite database path must be absolute.")
            return self

        required = {
            "host": self.host,
            "database_name": self.database_name,
            "username": self.username,
        }
        missing = [key for key, value in required.items() if not _clean(value)]
        if missing:
            raise ValueError(
                "PostgreSQL requires: " + ", ".join(missing) + "."
            )
        supported_engines = {
            None,
            "postgres",
            "postgresql",
            "postgresql+psycopg",
        }
        if self.engine not in supported_engines:
            raise ValueError("PostgreSQL uses the installed psycopg driver.")
        return self

    # -------------------------------------------------------------------------
    def to_database_settings(self):
        from server.configurations.settings import DatabaseSettings

        return DatabaseSettings(
            backend=self.backend,
            engine=("postgresql+psycopg" if self.backend == "postgresql" else None),
            host=self.host,
            port=self.port if self.backend == "postgresql" else None,
            database_name=self.database_name,
            username=self.username,
            password=self.password,
            ssl=self.ssl,
            ssl_ca=self.ssl_ca,
            connect_timeout=self.connect_timeout,
            insert_batch_size=self.insert_batch_size,
            sqlite_path=self.sqlite_path,
        )

    # -------------------------------------------------------------------------
    def public_dict(self) -> dict[str, Any]:
        values = self.model_dump(mode="json", exclude={"password"})
        values["password_configured"] = bool(self.password)
        return values

###############################################################################
class _StoredDatabaseConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)

    schema_version: Literal[1]
    backend: Literal["sqlite", "postgresql"]
    sqlite_path: str | None = None
    engine: str | None = None
    host: str | None = None
    port: int = 5432
    database_name: str | None = None
    username: str | None = None
    password_protected: str | None = None
    ssl: bool = False
    ssl_ca: str | None = None
    connect_timeout: int = 30
    insert_batch_size: int = 1000

###############################################################################
@dataclass(frozen=True)
class DatabaseConfigurationStatus:
    configuration: DatabaseConfiguration | None
    error: str | None

###############################################################################
def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    result = value.strip()
    return result or None

###############################################################################
def _safe_validation_message(error: ValidationError) -> str:
    details = error.errors(include_input=False)
    messages = [
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
        for item in details
    ]
    return "; ".join(messages) or "Database configuration is invalid."

###############################################################################
class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]

###############################################################################
def _protect_password(password: str) -> str:
    if not password:
        return ""
    if os.name != "nt":
        raise DatabaseConfigurationError(
            "Saving a PostgreSQL password requires Windows DPAPI."
        )

    raw = password.encode("utf-8")
    source_buffer = ctypes.create_string_buffer(raw)
    source = _DataBlob(
        len(raw), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_ubyte))
    )
    protected = _DataBlob()
    crypt32 = ctypes.WinDLL("Crypt32.dll", use_last_error=True)
    kernel32 = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
    protect = crypt32.CryptProtectData
    protect.argtypes = [
        ctypes.POINTER(_DataBlob),
        wintypes.LPCWSTR,
        ctypes.POINTER(_DataBlob),
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    protect.restype = wintypes.BOOL
    local_free = kernel32.LocalFree
    local_free.argtypes = [wintypes.HLOCAL]
    local_free.restype = wintypes.HLOCAL
    if not protect(
        ctypes.byref(source),
        "XREPORT database password",
        None,
        None,
        None,
        0x1,
        ctypes.byref(protected),
    ):
        raise DatabaseConfigurationError("Windows could not protect the database password.")
    try:
        return base64.b64encode(
            ctypes.string_at(protected.pbData, protected.cbData)
        ).decode("ascii")
    finally:
        local_free(protected.pbData)

###############################################################################
def _unprotect_password(value: str | None) -> str | None:
    if not value:
        return None
    if os.name != "nt":
        raise DatabaseConfigurationError(
            "This database password can only be opened by its Windows account."
        )
    try:
        raw = base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise DatabaseConfigurationError(
            "The protected database password is invalid; enter it again in Settings."
        ) from exc

    source_buffer = ctypes.create_string_buffer(raw)
    source = _DataBlob(
        len(raw), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_ubyte))
    )
    unprotected = _DataBlob()
    crypt32 = ctypes.WinDLL("Crypt32.dll", use_last_error=True)
    kernel32 = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
    unprotect = crypt32.CryptUnprotectData
    unprotect.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(_DataBlob),
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    unprotect.restype = wintypes.BOOL
    local_free = kernel32.LocalFree
    local_free.argtypes = [wintypes.HLOCAL]
    local_free.restype = wintypes.HLOCAL
    if not unprotect(
        ctypes.byref(source), None, None, None, None, 0x1, ctypes.byref(unprotected)
    ):
        raise DatabaseConfigurationError(
            "The saved database password is unavailable to this Windows account; enter it again in Settings."
        )
    try:
        return ctypes.string_at(unprotected.pbData, unprotected.cbData).decode("utf-8")
    finally:
        local_free(unprotected.pbData)

###############################################################################
def _stored_payload(configuration: DatabaseConfiguration) -> dict[str, Any]:
    values = configuration.model_dump(mode="json", exclude={"password"})
    values["schema_version"] = CONFIG_SCHEMA_VERSION
    values["password_protected"] = _protect_password(configuration.password or "")
    return values

###############################################################################
def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.chmod(temporary_path, 0o600)
        except OSError:
            # Windows relies on DPAPI for the secret; ACLs remain owned by the data directory.
            pass
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise DatabaseConfigurationError(
            "Could not safely save database settings in the application data directory."
        ) from exc

###############################################################################
def save_database_configuration(
    configuration: DatabaseConfiguration,
    *,
    path: Path = DATABASE_CONFIG_FILE_PATH,
    environment_file: Path | None = None,
) -> None:
    global _BOOTSTRAP_ERROR
    try:
        payload = json.dumps(
            _stored_payload(configuration), indent=2, sort_keys=True
        ).encode("utf-8") + b"\n"
    except DatabaseConfigurationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DatabaseConfigurationError("Could not encode database settings safely.") from exc
    with _CONFIG_LOCK:
        previous: bytes | None = None
        if path.is_file():
            try:
                previous = path.read_bytes()
            except OSError as exc:
                raise DatabaseConfigurationError(
                    "Could not read the current database settings before saving an update."
                ) from exc
        _atomic_write(path, payload)
        try:
            if environment_file is not None:
                _remove_legacy_environment_entries(environment_file)
        except DatabaseConfigurationError:
            try:
                if previous is None:
                    path.unlink(missing_ok=True)
                else:
                    _atomic_write(path, previous)
            except Exception as rollback_error:  # noqa: BLE001
                raise DatabaseConfigurationError(
                    "Legacy database settings could not be removed and the previous database configuration could not be restored."
                ) from rollback_error
            raise
        _BOOTSTRAP_ERROR = None
        _clear_legacy_process_environment()

###############################################################################
def read_database_configuration(
    *, path: Path = DATABASE_CONFIG_FILE_PATH
) -> DatabaseConfiguration:
    with _CONFIG_LOCK:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            stored = _StoredDatabaseConfiguration.model_validate(payload)
            settings = {
                key: value
                for key, value in stored.model_dump().items()
                if key not in {"schema_version", "password_protected"}
            }
            settings["password"] = _unprotect_password(stored.password_protected)
            if stored.sqlite_path is not None:
                settings["sqlite_path"] = Path(stored.sqlite_path)
            return DatabaseConfiguration.model_validate(settings)
        except FileNotFoundError as exc:
            raise DatabaseConfigurationError(
                "Database settings have not been configured. Use the Database section in Settings."
            ) from exc
        except DatabaseConfigurationError:
            raise
        except ValidationError as exc:
            raise DatabaseConfigurationError(
                "Database settings are invalid: " + _safe_validation_message(exc)
            ) from exc
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise DatabaseConfigurationError(
                "Database settings could not be read safely. Use the Database section in Settings to repair them."
            ) from exc

###############################################################################
def read_database_configuration_status(
    *, path: Path = DATABASE_CONFIG_FILE_PATH
) -> DatabaseConfigurationStatus:
    try:
        return DatabaseConfigurationStatus(
            configuration=read_database_configuration(path=path), error=None
        )
    except DatabaseConfigurationError as exc:
        return DatabaseConfigurationStatus(configuration=None, error=str(exc))

###############################################################################
def load_database_settings():
    if _BOOTSTRAP_ERROR:
        raise DatabaseConfigurationError(_BOOTSTRAP_ERROR)
    return read_database_configuration().to_database_settings()

###############################################################################
def _parse_bool(value: str | None, *, default: bool, name: str) -> bool:
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise DatabaseConfigurationError(f"Legacy {name} must be a boolean value.")

###############################################################################
def _legacy_configuration(values: Mapping[str, Any]) -> DatabaseConfiguration:
    database_url = _clean(_string(values.get("DATABASE_URL")))
    if database_url:
        raise DatabaseConfigurationError(
            "The legacy DATABASE_URL setting is unsupported. Select SQLite or enter PostgreSQL connection fields in Settings."
        )

    if _parse_bool(
        _string(values.get("EMBEDDED_DATABASE")),
        default=True,
        name="EMBEDDED_DATABASE",
    ):
        try:
            return DatabaseConfiguration(
                backend="sqlite",
                sqlite_path=DATABASE_FILE_PATH,
                insert_batch_size=_legacy_int(
                    values.get("DATABASE_INSERT_BATCH_SIZE"), 1000,
                    "DATABASE_INSERT_BATCH_SIZE",
                ),
            )
        except ValidationError as exc:
            raise DatabaseConfigurationError(
                "Legacy database settings need correction: " + _safe_validation_message(exc)
            ) from exc

    try:
        return DatabaseConfiguration(
            backend="postgresql",
            engine=_clean(_string(values.get("DATABASE_ENGINE"))) or "postgresql+psycopg",
            host=_clean(_string(values.get("DATABASE_HOST"))),
            port=_legacy_int(values.get("DATABASE_PORT"), 5432, "DATABASE_PORT"),
            database_name=_clean(_string(values.get("DATABASE_NAME"))),
            username=_clean(_string(values.get("DATABASE_USERNAME"))),
            password=_clean(_string(values.get("DATABASE_PASSWORD"))),
            ssl=_parse_bool(
                _string(values.get("DATABASE_SSL")), default=False, name="DATABASE_SSL"
            ),
            ssl_ca=_clean(_string(values.get("DATABASE_SSL_CA"))),
            connect_timeout=_legacy_int(
                values.get("DATABASE_CONNECT_TIMEOUT"), 30,
                "DATABASE_CONNECT_TIMEOUT",
            ),
            insert_batch_size=_legacy_int(
                values.get("DATABASE_INSERT_BATCH_SIZE"), 1000,
                "DATABASE_INSERT_BATCH_SIZE",
            ),
        )
    except ValidationError as exc:
        raise DatabaseConfigurationError(
            "Legacy database settings need correction: " + _safe_validation_message(exc)
        ) from exc

###############################################################################
def _legacy_int(value: Any, default: int, name: str) -> int:
    if value is None or not str(value).strip():
        return default
    try:
        return int(str(value).strip())
    except ValueError as exc:
        raise DatabaseConfigurationError(f"Legacy {name} must be an integer.") from exc

###############################################################################
def _string(value: Any) -> str | None:
    return None if value is None else str(value)

###############################################################################
def _clear_legacy_process_environment() -> None:
    for name in tuple(os.environ):
        if name == "EMBEDDED_DATABASE" or name.startswith("DATABASE_"):
            os.environ.pop(name, None)

###############################################################################
def _remove_legacy_environment_entries(path: Path) -> None:
    if not path.is_file():
        return
    try:
        source = path.read_text(encoding="utf-8")
        kept = [
            line
            for line in source.splitlines()
            if not re.match(
                r"^\s*(?:export\s+)?(?:EMBEDDED_DATABASE|DATABASE_[A-Z0-9_]+)\s*=",
                line,
                flags=re.IGNORECASE,
            )
        ]
        rendered = "\n".join(kept).rstrip("\n")
        if source.endswith("\n") or rendered:
            rendered += "\n"
        if rendered.encode("utf-8") != source.encode("utf-8"):
            _atomic_write(path, rendered.encode("utf-8"))
    except OSError as exc:
        raise DatabaseConfigurationError(
            "Database settings were saved, but legacy database entries could not be removed from the environment file."
        ) from exc

###############################################################################
def bootstrap_database_configuration(environment_file: Path) -> str | None:
    """Migrate supported legacy values once, then keep database keys out of process env."""
    global _BOOTSTRAP_ERROR
    with _CONFIG_LOCK:
        try:
            if DATABASE_CONFIG_FILE_PATH.is_file():
                read_database_configuration()
                _remove_legacy_environment_entries(environment_file)
                _clear_legacy_process_environment()
                _BOOTSTRAP_ERROR = None
                return None

            effective: dict[str, Any] = os.environ.copy()
            file_values = dotenv_values(environment_file)
            effective.update(
                {key: value for key, value in file_values.items() if value is not None}
            )
            configuration = _legacy_configuration(effective)
            save_database_configuration(
                configuration,
                environment_file=environment_file,
            )
            return None
        except DatabaseConfigurationError as exc:
            _BOOTSTRAP_ERROR = str(exc)
            _clear_legacy_process_environment()
            return _BOOTSTRAP_ERROR

###############################################################################
def clear_database_configuration_error() -> None:
    global _BOOTSTRAP_ERROR
    _BOOTSTRAP_ERROR = None


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "DatabaseConfiguration",
    "DatabaseConfigurationError",
    "DatabaseConfigurationStatus",
    "bootstrap_database_configuration",
    "clear_database_configuration_error",
    "load_database_settings",
    "read_database_configuration",
    "read_database_configuration_status",
    "save_database_configuration",
]
