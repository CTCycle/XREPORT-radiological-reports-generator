from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal, cast


InferenceDevice = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True)
class CorsSettings:
    allowed_origins: tuple[str, ...]
    allow_credentials: bool
    allow_methods: tuple[str, ...]
    allow_headers: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeEnvironmentSettings:
    inference_hf_local_only: bool
    inference_device: InferenceDevice
    cors: CorsSettings


def _environment_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be set to true or false")


def _environment_list(name: str, default: str = "") -> tuple[str, ...]:
    value = os.getenv(name, default)
    return tuple(item.strip() for item in value.split(",") if item.strip())


def get_runtime_environment_settings() -> RuntimeEnvironmentSettings:
    raw_device = os.getenv("INFERENCE_DEVICE", "auto").strip().lower() or "auto"
    if raw_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("INFERENCE_DEVICE must be auto, cpu, or cuda")
    device = cast(InferenceDevice, raw_device)

    allow_credentials = _environment_bool("CORS_ALLOW_CREDENTIALS", False)
    allowed_origins = _environment_list("CORS_ALLOWED_ORIGINS")
    if allow_credentials and "*" in allowed_origins:
        raise ValueError(
            "CORS_ALLOWED_ORIGINS cannot include '*' when "
            "CORS_ALLOW_CREDENTIALS is true"
        )

    return RuntimeEnvironmentSettings(
        inference_hf_local_only=_environment_bool(
            "INFERENCE_HF_LOCAL_ONLY", True
        ),
        inference_device=device,
        cors=CorsSettings(
            allowed_origins=allowed_origins,
            allow_credentials=allow_credentials,
            allow_methods=_environment_list(
                "CORS_ALLOW_METHODS", "GET,POST,PUT,PATCH,DELETE,OPTIONS"
            ),
            allow_headers=_environment_list(
                "CORS_ALLOW_HEADERS",
                "Authorization,Content-Type,X-XREPORT-Desktop-Token",
            ),
        ),
    )
