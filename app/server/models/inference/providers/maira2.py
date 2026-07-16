from __future__ import annotations

import base64
from collections.abc import Callable
import ipaddress
import re
from urllib.parse import urlsplit

import httpx

from server.configurations import InferenceSettings
from server.domain.inference import GenerationProfile, InferenceImage


MODEL_ID = "microsoft/maira-2"
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


###############################################################################
class Maira2Provider:
    """Client for the separately managed, loopback-only MAIRA-2 worker."""

    # -------------------------------------------------------------------------
    def __init__(self, settings: InferenceSettings) -> None:
        self.settings = settings
        self._validate_loopback_url(settings.maira2_worker_url)

    # -------------------------------------------------------------------------
    def availability(self) -> tuple[str, str | None]:
        if not self.settings.maira2_enabled:
            return "disabled", "MAIRA-2 worker integration is disabled."
        if not self.is_pinned_revision(self.settings.maira2_revision):
            return (
                "incompatible",
                "XREPORT_MAIRA2_REVISION must be an exact 40-character cached commit.",
            )
        try:
            response = httpx.get(
                f"{self.settings.maira2_worker_url.rstrip('/')}/health",
                timeout=httpx.Timeout(3.0),
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            return "runtime_unavailable", "The isolated MAIRA-2 worker is not running."
        if not isinstance(payload, dict) or payload.get("model") != MODEL_ID:
            return (
                "incompatible",
                "The worker did not identify the curated MAIRA-2 model.",
            )
        if payload.get("revision") != self.settings.maira2_revision:
            return (
                "incompatible",
                "The worker revision does not match XREPORT_MAIRA2_REVISION.",
            )
        if payload.get("status") != "ready":
            return "not_installed", str(
                payload.get("message") or "The pinned MAIRA-2 snapshot is unavailable."
            )
        return "ready", None

    # -------------------------------------------------------------------------
    def generate(
        self,
        *,
        repository_id: str,
        revision: str,
        profile: GenerationProfile,
        clinical_context: str,
        images: list[InferenceImage],
        should_stop: Callable[[], bool],
        report_progress: Callable[[int, int, dict[str, str]], None],
    ) -> dict[str, str]:
        if repository_id != MODEL_ID:
            raise ValueError(f"MAIRA-2 model is not allowlisted: {repository_id}")
        if revision != self.settings.maira2_revision or not self.is_pinned_revision(
            revision
        ):
            raise RuntimeError("MAIRA-2 requires the configured pinned worker revision")
        if len(images) != 1:
            raise ValueError("MAIRA-2 accepts exactly one frontal chest X-ray")
        status, message = self.availability()
        if status != "ready":
            raise RuntimeError(message or f"MAIRA-2 worker is not ready ({status})")
        if should_stop():
            return {}

        payload = {
            "model": repository_id,
            "revision": revision,
            "generation_profile": profile,
            "clinical_context": clinical_context,
            "image": base64.b64encode(images[0].data).decode("ascii"),
        }
        try:
            response = httpx.post(
                f"{self.settings.maira2_worker_url.rstrip('/')}/generate",
                json=payload,
                timeout=httpx.Timeout(
                    connect=5.0,
                    read=float(self.settings.model_timeout),
                    write=30.0,
                    pool=5.0,
                ),
            )
            response.raise_for_status()
            body = response.json()
        except httpx.ConnectError as exc:
            raise RuntimeError(
                "MAIRA-2 worker became unavailable during generation"
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError("MAIRA-2 generation timed out") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError(f"MAIRA-2 generation failed: {exc}") from exc
        report = body.get("findings") if isinstance(body, dict) else None
        if not isinstance(report, str) or not report.strip():
            raise RuntimeError("MAIRA-2 worker returned empty findings")
        if should_stop():
            return {}
        reports = {images[0].filename: f"Findings\n{report.strip()}"}
        report_progress(1, 1, reports)
        return reports

    # -------------------------------------------------------------------------
    @staticmethod
    def is_pinned_revision(revision: str | None) -> bool:
        return bool(revision and REVISION_PATTERN.fullmatch(revision))

    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_loopback_url(url: str) -> None:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "http"
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("MAIRA-2 worker URL must be a plain loopback HTTP origin")
        hostname = parsed.hostname
        try:
            if hostname is not None and ipaddress.ip_address(hostname).is_loopback:
                return
        except ValueError:
            pass
        raise ValueError("MAIRA-2 worker URL must use a loopback address")

    # -------------------------------------------------------------------------
    def unload(self) -> None:
        """The isolated worker owns model lifecycle."""
