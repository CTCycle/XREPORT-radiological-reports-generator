from __future__ import annotations

import argparse
import base64
import binascii
from io import BytesIO
import ipaddress
import os
from pathlib import Path
import re
import threading
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, status
from PIL import Image
from pydantic import BaseModel


MODEL_ID = "microsoft/maira-2"
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
MAX_IMAGE_BYTES = 64 * 1024 * 1024

# Defense in depth: Transformers and Hugging Face Hub must never use the network.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


###############################################################################
class GenerateRequest(BaseModel):
    model: Literal["microsoft/maira-2"]
    revision: str
    generation_profile: Literal["deterministic", "concise", "detailed"]
    clinical_context: str = ""
    image: str


###############################################################################
class WorkerRuntime:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.cache_dir = os.getenv("XREPORT_MAIRA2_CACHE_DIR", "").strip()
        self.revision = os.getenv("XREPORT_MAIRA2_REVISION", "").strip()
        self._lock = threading.Lock()
        self._model: Any = None
        self._processor: Any = None

    # -------------------------------------------------------------------------
    def readiness(self) -> tuple[str, str | None]:
        if not REVISION_PATTERN.fullmatch(self.revision):
            return (
                "incompatible",
                "XREPORT_MAIRA2_REVISION must be an exact 40-character commit.",
            )
        if not self.cache_dir or not (self.snapshot_path / "config.json").is_file():
            return (
                "not_installed",
                "The pinned MAIRA-2 snapshot is not present in the configured cache.",
            )
        return "ready", None

    # -------------------------------------------------------------------------
    @property
    def snapshot_path(self) -> Path:
        return (
            Path(self.cache_dir)
            / "models--microsoft--maira-2"
            / "snapshots"
            / self.revision
        )

    # -------------------------------------------------------------------------
    def generate(self, request: GenerateRequest) -> str:
        ready, message = self.readiness()
        if ready != "ready":
            raise RuntimeError(message or "MAIRA-2 worker is not ready")
        if request.revision != self.revision:
            raise ValueError("Requested revision does not match the worker revision")
        image = self._decode_image(request.image)
        model, processor = self._load()
        inputs = processor.format_and_preprocess_reporting_input(
            current_frontal=image,
            current_lateral=None,
            prior_frontal=None,
            indication=request.clinical_context.strip() or None,
            technique=None,
            comparison=None,
            prior_report=None,
            return_tensors="pt",
            get_grounding=False,
        ).to(model.device)

        import torch

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens(request.generation_profile),
                do_sample=False,
                use_cache=True,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        decoded = processor.decode(
            output[0][prompt_length:],
            skip_special_tokens=True,
        ).lstrip()
        converted = processor.convert_output_to_plaintext_or_grounded_sequence(decoded)
        findings = converted if isinstance(converted, str) else str(converted)
        if not findings.strip():
            raise RuntimeError("MAIRA-2 generated empty findings")
        return findings.strip()

    # -------------------------------------------------------------------------
    def _load(self) -> tuple[Any, Any]:
        with self._lock:
            if self._model is not None:
                return self._model, self._processor
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            if not torch.cuda.is_available():
                raise RuntimeError("MAIRA-2 worker requires a CUDA device")
            options = {
                "local_files_only": True,
                "trust_remote_code": True,
            }
            snapshot = str(self.snapshot_path)
            self._processor = AutoProcessor.from_pretrained(snapshot, **options)
            self._model = (
                AutoModelForCausalLM.from_pretrained(
                    snapshot,
                    dtype=torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16,
                    **options,
                )
                .eval()
                .to(torch.device("cuda"))
            )
            return self._model, self._processor

    # -------------------------------------------------------------------------
    @staticmethod
    def _decode_image(encoded: str) -> Image.Image:
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Image is not valid base64") from exc
        if not payload or len(payload) > MAX_IMAGE_BYTES:
            raise ValueError("Image payload must be between 1 byte and 64 MB")
        try:
            return Image.open(BytesIO(payload)).convert("RGB")
        except OSError as exc:
            raise ValueError("Image payload is not a supported image") from exc

    # -------------------------------------------------------------------------
    @staticmethod
    def _max_new_tokens(profile: str) -> int:
        return {"deterministic": 300, "concise": 200, "detailed": 300}[profile]


runtime = WorkerRuntime()
app = FastAPI(title="XREPORT MAIRA-2 Worker", docs_url=None, redoc_url=None)


###############################################################################
def _require_loopback(request: Request) -> None:
    peer = request.client.host if request.client else ""
    try:
        if ipaddress.ip_address(peer).is_loopback:
            return
    except ValueError:
        pass
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, detail="Loopback clients only"
    )


###############################################################################
@app.get("/health")
def health(request: Request) -> dict[str, str | None]:
    _require_loopback(request)
    worker_status, message = runtime.readiness()
    return {
        "status": worker_status,
        "message": message,
        "model": MODEL_ID,
        "revision": runtime.revision or None,
        "research_use_only": "true",
    }


###############################################################################
@app.post("/generate")
def generate(request: Request, payload: GenerateRequest) -> dict[str, str]:
    _require_loopback(request)
    try:
        findings = runtime.generate(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return {"findings": findings, "research_use_only": "true"}


###############################################################################
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the loopback-only XREPORT MAIRA-2 worker"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5010, type=int)
    args = parser.parse_args()
    try:
        if not ipaddress.ip_address(args.host).is_loopback:
            raise ValueError
    except ValueError as exc:
        raise SystemExit("MAIRA-2 worker host must be a loopback IP address") from exc

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
