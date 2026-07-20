# Local Inference Models

Last updated: 2026-07-20

## Safety Scope

All catalog models and generated reports are for research use only. They are not clinically approved, and every draft requires independent review, clinical correlation, and verification by qualified personnel.

## Catalog Contract

`GET /api/inference/models` lists curated model references, provider status, local availability, input semantics, capabilities, and revisions. The catalog combines entries from `settings/inference_models.json` with XREPORT checkpoints discovered under the checkpoints resource directory. XREPORT never automatically pulls or downloads model weights. Generation accepts `model_ref`, `generation_profile`, `clinical_context`, and image uploads.

The supported provider prefixes are `xreport:`, `ollama:`, and `huggingface:`. XREPORT checkpoints support independent images (up to 16 per request); the curated Ollama and Hugging Face entries currently support one image per request. The service also enforces a 64 MiB total image-payload limit.

## Ollama

- XREPORT checks the configured loopback runtime through `/api/tags`.
- Generation uses `/api/chat`, base64 image content, explicit timeouts, and the configured keep-alive value.
- Only curated, already installed model references can run.
- The current curated Ollama models accept one image per request.

## Hugging Face MedGemma

- Repository: `google/medgemma-1.5-4b-it`.
- Access terms must be accepted and the snapshot cached outside XREPORT before use.
- Set `XREPORT_HF_CACHE_DIR` to the existing cache root.
- Set `XREPORT_HF_MEDGEMMA_REVISION` to the exact 40-character cached commit.
- XREPORT uses `local_files_only=True`, `trust_remote_code=False`, and no network fallback.
- The runtime keeps at most one Hugging Face model loaded, uses `torch.inference_mode()`, and releases model memory when switching.
- MedGemma requests accept exactly one image.

If the revision is missing, mutable, or absent from the cache, the catalog reports the model as unavailable rather than attempting a download.
