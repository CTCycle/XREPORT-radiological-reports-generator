# Local Inference Models

Last updated: 2026-07-16

## Safety Scope

All catalog models and generated reports are for research use only. They are not clinically approved, and every draft requires independent review, clinical correlation, and verification by qualified personnel.

## Catalog Contract

`GET /api/inference/models` lists curated model references and their local availability. XREPORT never automatically pulls or downloads model weights. Generation accepts `model_ref`, `generation_profile`, `clinical_context`, and image uploads.

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

## MAIRA-2 Isolated Worker

- Repository: `microsoft/maira-2`, gated under the Microsoft Research License Agreement.
- MAIRA-2 is specialized for adult chest X-rays and generates a Findings draft. Other anatomy and modalities are out of scope.
- Its pinned custom model code runs only in the separate `app/maira2_worker` process because the official runtime contract requires `trust_remote_code=True` and Transformers 4.51.3.
- The worker accepts loopback IP clients only, defaults to `127.0.0.1:5010`, requires CUDA, and never starts automatically.
- Set `XREPORT_MAIRA2_CACHE_DIR` and the same exact `XREPORT_MAIRA2_REVISION` in the worker environment. Set `XREPORT_MAIRA2_ENABLED=true` in the main backend only after the worker is ready.
- Both Hugging Face offline flags are forced in the worker, and model/processor loading uses the pinned snapshot directory with `local_files_only=True`. There is no download fallback.

Start the isolated environment from the repository root after preparing the pinned cache:

```powershell
uv run --project app/maira2_worker python -m maira2_worker.app
```

The main backend rejects non-loopback worker URLs and accepts only `maira2:microsoft/maira-2` at the configured revision. All MAIRA-2 output remains research-only and requires qualified independent review; it is not clinically approved.
