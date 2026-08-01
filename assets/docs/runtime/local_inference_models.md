# Local Inference Models

Last updated: 2026-08-01

## Safety Scope

All catalogue models and generated reports are for research use only. They are not clinically approved, and every draft requires independent review, clinical correlation, and verification by qualified personnel.

## Catalogue Contract

`GET /api/inference/models` lists curated model references, provider status, local availability, input semantics, explicit output sections, processor metadata, and pinned revisions. The catalogue combines `settings/inference_models.json` with XREPORT checkpoints discovered under the checkpoints resource directory. XREPORT never downloads weights automatically.

The normal external path is one embedded Hugging Face Transformers provider. It does not require Ollama, llama.cpp, vLLM, or a separate model server. The service keeps one embedded model loaded at a time and unloads it before switching revisions.

The supported provider prefixes are `xreport:` and `huggingface:`. XREPORT checkpoints support up to 16 independent current images and retain their trained BEiT `224×224×3` preprocessing. Each external catalogue entry declares its own current-image limit and processor contract.

## Embedded Hugging Face Transformers

- Every entry declares a repository, exact 40-character commit revision, model loader, processor loader, adapter, dtype, quantization policy, prompt profile, output sections, and capability flags.
- The runtime uses `local_files_only=true`; it never resolves mutable refs or downloads weights during catalogue reads or generation.
- `trust_remote_code=true` is accepted only for an individually approved, revision-pinned manifest entry.
- Uploaded images are decoded with EXIF orientation applied, converted to RGB when needed, and passed to the selected processor at their original decoded resolution. The processor owns resizing, padding, cropping, rescaling, and normalization.
- Inference metadata records each image's original width and height plus the processed tensor dimensions returned by the processor.
- The catalogue does not mark a candidate `ready` until a real chest X-ray has produced non-empty, clinically structured report text. The MedGemma entry remains a candidate until that validation is recorded in the manifest.

### MedGemma 1.5 4B

Repository: `google/medgemma-1.5-4b-it`.

Access requires acceptance of the Health AI Developer Foundation terms of use. The terms are shown in the model catalogue. Set `HF_CACHE_DIR` to the existing local cache root. The pinned revision is maintained in `settings/inference_models.json`, not in environment configuration.

MedGemma is validated primarily for single-image tasks in this integration, so its current-image limit remains one and its output contract explicitly includes Findings and Impression. XReport does not synthesize or relabel sections for models whose manifest does not provide them.

## Custom XREPORT checkpoint

The custom XReport checkpoint is a separate Keras report generator. Its image path uses `BeitXRayImageEncoder` with the `microsoft/beit-base-patch16-224` encoder contract and deterministic 224-pixel preprocessing. This fixed path must remain unchanged unless the model is retrained.
