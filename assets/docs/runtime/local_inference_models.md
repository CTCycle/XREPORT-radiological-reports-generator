# Local Inference Models

Last updated: 2026-08-02

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
- The catalogue resolves `disabled`, `incompatible`, `gated`, `not_installed`, `unvalidated`, and `ready` states with explicit reasons. A Hugging Face model is `ready` only when its exact snapshot is complete, its runtime contract is valid, and an exact-revision receipt under `assets/QA/inference_validation/` records real inference. A manifest status flag alone is not evidence.

### MedGemma 1.5 4B

Repository: `google/medgemma-1.5-4b-it`.

Access requires acceptance of the Health AI Developer Foundations terms of use. The terms are shown in the model catalogue. Set `HF_CACHE_DIR` to the existing local cache root. The pinned revision is maintained in `settings/inference_models.json`, not in environment configuration.

MedGemma is retained as a standard loader path but remains gated and unvalidated. Its output contract is `raw_report`; XREPORT does not synthesize Findings or Impression sections for models whose manifest does not provide them.

### Configured Hugging Face catalogue

| Model | Exact revision | License | Loader / adapter | Contract | Initial state |
| --- | --- | --- | --- | --- | --- |
| MedGemma 1.5 4B | `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` | Health AI Developer Foundations terms | `AutoModelForImageTextToText` + `AutoProcessor` / `medgemma` | one image, raw report, BF16, gated terms | `gated` |
| MAIRA-2 | `795a2b1cd4a310624b4e3d14b5a23e41fd273deb` | MSRLA | metadata-only causal-LM/custom processor | frontal+lateral findings; remote code unapproved; Transformers `<4.52` | `incompatible` |
| CheXagent Impression | `4053d2f16626c1c355aa2b08c4c047beed98f94d` | CC-BY-4.0 | metadata-only custom path; pinned StanfordAIMI processor `8f19b53a2eceda4c33b0acec6c81fbc293ad80d0` | one image, impression | `disabled` |
| CXRMate-2 | `aa8e2d16470e20671acf049687b4707c9bf2f2b5` | Apache-2.0 | metadata-only custom causal-LM/processor | findings and impression; remote code unapproved | `disabled` |
| generate-cxr | `6609ed3b711769816141f0f6fdaa88310e1ea0cb` | Apache-2.0 | `BlipForConditionalGeneration` + `BlipProcessor` / `generate_cxr_blip` | one image, raw report, `indication:` prefix, FP32, max 512-token sequence | `not_installed` |

Only the standard MedGemma adapter path and the focused public BLIP adapter are implemented. No model weights, gated terms, auxiliary repositories, or model-specific dependency stacks are added.

### BLIP `generate-cxr`

The adapter calls `BlipProcessor(images=image, text="indication:" + indication, return_tensors="pt")` and decodes the complete encoder-decoder sequence. It does not remove prompt tokens. Reports are rejected when empty or malformed, and the exact decoded text is persisted unchanged. Display editors use only sections declared by the manifest.

Run the cache-only validator with an existing public/de-identified fixture:

```powershell
$env:KERAS_BACKEND = "torch"
app/server/.venv/Scripts/python.exe app/scripts/validate_inference_model.py --image C:/path/to/cxr.png
```

The command writes only JSON logs/receipts under `assets/QA/`; it never downloads weights or promotes the manifest. With the current unset `HF_CACHE_DIR`, validation is deferred and the ready count remains zero.

## Custom XREPORT checkpoint

The custom XReport checkpoint is a separate Keras report generator. Its image path uses `BeitXRayImageEncoder` with the `microsoft/beit-base-patch16-224` encoder contract and deterministic 224-pixel preprocessing. This fixed path must remain unchanged unless the model is retrained.
