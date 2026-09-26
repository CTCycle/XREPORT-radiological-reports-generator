# S35/S36 validation commands

All model work used the exact revisions in `case-manifest.json`, the isolated
resource root below, and sequential single-image cases. No command accepted
gated terms or accessed MedGemma.

```powershell
$repo = 'G:\Projects\Repositories\Active projects\XREPORT Radiological Reports'
$isolated = "$repo\runtimes\validation-s35-s36-20260926"
$env:XREPORT_RESOURCES_DIR = "$isolated\resources"
$env:XREPORT_VALIDATION_CACHE_ROOT = "$isolated\cache"
```

The exact model maintenance requests were:

```powershell
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:5003/api/inference/models/maintenance' -ContentType 'application/json' -Body (@{ model_ref = 'huggingface:aehrc/cxrmate-ed'; action = 'download'; revision = '68251c7605067ddbea330413aade032713fd2192' } | ConvertTo-Json)
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:5003/api/inference/models/maintenance' -ContentType 'application/json' -Body (@{ model_ref = 'huggingface:StanfordAIMI/CheXOne'; action = 'download'; revision = '0c350e6852ea08f9d9baf3b7595c1a10d4849927' } | ConvertTo-Json)
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:5003/api/inference/models/maintenance' -ContentType 'application/json' -Body (@{ model_ref = 'huggingface:aehrc/cxrmate-2'; action = 'download'; revision = 'aa8e2d16470e20671acf049687b4707c9bf2f2b5' } | ConvertTo-Json)
```

The S35 wrapper retained its exact existing case matrix:

```powershell
& "$repo\app\server\.venv\Scripts\python.exe" "$repo\app\scripts\validate_cxrmate_ed_sensitivity.py" `
  --image-dir "$repo\assets\QA\inference_validation_runs" `
  --fixture-provenance 'Public COVID-19 Image Data Collection; ieee8023/covid-chestxray-dataset; local approved PNG fixtures 000001-1.png, 000001-2.png, 000001-3.png; https://github.com/ieee8023/covid-chestxray-dataset' `
  --fixture-deidentification 'Public research fixtures; no direct identifiers; local validation only.' `
  --output "$isolated\cxrmate-ed-s35-receipt.json"
```

The shared independent-case runner was invoked for S36A and S36B:

```powershell
& "$repo\app\server\.venv\Scripts\python.exe" "$repo\app\scripts\validate_inference_model.py" --model-ref 'huggingface:StanfordAIMI/CheXOne' --case-manifest "$repo\assets\QA\validation_campaign\s35-s36-20260926\case-manifest.json" --receipt-path "$isolated\chexone-aggregate-technical-receipt.json" --fixture-provenance 'Public COVID-19 Image Data Collection; ieee8023/covid-chestxray-dataset; local approved PNG fixtures 000001-1.png, 000001-2.png, 000001-3.png; https://github.com/ieee8023/covid-chestxray-dataset' --fixture-deidentification 'Public research fixtures; no direct identifiers; local validation only.'
& "$repo\app\server\.venv\Scripts\python.exe" "$repo\app\scripts\validate_inference_model.py" --model-ref 'huggingface:aehrc/cxrmate-2' --case-manifest "$repo\assets\QA\validation_campaign\s35-s36-20260926\case-manifest.json" --receipt-path "$isolated\cxrmate2-aggregate-technical-receipt.json" --fixture-provenance 'Public COVID-19 Image Data Collection; ieee8023/covid-chestxray-dataset; local approved PNG fixtures 000001-1.png, 000001-2.png, 000001-3.png; https://github.com/ieee8023/covid-chestxray-dataset' --fixture-deidentification 'Public research fixtures; no direct identifiers; local validation only.'
```

Focused verification used the existing `app/server/.venv`, an isolated
`PYTHONPYCACHEPREFIX`/pytest cache, the adapter/provider/catalogue/manifest
tests, the new independent-case tests, and the adjacent S30/request-validation
regression files. The final test commands and counts are recorded in
`summary.md` after completion.
