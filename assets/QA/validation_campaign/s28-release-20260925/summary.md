# Release workflow validation: S28, S34, and S53

Date: 2026-09-25
Checkout: develop, HEAD 7d8eb47165258bc0d994b3946d75a69940fbf816; working tree dirty with the concurrent-admission, training-checkbox, and launcher-cache fixes.
Environment: Windows 11 Pro build 26200, Ryzen 5 5600H, 32 GiB RAM, RTX 3060 Laptop GPU (6 GiB VRAM), SQLite, official source launcher, isolated XREPORT_RESOURCES_DIR under runtimes/cache/release-validation-20260925/resources.

## S28 — Successful checkpoint evaluation: PASS (technical synthetic workflow)

- Restored the exact pinned microsoft/beit-base-patch16-224 encoder revision f02e8f77db4703e3fbd3766e3375a4619c5a4863, model.safetensors SHA-256 873a67558f17cdd9d9018eaf308288f6dc72d030af20391df9ccbb7e10be89cc; download receipt: ../s28/encoder-download-20260925.json.
- In the Dataset UI, loaded the S27 synthetic eight-image/eight-row fixture, built s28_release_20260925 (6 train, 2 validation), selected it on Training, and completed one CUDA epoch with checkpoint XREPORT_20260925T141533.
- Selected that checkpoint in Training, started the Evaluation Report through the UI, and observed a completed job and saved report. The report rendered after closing/reopening the view and browser reload. API values: loss 2.0254554748535156, accuracy 0.8571428656578064.
- Receipts: dataset-load.json, workflow-api-receipts.json, and browser observations below.
- This confirms software-path/evaluation persistence on synthetic data only. It is not evidence of clinical quality, representative performance, or scale.

## S34 — Custom XREPORT checkpoint inference: PASS (technical synthetic workflow)

- The live inference catalogue displayed six entries (five public and one custom). Chrome selection of the ready custom checkpoint rendered its card and exact identity: provider xreport, ref xreport:XREPORT_20260925T141533, local revision not reported.
- Because browser automation could not operate the native local file picker, the synthetic S27 image was sent as multipart to the running local application inference route. The application produced a report in 4.39 seconds and stored request 3f1d150dccda.
- The Reports route showed the persisted report and generation metadata, including exact provider/model ref, 64x64 source dimensions, 224x224x3 processed tensor, keras_checkpoint, fixed_224, and xreport_beit. The same detail remained after browser reload.
- This validates a custom checkpoint's end-to-end technical adapter/report/provenance path. It does not validate a user-selected image via the native picker or report quality. The fixture is artificial and the output is not clinical.

## S53 — Performance baseline: PARTIAL

- Hardware and one inference duration are captured in performance-sample.json. The run reported a one-epoch training duration of 1 second and an inference duration of 4.3877 seconds for one synthetic image.
- The epoch value is the API's reported training elapsed field, not total job wall time. Dataset processing wall time, application startup/readiness, peak process RAM/CUDA use, and packaged startup were not measured. One epoch and one image are too small for a release performance claim.
- Further measurements require repeatable timing instrumentation against a release build, representative data, and package variants.

## Route/accessibility follow-up

Chrome direct-loaded Dataset, Training, Reports, and Settings. Dataset and training records, saved report, and General/Data access/Advanced settings rendered. The Tips & Tricks panel remained keyboard-contained through a full Tab wrap; Escape closed it and returned focus to its trigger. No Settings were changed.

Evidence files: workflow-api-receipts.json, dataset-load.json, browser-workflow-observations.json, performance-sample.json.
