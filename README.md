# XREPORT Radiological Reports Generator

[![Release](https://img.shields.io/github/v/release/CTCycle/XREPORT-radiological-reports-generator?display_name=tag)](https://github.com/CTCycle/XREPORT-radiological-reports-generator/releases)
[![Python](https://img.shields.io/badge/python-%3E%3D3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Node.js](https://img.shields.io/badge/node.js-22.x-339933?logo=node.js)](https://nodejs.org/)
[![License](https://img.shields.io/github/license/CTCycle/XREPORT-radiological-reports-generator.svg)](LICENSE)
[![CI](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/CTCycle/XREPORT-radiological-reports-generator/actions/workflows/ci.yml)

Last updated: 2026-10-18

XREPORT is a local-first research application for preparing image/report
datasets, training and evaluating local models, and generating editable draft
radiology reports from X-ray images.

> **Research use only.** XREPORT is not a diagnostic device or a clinically
> approved reporting system. Models can omit findings or generate plausible
> but incorrect text. A qualified professional must independently review every
> draft before it informs any research or clinical decision.

## Main workflows

1. **Prepare data:** import images and a report table, review matches, and
   process the records you intend to use.
2. **Train or evaluate:** run a local training experiment, resume a checkpoint,
   or evaluate a saved model.
3. **Generate and review:** select a ready model, add de-identified images from
   one study, generate a draft, and review its declared report sections and
   provenance.

The model catalogue identifies each model's anatomy, image limit, readiness,
resource needs, access requirements, and output contract. Public models may
need an internet connection and additional storage for their first download;
verified local copies can be reused later. Model metrics and technical checks
do not establish clinical accuracy or safety.

## Start XREPORT

### Windows source launch

From the repository root, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1
```

Choose **Launch application**. The launcher prepares local runtimes as needed,
starts the backend and frontend, and opens the browser when the interface is
available. First launch can take several minutes. For the manual macOS/Linux
setup and operator details, see [Getting started](assets/docs/operations/getting_started.md).

### Windows desktop packages

Optional CPU and CUDA portable or MSI packages are published on
[GitHub Releases](https://github.com/CTCycle/XREPORT-radiological-reports-generator/releases).
Use CPU for general Windows hardware and CUDA only with supported NVIDIA
hardware and drivers. Portable packages require Microsoft WebView2. The
application keeps user data separately from the installed program.

## Data and safe use

Use only data you are authorized to process, and de-identify images and reports
before import. XREPORT keeps application data on the local machine by default;
public models are downloaded to local storage when selected. Back up important
datasets and checkpoints before resetting or removing application data.

Treat every generated report as an unverified draft. Check the text against
the source images and relevant clinical context, correct or discard unsupported
statements, and follow your institution's review and approval process.

## Application screenshots

These screenshots show representative interface workflows. They are not
clinical or model-quality evidence.

![Dataset image viewer](assets/figures/readme-dataset.png)

![Checkpoint evaluation report](assets/figures/readme-training.png)

![Public inference model catalogue](assets/figures/readme-inference.png)

![Inference image workflow](assets/figures/readme-inference-workflow.png)

![Inference draft review](assets/figures/readme-inference-report.png)

![Help and tips](assets/figures/readme-help-and-tips.png)

## Documentation

- [Documentation index](assets/docs/project_index.md) — documentation map and
  current project status.
- [Core workflows](assets/docs/operations/workflows.md) — dataset, training,
  inference, and validation journeys.
- [Troubleshooting](assets/docs/operations/troubleshooting.md) — startup,
  database, and common runtime problems.
- [Runtime configuration](assets/docs/runtime/configuration.md) — data roots,
  environment variables, and settings.
- [Local inference models](assets/docs/runtime/local_inference_models.md) —
  model catalogue, access, and technical limits.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
