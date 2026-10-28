"""Source-mode launcher entry point for XREPORT Studio."""

from __future__ import annotations

import os

os.environ["XREPORT_APP"] = "studio"
# Studio trains with Keras on PyTorch; set it before Keras is imported so neither
# the server nor its spawned training workers fall back to TensorFlow.
os.environ["KERAS_BACKEND"] = "torch"

from xreport_shared.source_server import serve  # noqa: E402


def main() -> None:
    from xreport_studio.app import app

    serve(app, port=int(os.getenv("STUDIO_FASTAPI_PORT", "5013")))


if __name__ == "__main__":
    main()
