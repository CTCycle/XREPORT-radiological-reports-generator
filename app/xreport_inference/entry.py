"""Source-mode launcher entry point for XREPORT Inference."""

from __future__ import annotations

import os

os.environ["XREPORT_APP"] = "xreport"

from xreport_shared.source_server import serve  # noqa: E402


###############################################################################
def main() -> None:
    from xreport_inference.app import app

    serve(app, port=int(os.getenv("FASTAPI_PORT", "5003")))


if __name__ == "__main__":
    main()
