"""Regenerate the checked-in OpenAPI snapshots for both XREPORT applications."""

from __future__ import annotations

import json
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from server.app import create_app  # noqa: E402

SHARED_DIR = APP_DIR / "shared"
PROFILES = ("xreport", "studio")

###############################################################################
def main() -> None:
    SHARED_DIR.mkdir(exist_ok=True)
    for profile in PROFILES:
        schema = create_app(profile).openapi()
        target = SHARED_DIR / f"openapi.{profile}.json"
        target.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
