from __future__ import annotations

import argparse
import functools
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


# -----------------------------------------------------------------------------
class SpaRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.should_serve_index():
            self.path = "/index.html"
        super().do_GET()

    # -------------------------------------------------------------------------
    def do_HEAD(self) -> None:
        if self.should_serve_index():
            self.path = "/index.html"
        super().do_HEAD()

    # -------------------------------------------------------------------------
    def should_serve_index(self) -> bool:
        requested_path = urlparse(self.path).path
        if requested_path == "/" or requested_path.startswith("/api/"):
            return False

        relative_path = requested_path.lstrip("/")
        target_path = os.path.join(self.directory or os.getcwd(), relative_path)
        return not os.path.isfile(target_path)


# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a SPA bundle with client-side route fallback."
    )
    parser.add_argument("--directory", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    return parser.parse_args()


# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    handler = functools.partial(SpaRequestHandler, directory=args.directory)
    with ThreadingHTTPServer((args.host, args.port), handler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
