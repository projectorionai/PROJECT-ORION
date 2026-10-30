"""Serve only the signed Android installer to a temporary HTTPS tunnel."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


APK_PATH = Path(__file__).resolve().parents[1] / "releases" / "android" / "orion-companion.apk"
DOWNLOAD_PATH = "/download/orion-companion.apk"


class ApkHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        self._send_apk(include_body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._send_apk(include_body=False)

    def _send_apk(self, *, include_body: bool) -> None:
        if urlsplit(self.path).path != DOWNLOAD_PATH:
            self.send_error(404)
            return
        try:
            size = APK_PATH.stat().st_size
            if not APK_PATH.is_file() or size <= 0:
                raise OSError("Installer is unavailable")
        except OSError:
            self.send_error(503, "Installer is unavailable")
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/vnd.android.package-archive")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", 'attachment; filename="orion-companion.apk"')
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if include_body:
            with APK_PATH.open("rb") as source:
                while chunk := source.read(65536):
                    self.wfile.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve only the ORION Android installer")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), ApkHandler)
    print(f"ORION installer available locally on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
