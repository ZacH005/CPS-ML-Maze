"""HTTPS page + orientation intake for scripts/phone_tilt_teleop.py.

Both the mobile page and the {beta, gamma} sample intake are served from the
SAME host:port over a single stdlib ThreadingHTTPServer - GET returns the
page, POST /orientation accepts a JSON sample. A single origin matters here:
a self-signed cert's browser trust exception only covers the exact host:port
the user navigated to and accepted, so a second port (e.g. a separate
WebSocket server) would need its own trust decision that the phone's browser
has no UI to grant for a background connection - it just fails silently.
"""
from __future__ import annotations

import json
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class PhoneOrientationServer:
    def __init__(
        self,
        host: str,
        http_port: int,
        ssl_context: ssl.SSLContext,
        html_path: str | Path,
    ):
        self._host = host
        self._http_port = http_port
        self._ssl_context = ssl_context
        self._html = Path(html_path).read_text(encoding="utf-8")

        self._lock = threading.Lock()
        self._beta = 0.0
        self._gamma = 0.0
        self._recv_time = 0.0
        self._has_sample = False

        self._http_server: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None

    def start(self) -> None:
        html = self._html
        ssl_context = self._ssl_context
        handshake_timeout_s = 10.0
        on_sample = self._on_sample

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self) -> None:
                # Do the TLS handshake here, inside THIS connection's own
                # ThreadingMixIn-spawned thread - not on the listening socket
                # and not inside the server's single accept loop. A stalled
                # or malformed handshake from one client (a phone browser's
                # speculative parallel connection, a stray LAN probe) then
                # only ever blocks its own thread, never other connections.
                self.request.settimeout(handshake_timeout_s)
                self.request = ssl_context.wrap_socket(self.request, server_side=True)
                self.request.settimeout(None)
                super().setup()

            def do_GET(self) -> None:  # noqa: N802 - stdlib-mandated name
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802 - stdlib-mandated name
                if self.path != "/orientation":
                    self.send_response(404)
                    self.end_headers()
                    return
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length > 0 else b""
                try:
                    data = json.loads(raw)
                    beta = float(data["beta"])
                    gamma = float(data["gamma"])
                except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                    self.send_response(400)
                    self.end_headers()
                    return
                on_sample(beta, gamma)
                self.send_response(204)
                self.end_headers()

            def log_message(self, fmt: str, *args: object) -> None:
                pass  # silence the default per-request access log

        class TLSHTTPServer(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address) -> None:
                pass  # routine failed/incomplete handshakes shouldn't spam the console

        server = TLSHTTPServer((self._host, self._http_port), Handler)
        self._http_server = server
        self._http_thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._http_thread.start()

    def latest(self) -> tuple[float, float, float] | None:
        """Return (beta_deg, gamma_deg, recv_monotonic_s), or None before the first sample."""
        with self._lock:
            if not self._has_sample:
                return None
            return self._beta, self._gamma, self._recv_time

    def stop(self) -> None:
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
        if self._http_thread is not None:
            self._http_thread.join(timeout=3.0)

    def _on_sample(self, beta: float, gamma: float) -> None:
        with self._lock:
            self._beta = beta
            self._gamma = gamma
            self._recv_time = time.monotonic()
            self._has_sample = True
