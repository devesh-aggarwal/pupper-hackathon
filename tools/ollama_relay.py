#!/usr/bin/env python3
"""Ollama behind a gzip relay, on the Mac:   python3 tools/ollama_relay.py [--port 11435] [--ollama URL]

The Pupper's WiFi sends slowly (a public hotspot measured ~11 KB/s up), and every brain turn sends
the whole conversation to Ollama: the prompt and 30 tools, ~60 KB of JSON, several seconds. JSON
gzips to about a fifth, but Ollama can't read a gzipped request. So the Pupper's brain gzips
(robot/gemma_brain.py, when GET /relay answers) and this relay unzips and passes the request on.
Plain requests (the eyes, ollama commands) pass through as they are. tools/brain_on_mac.sh runs it.
"""

import argparse
import gzip
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=11435, help="where the Pupper reaches Ollama (through the link)")
    ap.add_argument("--ollama", default="http://127.0.0.1:11437", help="the real Ollama")
    args = ap.parse_args()
    target = args.ollama.rstrip("/")

    class Relay(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _reply(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _pass(self, method: str) -> None:
            if self.path == "/relay":  # how the brain knows it may gzip
                return self._reply(200, json.dumps({"gzip": True}).encode())
            body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0)) if method == "POST" else None
            if body and self.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            req = urllib.request.Request(target + self.path, data=body, method=method,
                                         headers={"Content-Type": self.headers.get("Content-Type", "application/json")})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    self._reply(r.status, r.read(), r.headers.get("Content-Type", "application/json"))
            except urllib.error.HTTPError as e:
                self._reply(e.code, e.read(), e.headers.get("Content-Type", "application/json"))
            except (urllib.error.URLError, OSError) as e:
                self._reply(502, json.dumps({"error": f"relay: Ollama at {target} did not answer: {e}"}).encode())

        def do_GET(self):
            self._pass("GET")

        def do_POST(self):
            self._pass("POST")

        def do_HEAD(self):
            self._reply(200, b"")

    print(f"relay on 127.0.0.1:{args.port} -> {target} (gzip requests welcome)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Relay).serve_forever()


if __name__ == "__main__":
    main()
