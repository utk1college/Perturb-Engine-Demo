r"""Perturb Engine demo server. Offline, standard library HTTP server (no Flask needed).

Run from the repo root:  .venv311\Scripts\python demo_ui\app.py      then open http://127.0.0.1:8765
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never download at demo time
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import mimetypes  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402
from pathlib import Path  # noqa: E402

warnings.filterwarnings("ignore")
STATIC = Path(__file__).resolve().parent / "static"
mimetypes.add_type("font/ttf", ".ttf")
LOCK = threading.Lock()
DEMO = None
MAX_CHARS = 2000


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            path = "/index.html"
        if path == "/api/status":
            return self._json(200, {"device": DEMO.device, "load_seconds": round(DEMO.load_seconds, 1)})
        f = (STATIC / path.lstrip("/")).resolve()
        if STATIC not in f.parents or not f.is_file():
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        self._send(200, f.read_bytes(), ctype)

    def do_POST(self):
        if self.path != "/api/perturb":
            return self._send(404, b"not found", "text/plain")
        try:
            n = int(self.headers.get("Content-Length", 0))
            text = (json.loads(self.rfile.read(n) or b"{}").get("text") or "").strip()
        except Exception:
            return self._json(400, {"error": "Bad request."})
        if not text:
            return self._json(400, {"error": "Type a question first."})
        if len(text) > MAX_CHARS:
            return self._json(400, {"error": f"Please keep it under {MAX_CHARS} characters."})
        try:
            with LOCK:
                out = DEMO.run(text)
        except Exception as e:  # keep the demo alive and say what happened
            return self._json(500, {"error": f"Engine error: {type(e).__name__}: {e}"})
        self._json(200, out)


PRESETS = ["I have a fever, what should I do?",
           "Can a doctor treat a patient without consent if it is an emergency?",
           "What are the main differences between mitosis and meiosis?",
           "Does the moon have any effect on ocean tides?\nThanks in advance!"]


def _prewarm():
    """Compute the page's presets in the background so clicks on stage are instant (output is deterministic)."""
    t0 = time.time()
    for q in PRESETS:
        with LOCK:
            DEMO.run(q)
    print(f"Presets ready ({time.time() - t0:.0f}s)", flush=True)


def main():
    global DEMO
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    a = ap.parse_args()
    import torch
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    print(f"Loading spaCy, SBERT and DeBERTa NLI on {dev} (offline) ...", flush=True)
    t0 = time.time()
    from demo_engine import Demo
    DEMO = Demo(dev)
    DEMO.run("How do enzymes affect the rate of a chemical reaction?")  # warm-up so the first click is not slow
    print(f"Ready in {time.time() - t0:.1f}s  ->  http://{a.host}:{a.port}", flush=True)
    threading.Thread(target=_prewarm, daemon=True).start()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
