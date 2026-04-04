#!/usr/bin/env python3
"""
Dev server — serves viewer.html and exposes a /run endpoint that triggers
scraper.py for a single airline and streams its log output back as SSE.

Usage:
  python server.py
  open http://localhost:5173
"""

import asyncio
import csv
import io
import json
import os
import queue
import threading
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, send_file, stream_with_context

load_dotenv()

BASE = Path(__file__).parent
app = Flask(__name__)


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_file(BASE / "viewer.html")


@app.route("/airlines.csv")
def airlines_csv():
    path = BASE / "airlines.csv"
    if not path.exists():
        return ("airlines.csv not found", 404)
    return send_file(path, mimetype="text/csv")


@app.route("/results.csv")
def results_csv():
    path = BASE / "results.csv"
    if not path.exists():
        return ("", 204)  # not yet scraped — that's fine
    return send_file(path, mimetype="text/csv")


# ---------------------------------------------------------------------------
# Run scraper for a single airline — streams log lines as SSE
# ---------------------------------------------------------------------------

@app.route("/run/<airline_id>")
def run_scraper(airline_id: str):
    """
    Runs scraper.py --airline <id> in a background thread and streams its
    stdout/stderr back as Server-Sent Events.

    The client receives:
      data: <log line>           — progress messages
      data: __DONE__             — scrape finished successfully
      data: __ERROR__ <message>  — scrape failed
    """
    # Validate the airline id exists in airlines.csv
    airlines_path = BASE / "airlines.csv"
    if not airlines_path.exists():
        return jsonify({"error": "airlines.csv not found"}), 404

    with open(airlines_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ids = [r.get("id", "").strip() for r in rows]
    if airline_id not in ids:
        return jsonify({"error": f"Unknown airline id: {airline_id}"}), 400

    log_q: queue.Queue = queue.Queue()

    def worker():
        import subprocess, sys
        cmd = [sys.executable, str(BASE / "scraper.py"), "--airline", airline_id]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(BASE),
            )
            for line in proc.stdout:
                log_q.put(line.rstrip())
            proc.wait()
            if proc.returncode == 0:
                log_q.put("__DONE__")
            else:
                log_q.put(f"__ERROR__ Process exited with code {proc.returncode}")
        except Exception as exc:
            log_q.put(f"__ERROR__ {exc}")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def generate():
        while True:
            try:
                line = log_q.get(timeout=120)
            except queue.Empty:
                yield "data: __ERROR__ Timed out waiting for scraper\n\n"
                break
            yield f"data: {line}\n\n"
            if line.startswith("__DONE__") or line.startswith("__ERROR__"):
                break

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5173))
    print(f"BagCheck viewer → http://localhost:{port}")
    app.run(debug=False, port=port, threaded=True)
