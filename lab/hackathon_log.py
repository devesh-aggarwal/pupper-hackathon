"""Hackathon: the session log. Everything that happened, with pictures, for reading afterwards.

"Did the tool even run?" needs a record that survives the demo: every tool call and its reply,
every event the robot raised (an obstacle stop, a search result, a reminder), and what was said
in both directions (the web app posts Gemini's transcriptions to POST /transcript), each with the
camera frame of that moment where there is one.

  record(kind, **fields)   one JSON line in <log dir>/<day>/transcript.jsonl: t, clock, kind, fields
  save_frame(jpeg, tag)    the frame as <log dir>/<day>/frames/<clock>_<tag>.jpg (the newest
                           KEEP_FRAMES are kept)
  tail(n)                  the last n lines, for GET /transcript

The log dir is $PUPPER_LOG_DIR, else ~/pupper-logs (not /tmp: that is wiped at every boot).
Read it on the robot (python3 lab/hackathon_log.py [n]) or fetch it (./tools/pup get ~/pupper-logs .).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

log = logging.getLogger("transcript")

LOG_DIR = Path(os.environ.get("PUPPER_LOG_DIR", str(Path.home() / "pupper-logs")))
KEEP_FRAMES = 600
_last_prune = 0.0


def day_dir() -> Path:
    return LOG_DIR / time.strftime("%Y-%m-%d")


def transcript_file() -> Path:
    return day_dir() / "transcript.jsonl"


def record(kind: str, **fields) -> dict:
    """Append one line. Never raises: a full disk must not break a tool call."""
    entry = {"t": round(time.time(), 3), "clock": time.strftime("%H:%M:%S"), "kind": kind, **fields}
    try:
        f = transcript_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        with f.open("a") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except OSError as e:
        log.warning("could not write the transcript: %s", e)
    return entry


def save_frame(jpeg: bytes | None, tag: str) -> str | None:
    """Write the frame of this moment next to the transcript; the path, or None."""
    global _last_prune
    if not jpeg:
        return None
    tag = "".join(c if c.isalnum() else "_" for c in tag)[:30] or "frame"
    try:
        d = day_dir() / "frames"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{time.strftime('%H%M%S')}_{int(time.time() * 1000) % 1000:03d}_{tag}.jpg"
        path.write_bytes(jpeg)
        if time.time() - _last_prune > 60:  # keep the newest KEEP_FRAMES
            _last_prune = time.time()
            frames = sorted(d.glob("*.jpg"))
            for old in frames[:-KEEP_FRAMES]:
                old.unlink(missing_ok=True)
        return str(path)
    except OSError as e:
        log.warning("could not save a frame: %s", e)
        return None


def tail(n: int = 50) -> list[dict]:
    """The last n entries of today's transcript (oldest first)."""
    try:
        lines = transcript_file().read_text().splitlines()[-max(1, n):]
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def describe(entry: dict) -> str:
    """One entry as a line a person reads: time, who or what, and the gist."""
    k = entry.get("kind")
    if k == "tool":
        ok = "ok" if entry.get("ok") else "FAILED"
        return f"{entry['clock']}  tool {entry.get('name')}({entry.get('args')}) -> {ok}: {entry.get('message', '')}"
    if k == "speech":
        return f"{entry['clock']}  {entry.get('who', '?')}: {entry.get('text', '')}"
    if k == "event":
        return f"{entry['clock']}  event {entry.get('event')}: {entry.get('text', '')}"
    return f"{entry['clock']}  {k}: " + ", ".join(f"{a}={b}" for a, b in entry.items() if a not in ("t", "clock", "kind"))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    for e in tail(n):
        print(describe(e))
    print(f"({transcript_file()})")
