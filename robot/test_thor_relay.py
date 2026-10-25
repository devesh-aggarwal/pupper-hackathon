#!/usr/bin/env python3
"""Desk tests for Gemma on the Thor (tools/thor_relay.py): no Thor, no Mac servers, no robot.

    python3 lab/test.py brain thor   # these
    python3 robot/test_thor_relay.py # the same

A fake vLLM (OpenAI's /v1, as the Thor serves it) plays Gemma from a script. In front of it the real
relay, and in front of that the real brain (robot/gemma_brain.py's Brain and Ollama client, gzip
and all) and the eyes' Gemma client, on the fake robot: what the Pupper would send, end to end.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "vision"))
_TMP = Path(tempfile.mkdtemp(prefix="pupper_thor_test_"))
os.environ.setdefault("PUPPER_MEMORY_FILE", str(_TMP / "memory.json"))
os.environ.setdefault("PUPPER_LOG_DIR", str(_TMP / "logs"))

import gemma_brain as gb  # noqa: E402
import test_tools as tt  # noqa: E402  a fake ROS, and lab/ loaded
import thor_relay  # noqa: E402
from test_tools import check  # noqa: E402

THOR_MODEL = "gemma4-31b"


class FakeVLLM:
    """/v1/models and /v1/chat/completions (not streamed), from a script: each entry is {"content": str},
    {"calls": [(name, args)]}, (status, error text), or a function of the request returning one."""

    def __init__(self, script: list):
        self.script, self.requests = list(script), []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code: int, payload) -> None:
                data = (payload if isinstance(payload, str) else json.dumps(payload)).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(200, {"data": [{"id": THOR_MODEL}]} if self.path == "/v1/models" else {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(body)
                step = fake.script.pop(0) if fake.script else {"content": ""}
                if callable(step):
                    step = step(body)
                if isinstance(step, tuple):
                    return self._send(step[0], json.dumps({"error": {"message": step[1]}}))
                msg = {"role": "assistant", "content": step.get("content")}
                if step.get("calls"):
                    msg["content"] = None
                    msg["tool_calls"] = [{"id": f"chatcmpl-tool-{len(fake.requests)}{i}", "type": "function",
                                          "function": {"name": n, "arguments": json.dumps(a)}}
                                         for i, (n, a) in enumerate(step["calls"])]
                self._send(200, {"choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": 9000, "completion_tokens": 12}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class Rig:
    """The fake Thor, the relay in front of it, and the brain on the fake robot talking to the relay."""

    def __init__(self, script: list):
        self.thor = FakeVLLM(script)
        self.lines: list[str] = []
        self.relay = thor_relay.serve(0, thor_relay.Thor(self.thor.url), log=self.lines.append)
        threading.Thread(target=self.relay.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.relay.server_port}"
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.backend = gb.DeskBackend()
        with redirect_stdout(io.StringIO()):
            self.prompt, self.decls = self.backend.setup()
        self.said, self.calls = [], []
        real = self.backend.call

        async def recording(name, args):
            self.calls.append((name, args))
            return await real(name, args)

        self.backend.call, self.backend.after = recording, lambda: None
        self.llm = gb.Ollama(self.url, gb.MODEL)
        self.brain = gb.Brain(self.llm, self.prompt, self.decls, self.backend, self.said.append, log=lambda line: None)

    def turn(self, text: str) -> None:
        self.loop.run_until_complete(self.brain.turn(text))

    def close(self) -> None:
        self.relay.shutdown()
        self.relay.server_close()
        self.thor.close()
        self.backend.rig.close()
        self.loop.run_until_complete(asyncio.sleep(0))
        self.loop.close()


def translation_tests() -> None:
    print("Thor: Ollama's messages in OpenAI's terms, and back")
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "walk then sit", "images": ["QUJD"]},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "move", "arguments": {"direction": "forward"}}},
                                                                {"function": {"name": "play_trick", "arguments": {"name": "sit"}}}]},
            {"role": "tool", "tool_name": "move", "content": '{"success": true}'},
            {"role": "tool", "tool_name": "play_trick", "content": '{"success": false}'}]
    out = thor_relay.to_openai(msgs)
    ids = [c["id"] for c in out[2]["tool_calls"]]
    check("each tool reply carries the id of its own call", [m["tool_call_id"] for m in out[3:]] == ids and len(set(ids)) == 2, out)
    check("arguments go as JSON text", out[2]["tool_calls"][0]["function"]["arguments"] == '{"direction": "forward"}')
    check("an image becomes an image part (a data URL)", out[1]["content"][1]["image_url"]["url"] == "data:image/jpeg;base64,QUJD",
          out[1])
    back = thor_relay.from_openai({"content": None, "tool_calls": [{"id": "x", "function": {"name": "turn", "arguments": '{"angle": 90}'}}]})
    check("a reply's call comes back as Ollama's, arguments as an object", back["tool_calls"][0]["function"] ==
          {"name": "turn", "arguments": {"angle": 90}} and back["content"] == "", back)


def brain_tests() -> None:
    print("Thor: the brain, through the relay, to a fake Thor")
    rig = Rig([{"calls": [("move", {"direction": "forward", "duration": 2})]}, {"content": "I walked forward."}])
    try:
        check("the Pupper's checks see Gemma: the relay lists the name it asks for, and the Thor's",
              {gb.MODEL, THOR_MODEL} <= set(rig.llm.models()), rig.llm.models())
        check("the brain gzips its requests to the relay (the Pupper's WiFi is slow)", rig.llm._relay())
        rig.turn("Walk forward for two seconds.")
        first = rig.thor.requests[0]
        check("the Thor is asked for its own model, with the tools and thinking off",
              first["model"] == THOR_MODEL and len(first.get("tools", [])) == len(rig.decls)
              and first.get("chat_template_kwargs") == {"enable_thinking": False}, {k: first.get(k) for k in ("model", "chat_template_kwargs")})
        check("the system prompt reaches the Thor", first["messages"][0]["role"] == "system" and gb.VOICE_RULES in first["messages"][0]["content"])
        check("the Thor's call runs on the robot", rig.calls == [("move", {"direction": "forward", "duration": 2})], rig.calls)
        second = rig.thor.requests[1]["messages"]
        call_id = second[-2]["tool_calls"][0]["id"]
        check("the robot's reply goes back to the Thor as the answer to that call",
              second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == call_id and '"success": true' in second[-1]["content"],
              second[-2:])
        check("Pupper says the Thor's answer", rig.said == ["I walked forward."], rig.said)
        check("the relay logs each request with its time and tokens",
              any("/api/chat" in line and "9000 in" in line and "calls ['move']" in line for line in rig.lines), rig.lines)
    finally:
        rig.close()

    rig = Rig([(400, '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set')])
    try:
        rig.turn("Stand up.")
        check("tool calling off on the Thor: Pupper says its brain isn't answering",
              rig.said == ["Sorry, my brain isn't answering right now."], rig.said)
        check("... and the relay's log says how to turn it on", any("--enable-auto-tool-choice --tool-call-parser" in line
                                                                    for line in rig.lines), rig.lines)
    finally:
        rig.close()


def eyes_tests() -> None:
    print("Thor: the eyes' questions, through the relay")
    try:
        import eyes
    except ImportError as e:
        print(f"  (skipped: vision/eyes.py needs {e.name})")
        return
    rig = Rig([{"content": "yes"}, {"content": '{"box": [0.1, 0.2, 0.3, 0.4]}'}])
    try:
        jpeg = _jpeg()
        if jpeg is None:
            print("  (skipped: needs Pillow)")
            return
        g = eyes.Gemma(rig.url, gb.MODEL)
        check("the eyes' Ollama check finds Gemma", eyes.ollama_has(rig.url, gb.MODEL))
        check("a yes-or-no question about a frame is answered", g.exists(jpeg, "cup") is True)
        asked = rig.thor.requests[0]["messages"][0]["content"]
        check("... with the frame sent as an image, and the eyes' token cap",
              asked[1]["type"] == "image_url" and rig.thor.requests[0].get("max_tokens") == 3, rig.thor.requests[0].get("max_tokens"))
        raw = g.ask(jpeg, "Where is the cup? JSON.", json_mode=True)
        check("JSON mode is asked of the Thor, and its text comes back", raw.startswith('{"box"') and
              rig.thor.requests[1].get("response_format") == {"type": "json_object"}, raw)
    finally:
        rig.close()


def _jpeg() -> bytes | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (120, 80, 40)).save(buf, "JPEG")
    return buf.getvalue()


def cli_tests() -> None:
    print("Thor: tools/thor_relay.py --check")
    script = REPO / "tools" / "thor_relay.py"
    thor = FakeVLLM([{"content": "Hello there, I am Pupper."}, {"calls": [("stand_up", {})]}])
    out = subprocess.run([sys.executable, str(script), "--thor", thor.url, "--check"], capture_output=True, text=True, timeout=60)
    check("a Thor that talks and calls tools passes", out.returncode == 0 and "tool calls" in out.stdout and "ok" in out.stdout, out.stdout)
    thor.close()
    thor = FakeVLLM([{"content": "Hi."}, (400, '"auto" tool choice requires --enable-auto-tool-choice')])
    out = subprocess.run([sys.executable, str(script), "--thor", thor.url, "--check"], capture_output=True, text=True, timeout=60)
    check("tool calling off fails, saying how to turn it on", out.returncode == 1 and "--tool-call-parser" in out.stdout, out.stdout)
    thor.close()
    out = subprocess.run([sys.executable, str(script), "--thor", "http://127.0.0.1:9", "--check"], capture_output=True, text=True, timeout=60)
    check("no Thor: it says so (same network? vLLM up?)", out.returncode == 1 and "Same network" in out.stderr, out.stderr)


GROUPS = {"translation": translation_tests, "brain": brain_tests, "eyes": eyes_tests, "cli": cli_tests}


def main(groups: list[str]) -> None:
    for name in groups or GROUPS:
        if name not in GROUPS:
            sys.exit(f"unknown thor test group {name!r}; choose from {list(GROUPS)}")
        GROUPS[name]()
        print()
    if tt.FAIL:
        print(f"{tt.FAIL} check(s) failed, {tt.PASS} passed.")
        raise SystemExit(1)
    print(f"All {tt.PASS} thor checks passed.")


if __name__ == "__main__":
    main(sys.argv[1:])
