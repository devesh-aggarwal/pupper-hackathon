#!/usr/bin/env python3
"""Desk tests for the local brain (robot/gemma_brain.py): no Mac servers, no Ollama, no mic, no robot.

    python3 lab/test.py brain          # these
    python3 robot/test_gemma_brain.py  # the same

A fake Ollama (a little HTTP server here) plays Gemma's side from a script, and the fake robot
from test_tools.py runs the real lab/ tools, so a check sees the whole loop: what Gemma was
sent, which tool ran, and the reply that went back to Gemma.
"""

from __future__ import annotations

import array
import asyncio
import json
import math
import os
import subprocess
import urllib.request as urllib_request
import sys
import tempfile
import threading
import wave
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_TMP = Path(tempfile.mkdtemp(prefix="pupper_brain_test_"))
os.environ.setdefault("PUPPER_MEMORY_FILE", str(_TMP / "memory.json"))
os.environ.setdefault("PUPPER_LOG_DIR", str(_TMP / "logs"))

import gemma_brain as gb  # noqa: E402
import test_tools as tt  # noqa: E402  a fake ROS, and lab/ loaded
from test_tools import check  # noqa: E402


# ============================================================ a fake Ollama
class FakeOllama:
    """Answers /api/chat from a script: each entry is a reply message, (status, error text), or a
    function of the request body returning either. Every request is kept, to look at."""

    def __init__(self, script: list, models: tuple = ("gemma4:26b",)):
        self.script, self.models, self.requests = list(script), models, []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code: int, payload) -> None:
                data = json.dumps(payload).encode() if not isinstance(payload, str) else payload.encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(200, {"models": [{"name": n} for n in fake.models]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(body)
                step = fake.script.pop(0) if fake.script else {"content": ""}
                if callable(step):
                    step = step(body)
                if isinstance(step, tuple):
                    return self._send(step[0], json.dumps({"error": step[1]}))
                self._send(200, {"message": {"role": "assistant", **step}, "done": True})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def call(name: str, **args) -> dict:
    return {"function": {"name": name, "arguments": args}}


class Rig:
    """A brain on the fake robot, talking to a fake Ollama; what Pupper says is collected."""

    def __init__(self, script: list, **brain):
        self.ollama = FakeOllama(script)
        self.loop = asyncio.new_event_loop()  # one loop for the rig: the fake robot's job queue lives on it
        asyncio.set_event_loop(self.loop)
        self.backend = gb.DeskBackend()
        self.prompt, self.decls = self.backend.setup()
        self.said, self.lines, self.results = [], [], []
        real_call = self.backend.call

        async def recording_call(name, args):
            res = await real_call(name, args)
            self.results.append((name, args, res))
            return res

        self.backend.call = recording_call
        self.backend.after = lambda: None  # the fake robot's "...later" lines: not needed here
        self.brain = gb.Brain(gb.Ollama(self.ollama.url, gb.MODEL), self.prompt, self.decls,
                              self.backend, self.said.append, log=self.lines.append, **brain)

    def turn(self, text: str, image: bytes | None = None) -> None:
        self.loop.run_until_complete(self.brain.turn(text, image))

    def close(self, ollama: bool = True) -> None:
        if ollama:
            self.ollama.close()
        self.backend.rig.close()
        self.loop.run_until_complete(asyncio.sleep(0))  # let the cancelled job queue finish
        self.loop.close()


def tone(seconds: float, db: float, rate: int = 16000, hz: float = 220.0) -> bytes:
    amp = 32768 * 10 ** (db / 20) * math.sqrt(2)
    return array.array("h", [int(amp * math.sin(2 * math.pi * hz * i / rate)) for i in range(int(seconds * rate))]).tobytes()


def utterances(pcm: bytes, ep: gb.Endpointer) -> list[bytes]:
    out = []
    for i in range(0, len(pcm) - ep.frame_bytes + 1, ep.frame_bytes):
        u = ep.feed(pcm[i:i + ep.frame_bytes])
        if u:
            out.append(u)
    return out


# ============================================================ the tests
def translation_tests() -> None:
    print("Brain: the tools in Ollama's terms")
    loop = asyncio.new_event_loop()  # the fake robot's job queue needs one
    asyncio.set_event_loop(loop)
    rig = tt.Rig()
    decls = tt.tools.declarations(rig.ts)
    rig.close()
    loop.run_until_complete(asyncio.sleep(0))
    loop.close()
    tools = gb.ollama_tools(decls)
    text = json.dumps(tools)
    check("every tool is declared to Ollama", [t["function"]["name"] for t in tools] == [d["name"] for d in decls],
          f"{len(tools)} of {len(decls)}")
    check("no Gemini-style upper-case types are left (OBJECT, STRING, ...)",
          not any(f'"type": "{t}"' in text for t in ("OBJECT", "STRING", "NUMBER", "BOOLEAN", "INTEGER", "ARRAY")))
    move = next(t for t in tools if t["function"]["name"] == "move")["function"]["parameters"]
    check("move's parameters are a JSON Schema object with its enum kept",
          move["type"] == "object" and move["properties"]["direction"]["enum"] == ["forward", "backward", "left", "right"]
          and move["required"] == ["direction"], json.dumps(move)[:200])
    check("a tool without parameters gets an empty object",
          next(t for t in tools if t["function"]["name"] == "stand_up")["function"]["parameters"] == {"type": "object", "properties": {}})
    odd = gb.json_schema({"type": "OBJECT", "properties": {"type": {"type": "STRING"}}})
    check("a parameter that happens to be called 'type' survives", odd == {"type": "object", "properties": {"type": {"type": "string"}}}, odd)

    decl = {"parameters": {"properties": {"speed": {"type": "NUMBER"}, "n": {"type": "INTEGER"}, "run": {"type": "BOOLEAN"},
                                          "direction": {"type": "STRING"}}}}
    got = gb.coerce_args(decl, {"speed": "1.5", "n": 2.0, "run": "true", "direction": "forward", "duration": None})
    check("coerce: '1.5' -> 1.5, 2.0 -> 2, 'true' -> true, null dropped",
          got == {"speed": 1.5, "n": 2, "run": True, "direction": "forward"}, got)
    got = gb.coerce_args(decl, {"speed": "fast"})
    check("coerce: a value it can't read is left for the robot to refuse", got == {"speed": "fast"}, got)

    names = {"move", "stand_up"}
    words, calls = gb.calls_in_text('Sure!\n```json\n{"name": "move", "arguments": {"direction": "forward"}}\n```', names)
    check("a call written in a fence is found, and taken out of the words",
          calls == [("move", {"direction": "forward"})] and words == "Sure!", (words, calls))
    words, calls = gb.calls_in_text('[{"name": "stand_up", "args": {}}, {"name": "move", "parameters": {"direction": "left"}}]', names)
    check("a bare list of calls is found", calls == [("stand_up", {}), ("move", {"direction": "left"})] and words == "", calls)
    for text in ('I am a dog.', '{"name": "fly", "arguments": {}}', '{"say": "hi"}'):
        check(f"no call is found in {text!r}", gb.calls_in_text(text, names) == (text, []))

    check("the brain is Gemma 4 26B", gb.MODEL == "gemma4:26b", gb.MODEL)
    check("speech is cleaned of markdown", gb.clean_speech("**Sure!** I'll  _walk_\n now.") == "Sure! I'll walk now.")
    for raw, want in (("thought\n<channel|>", ""), ("thought\nI found your cup. It is on the table.", ""),
                      ("<|channel>thought\nThe user wants...<channel|>I am looking for your cup.", "I am looking for your cup."),
                      ("I thought about it: yes.", "I thought about it: yes."), ("Sure, I will sit.", "Sure, I will sit.")):
        check(f"Gemma 4's leaked thought {raw[:30]!r} is said as {want!r}", gb.drop_thought(raw) == want, gb.drop_thought(raw))


def loop_tests() -> None:
    print("Brain: the loop, with Gemma played by a fake Ollama")
    rig = Rig([{"content": "", "tool_calls": [call("move", direction="forward", duration="2")]},
               {"content": "I walked forward for two seconds."}])
    rig.turn("walk forward for 2 seconds")
    first, second = rig.ollama.requests[0], rig.ollama.requests[1]
    check("Gemma gets every tool, the prompt and the voice rules", len(first["tools"]) == len(rig.decls)
          and first["messages"][0]["role"] == "system" and gb.VOICE_RULES in first["messages"][0]["content"]
          and rig.prompt[:200] in first["messages"][0]["content"])
    check(f"with a context of {gb.NUM_CTX} tokens (Ollama's default would cut the tools off), not thinking (it's slow), kept loaded",
          first["options"]["num_ctx"] == gb.NUM_CTX and first["think"] is False and first["keep_alive"] == -1, first["options"])
    ran = rig.results[0] if rig.results else None
    check("Gemma's call ran on the robot, with '2' read as 2.0", ran is not None and ran[0] == "move"
          and ran[1] == {"direction": "forward", "duration": 2.0}, rig.results)
    tool_msg = second["messages"][-1]
    check("the robot's reply went back to Gemma word for word, as a tool message",
          tool_msg["role"] == "tool" and tool_msg["tool_name"] == "move" and json.loads(tool_msg["content"]) == ran[2],
          tool_msg)
    check("Gemma's own tool call is in the history before it", second["messages"][-2].get("tool_calls"))
    check("Pupper says Gemma's answer", rig.said == ["I walked forward for two seconds."], rig.said)
    twists = [m for _, topic, m in rig.backend.rig.robot.published if topic == "/llm_cmd_vel"]
    check("...and the fake robot really walked forward", any(m.linear.x > 0 for m in twists), f"{len(twists)} speed commands")
    rig.close()

    rig = Rig([{"content": "ready"}, {"content": "Hi there."}])
    rig.brain.warm()
    check("warm-up: Gemma reads the whole prompt and tools once, says nothing aloud",
          len(rig.ollama.requests) == 1 and len(rig.ollama.requests[0]["tools"]) == len(rig.decls) and rig.said == [], rig.said)
    rig.turn("hello")
    sent = [m["content"] for m in rig.ollama.requests[1]["messages"][1:]]
    check("... and the first real turn extends that exchange (so Ollama reuses what it read)",
          len(sent) == 3 and sent[0].startswith("[robot] Starting up") and sent[1] == "ready" and sent[2] == "hello", sent)
    rig.close()

    rig = Rig([{"content": "", "tool_calls": [call("move", direction="forward")]}])
    rig.brain.warm()
    check("warm-up: a call in its reply is not run, and that exchange isn't kept",
          rig.results == [] and rig.said == [] and rig.brain.history == [], (rig.results, rig.brain.history))
    rig.close()

    rig = Rig([{"content": "", "tool_calls": [call("find", what="cup")]},
               {"content": "thought\nI found your cup. It is on the table."}])
    rig.turn("where is my cup? find it")
    check("a thought that leaks after a tool result is never said (here it claimed a find that never happened)",
          rig.said == [] and rig.brain.history[-1]["content"] == "", (rig.said, rig.brain.history[-1]))
    rig.close()

    rig = Rig([{"content": "Zoom!", "tool_calls": [call("move", direction="forward", speed=9)]},
               {"content": "That's too fast for me: 1.5 metres a second at most."}])
    rig.turn("run forward at 9 metres a second")
    reply = json.loads(rig.ollama.requests[1]["messages"][-1]["content"])
    check("a refusal goes back to Gemma with the limit in it", reply["success"] is False and "1.5" in reply["message"], reply)
    check("words said with a call are said first, then the answer", rig.said == ["Zoom!", "That's too fast for me: 1.5 metres a second at most."], rig.said)
    rig.close()

    rig = Rig([{"content": '```json\n{"name": "stand_up", "arguments": {}}\n```'}, {"content": "I'm up."}])
    rig.turn("stand up")
    last = rig.ollama.requests[1]["messages"][-1]
    check("a call Gemma wrote out as text still runs", [r[0] for r in rig.results] == ["stand_up"], rig.results)
    check("...and its result goes back as a message Gemma can read", last["role"] == "user"
          and last["content"].startswith("[tool results, from the robot] stand_up ->"), last)
    check("...and the JSON is not read out loud", rig.said == ["I'm up."], rig.said)
    rig.close()

    rig = Rig([{"content": "", "tool_calls": [call("robot_status")]}] * 10)
    rig.turn("how are you")
    check(f"Gemma calling tools forever is stopped after {gb.MAX_ROUNDS}", len(rig.results) == gb.MAX_ROUNDS
          and any("stopped there" in line for line in rig.lines), len(rig.results))
    rig.close()

    rig = Rig([])
    rig.ollama.close()
    rig.brain.llm.timeout = 2
    rig.turn("hello")
    check("Ollama down: Pupper says so, nothing crashes", rig.said == ["Sorry, my brain isn't answering right now."], rig.said)
    rig.close(ollama=False)

    rig = Rig([{"content": "ok"}] * 40, keep=6)
    rig.turn("one", image=b"\xff\xd8jpeg1")
    check("a camera frame goes with its turn", rig.ollama.requests[0]["messages"][-1].get("images") == ["/9hqcGVnMQ=="],
          rig.ollama.requests[0]["messages"][-1].keys())
    rig.turn("two", image=b"\xff\xd8jpeg2")
    check("...and only the newest one is sent", [bool(m.get("images")) for m in rig.ollama.requests[1]["messages"][1:]]
          == [False, False, True])
    for i in range(5):
        rig.turn(f"turn {i}")
    check("history is trimmed, and starts with the person", len(rig.brain.history) <= 6 and rig.brain.history[0]["role"] == "user",
          [m["role"] for m in rig.brain.history])
    rig.close()

    rig = Rig([{"content": "A reminder!"}])
    rig.backend.alerts.EVENTS.emit("reminder", "call mom", for_gemini="[REMINDER #1] Tell the person: call mom.")
    got = gb.news(rig.backend)
    check("robot events become text for Gemma (news)", got == ["[REMINDER #1] Tell the person: call mom."], got)
    check("...once", gb.news(rig.backend) == [])
    rig.close()


def voice_tests() -> None:
    print("Brain: ears and voice (no mic needed)")
    quiet, loud = -60.0, -20.0
    ep = gb.Endpointer()
    pcm = tone(1.0, quiet, hz=50) + tone(1.2, loud) + tone(1.5, quiet, hz=50)
    got = utterances(pcm, ep)
    length = len(got[0]) / 2 / 16000 if got else 0
    check("a 1.2 s word over a quiet room is one utterance, with its tail", len(got) == 1 and 1.3 <= length <= 2.4,
          f"{len(got)} utterance(s), {length:.2f} s")
    ep = gb.Endpointer()
    got = utterances(tone(1.0, quiet, hz=50) + tone(0.2, loud) + tone(1.5, quiet, hz=50), ep)
    check("a 0.2 s click is not an utterance", got == [], f"{len(got)}")
    ep = gb.Endpointer()
    got = utterances(tone(1.0, -35, hz=50) + tone(1.0, -15) + tone(1.5, -35, hz=50), ep)
    check("in a loud hall the floor follows the room (speech is still heard)", len(got) == 1, f"{len(got)}")
    ep = gb.Endpointer(max_s=3.0)
    got = utterances(tone(0.5, quiet, hz=50) + tone(7.0, loud), ep)
    check("a sound that never ends is cut at max_s", len(got) >= 1 and len(got[0]) / 32000 <= 3.1, f"{len(got)}")

    wav = gb.to_wav(tone(0.5, loud), 16000)
    with wave.open(io.BytesIO(wav)) as w:
        ok = (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 16000, 8000)
    check("the WAV sent to whisper is 16 kHz mono 16-bit", ok)
    body, ctype = gb.multipart({"response_format": "json"}, "file", "speech.wav", wav, "audio/wav")
    boundary = ctype.split("boundary=")[1]
    check("the upload is a multipart form whisper-server reads", ctype.startswith("multipart/form-data")
          and b'name="response_format"\r\n\r\njson' in body and b'name="file"; filename="speech.wav"' in body
          and body.endswith(f"--{boundary}--\r\n".encode()) and wav in body)

    for raw, want in ((" Walk forward. ", "Walk forward."), ("[BLANK_AUDIO]", ""), ("(music) hi there", "hi there"),
                      (" ... ", ""), ("*coughs*", ""), (" Thank you.", ""), ("Thanks for watching!", ""), ("Ah.", ""),
                      ("Thank you, now sit down.", "Thank you, now sit down.")):
        check(f"whisper's {raw!r} is heard as {want!r}", gb.clean_heard(raw) == want, gb.clean_heard(raw))
    for text in ("Stop!", "stop.", "Pupper, stop.", "hey pupper stop", "Stop now!", "Freeze!"):
        check(f"{text!r} stops at once", bool(gb.STOP_WORDS.match(text)))
    for text in ("don't stop", "stop following me", "stop and sit", "bus stop"):
        check(f"{text!r} goes to Gemma, not straight to stop", not gb.STOP_WORDS.match(text))


def relay_tests() -> None:
    print("Brain: the gzip relay (tools/ollama_relay.py) in front of Ollama, for the Pupper's slow WiFi")
    ollama = FakeOllama([{"content": "Hello."}])
    port = 18000 + os.getpid() % 1000
    relay = subprocess.Popen([sys.executable, str(HERE.parent / "tools" / "ollama_relay.py"), "--port", str(port),
                              "--ollama", ollama.url], stdout=subprocess.DEVNULL)
    try:
        import time as _t
        for _ in range(50):
            try:
                urllib_request.urlopen(f"http://127.0.0.1:{port}/relay", timeout=1).close()
                break
            except OSError:
                _t.sleep(0.1)
        llm = gb.Ollama(f"http://127.0.0.1:{port}", gb.MODEL)
        m = llm.chat([{"role": "user", "content": "hi " * 2000}])
        check("through the relay, the brain sends gzip and Ollama gets the request whole",
              llm.gzip is True and m.get("content") == "Hello." and ollama.requests[0]["messages"][0]["content"] == "hi " * 2000
              and llm.sent < 1000, (llm.gzip, llm.sent, m))
        direct = gb.Ollama(ollama.url, gb.MODEL)
        direct.chat([{"role": "user", "content": "hi"}])
        check("straight to Ollama (no relay), it sends plain JSON", direct.gzip is False)
    finally:
        relay.kill()
        ollama.close()


def cli_tests() -> None:
    print("Brain: the command line, end to end (python3 robot/gemma_brain.py, typing, on the fake robot)")
    script = HERE / "gemma_brain.py"
    ollama = FakeOllama([{"content": ""},  # the warm-up
                         {"content": "", "tool_calls": [call("stand_up")]}, {"content": "Up I get!"}],
                        models=("llama3:8b", "gemma4:26b"))
    out = subprocess.run([sys.executable, str(script), "--ollama", ollama.url], input="stand up\nquit\n",
                         capture_output=True, text=True, timeout=60, env=os.environ).stdout
    check("its Gemma is gemma4:26b", "Gemma: gemma4:26b" in out, out[:300])
    check("it warms Gemma up before the first turn", "ready (" in out, out[:400])
    check("it runs Gemma's call on the fake robot and says the answer",
          "-> Gemma calls stand_up({})" in out and '"success": true' in out and "Pupper > Up I get!" in out, out[-600:])
    ollama.close()

    ollama = FakeOllama([{"content": "Hello there, I am Pupper."}, {"content": "", "tool_calls": [call("stand_up")]}])
    proc = subprocess.run([sys.executable, str(script), "--check", "--ollama", ollama.url, "--whisper", "http://127.0.0.1:9"],
                          capture_output=True, text=True, timeout=60)
    check("--check: Ollama reachable, Gemma talks, native tool calls ok, whisper missing (only --voice needs it)",
          proc.returncode == 0 and "talks" in proc.stdout and "native, ok" in proc.stdout and "only --voice needs it" in proc.stdout,
          proc.stdout[-500:])
    ollama.close()
    ollama2 = FakeOllama([{"content": "Hi."}, {"content": "", "tool_calls": [call("stand_up")]}])
    proc = subprocess.run([sys.executable, str(script), "--check", "--ollama", "http://127.0.0.1:9"],
                          capture_output=True, text=True, timeout=60)
    check("--check: Ollama unreachable says what to do (start Pupper from the Mac)",
          proc.returncode == 1 and "Start Pupper from the Mac" in proc.stdout, proc.stdout)
    proc = subprocess.run([sys.executable, str(script), "--check", "--voice", "--ollama", ollama2.url, "--whisper",
                           "http://127.0.0.1:9"], capture_output=True, text=True, timeout=60)
    check("--check --voice: no whisper is a failure (Pupper couldn't hear)",
          proc.returncode == 1 and "can't hear without it" in proc.stdout, proc.stdout[-300:])
    ollama2.close()
    ollama3 = FakeOllama([], models=("llama3:8b",))
    proc = subprocess.run([sys.executable, str(script), "--check", "--ollama", ollama3.url], capture_output=True, text=True, timeout=60)
    check("--check: no gemma4:26b says how to pull it", proc.returncode == 1 and "ollama pull gemma4:26b" in proc.stdout, proc.stdout[-300:])
    ollama3.close()
    proc = subprocess.run([sys.executable, str(script), "--ollama", "http://127.0.0.1:9"], capture_output=True, text=True, timeout=60)
    check("no Ollama: it says so, and points at --check", proc.returncode == 1 and "--check" in proc.stderr, proc.stderr)


GROUPS = {"translation": translation_tests, "loop": loop_tests, "voice": voice_tests, "relay": relay_tests, "cli": cli_tests}


def main(groups: list[str]) -> None:
    for name in groups or GROUPS:
        if name not in GROUPS:
            sys.exit(f"unknown brain test group {name!r}; choose from {list(GROUPS)}")
        GROUPS[name]()
        print()
    if tt.FAIL:
        print(f"{tt.FAIL} check(s) failed, {tt.PASS} passed.")
        raise SystemExit(1)
    print(f"All {tt.PASS} brain checks passed.")


if __name__ == "__main__":
    main(sys.argv[1:])
