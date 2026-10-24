#!/usr/bin/env python3
"""Pupper's brain on a local Gemma: it hears, talks, decides and calls tools, with no cloud in the loop.

The hackathon's rule is "on-device, no cloud", and the web app's brain, Gemini Live, is cloud. This
replaces it. It is robot/gemini_chat.py's loop with Gemma in Ollama instead of Gemini, plus a voice:
the same prompt and tools (GET /tools), every call sent to the robot server (POST /tools/<name>),
and its {success, message} given back to Gemma word for word.

    python3 robot/gemma_brain.py --check               # can this machine reach Gemma (and whisper) on the Mac?
    python3 robot/gemma_brain.py                       # type to Gemma: your tools on a fake robot (nothing moves)
    python3 robot/gemma_brain.py --robot               # type to Gemma: the real robot server (it moves!)
    python3 robot/gemma_brain.py --robot --voice       # talk to Pupper: mic, whisper, Gemma, Piper (it moves!)

Where things run (./run_pupper.sh on the Mac sets it all up: tools/brain_on_mac.sh):
  the Pi   this script, the robot server (localhost:8000), the mic, the speaker, Piper (speech)
  the Mac  Ollama with Gemma and whisper.cpp's server, over WiFi ($PUPPER_OLLAMA, $PUPPER_WHISPER)
The robot server stays on the Pi's localhost, so nothing on the network can move the robot. Run on
the Mac itself, without --robot, it talks to the Mac's own Ollama: a quick try on the fake robot.

Gemma calls tools natively (Ollama's tools), or, with a model or an Ollama that can't, by naming
them in a JSON reply (--tools json); auto tries native first and falls back. Robot events (a
reminder came due, a loud sound, a search finished) become turns of their own, so Pupper speaks
up without being asked, as it does with Gemini. Saying just "stop" stops the robot at once,
without waiting for Gemma.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import base64
import collections
import io
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gemini_chat import DeskBackend, RobotBackend  # noqa: E402  the same two robots Gemini's chat talks to

# On the Pupper, run_pupper.sh sets these to the Mac's servers; the defaults are this machine's own.
OLLAMA_URL = os.environ.get("PUPPER_OLLAMA", "http://localhost:11434")
WHISPER_URL = os.environ.get("PUPPER_WHISPER", "http://localhost:8178")
PIPER_VOICE = Path(os.environ.get("PUPPER_PIPER_VOICE", Path.home() / ".local/share/piper/en_US-lessac-medium.onnx"))
# Without --model, the first of these that Ollama has: the biggest Gemma first. The 26B is a
# mixture of experts, 4B active per token: fast for its size, and much the best at picking tools.
PREFERRED = ("gemma4:26b-a4b", "gemma4:26b", "gemma4:12b", "gemma4:e4b", "gemma4", "gemma4:e2b",
             "gemma3:12b", "gemma3:4b")
NUM_CTX = 16384  # the prompt and 30 tools are about 8k tokens: Ollama's default context would cut them off
MAX_ROUNDS = 6   # tool calls in a row before Gemma has to answer
MAX_CALLS = 24   # enough for a taught routine or a square; one bad reply cannot flood the queue
EVENT_TOOLS = {"stop", "robot_status", "look", "what_is_around", "recall", "list_reminders", "list_routines", "show_text"}
KEEP = 30        # messages of history kept (older ones are dropped, a turn at a time)
STOP_WORDS = re.compile(r"^\W*(?:(?:hey\W+)?pupper\W+)?(?:stop|halt|freeze)(?:\W+(?:it|now|pupper))*\W*$", re.I)

VOICE_RULES = (
    "How you talk here: everything you write is read out loud by a speech synthesizer, so write short, "
    "plain spoken sentences: no markdown, no lists, no emoji, no stage directions. To make the robot do "
    "something, call the tool; never only say that you will. After a tool replies, tell the person what "
    "really happened, from the reply. Queued means accepted, not finished: do not claim completion "
    "until robot_status or an outcome event confirms it. You do not receive continuous video here; "
    "Tool results include robot_state: running and queued say whether work remains; outcomes are "
    "actual completed jobs, with increasing IDs, success and message. Do not confuse older outcomes "
    "with a new job. If an outcome fails, explain it and reassess; do not repeat the failed action "
    "or continue dependent motion automatically. A refusal is feedback, not permission to improvise "
    "a different physical action. "
    "use look for a current visual question when no image is supplied. Messages from the robot's own systems (a reminder, a sound it "
    "heard, a search that finished) say so: answer those out loud too, as Pupper.")


# ============================================================ tools, in Ollama's terms
def json_schema(s):
    """Gemini's schema (type: OBJECT, STRING, ...) as the JSON Schema Ollama wants (object, string, ...)."""
    if isinstance(s, dict):
        return {k: v.lower() if k == "type" and isinstance(v, str) else json_schema(v) for k, v in s.items()}
    if isinstance(s, list):
        return [json_schema(v) for v in s]
    return s


def ollama_tools(decls: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {
        "name": d["name"], "description": d.get("description", ""),
        "parameters": json_schema(d.get("parameters") or {"type": "OBJECT", "properties": {}})}} for d in decls]


def coerce_args(decl: dict | None, args) -> dict:
    """Small models send "1.5" for 1.5, 2.0 for 2 and null for "not given": fix what is unambiguous,
    by the declaration. The robot checks everything else and says what was wrong."""
    if not isinstance(args, dict):
        return args  # let the server refuse it; never turn a broken move into default motion
    props = ((decl or {}).get("parameters") or {}).get("properties") or {}
    out = {}
    for k, v in (args if isinstance(args, dict) else {}).items():
        if v is None:
            continue
        t = str((props.get(k) or {}).get("type", "")).lower()
        if isinstance(v, str) and t in ("number", "integer"):
            try:
                v = float(v.strip())
            except ValueError:
                pass
        if isinstance(v, str) and t == "boolean" and v.strip().lower() in ("true", "false"):
            v = v.strip().lower() == "true"
        if t == "integer" and isinstance(v, float) and v.is_integer():
            v = int(v)
        out[k] = v
    return out


def native_calls(msg: dict) -> list[tuple[str, dict]]:
    calls = []
    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict) or not isinstance(tc.get("function"), dict):
            raise ModelReplyError("model returned a malformed native tool call")
        fn = tc.get("function") or {}
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass  # preserve malformed arguments for validation, rather than defaulting
        if not isinstance(fn.get("name"), str) or not fn["name"]:
            raise ModelReplyError("model returned a native call without a tool name")
        calls.append((fn["name"], args))
    return calls


FENCE = re.compile(r"```[a-z_]*\s*(.*?)```", re.S)


def calls_in_text(text: str, names) -> tuple[str, list[tuple[str, dict]]]:
    """A call written out instead of made (small models do this): {"name": "move", "arguments": {...}}
    as the whole reply, or in a ``` fence. Returns the words without it, and the calls."""
    candidates = [(m.group(1), m.group(0)) for m in FENCE.finditer(text)] + [(text, text)]
    for body, whole in candidates:
        try:
            data = json.loads(body.strip())
        except ValueError:
            continue
        if isinstance(data, dict) and "say" in data and "calls" in data:
            say, calls = structured_reply(data)
            return text.replace(whole, say).strip(), calls
        items = data if isinstance(data, list) else [data]
        calls = []
        for it in items:
            if isinstance(it, dict) and isinstance(it.get("name"), str) and it["name"] in names:
                args = it.get("arguments", it.get("args", it.get("parameters", {})))
                calls.append((it["name"], args))
        if calls and len(calls) == len(items):
            return text.replace(whole, "").strip(), calls
    return text, []


def structured_reply(data) -> tuple[str, list[tuple[str, dict]]]:
    if not isinstance(data, dict) or not isinstance(data.get("calls"), list) or not isinstance(data.get("say"), str):
        raise ModelReplyError("model returned an invalid tool response object")
    if any(not isinstance(c, dict) or not isinstance(c.get("name"), str) or not c["name"] for c in data["calls"]):
        raise ModelReplyError("model returned an invalid tool call")
    return data["say"], [(c["name"], c.get("args", {})) for c in data["calls"]]


def reply_schema(names: list[str]) -> dict:
    """--tools json: every reply is {say, calls}, enforced by Ollama's structured output."""
    return {"type": "object", "required": ["say", "calls"], "properties": {
        "say": {"type": "string"},
        "calls": {"type": "array", "items": {"type": "object", "required": ["name", "args"], "properties": {
            "name": {"type": "string", "enum": names}, "args": {"type": "object"}}}}}}


def json_mode_prompt(decls: list[dict]) -> str:
    lines = ['Reply with JSON only, always in this form: {"say": "what you say out loud, or empty", '
             '"calls": [{"name": "a tool", "args": {...}}]}. calls is [] when you only talk. After calls you '
             "get their results, and reply again in the same form. The tools:"]
    for d in decls:
        params = (d.get("parameters") or {})
        required = set(params.get("required") or [])
        lines.append(f"- {d['name']}: {d.get('description', '')}")
        for k, p in (params.get("properties") or {}).items():
            enum = f" One of: {', '.join(map(str, p['enum']))}." if p.get("enum") else ""
            lines.append(f"    {k} ({str(p.get('type', '')).lower()}{', required' if k in required else ''}): "
                         f"{p.get('description', '')}{enum}")
    return "\n".join(lines)


def drop_thought(text: str) -> str:
    """Gemma 4's thought channel, should any leak into a reply: "thought ... <channel|>answer" keeps
    the answer; a thought that never closes is all thought, and none of it is for the person."""
    text = (text or "").strip()
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.S | re.I)
    text = re.split(r"<think(?:ing)?>", text, maxsplit=1, flags=re.I)[0]
    if "<channel|>" in text:
        text = text.split("<channel|>")[-1]
    elif re.match(r"(<\|channel>)?thought\b", text):
        return ""
    return re.sub(r"<\|?[a-z_]+\|?>", "", text).strip()


def clean_speech(text: str) -> str:
    """What the synthesizer should read: no markdown marks, one line."""
    return " ".join(re.sub(r"[*_#`>]+", "", text or "").split())


# ============================================================ Gemma, in Ollama on the Mac
class OllamaError(RuntimeError):
    pass


class ModelReplyError(OllamaError):
    """Malformed model protocol: safe to request one correction because no tool ran."""


class Ollama:
    def __init__(self, url: str, model: str, num_ctx: int = NUM_CTX, timeout: float = 180.0):
        self.url, self.model, self.num_ctx, self.timeout = url.rstrip("/"), model, num_ctx, timeout
        # Thinking off: on, Gemma 4 takes 3.4 s to reply instead of 0.5 s, for the same tool choices
        # (tools/brain_bench.py). Off, its thought sometimes leaks into a reply after a tool result;
        # drop_thought() keeps that from being said. None: a model that can't be told (Ollama says so).
        self.think: bool | None = False
        self.last: dict = {}  # the last reply, for its timings

    def _req(self, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise OllamaError(f"Ollama said {e.code}: {e.read().decode(errors='replace')[:300]}") from None

    def models(self) -> list[str]:
        return [m.get("name", "") for m in self._req("/api/tags", timeout=5).get("models", [])]

    def chat(self, messages: list[dict], tools: list[dict] | None = None, fmt: dict | None = None) -> dict:
        # keep_alive -1: stay loaded (a reload is 10-30 s); num_ctx must match what the eyes'
        # requests get (OLLAMA_CONTEXT_LENGTH, tools/brain_on_mac.sh), or Ollama reloads between them.
        body = {"model": self.model, "messages": messages, "stream": False, "keep_alive": -1,
                "options": {"num_ctx": self.num_ctx, "num_predict": 2048, "temperature": 0.3}}
        if self.think is not None:
            body["think"] = self.think
        if tools:
            body["tools"] = tools
        if fmt:
            body["format"] = fmt
        try:
            self.last = self._req("/api/chat", body)
        except OllamaError as e:
            if "think" in str(e).lower() and self.think is not None:  # this model can't be told: don't
                self.think = None
                return self.chat(messages, tools, fmt)
            raise
        return self.last.get("message") or {}


def pick_model(names: list[str]) -> str | None:
    for want in PREFERRED:
        for n in names:
            if n == want or n == want + ":latest" or n.startswith(want + "-"):
                return n
    return next((n for n in names if n.startswith("gemma")), None)


# ============================================================ the loop
class Brain:
    """One conversation with Gemma: a turn goes in, Gemma talks and calls tools until it is done."""

    def __init__(self, llm: Ollama, prompt: str, decls: list[dict], backend, say, mode: str = "auto",
                 keep: int = KEEP, log=print):
        self.llm, self.backend, self.say, self.log, self.keep = llm, backend, say, log, keep
        self.prompt, self.mode = prompt, mode  # auto: native, until Ollama says this model can't
        self.decls = {d["name"]: d for d in decls}
        self.tools = ollama_tools(decls)
        self.history: list[dict] = []
        self.generation = 0
        self.turn_starts: list[int] = []
        self.reset_next_turn = False

    async def stop(self) -> dict:
        """Invalidate an in-flight decision before stopping; its late reply must never act."""
        self.generation += 1
        self.reset_next_turn = True
        try:
            return await self.backend.call("stop", {})
        except Exception as e:
            self.log(f"  !! stop could not be confirmed: {e}")
            return {"success": False, "message": "stop could not be confirmed; use the physical e-stop"}

    def system(self) -> dict:
        text = VOICE_RULES + "\n\n" + self.prompt
        if self.mode == "json":
            text += "\n\n" + json_mode_prompt(list(self.decls.values()))
        return {"role": "system", "content": text}

    async def turn(self, text: str, image: bytes | None = None, source: str = "person") -> None:
        if source == "person" and STOP_WORDS.match(text):
            result = await self.stop()
            self.say("Stopped." if result.get("success") else result["message"])
            return
        if self.reset_next_turn:
            self.history, self.turn_starts = [], []
            self.reset_next_turn = False
        generation = self.generation
        self.turn_starts.append(len(self.history))
        refused: dict[str, dict] = {}
        calls_left = MAX_CALLS
        repairs = 0
        for m in self.history:
            m.pop("images", None)  # only the newest frame is worth its tokens
        msg = {"role": "user", "content": text if source == "person" else
               "[robot event, not a new request or permission to move] " + text}
        if image:
            msg["images"] = [base64.b64encode(image).decode()]
        self.history.append(msg)
        try:
            for _ in range(MAX_ROUNDS):
                t0 = time.time()
                try:
                    said, calls, native = await asyncio.to_thread(self._ask)
                except ModelReplyError:
                    if repairs:
                        raise
                    repairs += 1
                    self.history.append({"role": "user", "content":
                        '[robot] Your last response was malformed. No tool from it was run. '
                        'Reply with a valid response: {"say":"short spoken answer","calls":'
                        '[{"name":"tool name","args":{}}]}. Use an empty calls list to only speak.'})
                    continue
                if generation != self.generation:
                    # _ask has finished writing history, so no worker is left mutating it.
                    self.history = []
                    self.turn_starts = []
                    return
                self.log(f"  (Gemma: {time.time() - t0:.1f} s)")
                if not calls:
                    if said:
                        self.say(said)
                    break
                results = []
                failed = False
                after = 0
                observation_error = None
                if hasattr(self.backend, "begin_batch"):
                    try:
                        after = await self.backend.begin_batch()
                    except (urllib.error.URLError, OSError, ValueError) as e:
                        observation_error = str(e)
                if len(calls) > calls_left:
                    self.say("That plan has too many actions. Please split it into smaller steps; this batch was not run.")
                    # This assistant message contains unanswered calls; do not carry it forward.
                    self.history = self.history[:self.turn_starts[-1]]
                    self.turn_starts.pop()
                    return
                calls_left -= len(calls)
                for name, args in calls:
                    if generation != self.generation:
                        self.history = []
                        self.turn_starts = []
                        return
                    args = coerce_args(self.decls.get(name), args)
                    signature = json.dumps([name, args], sort_keys=True)
                    self.log(f"  -> Gemma calls {name}({json.dumps(args)})")
                    try:
                        if observation_error and name != "stop":
                            result = {"success": False, "message": f"not run: robot state unavailable: {observation_error}"}
                        elif failed and name != "stop":
                            result = {"success": False, "message": "not run: an earlier call in this batch failed. Reassess using its result before continuing."}
                        elif signature in refused and name != "stop":
                            result = {"success": False, "message": f"already refused this turn; not retried: {refused[signature]['message']}"}
                        elif not isinstance(args, dict):
                            result = {"success": False, "message": f"{name}: arguments must be a JSON object; call was not run"}
                        elif source != "person" and name not in EVENT_TOOLS:
                            result = {"success": False, "message": "not run: a robot event is not permission for a new action. Report the event; wait for the person."}
                        elif name not in self.decls:
                            result = {"success": False, "message": f"unknown or disabled tool {name}; available: {sorted(self.decls)}"}
                        else:
                            result = await self.backend.call(name, args)
                    except Exception as e:  # e.g. the robot server is not running
                        result = {"success": False, "message": f"{type(e).__name__}: {e}"}
                    self.log(f"  <- reply Gemma hears now: {json.dumps(result)}")
                    if hasattr(self.backend, "after"):
                        self.backend.after()
                    results.append((name, result))
                    failed = failed or not result.get("success", False)
                    state = result.get("robot_state", {})
                    failed = failed or any(o.get("id", 0) > after and not o.get("success")
                                           for o in state.get("outcomes", []))
                    if not result.get("success", False):
                        refused[signature] = result
                if hasattr(self.backend, "finish_batch") and any(r.get("success") for _, r in results):
                    try:
                        state = await self.backend.finish_batch()
                        results[-1][1]["robot_state"] = state
                        failed = failed or any(o.get("id", 0) > after and not o.get("success")
                                               for o in state.get("outcomes", []))
                    except (urllib.error.URLError, OSError, ValueError) as e:
                        results[-1][1]["robot_state"] = {"unavailable": str(e)}
                if generation != self.generation:
                    self.history = []
                    self.turn_starts = []
                    return
                changed_memory = {"remember", "forget", "switch_profile", "found_object", "finish_learning", "forget_routine"}
                if any(name in changed_memory and result.get("success") for name, result in results):
                    try:
                        self.prompt = await self.backend.refresh_prompt()
                    except (urllib.error.URLError, OSError, ValueError) as e:
                        self.log(f"  !! memory context refresh failed: {e}")
                if generation != self.generation:
                    self.history = []
                    self.turn_starts = []
                    return
                if said and not failed:
                    self.say(said)  # never speak a failed action's optimistic preamble
                for _, result in results[:-1]:
                    result.pop("robot_state", None)  # one fresh snapshot per batch, not one per action
                self._add_results(results, native)
            else:
                self.log(f"  (Gemma called tools {MAX_ROUNDS} times in a row: stopped there)")
                self.say("I reached my limit for this request. I haven't finished it; please give me one step at a time.")
        except (OllamaError, urllib.error.URLError, OSError) as e:
            if generation != self.generation:
                self.history = []
                self.turn_starts = []
                return
            self.log(f"  !! Gemma did not answer: {e}")
            self.say("Sorry, my brain isn't answering right now.")
        self._trim()

    def warm(self) -> float:
        """Gemma reads the whole prompt and tools once, before anyone speaks, and the exchange stays as
        the start of the conversation: Ollama reuses a cached prompt reliably only when the next
        request extends it (Gemma 4's sliding-window attention), so the first real turn then takes
        about a second, not ten. A call in the reply is not run (and the exchange isn't kept). Also
        settles whether this model calls tools natively or needs JSON replies. Returns the seconds."""
        t0 = time.time()
        self.turn_starts = []
        self.history = [{"role": "user", "content": "[robot] Starting up; nobody has spoken yet. Reply with one word: ready."}]
        try:
            _, calls, _ = self._ask()
        except BaseException:
            self.history = []
            raise
        if calls:
            self.history = []
        return time.time() - t0

    def _ask(self) -> tuple[str, list[tuple[str, dict]], bool]:
        """One reply from Gemma: (what to say, the calls, whether they were native tool calls)."""
        self._fit_context()
        messages = [self.system()] + self.history
        if self.mode in ("auto", "native"):
            try:
                m = self.llm.chat(messages, tools=self.tools)
            except OllamaError as e:
                if self.mode == "auto" and "does not support tools" in str(e):
                    self.log(f"  ({self.llm.model} can't call tools natively here: switching to JSON replies)")
                    self.mode = "json"
                    return self._ask()
                raise
            content = drop_thought(m.get("content") or "")
            calls, native = native_calls(m), True
            if not calls:
                content, calls = calls_in_text(content, self.decls)
                native = False
                if not calls and content.lstrip().startswith(("{", "[", "```")):
                    raise ModelReplyError("unrecognized tool protocol in model text")
            reply = {"role": "assistant", "content": drop_thought(m.get("content") or "")}
            if m.get("tool_calls"):
                reply["tool_calls"] = m["tool_calls"]
            self.history.append(reply)
            return clean_speech(content), calls, native
        m = self.llm.chat(messages, fmt=reply_schema(sorted(self.decls)))
        raw = m.get("content") or ""
        self.history.append({"role": "assistant", "content": raw})
        try:
            data = json.loads(raw)
        except ValueError:
            raise ModelReplyError("model returned invalid JSON instead of a tool response") from None
        say, calls = structured_reply(data)
        return clean_speech(drop_thought(say)), calls, False

    def _add_results(self, results: list[tuple[str, dict]], native: bool) -> None:
        if native:
            for name, result in results:
                self.history.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
            return
        # Calls Gemma wrote out (or JSON mode): the results go back as a message it can read.
        self.history.append({"role": "user", "content": "[tool results, from the robot] " + "; ".join(
            f"{name} -> {json.dumps(result)}" for name, result in results)
            + (" Reply in the same JSON form." if self.mode == "json" else " Now tell the person what happened.")})

    def _trim(self) -> None:
        # Half at a time: every trim changes the start of the conversation, and Gemma re-reads the
        # whole prompt (about 10 s), so it should happen rarely.
        if len(self.history) > self.keep:
            target = len(self.history) - self.keep // 2
            start = next((i for i in self.turn_starts if i >= target), self.turn_starts[-1] if self.turn_starts else 0)
            self.history = self.history[start:]
            self.turn_starts = [i - start for i in self.turn_starts if i >= start]

    def _fit_context(self) -> None:
        """Bound the request by size as well as message count, keeping whole turns.

        Four UTF-8 bytes per token is an estimate, not a tokenizer. Leave room for the
        reply and reject an oversized current turn rather than let Ollama silently crop it.
        """
        budget = max(1024, (self.llm.num_ctx - 2048) * 4)
        while True:
            # JPEG base64 is not text tokens. The reply reserve also leaves room for an image.
            history = [{k: v for k, v in m.items() if k != "images"} for m in self.history]
            payload = [self.system(), history] + ([self.tools] if self.mode != "json" else [])
            size = len(json.dumps(payload, ensure_ascii=False).encode())
            if size <= budget:
                return
            if len(self.turn_starts) < 2:
                raise OllamaError("the current request and tools exceed the context budget; use fewer tools or a shorter request")
            start = self.turn_starts[1]
            self.history = self.history[start:]
            self.turn_starts = [i - start for i in self.turn_starts[1:]]


# ============================================================ what the robot has to say
def _post(backend, path: str, payload: dict | None = None) -> dict:
    req = urllib.request.Request(backend.url + path, data=json.dumps(payload or {}).encode(),
                                 headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def news(backend) -> list[str]:
    """What the robot has to tell Gemma since the last look: events (GET /events), and, on the real
    robot, reminders that came due. The face (the web app) claims those first, shows the card and
    hands them over as an event (POST /brain/announce); with no face, we claim them a minute late."""
    out = []
    for ev in backend.events():
        print(f"  [event {ev['kind']}] {ev['text']}" + (f"  (screen: {ev['display']['text']})" if ev.get("display") else ""))
        if ev.get("for_gemini"):
            out.append(ev["for_gemini"])
    if isinstance(backend, RobotBackend):
        for r in _post(backend, "/reminders/claim?min_late_s=60").get("due", []):
            print(f"  [reminder #{r['id']}] {r['message']}")
            out.append(r["announce"])
            _post(backend, f"/reminders/{r['id']}/delivered", {"how": "gemma"})
    return out


def transcript(backend, who: str, text: str) -> None:
    """The session log (lab/hackathon_log.py) gets what was said, as it does from the web app."""
    if isinstance(backend, RobotBackend) and text:
        try:
            _post(backend, "/transcript", {"who": who, "text": text})
        except (urllib.error.URLError, OSError):
            pass


def brain_state(backend, listening: bool, speaking: bool, caption: str | None) -> None:
    """What the face shows for us: listening, speaking, the caption (POST /brain/state)."""
    if isinstance(backend, RobotBackend):
        try:
            _post(backend, "/brain/state", {"listening": listening, "speaking": speaking, "caption": caption})
        except (urllib.error.URLError, OSError):
            pass


def camera_frame(backend) -> bytes | None:
    if not isinstance(backend, RobotBackend):
        return None
    try:
        with urllib.request.urlopen(backend.url + "/camera/image", timeout=3) as r:
            return r.read() if r.status == 200 else None
    except (urllib.error.URLError, OSError):
        return None


# ============================================================ ears and voice (on the Pi)
class Endpointer:
    """Cuts 16-bit mono audio into utterances by loudness over the room's own floor.

    The floor follows the quiet frames (it drops at once and rises slowly), so it holds in a quiet
    lab and in a loud hackathon hall. Speech starts after start_s of frames start_db over the
    floor and ends after end_s back under it; pre_s of audio before the start is kept, so the
    first syllable isn't cut. Utterances shorter than min_s (a cough, a clack) are dropped."""

    def __init__(self, rate: int = 16000, frame_ms: int = 30, start_db: float = 10.0, start_s: float = 0.15,
                 end_s: float = 0.8, min_s: float = 0.4, max_s: float = 15.0, pre_s: float = 0.3):
        self.frame_s = frame_ms / 1000
        self.frame_bytes = int(rate * self.frame_s) * 2
        self.start_db, self.start_s, self.end_s, self.min_s, self.max_s = start_db, start_s, end_s, min_s, max_s
        self.pre_frames = max(1, round(pre_s / self.frame_s))
        self.floor: float | None = None
        self.reset()

    def reset(self) -> None:
        self.pre: collections.deque = collections.deque(maxlen=self.pre_frames)
        self.buf: list[bytes] = []
        self.loud_run = self.quiet_run = 0
        self.speaking = False

    @staticmethod
    def level_db(frame: bytes) -> float:
        a = array.array("h", frame)
        if sys.byteorder == "big":
            a.byteswap()
        rms = math.sqrt(sum(x * x for x in a) / max(1, len(a))) / 32768.0
        return 20 * math.log10(rms) if rms > 1e-5 else -100.0

    def feed(self, frame: bytes) -> bytes | None:
        """One frame in; an utterance out when one just ended, else None."""
        db = self.level_db(frame)
        if self.floor is None:
            self.floor = db
        if not self.speaking:
            self.floor = db if db < self.floor else self.floor + 0.02 * (db - self.floor)
            self.pre.append(frame)
            self.loud_run = self.loud_run + 1 if db > self.floor + self.start_db else 0
            if self.loud_run * self.frame_s >= self.start_s:
                self.speaking, self.buf, self.quiet_run = True, list(self.pre), 0
            return None
        self.buf.append(frame)
        self.quiet_run = self.quiet_run + 1 if db < self.floor + self.start_db - 3 else 0
        length = len(self.buf) * self.frame_s
        if self.quiet_run * self.frame_s >= self.end_s or length >= self.max_s:
            pcm, speech_s = b"".join(self.buf), length - self.quiet_run * self.frame_s
            self.reset()
            return pcm if speech_s >= self.min_s else None
        return None


VAD_MODEL = Path(os.environ.get("PUPPER_VAD_MODEL", Path.home() / ".cache/pupper/silero_vad.onnx"))


class SileroEndpointer:
    """Cuts 16 kHz audio into utterances with Silero VAD (a small speech detector, ONNX, ~2 MB). It
    hears speech, not loudness, so a noisy hall, the motors or music don't keep an utterance open
    (by loudness alone, a busy room made 15 s "utterances" of noise that whisper turned to nonsense).
    Used like Endpointer: feed() 32 ms frames; an utterance comes out when it has ended."""

    def __init__(self, model: Path = VAD_MODEL, rate: int = 16000, start_s: float = 0.1, end_s: float = 0.7,
                 min_s: float = 0.3, max_s: float = 8.0, pre_s: float = 0.3, on: float = 0.5, off: float = 0.35):
        import numpy as np
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        self.np, self.sess = np, ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
        self.rate, self.n = rate, 512  # Silero's frame at 16 kHz, plus 64 samples of the last one as context
        self.frame_bytes, self.frame_s = self.n * 2, self.n / rate
        self.start_s, self.end_s, self.min_s, self.max_s, self.on, self.off = start_s, end_s, min_s, max_s, on, off
        self.pre_frames = max(1, round(pre_s / self.frame_s))
        self.reset()

    def reset(self) -> None:
        np = self.np
        self.state, self.context = np.zeros((2, 1, 128), np.float32), np.zeros((1, 64), np.float32)
        self.pre: collections.deque = collections.deque(maxlen=self.pre_frames)
        self.buf: list[bytes] = []
        self.voiced_run = self.quiet_run = 0
        self.speaking = False

    def prob(self, frame: bytes) -> float:
        np = self.np
        x = np.frombuffer(frame, np.int16).astype(np.float32)[None, :] / 32768.0
        out, self.state = self.sess.run(None, {"input": np.concatenate([self.context, x], axis=1), "state": self.state,
                                               "sr": np.array(self.rate, np.int64)})
        self.context = x[:, -64:]
        return float(np.asarray(out).ravel()[0])

    def feed(self, frame: bytes) -> bytes | None:
        p = self.prob(frame)
        if not self.speaking:
            self.pre.append(frame)
            self.voiced_run = self.voiced_run + 1 if p >= self.on else 0
            if self.voiced_run * self.frame_s >= self.start_s:
                self.speaking, self.buf, self.quiet_run = True, list(self.pre), 0
            return None
        self.buf.append(frame)
        self.quiet_run = self.quiet_run + 1 if p < self.off else 0
        length = len(self.buf) * self.frame_s
        if self.quiet_run * self.frame_s >= self.end_s or length >= self.max_s:
            pcm, speech_s = b"".join(self.buf), length - self.quiet_run * self.frame_s
            self.reset()
            return pcm if speech_s >= self.min_s else None
        return None


def endpointer(rate: int = 16000):
    """Silero when its model and onnxruntime are here (robot/setup_voice.sh), else loudness."""
    try:
        if VAD_MODEL.exists():
            return SileroEndpointer(VAD_MODEL, rate)
    except Exception as e:  # e.g. no onnxruntime
        print(f"  (Silero VAD unavailable: {e}; cutting speech by loudness)")
    return Endpointer(rate, max_s=8.0)


class Mic:
    """The mic through arecord (16 kHz mono, the PipeWire default unless --mic says otherwise)."""

    def __init__(self, device: str = "default", rate: int = 16000):
        self.device, self.rate = device, rate

    def run(self, on_utterance, muted) -> None:
        """Blocks: calls on_utterance(pcm) for each utterance, ignoring the mic while muted() is true."""
        ep = endpointer(self.rate)
        print(f"Ears: {type(ep).__name__} ({'speech detector' if isinstance(ep, SileroEndpointer) else 'loudness'})", flush=True)
        cmd = ["arecord", "-q", "-D", self.device, "-f", "S16_LE", "-r", str(self.rate), "-c", "1", "-t", "raw"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        try:
            while True:
                chunk = proc.stdout.read(ep.frame_bytes)
                if len(chunk) < ep.frame_bytes:
                    break
                if muted():  # Pupper is talking: don't transcribe its own voice
                    ep.reset()
                    continue
                pcm = ep.feed(chunk)
                if pcm:
                    on_utterance(pcm)
        finally:
            proc.kill()
        print(f"!! the mic stopped (arecord -D {self.device}): is another program holding it?", flush=True)
        os._exit(1)  # a brain with no ears must not sit there looking alive


def to_wav(pcm: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def multipart(fields: dict[str, str], file_field: str, filename: str, data: bytes, ctype: str) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
             for k, v in fields.items()]
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
                 f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n")
    return b"".join(parts) + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def clean_heard(text: str) -> str:
    """whisper's words without its notes ([BLANK_AUDIO], (music)); "" if nothing was said."""
    text = " ".join(re.sub(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*", " ", text or "").split())
    return text if re.search(r"[A-Za-z]", text) else ""


class Whisper:
    """Speech to text: whisper.cpp's server (whisper-server) on the Mac, POST /inference."""

    def __init__(self, url: str, timeout: float = 30.0):
        self.url, self.timeout = url.rstrip("/"), timeout

    # Words whisper should expect: it spells what it already knows (a hint, not a restriction).
    PROMPT = ("Pupper, stand up, sit down, lie down, walk forward, turn left, turn right, push-up, "
              "shake your paw, follow me, stop, remind me, remember, find my cup.")

    def transcribe(self, pcm: bytes, rate: int = 16000) -> str:
        body, ctype = multipart({"response_format": "json", "temperature": "0.0", "prompt": self.PROMPT},
                                "file", "speech.wav", to_wav(pcm, rate), "audio/wav")
        req = urllib.request.Request(self.url + "/inference", data=body, headers={"Content-Type": ctype})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return clean_heard(json.loads(r.read().decode()).get("text", ""))


class Speaker:
    """Says things out loud one at a time, in a thread of its own, so the robot acts while Pupper talks.

    piper (a neural voice, fast on the Pi 5) when it and its voice are installed, else espeak-ng,
    else it only prints."""

    def __init__(self, engine: str = "auto", piper_voice: Path = PIPER_VOICE, device: str | None = None):
        local = Path.home() / ".local/bin/piper"  # pip install --user puts it here, often not on the PATH
        self.piper = shutil.which("piper") or (str(local) if local.exists() else "piper")
        if engine == "auto":
            engine = ("piper" if Path(self.piper).exists() and Path(piper_voice).exists()
                      else "espeak" if shutil.which("espeak-ng") else "print")
        self.engine, self.piper_voice, self.device = engine, Path(piper_voice), device
        self.voice = None  # Piper's voice, loaded once: the command line reloads it for every sentence (1-2 s)
        if engine == "piper":
            try:
                try:
                    from piper import PiperVoice  # piper-tts 1.3
                except ImportError:
                    from piper.voice import PiperVoice  # piper-tts 1.2
                self.voice = PiperVoice.load(str(self.piper_voice))
            except Exception as e:
                print(f"  (Piper's voice not loaded in this process: {e}; using its command line)")
        self.q: queue.Queue = queue.Queue()
        self.speaking = threading.Event()
        self.quiet_until = 0.0
        self.saying: str | None = None  # the sentence being said now (the face's caption)
        self.on_change = lambda: None   # called when Pupper starts or stops saying something
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, text: str) -> None:
        self.q.put(text)

    def busy(self) -> bool:
        return self.speaking.is_set() or not self.q.empty() or time.time() < self.quiet_until

    def _run(self) -> None:
        while True:
            text = self.q.get()
            self.speaking.set()
            self.saying = text
            self.on_change()
            try:
                self._speak(text)
            except (OSError, subprocess.CalledProcessError) as e:
                print(f"  (speech failed: {e})")
            finally:
                if self.q.empty():
                    self.quiet_until = time.time() + 0.4  # the room's echo of the last word
                    self.speaking.clear()
                    self.saying = None
                    self.on_change()

    def _speak(self, text: str) -> None:
        aplay = ["aplay", "-q"] + (["-D", self.device] if self.device else [])
        if self.voice is not None:
            t0, buf = time.time(), io.BytesIO()
            with wave.open(buf, "wb") as w:
                (getattr(self.voice, "synthesize_wav", None) or self.voice.synthesize)(text, w)
            print(f"  (voice: {time.time() - t0:.1f} s to make the sound)", flush=True)
            subprocess.run(aplay + ["-"], input=buf.getvalue(), check=True)
        elif self.engine == "piper":
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                subprocess.run([self.piper, "-m", str(self.piper_voice), "-f", f.name], input=text.encode(),
                               check=True, capture_output=True)
                subprocess.run(aplay + [f.name], check=True)
        elif self.engine == "espeak":
            subprocess.run(["espeak-ng", "-s", "165", text], check=True, capture_output=True)


# ============================================================ talking to Pupper
async def type_loop(brain: Brain, backend) -> None:
    print("Type to Pupper ('quit' to exit). Robot events come in with your next line.\n")
    while True:
        user = (await asyncio.to_thread(input, "You > ")).strip()
        if user.lower() in ("quit", "exit"):
            return
        if user and STOP_WORDS.match(user):
            await brain.turn(user)
            continue
        for event in await asyncio.to_thread(news, backend):
            await brain.turn(event, source="robot")
        if user:
            await brain.turn(user)


async def voice_loop(brain: Brain, backend, mic: Mic, stt: Whisper, speaker: Speaker, camera: bool) -> None:
    """The mic (a thread) cuts utterances; whisper (another thread) turns them into words; robot
    news is polled every second; each goes to Gemma as a turn, one at a time, in order."""
    loop = asyncio.get_running_loop()
    inbox: asyncio.Queue = asyncio.Queue()
    heard: queue.Queue = queue.Queue()

    def listen() -> None:
        while True:
            pcm = heard.get()
            t0 = time.time()
            try:
                text = stt.transcribe(pcm, mic.rate)
                print(f"  (whisper: {time.time() - t0:.1f} s for {len(pcm) / 2 / mic.rate:.1f} s of speech)", flush=True)
            except (urllib.error.URLError, OSError, ValueError) as e:
                print(f"  !! whisper did not answer: {e}")
                continue
            if not text:
                continue
            if STOP_WORDS.match(text):  # "stop" doesn't wait for Gemma
                print("  [stop heard: stopping now]")
                asyncio.run_coroutine_threadsafe(brain.stop(), loop)
                continue
            loop.call_soon_threadsafe(inbox.put_nowait, (brain.generation, "person", text))

    def publish() -> None:  # to the face: on each change, and every second (so it knows we're alive)
        brain_state(backend, not speaker.busy(), speaker.speaking.is_set(), speaker.saying)

    speaker.on_change = publish

    async def poll_news() -> None:
        while True:
            await asyncio.to_thread(publish)
            try:
                for text in await asyncio.to_thread(news, backend):
                    inbox.put_nowait((brain.generation, "robot", text))
            except (urllib.error.URLError, OSError) as e:
                print(f"  !! the robot server did not answer: {e}")
            await asyncio.sleep(1.0)

    threading.Thread(target=mic.run, args=(heard.put, speaker.busy), daemon=True).start()
    threading.Thread(target=listen, daemon=True).start()
    asyncio.create_task(poll_news())
    print("Listening. Talk to Pupper (Ctrl+C to stop).\n")
    while True:
        generation, who, text = await inbox.get()
        if generation != brain.generation:
            continue  # a stop cancels requests waiting behind the old decision too
        if who == "person":
            print(f"You > {text}")
            transcript(backend, "person", text)
        image = await asyncio.to_thread(camera_frame, backend) if camera else None
        if generation == brain.generation:
            await brain.turn(text, image, source=who)


# ============================================================ checks, and main
def check(ollama_url: str, whisper_url: str, model: str | None, voice: bool = False) -> int:
    """Is the Mac reachable from here, and can its Gemma call a
    tool? With voice, whisper must answer too. Exits non-zero, saying what to fix, if not."""
    print(f"Ollama at {ollama_url} ... ", end="", flush=True)
    try:
        names = Ollama(ollama_url, "").models()
    except (OllamaError, urllib.error.URLError, OSError) as e:
        print(f"unreachable ({e}).\n  Start Pupper from the Mac (./run_pupper.sh there): it runs Ollama and"
              " reaches the Pupper through the ssh connection.")
        return 1
    print("ok: " + (", ".join(names) or "no models"))
    if model and model not in names and model + ":latest" not in names:
        print(f"  {model} isn't pulled there: ollama pull {model}")
        return 1
    chosen = model or pick_model(names)
    if not chosen:
        print("  No Gemma pulled. On the Mac: ollama pull gemma4:26b")
        return 1
    llm = Ollama(ollama_url, chosen)
    t0 = time.time()
    try:
        m = llm.chat([{"role": "user", "content": "Say hello in five words."}])
        print(f"  {chosen} talks ({time.time() - t0:.1f} s, the first call loads it): {m.get('content', '').strip()[:80]!r}")
        t0 = time.time()
        tool = [{"name": "stand_up", "description": "Stand up, ready to walk."}]
        m = llm.chat([{"role": "user", "content": "Please stand up."}], tools=ollama_tools(tool))
        calls = native_calls(m) or calls_in_text(m.get("content") or "", {"stand_up"})[1]
        print(f"  tool calls ({time.time() - t0:.1f} s): " + ("native, ok" if native_calls(m) else
              "written out as text (--tools auto copes)" if calls else
              f"none ({(m.get('content') or '').strip()[:60]!r}); try --tools json"))
    except OllamaError as e:
        print(f"  {chosen}: {e}" + ("  (--tools auto falls back to JSON replies)" if "does not support tools" in str(e) else ""))
    print(f"whisper at {whisper_url} ... ", end="", flush=True)
    try:
        urllib.request.urlopen(whisper_url, timeout=3).close()
        print("ok")
    except (urllib.error.URLError, OSError) as e:
        print(f"unreachable ({e})" + (": Pupper can't hear without it" if voice else ": only --voice needs it"))
        return 1 if voice else 0
    return 0


async def run(args, model: str) -> None:
    backend = RobotBackend(args.robot) if args.robot else DeskBackend()
    prompt, decls = backend.setup()
    if args.only:
        wanted = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = sorted(set(wanted) - {d["name"] for d in decls})
        if unknown:
            sys.exit(f"--only: no tool {unknown}; the tools are {sorted(d['name'] for d in decls)}")
        decls = [d for d in decls if d["name"] in wanted]
    if args.show:
        print("=== What Gemma reads: the prompt ===\n" + VOICE_RULES + "\n\n" + prompt + "\n")
        print("=== ... and the tools ===\n" + json.dumps(ollama_tools(decls), indent=2) + "\n")
    print(f"Gemma: {model} at {args.ollama}, with {backend.name}.")
    print(f"Its {len(decls)} tools: " + ", ".join(d["name"] for d in decls))

    llm = Ollama(args.ollama, model, args.num_ctx)
    if args.voice:
        speaker = Speaker(args.tts, args.piper_voice, args.speaker)
        print(f"Voice: {speaker.engine}; ears: whisper at {args.whisper}; mic: {args.mic}")
        if speaker.engine == "print":
            print("!! no Piper and no espeak-ng: Pupper will only print what it says (robot/setup_voice.sh)")

        def say(text: str) -> None:
            print(f"Pupper > {text}")
            speaker.say(text)
            transcript(backend, "pupper", text)

        brain = Brain(llm, prompt, decls, backend, say, args.tools)
    else:
        brain = Brain(llm, prompt, decls, backend, lambda text: print(f"Pupper > {text}"), args.tools)
    print("Gemma is reading the prompt and tools once, so the first turn is quick...", flush=True)
    try:
        print(f"  ready ({await asyncio.to_thread(brain.warm):.1f} s)")
    except (OllamaError, urllib.error.URLError, OSError) as e:
        print(f"  !! Gemma did not answer: {e}")
    if args.voice:
        await voice_loop(brain, backend, Mic(args.mic), Whisper(args.whisper), speaker, args.camera)
    else:
        await type_loop(brain, backend)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", nargs="?", const="http://localhost:8000", metavar="URL",
                    help="the real robot server (default http://localhost:8000): Pupper moves!")
    ap.add_argument("--voice", action="store_true", help="talk and listen (mic, whisper, Piper) instead of typing")
    ap.add_argument("--check", action="store_true", help="check the Mac's Ollama and whisper from here, and exit")
    ap.add_argument("--ollama", default=OLLAMA_URL, help=f"Ollama on the Mac (default {OLLAMA_URL}; $PUPPER_OLLAMA)")
    ap.add_argument("--model", default=os.environ.get("PUPPER_GEMMA"),
                    help="the Gemma in Ollama (default: the biggest one it has; $PUPPER_GEMMA)")
    ap.add_argument("--tools", choices=["auto", "native", "json"], default="auto",
                    help="how Gemma calls tools: Ollama's tools, or JSON replies (auto: native, JSON if it can't)")
    ap.add_argument("--only", help="comma-separated tool names to declare (a small model picks better from fewer)")
    ap.add_argument("--num-ctx", type=int, default=NUM_CTX, help=f"Gemma's context in tokens (default {NUM_CTX})")
    ap.add_argument("--camera", action="store_true", help="--voice --robot: send the newest camera frame with each turn")
    ap.add_argument("--whisper", default=WHISPER_URL, help=f"whisper.cpp's server (default {WHISPER_URL}; $PUPPER_WHISPER)")
    ap.add_argument("--mic", default="default", help="the arecord device (default: PipeWire's default source)")
    ap.add_argument("--speaker", default=None, help="the aplay device for Piper (default: the default sink)")
    ap.add_argument("--tts", choices=["auto", "piper", "espeak", "print"], default="auto")
    ap.add_argument("--piper-voice", type=Path, default=PIPER_VOICE, help=f"the Piper voice (default {PIPER_VOICE})")
    ap.add_argument("--show", action="store_true", help="print the whole prompt and every tool declaration first")
    args = ap.parse_args()

    if args.check:
        sys.exit(check(args.ollama, args.whisper, args.model, args.voice))
    model = args.model
    if not model:
        try:
            model = pick_model(Ollama(args.ollama, "").models())
        except (OllamaError, urllib.error.URLError, OSError) as e:
            sys.exit(f"Can't reach Ollama at {args.ollama} ({e}). python3 robot/gemma_brain.py --check helps.")
        if not model:
            sys.exit(f"Ollama at {args.ollama} has no Gemma: on the Mac, ollama pull gemma4:26b")

    try:
        asyncio.run(run(args, model))  # setup inside the loop: the fake robot's job queue lives on it
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
