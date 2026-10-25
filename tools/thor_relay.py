#!/usr/bin/env python3
"""Gemma on the Thor, in Ollama's terms, on the Mac:   python3 tools/thor_relay.py --thor http://10.42.0.1:8000

The Thor (a Jetson with a GPU) serves Gemma 4 31B with vLLM, which speaks OpenAI's API (/v1). The
Pupper's brain (robot/gemma_brain.py) and eyes (vision/eyes.py) speak Ollama's (/api/chat,
/api/generate). This stands where tools/ollama_relay.py stands (port 11435, which the link makes
localhost on the Pupper) and translates, so nothing on the Pupper changes: ./run_thor.sh runs it.

  /api/chat      the brain: messages, tools and tool calls, images -> /v1/chat/completions
  /api/generate  the eyes: a prompt and an image (and "format": "json") -> /v1/chat/completions
  /api/tags      lists the Thor's model, and also the name the Pupper asks for (gemma4:26b), which
                 is answered by the Thor's model: the Pupper's code checks for that name
  /relay         says gzip is welcome, as tools/ollama_relay.py does (the Pupper's WiFi is slow)

The Thor's vLLM must be started with tool calling on (--enable-auto-tool-choice and a
--tool-call-parser), or it refuses the brain's requests; this says so, with the fix.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ASKED_AS = "gemma4:26b"  # the model name the Pupper's brain and eyes ask Ollama for
TOOL_FIX = ("the Thor's vLLM has tool calling off. On the Thor, restart it with tool calling on: add "
            "--enable-auto-tool-choice --tool-call-parser gemma4 to the vllm serve line in ~/setup_endpoint.sh, "
            "then sudo bash ~/setup_endpoint.sh")


def to_openai(messages: list[dict]) -> list[dict]:
    """Ollama's messages as OpenAI's: tool calls get ids, each tool reply the id of its call, and
    images become image parts."""
    out, pending = [], []  # pending: (id, name) of calls not yet answered
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for j, tc in enumerate(m["tool_calls"]):
                fn = tc.get("function") or {}
                args = fn.get("arguments") or {}
                cid = tc.get("id") or f"call_{i}_{j}"
                calls.append({"id": cid, "type": "function", "function": {
                    "name": fn.get("name", ""), "arguments": args if isinstance(args, str) else json.dumps(args)}})
                pending.append((cid, fn.get("name", "")))
            out.append({"role": "assistant", "content": m.get("content") or None, "tool_calls": calls})
        elif role == "tool":
            name = m.get("tool_name") or m.get("name")
            k = next((k for k, (_, n) in enumerate(pending) if n == name), 0 if pending else None)
            cid = pending.pop(k)[0] if k is not None else f"call_{i}"
            out.append({"role": "tool", "tool_call_id": cid, "content": m.get("content") or ""})
        elif m.get("images"):
            parts = [{"type": "text", "text": m.get("content") or ""}] + [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} for b64 in m["images"]]
            out.append({"role": role, "content": parts})
        else:
            out.append({"role": role, "content": m.get("content") or ""})
    return out


def from_openai(msg: dict) -> dict:
    """OpenAI's reply message as Ollama's: arguments as objects, not strings."""
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments") or "{}"
        try:
            args = json.loads(args) if isinstance(args, str) else args
        except ValueError:
            args = {}
        calls.append({"id": tc.get("id"), "function": {"name": fn.get("name", ""), "arguments": args}})
    out = {"role": "assistant", "content": msg.get("content") or ""}
    if calls:
        out["tool_calls"] = calls
    return out


class Thor:
    def __init__(self, url: str, model: str | None = None, timeout: float = 600.0):
        self.url, self.timeout = url.rstrip("/"), timeout
        self.model = model or self.models()[0]

    def _req(self, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            return json.loads(r.read().decode())

    def models(self) -> list[str]:
        return [m["id"] for m in self._req("/v1/models", timeout=5).get("data", [])]

    def complete(self, messages: list[dict], tools=None, temperature=None, max_tokens=None, json_mode=False) -> dict:
        body = {"model": self.model, "messages": messages,
                "chat_template_kwargs": {"enable_thinking": False}}  # as the brain asks Ollama: think off
        if tools:
            body["tools"] = tools
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens:
            body["max_tokens"] = max_tokens
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return self._req("/v1/chat/completions", body)


def ollama_reply(thor: Thor, path: str, req: dict) -> dict:
    """One Ollama request (/api/chat or /api/generate), answered by the Thor, in Ollama's terms."""
    t0 = time.time()
    opts = req.get("options") or {}
    if path == "/api/chat":
        r = thor.complete(to_openai(req.get("messages") or []), tools=req.get("tools"),
                          temperature=opts.get("temperature"), max_tokens=opts.get("num_predict"),
                          json_mode=req.get("format") == "json")
    else:  # /api/generate: a prompt, maybe images
        m = {"role": "user", "content": req.get("prompt") or ""}
        if req.get("images"):
            m["images"] = req["images"]
        r = thor.complete(to_openai([m]), temperature=opts.get("temperature"),
                          max_tokens=opts.get("num_predict"), json_mode=req.get("format") == "json")
    msg = from_openai(((r.get("choices") or [{}])[0]).get("message") or {})
    usage = r.get("usage") or {}
    out = {"model": req.get("model", thor.model), "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "done": True, "done_reason": "stop", "total_duration": int((time.time() - t0) * 1e9),
           "prompt_eval_count": usage.get("prompt_tokens", 0), "eval_count": usage.get("completion_tokens", 0)}
    if path == "/api/chat":
        out["message"] = msg
    else:
        out["response"] = msg["content"]
    return out


def serve(port: int, thor: Thor, log=print) -> ThreadingHTTPServer:
    class Relay(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _reply(self, code: int, payload) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_HEAD(self):
            self._reply(200, {})

        def do_GET(self):
            if self.path == "/relay":  # how the brain knows it may gzip (and run_thor.sh, which relay this is)
                return self._reply(200, {"gzip": True, "backend": "thor", "thor": thor.url, "model": thor.model})
            if self.path == "/api/version":
                return self._reply(200, {"version": "thor-relay"})
            if self.path == "/api/tags":
                return self._reply(200, {"models": [{"name": thor.model}, {"name": ASKED_AS}]})
            if self.path == "/api/ps":
                return self._reply(200, {"models": [{"name": thor.model}]})
            self._reply(404, {"error": f"thor relay: no {self.path}"})

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            if body and self.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            try:
                req = json.loads(body or b"{}")
            except ValueError:
                return self._reply(400, {"error": "thor relay: the request isn't JSON"})
            if self.path not in ("/api/chat", "/api/generate"):
                if self.path in ("/api/show", "/api/pull"):  # ollama commands: the Thor's model is always there
                    return self._reply(200, {"status": "success", "model": thor.model})
                return self._reply(404, {"error": f"thor relay: no {self.path}"})
            if req.get("stream"):
                return self._reply(400, {"error": "thor relay: streaming isn't translated; send stream: false"})
            if not req.get("messages") and not req.get("prompt"):  # an Ollama load or unload request
                return self._reply(200, {"model": req.get("model", thor.model), "done": True, "response": "", "message": {"role": "assistant", "content": ""}})
            t0 = time.time()
            try:
                out = ollama_reply(thor, self.path, req)
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="replace")
                if "tool" in text.lower() and ("auto" in text.lower() or "parser" in text.lower()):
                    text = TOOL_FIX + f" (vLLM said: {text[:200]})"
                log(f"!! {self.path}: the Thor said {e.code}: {text[:300]}")
                return self._reply(e.code, {"error": f"the Thor said {e.code}: {text[:400]}"})
            except (urllib.error.URLError, OSError) as e:
                log(f"!! {self.path}: the Thor at {thor.url} did not answer: {e}")
                return self._reply(502, {"error": f"thor relay: the Thor at {thor.url} did not answer: {e}"})
            calls = [c["function"]["name"] for c in (out.get("message") or {}).get("tool_calls", [])]
            log(f"{self.path} {time.time() - t0:5.1f} s  {out['prompt_eval_count']} in, {out['eval_count']} out"
                + (f"  calls {calls}" if calls else ""))
            self._reply(200, out)

    return ThreadingHTTPServer(("127.0.0.1", port), Relay)  # this Mac only: the link carries it to the Pupper


def check(thor: Thor) -> int:
    """Can the Thor talk, and call a tool? Says what to fix if not."""
    print(f"Thor at {thor.url}: model {thor.model}")
    try:
        t0 = time.time()
        r = ollama_reply(thor, "/api/chat", {"messages": [{"role": "user", "content": "Say hello in five words."}]})
        print(f"  talks ({time.time() - t0:.1f} s): {r['message']['content'].strip()[:80]!r}")
        t0 = time.time()
        tool = [{"type": "function", "function": {"name": "stand_up", "description": "Stand up, ready to walk.",
                                                  "parameters": {"type": "object", "properties": {}}}}]
        r = ollama_reply(thor, "/api/chat", {"messages": [{"role": "user", "content": "Please stand up."}], "tools": tool})
        calls = [c["function"]["name"] for c in r["message"].get("tool_calls", [])]
        print(f"  tool calls ({time.time() - t0:.1f} s): " + ("ok" if calls == ["stand_up"] else
              f"none (it said {r['message']['content'].strip()[:80]!r}); is the --tool-call-parser right?"))
        return 0 if calls == ["stand_up"] else 1
    except urllib.error.HTTPError as e:
        text = e.read().decode(errors="replace")
        print(f"  !! the Thor said {e.code}: {text[:300]}")
        if "tool" in text.lower():
            print(f"  -> {TOOL_FIX}")
        return 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thor", required=True, help="the Thor's vLLM, e.g. http://10.42.0.1:8000")
    ap.add_argument("--port", type=int, default=11435, help="where the Pupper reaches Gemma (through the link)")
    ap.add_argument("--model", help="the Thor's model (default: the one it serves)")
    ap.add_argument("--check", action="store_true", help="check the Thor talks and calls a tool, and exit")
    args = ap.parse_args()
    try:
        thor = Thor(args.thor, args.model)
    except (urllib.error.URLError, OSError, IndexError, ValueError) as e:
        raise SystemExit(f"!! the Thor at {args.thor} didn't answer /v1/models ({e}). Same network? Is vLLM up?")
    if args.check:
        raise SystemExit(check(thor))
    server = serve(args.port, thor, log=lambda line: print(line, flush=True))
    print(f"thor relay on 127.0.0.1:{args.port} -> {thor.url} ({thor.model}); gzip requests welcome", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
