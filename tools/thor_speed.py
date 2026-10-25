#!/usr/bin/env python3
"""How fast is Gemma on the Thor, really?   python3 tools/thor_speed.py [http://10.42.0.1:8000]

One "42 tokens in 5.1 s" number mixes three things. This measures them apart, streamed:
  first token  the wait before the first word (network + reading the prompt)
  decode       words per second once it is writing (what a long answer costs)
  Pupper turn  a real-sized brain request: the lab prompt and the 30 tools (~9k tokens), first
               cold, then again (vLLM's prefix cache: what every later turn costs), then with a
               tool call, the way the brain asks
Run it from the Mac, on the Thor's network. Stdlib only. With no address it tries thornet, then LatentSpace.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def stream(url: str, body: dict) -> tuple[float, float, int, str, list]:
    """(seconds to the first token, seconds in all, tokens out, text, tool call names)."""
    body = {**body, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, first, text, n, calls = time.time(), None, "", 0, []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                n = d["usage"].get("completion_tokens", n)
            for c in d.get("choices") or []:
                delta = c.get("delta") or {}
                if (delta.get("content") or delta.get("tool_calls")) and first is None:
                    first = time.time() - t0
                text += delta.get("content") or ""
                for tc in delta.get("tool_calls") or []:
                    if (tc.get("function") or {}).get("name"):
                        calls.append(tc["function"]["name"])
    return first or float("nan"), time.time() - t0, n, text, calls


def pupper_request() -> tuple[str, list]:
    """The brain's real system prompt and tools, from the repo, on the fake robot (nothing moves)."""
    import contextlib, io, os, tempfile
    os.environ.setdefault("PUPPER_MEMORY_FILE", str(Path(tempfile.mkdtemp()) / "memory.json"))
    sys.path.insert(0, str(REPO / "robot"))
    import gemma_brain as gb
    with contextlib.redirect_stdout(io.StringIO()):
        prompt, decls = gb.DeskBackend().setup()
    return gb.VOICE_RULES + "\n\n" + prompt, gb.ollama_tools(decls)


def main() -> None:
    urls = sys.argv[1:] or ["http://10.42.0.1:8000", "http://192.168.2.5:8000"]
    url = model = None
    for u in urls:
        try:
            with urllib.request.urlopen(u.rstrip("/") + "/v1/models", timeout=3) as r:
                url, model = u.rstrip("/"), json.loads(r.read())["data"][0]["id"]
                break
        except OSError:
            print(f"no Thor at {u}")
    if not url:
        sys.exit("!! no Thor answered. Is this Mac on its network?")
    print(f"Thor at {url}, model {model}\n")

    t0 = time.time()
    urllib.request.urlopen(url + "/v1/models", timeout=5).read()
    print(f"network round trip         {1000 * (time.time() - t0):6.0f} ms")

    first, total, n, _, _ = stream(url, {"model": model, "max_tokens": 1,
                                         "messages": [{"role": "user", "content": "Hi"}]})
    print(f"first token, tiny prompt   {first:6.2f} s")

    rates = []
    for _ in range(2):
        first, total, n, _, _ = stream(url, {"model": model, "max_tokens": 200, "temperature": 0,
                                             "messages": [{"role": "user", "content": "Write 150 words about dogs."}]})
        rates.append((n - 1) / max(total - first, 1e-6))
    print(f"decode (200 tokens, x2)    {rates[0]:6.1f}, {rates[1]:.1f} tokens/s   <- the real writing speed")

    try:
        system, tools = pupper_request()
    except Exception as e:  # no PyYAML etc.: skip the Pupper-sized part
        print(f"\n(Pupper-sized test skipped: {type(e).__name__}: {e})")
        return
    msgs = lambda text: [{"role": "system", "content": system}, {"role": "user", "content": text}]
    print(f"\nPupper-sized: {len(system)} chars of prompt, {len(tools)} tools")
    for label, text in [("cold (prompt read)", "Hi Pupper, how are you?"), ("again (cached)", "Hi Pupper, what's up?")]:
        first, total, n, said, _ = stream(url, {"model": model, "max_tokens": 60, "temperature": 0.3,
                                                "messages": msgs(text), "tools": tools})
        print(f"  {label:20} first token {first:5.2f} s, whole reply {total:5.2f} s, {n} tokens: {said.strip()[:60]!r}")
    try:
        first, total, n, said, calls = stream(url, {"model": model, "max_tokens": 120, "temperature": 0.3,
                                                    "messages": msgs("Walk forward for two seconds."), "tools": tools})
        print(f"  {'tool call':20} first token {first:5.2f} s, whole reply {total:5.2f} s, {n} tokens, calls {calls or 'none'}")
    except urllib.error.HTTPError as e:
        print(f"  tool call: the Thor said {e.code}: {e.read().decode()[:200]}")
        print("  -> tool calling is off: add --enable-auto-tool-choice --tool-call-parser gemma4 to vllm serve")


if __name__ == "__main__":
    main()
