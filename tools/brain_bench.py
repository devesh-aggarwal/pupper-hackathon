#!/usr/bin/env python3
"""Which Gemma makes the best brain? Each model hears the same 17 commands, through the brain's real
prompt and tools (robot/gemma_brain.py), on the fake robot (nothing moves), and is scored on what
it does.

    python3 tools/brain_bench.py gemma4:26b gemma4:26b-a4b-it-qat gemma4:26b-mlx

  right   the call the command needs, with the right arguments (and for a refusal, the limit said)
  clean   right, and nothing else that moves the robot (no extra tricks, no stray walks)
  junk    speech with markup or leftover thinking in it (Pupper would read it out)
  reply   the median time to Gemma's first reply in a turn (what you wait for after speaking)
  first   the first turn after loading: Gemma reads the whole prompt and tools once (later turns reuse it)

Every model must already be pulled (ollama pull <model>). Each command is a fresh conversation.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROBOT = Path(__file__).resolve().parent.parent / "robot"
_TMP = Path(tempfile.mkdtemp(prefix="pupper_bench_"))
os.environ.setdefault("PUPPER_MEMORY_FILE", str(_TMP / "memory.json"))  # never your real memory
os.environ.setdefault("PUPPER_LOG_DIR", str(_TMP / "logs"))
sys.path.insert(0, str(ROBOT))
import gemma_brain as gb  # noqa: E402

MOTION = {"move", "turn", "arc", "play_trick", "interactive_action", "follow", "go_to", "find", "stand_up", "lie_down"}


def called(calls, tool, **want) -> bool:
    """Was `tool` called with these arguments? A value can be a test (a function of the argument)."""
    for name, args in calls:
        if name == tool and all(v(args.get(k)) if callable(v) else args.get(k) == v for k, v in want.items()):
            return True
    return False


def num(x):
    return lambda got: isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - x) < 0.01


def has_text(s):
    return lambda got: isinstance(got, str) and s in got.lower()


# (what the person says, is it right, the tools a clean answer may use, the most calls it needs)
CASES = [
    ("Stand up.", lambda c, s: called(c, "stand_up"), "stand_up robot_status", 1),
    ("Walk forward for two seconds.", lambda c, s: called(c, "move", direction="forward", duration=num(2)), "move stand_up robot_status", 2),
    ("Turn left 90 degrees.", lambda c, s: called(c, "turn", direction="left", angle=num(90)), "turn stand_up robot_status", 2),
    ("Turn around.", lambda c, s: called(c, "turn", angle=num(180)), "turn stand_up robot_status", 2),
    ("Step to the right a little.", lambda c, s: called(c, "move", direction="right"), "move stand_up robot_status", 2),
    ("Do a push-up!", lambda c, s: called(c, "play_trick", name="push_up"), "play_trick stand_up robot_status", 2),
    ("Sit down and give me your left paw.", lambda c, s: called(c, "play_trick", name="sit_stay_shake")
     and called(c, "interactive_action", action="shake_left"), "play_trick interactive_action robot_status", 3),
    ("Remind me in 20 minutes to take the pizza out.", lambda c, s: called(c, "set_reminder", in_minutes=num(20),
                                                                           message=has_text("pizza")), "set_reminder", 1),
    ("Remember that I like tennis balls.", lambda c, s: called(c, "remember", fact=has_text("tennis")), "remember", 1),
    ("Put WAKE UP on your screen in red.", lambda c, s: called(c, "show_text", text=has_text("wake up"), color="red"), "show_text", 1),
    ("Where is my cup? Go find it.", lambda c, s: called(c, "find", what=has_text("cup")), "find recall look robot_status", 2),
    ("Follow me.", lambda c, s: called(c, "follow", enabled=True), "follow", 1),
    ("Lie down.", lambda c, s: called(c, "lie_down"), "lie_down", 1),
    ("Walk in a square.", lambda c, s: sum(n == "move" for n, _ in c) >= 4 and sum(n == "turn" for n, _ in c) >= 3,
     "move turn stand_up robot_status", 9),
    ("Curve to the left in a half circle.", lambda c, s: called(c, "arc", direction="left", angle=num(180)), "arc stand_up robot_status", 2),
    ("Hi Pupper, how are you today?", lambda c, s: not any(n in MOTION for n, _ in c) and len(s) > 5, "robot_status recall", 1),
    ("Run forward at nine metres per second.", lambda c, s: "1.5" in s or "too fast" in s.lower(), "move stand_up robot_status", 3),
]
JUNK = ("<", "```", "thought", "{", "tool_call", "|>")


async def one(url: str, model: str, text: str, right, allowed: str, most: int, verbose: bool) -> dict:
    backend = gb.DeskBackend()
    with contextlib.redirect_stdout(io.StringIO()):  # the fake robot's start-up lines
        prompt, decls = backend.setup()
    calls, said, replies = [], [], []
    run = backend.call

    async def recorded(name, args):
        calls.append((name, args))
        return await run(name, args)

    backend.call, backend.after = recorded, lambda: None
    llm = gb.Ollama(url, model)
    chat = llm.chat

    def timed(*a, **k):
        t0 = time.time()
        m = chat(*a, **k)
        replies.append(time.time() - t0)
        return m

    llm.chat = timed
    brain = gb.Brain(llm, prompt, decls, backend, said.append, "auto", log=lambda line: None)
    await brain.turn(text)
    backend.rig.close()
    speech = " ".join(said)
    ok = bool(right(calls, speech))
    res = {"ok": ok, "clean": ok and len(calls) <= most and all(n in allowed.split() for n, _ in calls),
           "junk": not speech or any(j in speech for j in JUNK), "reply_s": replies[0] if replies else float("nan"),
           "mode": brain.mode}
    if verbose:
        mark = "clean" if res["clean"] else "right" if ok else "WRONG"
        print(f"  {mark} {res['reply_s']:5.1f} s  {text:47} {calls} {speech[:90]!r}")
    return res


async def bench(url: str, model: str, verbose: bool) -> dict:
    t0 = time.time()
    gb.Ollama(url, model).chat([{"role": "user", "content": "Hi"}])  # load it
    load_s = time.time() - t0
    results = [await one(url, model, *case, verbose) for case in CASES]
    gb.Ollama(url, model)._req("/api/generate", {"model": model, "keep_alive": 0})  # unload: room for the next
    return {"model": model, "right": sum(r["ok"] for r in results), "clean": sum(r["clean"] for r in results),
            "junk": sum(r["junk"] for r in results), "reply_s": statistics.median(r["reply_s"] for r in results[1:]),
            "first_s": results[0]["reply_s"], "load_s": load_s, "mode": results[-1]["mode"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("-q", "--quiet", action="store_true", help="only the table, not every command")
    args = ap.parse_args()
    loop = asyncio.new_event_loop()  # the fake robot's job queue lives on it
    asyncio.set_event_loop(loop)
    rows = []
    for model in args.models:
        print(f"{model}:")
        try:
            rows.append(loop.run_until_complete(bench(args.ollama, model, not args.quiet)))
        except (gb.OllamaError, OSError) as e:
            print(f"  skipped: {e}")
    print(f"\n{'model':48} right  clean  junk  reply   first   tools")
    for r in sorted(rows, key=lambda r: (-r["clean"], -r["right"], r["reply_s"])):
        print(f"{r['model']:48} {r['right']:2}/{len(CASES)}  {r['clean']:2}/{len(CASES)}  {r['junk']:4}  "
              f"{r['reply_s']:4.1f} s {r['first_s']:5.1f} s   {r['mode']}")


if __name__ == "__main__":
    main()
