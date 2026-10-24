"""Hackathon, part A: memory. Pupper remembers people, facts and where things are.

The lab's Pupper forgets everything when the browser reconnects. For the hackathon
it keeps a small memory file on the robot, outside the repo (never committed):

    ~/.config/gemini-pupper/memory.json      (or $PUPPER_MEMORY_FILE)

What is in it:
  profiles      one per person Pupper knows ("devesh", "guest", ...): facts about
                them ("likes tennis balls", "wakes up at 9"), and where their
                things were last seen ("keys": "on the desk", 14:02)
  active        the profile Pupper is talking to right now
  reminders     (not used: reminders are kept by robot/reminder_store.py)
  routines      tricks taught on the fly (hackathon_learn.py)
  settings      small switches, e.g. sound alerts on/off

How Gemini gets it: the robot server appends memory_prompt() to lab/prompt.md
(GET /tools), so a new session starts knowing the person. During a session the
remember / recall / forget / switch_profile tools below read and write it.

Easy reset, three ways: the forget tool ("forget everything"), the Memory card in the
app's debug view, or on the robot:

    python3 lab/hackathon_memory.py show        (or: summary, one line)
    python3 lab/hackathon_memory.py reset [profile|all]
    python3 lab/hackathon_memory.py switch <name>

Same two halves as every lab file: what the robot does (the store and the
behaviors), then what Gemini reads (the @tools).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

if __name__ == "__main__":  # the CLI: a fake ROS, so it runs on any laptop (like lab/chat.py)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "robot"))
    import test_tools  # noqa: F401
from robot_api import Result, Robot  # noqa: E402
from tool_registry import Param, tool  # noqa: E402

DEFAULT_PROFILE = "guest"
MAX_FACTS = 200  # per profile; the oldest go when it is full
MAX_FACT_CHARS = 300

# Where the memory lives. Tests point this at a temporary file.
MEMORY_FILE = Path(os.environ.get("PUPPER_MEMORY_FILE") or Path.home() / ".config" / "gemini-pupper" / "memory.json")


# ============================================================ what the robot does: the store
def _empty() -> dict:
    return {"active": DEFAULT_PROFILE, "profiles": {}, "reminders": [], "routines": {}, "settings": {}}


def load() -> dict:
    """The whole memory. A missing or broken file is an empty memory, never a crash."""
    try:
        data = json.loads(MEMORY_FILE.read_text())
        if not isinstance(data, dict):
            raise ValueError("not a dict")
    except (OSError, ValueError):
        return _empty()
    base = _empty()
    base.update(data)
    return base


def save(data: dict) -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = MEMORY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(MEMORY_FILE)  # atomic: a crash mid-write never leaves half a file


def profile_name(name: str) -> str:
    """"Devesh A." -> "devesh_a": one spelling per person."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return cleaned or DEFAULT_PROFILE


def active_profile(data: dict) -> dict:
    """The active profile's record (created empty if needed)."""
    name = data.get("active") or DEFAULT_PROFILE
    return data["profiles"].setdefault(name, {"facts": [], "objects": {}, "created": time.time()})


def stamp(t: float | None = None) -> str:
    return datetime.fromtimestamp(t or time.time()).strftime("%a %H:%M")


def ago(t: float, now: float | None = None) -> str:
    """"3 min ago", "2 h ago", "yesterday", in words Gemini can say out loud."""
    s = max(0.0, (now or time.time()) - t)
    if s < 60:
        return "just now"
    if s < 3600:
        return f"{int(s // 60)} min ago"
    if s < 86400:
        return f"{int(s // 3600)} h ago"
    return "yesterday" if s < 2 * 86400 else f"{int(s // 86400)} days ago"


# ============================================================ what the robot does: behaviors
DEAF = re.compile(r"\b(deaf|hard of hearing|hearing[- ]impaired|can'?t hear|cannot hear|lip[- ]?read\w*|sign language)\b", re.I)


def needs_text() -> bool:
    """Is the person Pupper is talking with deaf or hard of hearing (a fact it remembers)? Only then does
    the screen show text: captions, show_text, reminder and alert cards. Everyone else just hears Pupper."""
    return any(DEAF.search(f["text"]) for f in active_profile(load())["facts"])


def remember(fact: str) -> Result:
    """Save one fact about the active person."""
    fact = " ".join(fact.split())[:MAX_FACT_CHARS]
    if len(fact) < 3:
        return Result(False, "nothing to remember: say the fact in a few words, e.g. 'likes tennis balls'")
    data = load()
    prof = active_profile(data)
    if any(f["text"].lower() == fact.lower() for f in prof["facts"]):
        return Result(True, f"already remembered for {data['active']}: {fact}")
    prof["facts"].append({"text": fact, "time": time.time()})
    del prof["facts"][:-MAX_FACTS]
    save(data)
    return Result(True, f"remembered for {data['active']}: {fact} ({len(prof['facts'])} facts now)")


def remember_object(name: str, where: str) -> Result:
    """Where one of the person's things was last seen (one place per thing)."""
    name, where = " ".join(name.split()).lower()[:60], " ".join(where.split())[:MAX_FACT_CHARS]
    if not name or len(where) < 2:
        return Result(False, "say what the thing is and where it is, e.g. keys, on the desk")
    data = load()
    prof = active_profile(data)
    prof["objects"][name] = {"where": where, "time": time.time()}
    save(data)
    return Result(True, f"noted for {data['active']}: {name} last seen {where} ({stamp()})")


def where_is(name: str) -> str | None:
    """"on the desk (12 min ago)" if the active person's thing was seen, else None."""
    name = " ".join(name.split()).lower()
    prof = active_profile(load())
    hit = prof["objects"].get(name) or next((v for k, v in prof["objects"].items() if name in k or k in name), None)
    return f"{hit['where']} ({ago(hit['time'])})" if hit else None


def recall() -> Result:
    """Everything Pupper knows about the active person, in words."""
    data = load()
    prof = active_profile(data)
    others = sorted(p for p in data["profiles"] if p != data["active"])
    lines = [f"talking with: {data['active']}" + (f" (other people I know: {', '.join(others)})" if others else "")]
    if prof["facts"]:
        lines.append("facts: " + "; ".join(f["text"] for f in prof["facts"]))
    else:
        lines.append("facts: none yet")
    if prof["objects"]:
        lines.append("things: " + "; ".join(f"{k} {v['where']} ({ago(v['time'])})" for k, v in prof["objects"].items()))
    if data["routines"]:
        lines.append("learned routines: " + ", ".join(data["routines"]))
    return Result(True, ". ".join(lines))


def forget(what: str) -> Result:
    """Drop the facts (and things) that mention ``what``; "everything" clears the active profile."""
    what = " ".join(what.split()).lower()
    if not what:
        return Result(False, "say what to forget: a word from the fact, or 'everything'")
    data = load()
    prof = active_profile(data)
    if what in ("everything", "all", "everything about me"):
        n = len(prof["facts"]) + len(prof["objects"])
        prof["facts"], prof["objects"] = [], {}
        save(data)
        return Result(True, f"forgot everything about {data['active']} ({n} entries)")
    gone = [f["text"] for f in prof["facts"] if what in f["text"].lower()]
    prof["facts"] = [f for f in prof["facts"] if what not in f["text"].lower()]
    gone_things = [k for k in prof["objects"] if what in k or what in prof["objects"][k]["where"].lower()]
    for k in gone_things:
        del prof["objects"][k]
    if not gone and not gone_things:
        return Result(False, f"nothing about '{what}' in {data['active']}'s memory; facts: "
                             + ("; ".join(f["text"] for f in prof["facts"]) or "none"))
    save(data)
    return Result(True, f"forgot {len(gone) + len(gone_things)}: " + "; ".join(gone + gone_things))


def switch_profile(name: str) -> Result:
    """Talk to someone else (a new name makes a new, empty profile)."""
    pname = profile_name(name)
    data = load()
    new = pname not in data["profiles"]
    data["active"] = pname
    prof = active_profile(data)
    save(data)
    if new:
        return Result(True, f"nice to meet you, {pname}: new profile, nothing remembered yet")
    return Result(True, f"switched to {pname}: {len(prof['facts'])} facts, {len(prof['objects'])} things remembered")


def reset_memory(scope: str = "profile") -> Result:
    """The app's and the CLI's reset: the active profile, or all memory (reminders and routines too)."""
    data = load()
    if scope == "all":
        save(_empty())
        return Result(True, "memory wiped: all profiles, reminders, routines and settings")
    name = data["active"]
    data["profiles"].pop(name, None)
    active_profile(data)
    save(data)
    return Result(True, f"profile {name} reset")


def memory_prompt() -> str:
    """The section the robot server appends to the system prompt at connect time."""
    data = load()
    prof = active_profile(data)
    lines = [f"# What you remember (from {MEMORY_FILE.name}, use the memory tools to change it)", "",
             f"- You are talking with: {data['active']}."]
    if data["active"] == DEFAULT_PROFILE:
        lines.append("- You don't know this person's name yet. Ask, then call switch_profile with it.")
    for f in prof["facts"][-40:]:
        lines.append(f"- {f['text']}")
    for k, v in prof["objects"].items():
        lines.append(f"- Their {k} was last seen {v['where']} ({ago(v['time'])}).")
    if data["routines"]:
        lines.append("- Routines you have learned (do_routine): " + ", ".join(data["routines"]) + ".")
    pending = [r for r in data["reminders"] if not r.get("done")]
    if pending:
        lines.append("- Reminders set: " + "; ".join(f"{r['message']} at {r.get('when_text', '?')}" for r in pending) + ".")
    return "\n".join(lines)


# ============================================================ what Gemini reads
@tool(
    "remember",
    "Save a fact about the person you are talking with, so you still know it after a restart:"
    " their name, likes, routine, who they are to you. Call it whenever they tell you something"
    " worth keeping ('I'm allergic to cats' -> remember(fact='allergic to cats')). Short, one fact"
    " per call. To note where a thing is, use found_object instead.",
    fact=Param("string", "The fact, in a few words, e.g. 'likes tennis balls', 'wakes up at 9'."),
)
async def remember_tool(robot: Robot, fact: str) -> Result:
    return remember(fact)


@tool(
    "recall",
    "What you remember about the person you are talking with: their facts, where their things were"
    " last seen, and the routines you have learned. Call it when asked 'what do you know about me',"
    " 'where are my keys', or when you need context.",
)
async def recall_tool(robot: Robot) -> Result:
    return recall()


@tool(
    "forget",
    "Forget facts about the current person. what='everything' clears their whole profile (ask"
    " 'are you sure?' first); any other word forgets the facts and things that mention it.",
    what=Param("string", "A word from the fact to forget, or 'everything'."),
)
async def forget_tool(robot: Robot, what: str) -> Result:
    return forget(what)


@tool(
    "switch_profile",
    "Start talking with a different person, by name. Each person has their own memory. Use it when"
    " someone introduces themselves ('I'm Devesh' -> switch_profile(name='Devesh')) or when a new"
    " person takes over. A new name starts an empty profile.",
    name=Param("string", "The person's first name."),
)
async def switch_profile_tool(robot: Robot, name: str) -> Result:
    return switch_profile(name)


# ============================================================ CLI: show / reset / switch
def _cli(argv: list[str]) -> int:
    cmd = argv[0] if argv else "show"
    if cmd == "show":
        data = load()
        print(f"memory file: {MEMORY_FILE}")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    elif cmd == "reset":
        print(reset_memory(argv[1] if len(argv) > 1 else "profile").message)
    elif cmd == "switch" and len(argv) > 1:
        print(switch_profile(argv[1]).message)
    elif cmd == "summary":  # one line, for run_pupper.sh
        data = load()
        prof = active_profile(data)
        pending = [r for r in data["reminders"] if not r.get("done")]
        print(f"talking with {data['active']} ({len(prof['facts'])} facts, {len(prof['objects'])} things);"
              f" {len(pending)} reminder(s); routines: {', '.join(data['routines']) or 'none'}")
    else:
        print(__doc__.split("Easy reset")[1].split("Same two halves")[0].strip())
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
